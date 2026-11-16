//! Exports a workspace's pipeline as a Python script, so anyone can reproduce a result without
//! the app.
//!
//! The script uses polars and reads the same files: it loads each source the target needs, adds
//! the derived fields, makes the joins with the engine's rules and applies a selection, then
//! writes the result to Parquet (or prints it). Formulas become polars expressions with the same
//! meaning (see [`crate::formula`]), and joins call small functions defined in the script that
//! follow `dv-query`'s rules: every left row is kept, null keys never match, an ambiguous key
//! join is refused, right key columns are dropped and clashing names get `_right`.
//!
//! Only local table files that polars reads without extra packages can be exported for now:
//! CSV, TSV, JSON, NDJSON, Parquet and Arrow IPC. Text formats are read with the column types the
//! engine inferred, so both sides see the same types.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use arrow::datatypes::{DataType, TimeUnit};
use dv_model::field::{Dataset, Role, RoleSource};
use dv_model::profile::Value;
use dv_model::selection::{Predicate, Selection};
use dv_model::units::Unit;
use dv_spec::workspace::{Join, Source, Workspace};

use crate::{Context, FormulaError, compile};

/// A pipeline that can't be exported.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExportError {
    #[error("no source or join {0:?}")]
    Missing(String),
    #[error("{0} depends on itself")]
    Cycle(String),
    #[error("{at}: {error}")]
    Formula { at: String, error: FormulaError },
    #[error("{at}: {message}")]
    Unsupported { at: String, message: String },
}

/// What to export: a workspace, the datasets its sources opened as, and where its files are.
#[derive(Debug, Clone, Copy)]
pub struct Export<'a> {
    pub workspace: &'a Workspace,
    /// Each source's dataset as opened, before the workspace's fixes.
    pub datasets: &'a BTreeMap<String, Dataset>,
    /// The folder relative source paths are under: usually the workspace file's.
    pub base: &'a Path,
    /// Rows to keep from the target, if not all of them.
    pub selection: Option<&'a Selection>,
}

impl Export<'_> {
    /// A polars script that computes `target`, a source or join of the workspace.
    pub fn python(&self, target: &str) -> Result<String, ExportError> {
        let mut order = Vec::new();
        self.visit(target, &mut Vec::new(), &mut order)?;
        let mut names = Names::default();
        let vars: BTreeMap<&str, String> =
            order.iter().map(|n| (n.as_str(), names.var(n))).collect();
        let used = order.iter().filter_map(|n| self.workspace.joins.get(n));
        let key_joins = used.clone().any(|j| matches!(j, Join::Key { .. }));
        let time_joins = used.clone().any(|j| matches!(j, Join::Time { .. }));
        let ranges = self.selection.is_some_and(|s| {
            s.predicates
                .iter()
                .any(|p| matches!(p, Predicate::Range { .. }))
        });

        let mut body = String::new();
        for name in &order {
            let var = &vars[name.as_str()];
            match (
                self.workspace.sources.get(name),
                self.workspace.joins.get(name),
            ) {
                (Some(source), _) => self.source(&mut body, name, var, source)?,
                (None, Some(join)) => {
                    let (left, right) = match join {
                        Join::Key { left, right, .. } | Join::Time { left, right, .. } => {
                            (&vars[left.as_str()], &vars[right.as_str()])
                        }
                    };
                    writeln!(body, "{var} = {}", join_call(name, join, left, right)?).unwrap();
                    body.push('\n');
                }
                (None, None) => return Err(ExportError::Missing(name.clone())),
            }
        }
        let mut result = vars[target].clone();
        if let Some(s) = self.selection.filter(|s| !s.is_empty()) {
            let preds: Vec<String> = s.predicates.iter().map(|p| predicate(p, &result)).collect();
            let selected = names.var(&format!("{target}_selected"));
            writeln!(
                body,
                "# The selection.\n{selected} = {result}.filter(\n    {},\n)\n",
                preds.join(",\n    ")
            )
            .unwrap();
            result = selected;
        }

        let mut out = String::new();
        writeln!(
            out,
            "\"\"\"{} from a Data Ventura workspace, with polars.\n",
            py_doc(target)
        )
        .unwrap();
        out.push_str(
            "Exported by the Data Ventura engine. Needs polars 1.31 or later. Run:\n\n    \
             python this-script.py [result.parquet]\n\n\
             With a path, the result is written there as Parquet; without one, it's printed.\n\
             \"\"\"\n\n",
        );
        out.push_str("import sys\n");
        if time_joins {
            out.push_str("from datetime import timedelta\n");
        }
        out.push_str("from pathlib import Path\n\nimport polars as pl\n\n");
        writeln!(
            out,
            "# The workspace's folder: relative source paths are under it.\nBASE = Path({})\n",
            py_str(&self.base.to_string_lossy())
        )
        .unwrap();
        if key_joins {
            out.push_str(KEY_JOIN);
        }
        if time_joins {
            out.push_str(TIME_JOIN);
        }
        if ranges {
            out.push_str(IN_RANGE);
        }
        out.push_str(&body);
        writeln!(
            out,
            "if __name__ == \"__main__\":\n    if len(sys.argv) > 1:\n        \
             {result}.write_parquet(sys.argv[1])\n    else:\n        print({result})"
        )
        .unwrap();
        Ok(out)
    }

    /// Adds `name` and what it depends on to `order`, dependencies first.
    fn visit(
        &self,
        name: &str,
        path: &mut Vec<String>,
        order: &mut Vec<String>,
    ) -> Result<(), ExportError> {
        if order.iter().any(|n| n == name) {
            return Ok(());
        }
        if path.iter().any(|n| n == name) {
            return Err(ExportError::Cycle(name.into()));
        }
        if !self.workspace.sources.contains_key(name) {
            let Some(join) = self.workspace.joins.get(name) else {
                return Err(ExportError::Missing(name.into()));
            };
            path.push(name.into());
            let (Join::Key { left, right, .. } | Join::Time { left, right, .. }) = join;
            self.visit(left, path, order)?;
            self.visit(right, path, order)?;
            path.pop();
        }
        order.push(name.into());
        Ok(())
    }

    fn source(
        &self,
        out: &mut String,
        name: &str,
        var: &str,
        source: &Source,
    ) -> Result<(), ExportError> {
        let at = format!("sources.{name}");
        let unsupported = |message: String| ExportError::Unsupported {
            at: at.clone(),
            message,
        };
        let opened = self
            .datasets
            .get(name)
            .ok_or_else(|| ExportError::Missing(name.into()))?;
        let dataset = fixed(self.workspace, name, opened);
        if source.connection.is_some() || source.url.contains("://") {
            return Err(unsupported(
                "only local files can be exported to Python for now".into(),
            ));
        }
        let format = source.format.clone().unwrap_or_else(|| {
            Path::new(&source.url)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
        });
        let (reader, typed) = match format.as_str() {
            "csv" => ("read_csv", true),
            "tsv" | "tab" => ("read_csv", true),
            "ndjson" | "jsonl" => ("read_ndjson", true),
            "json" => ("read_json", true),
            "parquet" | "pq" => ("read_parquet", false),
            "arrow" | "feather" | "ipc" => ("read_ipc", false),
            "arrows" => ("read_ipc_stream", false),
            other => {
                return Err(unsupported(format!(
                    "{other:?} files can't be exported to Python yet"
                )));
            }
        };
        if source.part.is_some() {
            return Err(unsupported(
                "a part of a file can't be exported to Python yet".into(),
            ));
        }
        let path = if Path::new(&source.url).is_absolute() {
            format!("Path({})", py_str(&source.url))
        } else {
            format!("BASE / {}", py_str(&source.url))
        };
        let mut args = vec![path];
        if matches!(format.as_str(), "tsv" | "tab") {
            args.push("separator=\"\\t\"".into());
        }
        if typed {
            let types: Vec<String> = dataset
                .fields
                .iter()
                .filter_map(|f| {
                    polars_type(&f.data_type).map(|t| format!("{}: {t}", py_str(&f.name)))
                })
                .collect();
            if !types.is_empty() {
                args.push(format!(
                    "schema_overrides={{\n        {},\n    }}",
                    types.join(",\n        ")
                ));
            }
        }
        writeln!(
            out,
            "{var} = pl.{reader}(\n    {},\n)",
            args.join(",\n    ")
        )
        .unwrap();
        let notes: Vec<String> = self
            .workspace
            .fixes
            .iter()
            .filter(|f| f.source == name)
            .filter_map(|f| {
                let what: Vec<String> = f
                    .role
                    .iter()
                    .map(|r| format!("is a {r}"))
                    .chain(f.unit.iter().map(|u| format!("is in {u}")))
                    .collect();
                (!what.is_empty()).then(|| format!("{} {}", f.field, what.join(" and ")))
            })
            .collect();
        if !notes.is_empty() {
            writeln!(out, "# Fixed in the workspace: {}.", notes.join("; ")).unwrap();
        }

        let ctx = Context::new(&dataset);
        let mut derived = Vec::new();
        for (field, d) in self
            .workspace
            .fields
            .iter()
            .filter(|(_, d)| d.source == name)
        {
            let formula = compile(&d.formula, &ctx).map_err(|error| ExportError::Formula {
                at: format!("fields.{field}"),
                error,
            })?;
            derived.push(format!(
                "    # {}\n    ({}).alias({}),",
                d.formula.replace('\n', " "),
                formula.python,
                py_str(field)
            ));
        }
        if !derived.is_empty() {
            writeln!(
                out,
                "{var} = {var}.with_columns(\n{}\n)",
                derived.join("\n")
            )
            .unwrap();
        }
        out.push('\n');
        Ok(())
    }
}

/// A source's dataset with the workspace's role and unit fixes applied.
pub fn fixed(workspace: &Workspace, source: &str, dataset: &Dataset) -> Dataset {
    let mut d = dataset.clone();
    for fix in workspace.fixes.iter().filter(|f| f.source == source) {
        let Some(field) = d.fields.iter_mut().find(|f| f.name == fix.field) else {
            continue;
        };
        if let Some(role) = fix.role.as_deref().and_then(role) {
            field.role = role;
            field.role_source = RoleSource::Fixed;
        }
        if let Some(u) = &fix.unit {
            field.unit = Unit::parse(u).ok();
            field.unit_text = Some(u.clone());
        }
    }
    d
}

fn role(name: &str) -> Option<Role> {
    Some(match name {
        "time" => Role::Time,
        "latitude" => Role::Latitude,
        "longitude" => Role::Longitude,
        "x" => Role::X,
        "y" => Role::Y,
        "vertical" => Role::Vertical,
        "geometry" => Role::Geometry,
        "identifier" => Role::Identifier,
        "category" => Role::Category,
        "measure" => Role::Measure,
        "text" => Role::Text,
        "other" => Role::Other,
        _ => return None,
    })
}

fn join_call(name: &str, join: &Join, left: &str, right: &str) -> Result<String, ExportError> {
    let list = |v: &[String]| {
        let items: Vec<String> = v.iter().map(|s| py_str(s)).collect();
        format!("[{}]", items.join(", "))
    };
    Ok(match join {
        Join::Key {
            left_keys,
            right_keys,
            duplicates,
            order_by,
            ..
        } => {
            let mut call = format!(
                "key_join({left}, {right}, {}, {}",
                list(left_keys),
                list(right_keys)
            );
            if let Some(d) = duplicates {
                write!(call, ", duplicates={}", py_str(d)).unwrap();
            }
            if let Some(o) = order_by {
                write!(call, ", order_by={}", py_str(o)).unwrap();
            }
            call + ")"
        }
        Join::Time {
            left_time,
            right_time,
            tolerance,
            direction,
            match_on,
            ..
        } => {
            let us =
                dv_spec::parse_duration(tolerance).ok_or_else(|| ExportError::Unsupported {
                    at: format!("joins.{name}"),
                    message: format!("can't read the tolerance {tolerance:?}"),
                })?;
            let mut call = format!(
                "time_join({left}, {right}, {}, {}, timedelta(microseconds={us}), {}",
                py_str(left_time),
                py_str(right_time),
                py_str(direction)
            );
            if !match_on.is_empty() {
                let pairs: Vec<String> = match_on
                    .iter()
                    .map(|[l, r]| format!("({}, {})", py_str(l), py_str(r)))
                    .collect();
                write!(call, ", by=[{}]", pairs.join(", ")).unwrap();
            }
            call + ")"
        }
    })
}

/// A selection predicate as a polars filter on `frame`, with dv-model's meaning.
fn predicate(p: &Predicate, frame: &str) -> String {
    let c = format!("pl.col({})", py_str(p.field()));
    let bound = |b: &Option<f64>| b.map_or("None".into(), py_float);
    match p {
        Predicate::Range { field, min, max } => format!(
            "in_range({frame}, {}, {}, {})",
            py_str(field),
            bound(min),
            bound(max)
        ),
        Predicate::OneOf { values, .. } => {
            let items: Vec<String> = values
                .iter()
                .map(|v| match v {
                    Value::Integer(i) => i.to_string(),
                    Value::Float(f) => py_float(*f),
                    Value::Boolean(b) => (if *b { "True" } else { "False" }).into(),
                    Value::Text(s) => py_str(s),
                })
                .collect();
            format!("{c}.is_in([{}])", items.join(", "))
        }
        Predicate::IsNull { .. } => format!("{c}.is_null()"),
        Predicate::NotNull { .. } => format!("{c}.is_not_null()"),
    }
}

/// The polars type for an engine column type, when polars has the same one.
fn polars_type(dt: &DataType) -> Option<String> {
    Some(match dt {
        DataType::Boolean => "pl.Boolean".into(),
        DataType::Int8 => "pl.Int8".into(),
        DataType::Int16 => "pl.Int16".into(),
        DataType::Int32 => "pl.Int32".into(),
        DataType::Int64 => "pl.Int64".into(),
        DataType::UInt8 => "pl.UInt8".into(),
        DataType::UInt16 => "pl.UInt16".into(),
        DataType::UInt32 => "pl.UInt32".into(),
        DataType::UInt64 => "pl.UInt64".into(),
        DataType::Float32 => "pl.Float32".into(),
        DataType::Float64 => "pl.Float64".into(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "pl.String".into(),
        DataType::Date32 | DataType::Date64 => "pl.Date".into(),
        DataType::Timestamp(unit, tz) => {
            // polars has no seconds unit; microseconds hold the same instants.
            let unit = match unit {
                TimeUnit::Second | TimeUnit::Microsecond => "us",
                TimeUnit::Millisecond => "ms",
                TimeUnit::Nanosecond => "ns",
            };
            match tz {
                Some(tz) => format!("pl.Datetime({}, time_zone={})", py_str(unit), py_str(tz)),
                None => format!("pl.Datetime({})", py_str(unit)),
            }
        }
        _ => return None,
    })
}

/// Python variable names for sources and joins: valid identifiers, unique, and clear of the
/// script's own names.
#[derive(Default)]
struct Names {
    taken: HashSet<String>,
}

const RESERVED: [&str; 13] = [
    "pl",
    "sys",
    "Path",
    "BASE",
    "timedelta",
    "key_join",
    "time_join",
    "in_range",
    "None",
    "True",
    "False",
    "print",
    "len",
];

const KEYWORDS: [&str; 35] = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

impl Names {
    fn var(&mut self, name: &str) -> String {
        let mut v: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        if v.is_empty() || v.starts_with(|c: char| c.is_ascii_digit()) {
            v.insert(0, '_');
        }
        if RESERVED.contains(&v.as_str()) || KEYWORDS.contains(&v.as_str()) {
            v.push_str("_df");
        }
        let mut out = v.clone();
        let mut i = 2;
        while !self.taken.insert(out.clone()) {
            out = format!("{v}_{i}");
            i += 1;
        }
        out
    }
}

/// A Python string literal. JSON's escapes are all valid in Python.
pub(crate) fn py_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                write!(out, "\\u{:04x}", c as u32).unwrap();
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A Python float literal that reads back as the same number.
pub(crate) fn py_float(x: f64) -> String {
    if x.is_nan() {
        "float(\"nan\")".into()
    } else if x.is_infinite() {
        if x > 0.0 {
            "float(\"inf\")"
        } else {
            "-float(\"inf\")"
        }
        .into()
    } else {
        // Rust's shortest round-trip form, which Python reads as the same double.
        format!("{x:?}")
    }
}

/// Text that's safe inside a docstring.
fn py_doc(s: &str) -> String {
    s.replace('\\', "\\\\").replace("\"\"\"", "\\\"\\\"\\\"")
}

const KEY_JOIN: &str = r#"def key_join(left, right, left_on, right_on, duplicates=None, order_by=None):
    """Joins right onto left by equal keys, with the engine's rules.

    Every left row is kept, with nulls where nothing matches, and null keys never match. When a
    left row matches several right rows, `duplicates` says what to do: "keep_all", "first" or
    "last" by `order_by`, or "mean", "sum", "min", "max" or "count" of the other right columns
    (numeric ones only for "mean" and "sum"). Without it, such a join is refused. Right key
    columns are dropped, and other right columns whose names clash with the left's get "_right".
    """
    if duplicates is None:
        counts = right.group_by(right_on).len(name="__dv_n")
        matches = left.select(left_on).join(counts, left_on=left_on, right_on=right_on, how="left")
        ambiguous = matches.filter(pl.col("__dv_n") > 1).height
        if ambiguous:
            raise ValueError(
                f"{ambiguous} of {left.height} rows match more than one row; "
                "choose keep all, first, last or an aggregate"
            )
    elif duplicates in ("first", "last"):
        last = duplicates == "last"
        right = right.sort(order_by, descending=last, nulls_last=not last, maintain_order=True)
        right = right.unique(right_on, keep="first", maintain_order=True)
    elif duplicates != "keep_all":
        numeric = duplicates in ("mean", "sum")
        others = [
            c for c, t in right.schema.items()
            if c not in right_on and (not numeric or t.is_numeric())
        ]
        right = right.group_by(right_on).agg(getattr(pl.col(c), duplicates)() for c in others)
    keys = [f"__dv_key{i}" for i in range(len(right_on))]
    names = {
        c: keys[right_on.index(c)] if c in right_on else f"{c}_right" if c in left.columns else c
        for c in right.columns
    }
    right = right.rename(names)
    joined = left.join(right, left_on=left_on, right_on=keys, how="left", coalesce=False)
    return joined.drop(keys)


"#;

const TIME_JOIN: &str = r#"def time_join(left, right, left_time, right_time, tolerance, direction, by=()):
    """Joins right onto left by nearest time, with the engine's rules.

    Each left row takes the right row closest in time ("nearest"), the last at or before it
    ("backward") or the first at or after it ("forward"), only within `tolerance` (inclusive) and
    only among right rows whose `by` columns equal the left's (pairs of left and right names).
    Every left row is kept; rows with a null time or key match nothing. The right `by` columns are
    dropped, and other right columns whose names clash with the left's get "_right".
    """
    left_by = [l for l, _ in by]
    right_by = [r for _, r in by]
    keys = [f"__dv_by{i}" for i in range(len(by))]
    names = {
        c: f"{c}_right" if c in left.columns else c for c in right.columns if c not in right_by
    }
    right = right.select(
        pl.col(right_time).alias("__dv_t"),
        *[pl.col(r).alias(k) for r, k in zip(right_by, keys)],
        *[pl.col(c).alias(n) for c, n in names.items()],
    )
    right = right.drop_nulls(["__dv_t", *keys]).sort("__dv_t", maintain_order=True)
    left = left.with_row_index("__dv_row")
    usable = pl.all_horizontal(pl.col(c).is_not_null() for c in [left_time, *left_by])
    matched = (
        left.filter(usable)
        .with_columns(
            pl.col(left_time).alias("__dv_t"),
            *[pl.col(l).alias(k) for l, k in zip(left_by, keys)],
        )
        .sort("__dv_t", maintain_order=True)
        .join_asof(
            right,
            on="__dv_t",
            by=keys or None,
            strategy=direction,
            tolerance=tolerance,
        )
        .drop(["__dv_t", *keys])
    )
    rest = left.filter(~usable)
    return pl.concat([matched, rest], how="diagonal_relaxed").sort("__dv_row").drop("__dv_row")


"#;

const IN_RANGE: &str = r#"def in_range(frame, name, lo, hi):
    """Rows where lo <= name <= hi (either bound may be None), compared as numbers: timestamps as
    microseconds since the epoch and dates as days. Nulls and NaN never match."""
    dtype = frame.schema[name]
    if dtype == pl.Date:
        x = pl.col(name).cast(pl.Int32).cast(pl.Float64)
    elif isinstance(dtype, pl.Datetime):
        x = pl.col(name).dt.epoch("us").cast(pl.Float64)
    else:
        x = pl.col(name).cast(pl.Float64)
    keep = x.is_not_null() & x.is_not_nan()
    if lo is not None:
        keep = keep & (x >= lo)
    if hi is not None:
        keep = keep & (x <= hi)
    return keep


"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals() {
        assert_eq!(py_str("say \"hi\"\\\n"), r#""say \"hi\"\\\n""#);
        assert_eq!(py_float(7.0), "7.0");
        assert_eq!(py_float(0.1), "0.1");
        assert_eq!(py_float(-273.15), "-273.15");
        assert_eq!(py_float(1e300), "1e300");
    }

    #[test]
    fn names() {
        let mut n = Names::default();
        assert_eq!(n.var("trips"), "trips");
        assert_eq!(n.var("trip-types"), "trip_types");
        assert_eq!(n.var("trip types"), "trip_types_2");
        assert_eq!(n.var("2020"), "_2020");
        assert_eq!(n.var("pl"), "pl_df");
        assert_eq!(n.var("class"), "class_df");
    }
}
