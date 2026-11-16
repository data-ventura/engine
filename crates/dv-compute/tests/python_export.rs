//! Exports workspace pipelines as polars scripts, runs them with Python on the table fixtures and
//! checks every value against the engine's own result: DataFusion for sources, derived fields and
//! selections, and dv-query's key and time joins. The workspaces are `fixtures/spec/export.dvspec`
//! and `fixtures/spec/example.dvspec`.
//!
//! Python comes from `dv_py::testing::python` (a uv venv with polars, or `DV_PYTHON`). Without it
//! the tests skip, except under CI, where they fail.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch};
use arrow::compute::{cast, concat_batches};
use arrow::datatypes::{DataType, Float64Type, Int32Type, Int64Type, TimeUnit};
use datafusion::prelude::{DataFrame, ident};
use dv_compute::export::fixed;
use dv_compute::{Context, Export, ExportError, compile};
use dv_data::budget::Budget;
use dv_data::table::Table;
use dv_model::field::{Dataset, Family};
use dv_model::profile::Value;
use dv_model::selection::{Predicate, Selection};
use dv_query::Engine;
use dv_query::join::{Aggregate, Duplicates, KeyJoin, key_join};
use dv_query::time::{Direction, TimeJoin, time_join};
use dv_spec::Workspace;
use dv_spec::workspace::Join;

fn spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/spec")
}

fn python() -> Option<PathBuf> {
    let py = dv_py::testing::python();
    if py.is_none() {
        assert!(
            std::env::var_os("CI").is_none(),
            "CI needs Python with polars: install uv or set DV_PYTHON"
        );
        eprintln!("skipping: no Python (install uv or set DV_PYTHON)");
    }
    py
}

/// A workspace, its sources opened, and an engine with them registered.
struct Loaded {
    workspace: Workspace,
    base: PathBuf,
    datasets: BTreeMap<String, Dataset>,
    engine: Engine,
}

async fn load(file: &str) -> Loaded {
    let base = spec_dir();
    let workspace = Workspace::from_toml(&std::fs::read_to_string(base.join(file)).unwrap())
        .unwrap_or_else(|e| panic!("{file}: {e}"));
    let engine = Engine::new(Budget::new(1 << 30)).unwrap();
    let mut datasets = BTreeMap::new();
    for (name, source) in &workspace.sources {
        if source.url.contains("://") {
            continue;
        }
        let table = Table::open(base.join(&source.url)).unwrap();
        datasets.insert(
            name.clone(),
            Dataset::from_schema(name, Family::Table, table.schema()),
        );
        engine.register(name, &table).await.unwrap();
    }
    Loaded {
        workspace,
        base,
        datasets,
        engine,
    }
}

impl Loaded {
    fn export(&self, target: &str, selection: Option<&Selection>) -> Result<String, ExportError> {
        Export {
            workspace: &self.workspace,
            datasets: &self.datasets,
            base: &self.base,
            selection,
        }
        .python(target)
    }

    /// The engine's result for a source or join, the way the app computes it.
    async fn frame(&self, name: &str) -> dv_query::Result<DataFrame> {
        let ws = &self.workspace;
        if ws.sources.contains_key(name) {
            let dataset = fixed(ws, name, &self.datasets[name]);
            let ctx = Context::new(&dataset);
            let mut cols: Vec<_> = dataset.fields.iter().map(|f| ident(&f.name)).collect();
            for (field, d) in ws.fields.iter().filter(|(_, d)| d.source == name) {
                let f = compile(&d.formula, &ctx).unwrap_or_else(|e| panic!("{field}: {e}"));
                cols.push(f.expr.alias(field));
            }
            return Ok(self.engine.table(name).await?.select(cols)?);
        }
        match &ws.joins[name] {
            Join::Key {
                left,
                right,
                left_keys,
                right_keys,
                duplicates,
                order_by,
            } => {
                let order_by = || order_by.clone().unwrap();
                let duplicates = duplicates.as_deref().map(|d| match d {
                    "keep_all" => Duplicates::KeepAll,
                    "first" => Duplicates::First {
                        order_by: order_by(),
                    },
                    "last" => Duplicates::Last {
                        order_by: order_by(),
                    },
                    "mean" => Duplicates::Aggregate(Aggregate::Mean),
                    "sum" => Duplicates::Aggregate(Aggregate::Sum),
                    "min" => Duplicates::Aggregate(Aggregate::Min),
                    "max" => Duplicates::Aggregate(Aggregate::Max),
                    _ => Duplicates::Aggregate(Aggregate::Count),
                });
                let spec = KeyJoin {
                    left_keys: left_keys.clone(),
                    right_keys: right_keys.clone(),
                    duplicates,
                };
                let (l, r) = (
                    Box::pin(self.frame(left)).await?,
                    Box::pin(self.frame(right)).await?,
                );
                Ok(key_join(l, r, &spec).await?.0)
            }
            Join::Time {
                left,
                right,
                left_time,
                right_time,
                tolerance,
                direction,
                match_on,
            } => {
                let spec = TimeJoin {
                    left_time: left_time.clone(),
                    right_time: right_time.clone(),
                    tolerance_us: dv_spec::parse_duration(tolerance).unwrap(),
                    direction: match direction.as_str() {
                        "backward" => Direction::Backward,
                        "forward" => Direction::Forward,
                        _ => Direction::Nearest,
                    },
                    by: match_on
                        .iter()
                        .map(|[l, r]| (l.clone(), r.clone()))
                        .collect(),
                };
                let (l, r) = (
                    Box::pin(self.frame(left)).await?,
                    Box::pin(self.frame(right)).await?,
                );
                Ok(time_join(&self.engine, l, r, &spec).await?.0)
            }
        }
    }

    async fn engine_result(
        &self,
        target: &str,
        selection: Option<&Selection>,
    ) -> dv_query::Result<RecordBatch> {
        let mut df = self.frame(target).await?;
        if let Some(s) = selection {
            let schema = df.schema().as_arrow().clone();
            if let Some(filter) = dv_query::selection_expr(s, &schema)? {
                df = df.filter(filter)?;
            }
        }
        let schema = df.schema().as_arrow().clone();
        let batches = df.collect().await?;
        Ok(concat_batches(&std::sync::Arc::new(schema), &batches)?)
    }
}

/// Runs a script; on success, reads what it wrote. On failure, returns its stderr.
fn run(py: &Path, script: &str, name: &str) -> Result<RecordBatch, String> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("python-export");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{name}.py"));
    let out = dir.join(format!("{name}.parquet"));
    let _ = std::fs::remove_file(&out);
    std::fs::write(&file, script).unwrap();
    let res = Command::new(py)
        .arg("-I")
        .arg(&file)
        .arg(&out)
        .output()
        .unwrap();
    if !res.status.success() {
        return Err(String::from_utf8_lossy(&res.stderr).into_owned());
    }
    let table = Table::open(&out).unwrap();
    let batches = table
        .batches()
        .unwrap()
        .collect::<dv_data::Result<Vec<_>>>()
        .unwrap();
    Ok(concat_batches(table.schema(), &batches).unwrap())
}

/// A value, normalized so types that differ only in width or encoding compare equal.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
enum Cell {
    Null,
    Bool(bool),
    Num(f64),
    Text(String),
}

fn cells(c: &ArrayRef) -> Vec<Cell> {
    let n = c.len();
    let col: Vec<Cell> = match c.data_type() {
        DataType::Boolean => c
            .as_boolean()
            .iter()
            .map(|v| v.map(Cell::Bool))
            .collect::<Vec<_>>(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            let s = cast(c, &DataType::Utf8).unwrap();
            s.as_string::<i32>()
                .iter()
                .map(|v| v.map(|v| Cell::Text(v.into())))
                .collect()
        }
        DataType::Timestamp(_, _) => {
            let us = cast(c, &DataType::Timestamp(TimeUnit::Microsecond, None)).unwrap();
            let us = cast(&us, &DataType::Int64).unwrap();
            us.as_primitive::<Int64Type>()
                .iter()
                .map(|v| v.map(|v| Cell::Num(v as f64)))
                .collect()
        }
        DataType::Date32 | DataType::Date64 => {
            let d = cast(c, &DataType::Date32).unwrap();
            let d = cast(&d, &DataType::Int32).unwrap();
            d.as_primitive::<Int32Type>()
                .iter()
                .map(|v| v.map(|v| Cell::Num(v as f64)))
                .collect()
        }
        dt if dt.is_numeric() => {
            let f = cast(c, &DataType::Float64).unwrap();
            f.as_primitive::<Float64Type>()
                .iter()
                .map(|v| v.map(Cell::Num))
                .collect()
        }
        other => panic!("can't compare {other} columns"),
    }
    .into_iter()
    .map(|v| v.unwrap_or(Cell::Null))
    .collect();
    assert_eq!(col.len(), n);
    col
}

/// Rows in a canonical order, since neither side promises one.
fn rows(b: &RecordBatch) -> Vec<Vec<Cell>> {
    let cols: Vec<Vec<Cell>> = b.columns().iter().map(cells).collect();
    let mut rows: Vec<Vec<Cell>> = (0..b.num_rows())
        .map(|i| cols.iter().map(|c| c[i].clone()).collect())
        .collect();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    rows
}

fn assert_same(what: &str, engine: &RecordBatch, python: &RecordBatch) {
    let names = |b: &RecordBatch| -> Vec<String> {
        b.schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    };
    assert_eq!(names(python), names(engine), "{what}: columns");
    assert_eq!(python.num_rows(), engine.num_rows(), "{what}: rows");
    let columns = names(engine);
    for (i, (e, p)) in rows(engine).iter().zip(rows(python)).enumerate() {
        for ((e, p), name) in e.iter().zip(&p).zip(&columns) {
            let ok = match (e, p) {
                (Cell::Num(x), Cell::Num(y)) => {
                    x == y || (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-9 * x.abs().max(1.0)
                }
                (e, p) => e == p,
            };
            assert!(ok, "{what}, row {i}, {name}: engine {e:?}, python {p:?}");
        }
    }
}

#[tokio::test]
async fn exported_scripts_match_the_engine() {
    let Some(py) = python() else { return };
    let w = load("export.dvspec").await;
    let targets = [
        "trips",
        "obs",
        "trip_types",
        "fee_all",
        "fee_first",
        "fee_last",
        "fee_mean",
        "fee_count",
        "trip_weather",
        "visit_before",
        "visit_after",
    ];
    for target in targets {
        let script = w.export(target, None).unwrap();
        let python = run(&py, &script, target)
            .unwrap_or_else(|e| panic!("{target}: the script failed:\n{e}\n{script}"));
        let engine = w.engine_result(target, None).await.unwrap();
        assert!(engine.num_rows() > 0, "{target}: no rows");
        assert_same(target, &engine, &python);
    }
}

#[tokio::test]
async fn selections_match_the_engine() {
    let Some(py) = python() else { return };
    let w = load("export.dvspec").await;
    // 2015-01-05 to 2015-01-20, in microseconds.
    let (from, to) = (1_420_416_000e6, 1_421_712_000e6);
    let selection = Selection {
        predicates: vec![
            Predicate::Range {
                field: "fare".into(),
                min: Some(5.0),
                max: Some(20.0),
            },
            Predicate::Range {
                field: "pickup".into(),
                min: Some(from),
                max: Some(to),
            },
            Predicate::Range {
                field: "change".into(),
                min: None,
                max: Some(0.0),
            },
            Predicate::OneOf {
                field: "name".into(),
                values: vec![
                    Value::Text("Cash".into()),
                    Value::Text("Credit card".into()),
                ],
            },
            Predicate::NotNull {
                field: "temp".into(),
            },
        ],
        generation: 1,
    };
    let script = w.export("trip_weather", Some(&selection)).unwrap();
    let python = run(&py, &script, "trip_weather_selected").unwrap();
    let engine = w
        .engine_result("trip_weather", Some(&selection))
        .await
        .unwrap();
    assert!(engine.num_rows() > 10, "the selection kept too little");
    assert_same("trip_weather, selected", &engine, &python);
}

#[tokio::test]
async fn ambiguous_joins_are_refused_by_both() {
    let Some(py) = python() else { return };
    let w = load("export.dvspec").await;
    let engine = w.engine_result("fee_ambiguous", None).await.unwrap_err();
    let script = w.export("fee_ambiguous", None).unwrap();
    let python = run(&py, &script, "fee_ambiguous").unwrap_err();
    // Both count the trips paid by card: payment 1 has three fees, 3 has two.
    let message = engine.to_string();
    assert!(message.contains("301 of 500 rows"), "{message}");
    assert!(python.contains("ValueError: 301 of 500 rows"), "{python}");
}

#[tokio::test]
async fn the_example_workspace_exports() {
    let Some(py) = python() else { return };
    let w = load("example.dvspec").await;
    let script = w.export("trip_weather", None).unwrap();
    assert!(script.contains("# Fixed in the workspace: payment is a category; distance is in mi."));
    let python = run(&py, &script, "example").unwrap();
    let engine = w.engine_result("trip_weather", None).await.unwrap();
    assert_same("example trip_weather", &engine, &python);
    // The remote source isn't needed for this target, but can't be exported on its own.
    let mut datasets = w.datasets.clone();
    datasets.insert(
        "era5".into(),
        Dataset::from_schema("era5", Family::Array, &arrow::datatypes::Schema::empty()),
    );
    let err = Export {
        workspace: &w.workspace,
        datasets: &datasets,
        base: &w.base,
        selection: None,
    }
    .python("era5")
    .unwrap_err();
    assert!(matches!(err, ExportError::Unsupported { .. }), "{err}");
}
