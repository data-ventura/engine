//! Joins by key, with explicit rules and a report of what matched.
//!
//! A join never hides a bad match. Every left row is kept: unmatched rows get nulls and are
//! counted. Null keys never match. When a left row matches several right rows, the join needs a
//! [`Duplicates`] policy; without one it refuses and says how many rows are ambiguous, rather than
//! picking. Right key columns are dropped, and other right columns whose names clash with the
//! left's get a `_right` suffix.

use std::collections::HashSet;

use arrow::array::AsArray;
use arrow::datatypes::Int64Type;
use datafusion::common::JoinType;
use datafusion::functions_aggregate::expr_fn::{avg, count, max, min, sum};
use datafusion::functions_window::expr_fn::row_number;
use datafusion::logical_expr::{ExprFunctionExt, SortExpr, col, lit};
use datafusion::prelude::{DataFrame, Expr, ident};
use serde::Serialize;

use crate::Result;

/// What to do when a left row matches several right rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Duplicates {
    /// Keep every match: the left row repeats once per match.
    KeepAll,
    /// Keep the match with the smallest value of a column.
    First { order_by: String },
    /// Keep the match with the largest value of a column.
    Last { order_by: String },
    /// Combine the matches: every right column that isn't a key is aggregated. A mean or sum
    /// keeps only numeric columns.
    Aggregate(Aggregate),
}

/// How [`Duplicates::Aggregate`] combines matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregate {
    Mean,
    Sum,
    Min,
    Max,
    Count,
}

/// A join by equal keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyJoin {
    pub left_keys: Vec<String>,
    pub right_keys: Vec<String>,
    /// Required when any left row matches more than one right row.
    pub duplicates: Option<Duplicates>,
}

/// What a join matched, for the join inspector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct JoinReport {
    pub left_rows: u64,
    /// Left rows with at least one match.
    pub matched: u64,
    /// Left rows with no match, kept with nulls.
    pub unmatched: u64,
    /// Left rows with more than one match.
    pub ambiguous: u64,
}

/// A join that can't go ahead as specified.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JoinError {
    #[error(
        "{ambiguous} of {left_rows} rows match more than one row; choose keep all, first, last or an aggregate"
    )]
    Ambiguous { ambiguous: u64, left_rows: u64 },
    #[error("{0}")]
    Spec(String),
}

const MATCHES: &str = "__dv_matches";
const ROW: &str = "__dv_row";

/// Joins `right` onto `left` by key. Returns the lazy result and the match report; the report runs
/// a small aggregate query over the keys.
pub async fn key_join(
    left: DataFrame,
    right: DataFrame,
    spec: &KeyJoin,
) -> Result<(DataFrame, JoinReport)> {
    if spec.left_keys.is_empty() || spec.left_keys.len() != spec.right_keys.len() {
        return Err(
            JoinError::Spec("a join needs the same number of keys on each side".into()).into(),
        );
    }
    let left_keys: Vec<&str> = spec.left_keys.iter().map(String::as_str).collect();
    let right_keys: Vec<&str> = spec.right_keys.iter().map(String::as_str).collect();

    let report = report(&left, &right, &left_keys, &right_keys).await?;
    let right = match &spec.duplicates {
        None if report.ambiguous > 0 => {
            return Err(JoinError::Ambiguous {
                ambiguous: report.ambiguous,
                left_rows: report.left_rows,
            }
            .into());
        }
        None | Some(Duplicates::KeepAll) => right,
        Some(Duplicates::First { order_by }) => pick(right, &right_keys, order_by, true)?,
        Some(Duplicates::Last { order_by }) => pick(right, &right_keys, order_by, false)?,
        Some(Duplicates::Aggregate(agg)) => aggregate(right, &right_keys, *agg)?,
    };

    // Rename right columns: keys to private names (dropped after the join), clashes to `_right`.
    let left_names: HashSet<String> = left
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let key_alias = |i: usize| format!("__dv_key{i}");
    let renamed: Vec<Expr> = right
        .schema()
        .fields()
        .iter()
        .map(|f| {
            let name = f.name();
            let out = match right_keys.iter().position(|k| k == name) {
                Some(i) => key_alias(i),
                None if left_names.contains(name) => format!("{name}_right"),
                None => name.clone(),
            };
            ident(name).alias(out)
        })
        .collect();
    let right = right.select(renamed)?;
    let aliases: Vec<String> = (0..right_keys.len()).map(key_alias).collect();
    let alias_refs: Vec<&str> = aliases.iter().map(String::as_str).collect();
    let joined = left.join(right, JoinType::Left, &left_keys, &alias_refs, None)?;
    let keep: Vec<Expr> = joined
        .schema()
        .fields()
        .iter()
        .filter(|f| !aliases.contains(f.name()))
        .map(|f| ident(f.name()))
        .collect();
    Ok((joined.select(keep)?, report))
}

async fn report(
    left: &DataFrame,
    right: &DataFrame,
    left_keys: &[&str],
    right_keys: &[&str],
) -> Result<JoinReport> {
    let aliases: Vec<String> = (0..right_keys.len())
        .map(|i| format!("__dv_rk{i}"))
        .collect();
    let counts = right.clone().aggregate(
        right_keys
            .iter()
            .zip(&aliases)
            .map(|(k, a)| ident(*k).alias(a))
            .collect(),
        vec![count(lit(1)).alias(MATCHES)],
    )?;
    let alias_refs: Vec<&str> = aliases.iter().map(String::as_str).collect();
    let keys_only = left
        .clone()
        .select(left_keys.iter().map(|k| ident(*k)).collect::<Vec<_>>())?;
    let matched = keys_only.join(counts, JoinType::Left, left_keys, &alias_refs, None)?;
    let n = col(MATCHES);
    let summary = matched.aggregate(
        vec![],
        vec![
            count(lit(1)).alias("rows"),
            count(n.clone()).alias("matched"),
            sum(n.gt(lit(1)).cast_to_int()).alias("ambiguous"),
        ],
    )?;
    let batches = summary.collect().await?;
    let b = &batches[0];
    let get = |i: usize| -> u64 {
        let c = arrow::compute::cast(b.column(i), &arrow::datatypes::DataType::Int64)
            .expect("counts cast to Int64");
        c.as_primitive::<Int64Type>()
            .iter()
            .next()
            .flatten()
            .unwrap_or(0) as u64
    };
    let (rows, matched) = (get(0), get(1));
    Ok(JoinReport {
        left_rows: rows,
        matched,
        unmatched: rows - matched,
        ambiguous: get(2),
    })
}

trait CastToInt {
    fn cast_to_int(self) -> Expr;
}

impl CastToInt for Expr {
    fn cast_to_int(self) -> Expr {
        datafusion::logical_expr::cast(self, arrow::datatypes::DataType::Int64)
    }
}

/// One right row per key: the first or last by `order_by`. Ties keep an arbitrary one of them.
fn pick(right: DataFrame, keys: &[&str], order_by: &str, first: bool) -> Result<DataFrame> {
    let names: Vec<String> = right
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let order = SortExpr::new(ident(order_by), first, !first);
    let rn = row_number()
        .partition_by(keys.iter().map(|k| ident(*k)).collect())
        .order_by(vec![order])
        .build()?
        .alias(ROW);
    let ranked = right.window(vec![rn])?.filter(col(ROW).eq(lit(1u64)))?;
    Ok(ranked.select(names.iter().map(ident).collect::<Vec<_>>())?)
}

/// One right row per key, with the other columns aggregated (numeric ones only for mean and sum).
fn aggregate(right: DataFrame, keys: &[&str], agg: Aggregate) -> Result<DataFrame> {
    let numeric_only = matches!(agg, Aggregate::Mean | Aggregate::Sum);
    let others: Vec<String> = right
        .schema()
        .fields()
        .iter()
        .filter(|f| !keys.contains(&f.name().as_str()))
        .filter(|f| !numeric_only || f.data_type().is_numeric())
        .map(|f| f.name().clone())
        .collect();
    let aggs = others
        .iter()
        .map(|n| {
            let c = ident(n);
            let e = match agg {
                Aggregate::Mean => avg(c),
                Aggregate::Sum => sum(c),
                Aggregate::Min => min(c),
                Aggregate::Max => max(c),
                Aggregate::Count => count(c),
            };
            e.alias(n)
        })
        .collect();
    Ok(right.aggregate(keys.iter().map(|k| ident(*k)).collect(), aggs)?)
}
