//! Joins by nearest time, within a tolerance.
//!
//! Each left row takes the right row closest in time (or the closest before or after it), only
//! if it's within the tolerance, and optionally only among right rows with the same `by` keys.
//! There's no default tolerance: a nearest match without a limit would match anything. Rows
//! outside it, rows with a null time or key, and rows with no candidate are kept with nulls and
//! counted. Naive timestamps can't join zoned ones: one side must be given a time zone first.
//!
//! Both sides are read into memory and matched by binary search, so this suits joining a table
//! to another of similar or smaller size; the result is an in-memory table.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, UInt32Array};
use arrow::compute::{cast, concat_batches, take};
use arrow::datatypes::{DataType, Field, Int64Type, Schema, TimeUnit};
use datafusion::prelude::DataFrame;
use serde::Serialize;

use crate::join::JoinError;
use crate::{Engine, Result};

/// Which right rows a left row may match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// The closest in either direction; a tie goes to the earlier row.
    Nearest,
    /// The last at or before the left time.
    Backward,
    /// The first at or after the left time.
    Forward,
}

/// A join by nearest time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeJoin {
    pub left_time: String,
    pub right_time: String,
    /// The largest gap a match may have, in microseconds (inclusive).
    pub tolerance_us: i64,
    pub direction: Direction,
    /// Pairs of (left, right) columns that must also be equal.
    pub by: Vec<(String, String)>,
}

/// What a time join matched, for the join inspector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TimeJoinReport {
    pub left_rows: u64,
    pub matched: u64,
    /// Left rows kept with nulls: no candidate within the tolerance, or a null time or key.
    pub unmatched: u64,
    /// Mean gap between matched times, in microseconds.
    pub mean_gap_us: Option<f64>,
    /// Largest gap between matched times, in microseconds.
    pub max_gap_us: Option<i64>,
}

fn spec_err(msg: String) -> crate::Error {
    JoinError::Spec(msg).into()
}

/// A time column as microseconds since the epoch (UTC for zoned columns), and its zone.
type Times = (Vec<Option<i64>>, Option<Arc<str>>);

fn micros(batch: &RecordBatch, name: &str) -> Result<Times> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| spec_err(format!("no time column {name:?}")))?;
    let tz = match col.data_type() {
        DataType::Timestamp(_, tz) => tz.clone(),
        DataType::Date32 | DataType::Date64 => None,
        other => return Err(spec_err(format!("{name:?} is {other}, not a time"))),
    };
    let us = cast(col, &DataType::Timestamp(TimeUnit::Microsecond, tz.clone()))?;
    let us = cast(&us, &DataType::Int64)?;
    Ok((us.as_primitive::<Int64Type>().iter().collect(), tz))
}

/// Each row's `by` key as text, or `None` if any part is null (null keys never match).
fn keys(batch: &RecordBatch, names: &[&str]) -> Result<Vec<Option<String>>> {
    let cols: Vec<ArrayRef> = names
        .iter()
        .map(|n| {
            let c = batch
                .column_by_name(n)
                .ok_or_else(|| spec_err(format!("no column {n:?}")))?;
            Ok(cast(c, &DataType::Utf8)?)
        })
        .collect::<Result<_>>()?;
    Ok((0..batch.num_rows())
        .map(|i| {
            let mut key = String::new();
            for c in &cols {
                if c.is_null(i) {
                    return None;
                }
                key.push_str(c.as_string::<i32>().value(i));
                key.push('\u{1f}');
            }
            Some(key)
        })
        .collect())
}

/// Joins `right` onto `left` by nearest time. See the module docs for the rules.
pub async fn time_join(
    engine: &Engine,
    left: DataFrame,
    right: DataFrame,
    spec: &TimeJoin,
) -> Result<(DataFrame, TimeJoinReport)> {
    if spec.tolerance_us < 0 {
        return Err(spec_err("the tolerance can't be negative".into()));
    }
    let left_schema = Arc::new(left.schema().as_arrow().clone());
    let right_schema = Arc::new(right.schema().as_arrow().clone());
    let left = concat_batches(&left_schema, &left.collect().await?)?;
    let right = concat_batches(&right_schema, &right.collect().await?)?;

    let (lt, ltz) = micros(&left, &spec.left_time)?;
    let (rt, rtz) = micros(&right, &spec.right_time)?;
    if ltz.is_some() != rtz.is_some() {
        let naive = if ltz.is_none() {
            &spec.left_time
        } else {
            &spec.right_time
        };
        return Err(spec_err(format!(
            "{naive:?} has no time zone but the other side does; give it a time zone (or declare it UTC) first"
        )));
    }
    let left_by: Vec<&str> = spec.by.iter().map(|(l, _)| l.as_str()).collect();
    let right_by: Vec<&str> = spec.by.iter().map(|(_, r)| r.as_str()).collect();
    let lk = keys(&left, &left_by)?;
    let rk = keys(&right, &right_by)?;

    // Right rows by key, sorted by time; rows with equal times stay in input order.
    let mut groups: HashMap<String, Vec<(i64, u32)>> = HashMap::new();
    for (i, (t, k)) in rt.iter().zip(&rk).enumerate() {
        if let (Some(t), Some(k)) = (t, k) {
            groups.entry(k.clone()).or_default().push((*t, i as u32));
        }
    }
    for g in groups.values_mut() {
        g.sort_by_key(|(t, _)| *t);
    }

    let mut picks = Vec::with_capacity(left.num_rows());
    let (mut matched, mut gap_sum, mut gap_max) = (0u64, 0f64, 0i64);
    for (t, k) in lt.iter().zip(&lk) {
        let pick = match (t, k.as_ref().and_then(|k| groups.get(k))) {
            (Some(t), Some(g)) => nearest(g, *t, spec.direction, spec.tolerance_us),
            _ => None,
        };
        if let Some((row, gap)) = pick {
            matched += 1;
            gap_sum += gap as f64;
            gap_max = gap_max.max(gap);
            picks.push(Some(row));
        } else {
            picks.push(None);
        }
    }

    // Right columns, minus the `by` keys; clashing names get `_right`.
    let indices = UInt32Array::from(picks);
    let left_names: Vec<&str> = left_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    let mut fields: Vec<Field> = left_schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let mut columns: Vec<ArrayRef> = left.columns().to_vec();
    for (f, c) in right_schema.fields().iter().zip(right.columns()) {
        if right_by.contains(&f.name().as_str()) {
            continue;
        }
        let name = if left_names.contains(&f.name().as_str()) {
            format!("{}_right", f.name())
        } else {
            f.name().clone()
        };
        fields.push(Field::new(name, f.data_type().clone(), true));
        columns.push(take(c, &indices, None)?);
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
    let rows = left.num_rows() as u64;
    let report = TimeJoinReport {
        left_rows: rows,
        matched,
        unmatched: rows - matched,
        mean_gap_us: (matched > 0).then(|| gap_sum / matched as f64),
        max_gap_us: (matched > 0).then_some(gap_max),
    };
    Ok((engine.context().read_batch(batch)?, report))
}

/// The right row to match at time `t` in a time-sorted group, and its gap.
fn nearest(g: &[(i64, u32)], t: i64, dir: Direction, tolerance: i64) -> Option<(u32, i64)> {
    // The last row at or before t (the last of equal times), and the first at or after it.
    let after = g.partition_point(|(x, _)| *x < t);
    let before = g.partition_point(|(x, _)| *x <= t).checked_sub(1);
    let back = before.map(|i| (g[i].1, t - g[i].0));
    let fwd = g.get(after).map(|(x, r)| (*r, x - t));
    let pick = match dir {
        Direction::Backward => back,
        Direction::Forward => fwd,
        Direction::Nearest => match (back, fwd) {
            (Some(b), Some(f)) => Some(if f.1 < b.1 { f } else { b }),
            (b, f) => b.or(f),
        },
    };
    pick.filter(|(_, gap)| *gap <= tolerance)
}
