//! Column profiles: what a dataset contains, shown the moment it opens.
//!
//! A profile mixes two kinds of facts, and always says which is which. Exact facts (row count, nulls,
//! minimum, maximum) come from the source's own metadata. Everything else (mean, spread, quantiles,
//! histograms, distinct and frequent values) is computed from a sample, and records the sample's size.

use std::collections::HashMap;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type, Int64Type, TimeUnit};
use rayon::prelude::*;
use serde::Serialize;

/// Bins in a sampled histogram.
pub const HISTOGRAM_BINS: usize = 40;
/// Frequent values kept for text and low-cardinality columns.
pub const TOP_VALUES: usize = 10;
/// Numeric columns with at most this many distinct sampled values also get frequent values.
const CATEGORICAL_MAX_DISTINCT: usize = 20;
/// Quantiles reported for numeric columns.
pub const QUANTILES: [f64; 5] = [0.01, 0.25, 0.5, 0.75, 0.99];
/// The histogram spans these quantiles; values outside are counted as below or above.
const HISTOGRAM_SPAN: (f64, f64) = (0.005, 0.995);

/// How a column is treated in a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Integer,
    Float,
    Boolean,
    /// Microseconds since the Unix epoch, UTC.
    Timestamp,
    /// Days since the Unix epoch.
    Date,
    Text,
    Other,
}

impl Kind {
    pub fn of(dt: &DataType) -> Self {
        use DataType::*;
        match dt {
            Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 => Kind::Integer,
            Float16 | Float32 | Float64 | Decimal128(..) | Decimal256(..) => Kind::Float,
            Boolean => Kind::Boolean,
            Timestamp(..) => Kind::Timestamp,
            Date32 | Date64 => Kind::Date,
            Utf8 | LargeUtf8 | Utf8View => Kind::Text,
            Dictionary(_, v) if Kind::of(v) == Kind::Text => Kind::Text,
            _ => Kind::Other,
        }
    }

    fn is_numeric(self) -> bool {
        matches!(
            self,
            Kind::Integer | Kind::Float | Kind::Timestamp | Kind::Date
        )
    }
}

/// A single value, in a column's [`Kind`].
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Text(String),
}

impl Value {
    /// The first value of an array, converted for its kind. `None` if it's null or unsupported.
    pub fn from_array(a: &ArrayRef) -> Option<Self> {
        if a.is_empty() || a.is_null(0) {
            return None;
        }
        let kind = Kind::of(a.data_type());
        match kind {
            Kind::Integer | Kind::Timestamp | Kind::Date => {
                let a = numeric_as(a, kind, &DataType::Int64)?;
                Some(Value::Integer(a.as_primitive::<Int64Type>().value(0)))
            }
            Kind::Float => {
                let a = cast(a, &DataType::Float64).ok()?;
                Some(Value::Float(a.as_primitive::<Float64Type>().value(0)))
            }
            Kind::Boolean => Some(Value::Boolean(a.as_boolean().value(0))),
            Kind::Text => {
                let a = cast(a, &DataType::Utf8).ok()?;
                Some(Value::Text(a.as_string::<i32>().value(0).to_owned()))
            }
            Kind::Other => None,
        }
    }
}

/// Exact facts about a column from the source's metadata. Any of them may be unknown.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ExactStats {
    pub nulls: Option<u64>,
    pub min: Option<Value>,
    pub max: Option<Value>,
}

/// A dataset's profile.
#[derive(Debug, Clone, Serialize)]
pub struct Profile {
    /// Rows in the whole dataset.
    pub rows: u64,
    /// Rows the sampled facts were computed from.
    pub sample_rows: usize,
    pub columns: Vec<ColumnProfile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColumnProfile {
    pub name: String,
    pub kind: Kind,
    /// The source's type, for display.
    pub data_type: String,
    pub exact: ExactStats,
    pub sample: SampleStats,
}

/// Facts computed from the sample.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SampleStats {
    pub nulls: usize,
    /// NaN values in float columns. They're left out of every numeric statistic.
    pub nans: usize,
    /// Distinct non-null values in the sample.
    pub distinct: usize,
    pub numeric: Option<NumericStats>,
    /// The most frequent non-null values with their sample counts, most frequent first.
    pub top: Vec<(Value, usize)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NumericStats {
    pub count: usize,
    pub mean: f64,
    /// Sample standard deviation (n − 1). Zero for a single value.
    pub std: f64,
    /// Values at [`QUANTILES`], by linear interpolation between closest ranks.
    pub quantiles: [f64; QUANTILES.len()],
    pub histogram: Histogram,
}

/// Equal-width bins over `[lo, hi]`. The last bin includes `hi`.
#[derive(Debug, Clone, Serialize)]
pub struct Histogram {
    pub lo: f64,
    pub hi: f64,
    pub counts: Vec<u64>,
    /// Values below `lo` or above `hi`, so outliers don't squash the bins.
    pub below: u64,
    pub above: u64,
}

/// Profiles every column of a sample. `exact` holds the source's exact facts, in schema order.
pub fn profile(sample: &RecordBatch, rows: u64, exact: Vec<ExactStats>) -> Profile {
    let schema = sample.schema();
    let columns = schema
        .fields()
        .par_iter()
        .zip(sample.columns().par_iter())
        .zip(exact.into_par_iter())
        .map(|((field, col), exact)| {
            let kind = Kind::of(field.data_type());
            ColumnProfile {
                name: field.name().clone(),
                kind,
                data_type: field.data_type().to_string(),
                exact,
                sample: sample_stats(col, kind),
            }
        })
        .collect();
    Profile {
        rows,
        sample_rows: sample.num_rows(),
        columns,
    }
}

fn sample_stats(col: &ArrayRef, kind: Kind) -> SampleStats {
    let nulls = col.null_count();
    if kind.is_numeric() {
        return numeric_sample(col, kind, nulls);
    }
    let mut counts: HashMap<Value, usize> = HashMap::new();
    match kind {
        Kind::Boolean => {
            for v in col.as_boolean().iter().flatten() {
                *counts.entry(Value::Boolean(v)).or_default() += 1;
            }
        }
        Kind::Text => {
            if let Ok(s) = cast(col, &DataType::Utf8) {
                for v in s.as_string::<i32>().iter().flatten() {
                    *counts.entry(Value::Text(v.to_owned())).or_default() += 1;
                }
            }
        }
        _ => {}
    }
    SampleStats {
        nulls,
        nans: 0,
        distinct: counts.len(),
        numeric: None,
        top: top(counts),
    }
}

fn numeric_sample(col: &ArrayRef, kind: Kind, nulls: usize) -> SampleStats {
    let Some(f) = numeric_as(col, kind, &DataType::Float64) else {
        return SampleStats {
            nulls,
            ..Default::default()
        };
    };
    let mut v: Vec<f64> = f.as_primitive::<Float64Type>().iter().flatten().collect();
    let before = v.len();
    v.retain(|x| !x.is_nan());
    let nans = before - v.len();
    v.sort_unstable_by(f64::total_cmp);

    // Runs of equal values in sorted order give distinct counts and frequencies in one pass.
    let mut runs: Vec<(f64, usize)> = Vec::new();
    for &x in &v {
        match runs.last_mut() {
            Some((y, n)) if *y == x => *n += 1,
            _ => runs.push((x, 1)),
        }
    }
    let distinct = runs.len();
    let top_values = if distinct <= CATEGORICAL_MAX_DISTINCT {
        let as_value = |x: f64| match kind {
            Kind::Float => Value::Float(x),
            _ => Value::Integer(x as i64),
        };
        top(runs.iter().map(|&(x, n)| (as_value(x), n)))
    } else {
        Vec::new()
    };
    SampleStats {
        nulls,
        nans,
        distinct,
        numeric: numeric_stats(&v),
        top: top_values,
    }
}

/// Mean, spread, quantiles and histogram of sorted, NaN-free values.
fn numeric_stats(sorted: &[f64]) -> Option<NumericStats> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    // Welford's update keeps the variance accurate for large offsets such as timestamps.
    let (mut mean, mut m2) = (0.0, 0.0);
    for (i, &x) in sorted.iter().enumerate() {
        let d = x - mean;
        mean += d / (i + 1) as f64;
        m2 += d * (x - mean);
    }
    let std = if n > 1 {
        (m2 / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    let quantiles = QUANTILES.map(|q| quantile(sorted, q));
    let histogram = histogram(
        sorted,
        quantile(sorted, HISTOGRAM_SPAN.0),
        quantile(sorted, HISTOGRAM_SPAN.1),
        HISTOGRAM_BINS,
    );
    Some(NumericStats {
        count: n,
        mean,
        std,
        quantiles,
        histogram,
    })
}

/// The `q` quantile of sorted values, interpolating linearly between closest ranks
/// (DuckDB's `quantile_cont`, NumPy's default).
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = (sorted.len() - 1) as f64 * q;
    let (i, frac) = (pos.floor() as usize, pos.fract());
    match sorted.get(i + 1) {
        Some(&next) if frac > 0.0 => sorted[i] + (next - sorted[i]) * frac,
        _ => sorted[i],
    }
}

/// Counts values into `bins` equal-width bins over `[lo, hi]`.
pub fn histogram(values: &[f64], lo: f64, hi: f64, bins: usize) -> Histogram {
    let bins = if hi > lo { bins } else { 1 };
    let mut h = Histogram {
        lo,
        hi,
        counts: vec![0; bins],
        below: 0,
        above: 0,
    };
    let width = (hi - lo) / bins as f64;
    for &x in values {
        if x < lo {
            h.below += 1;
        } else if x > hi {
            h.above += 1;
        } else {
            let b = if width > 0.0 {
                ((x - lo) / width) as usize
            } else {
                0
            };
            h.counts[b.min(bins - 1)] += 1;
        }
    }
    h
}

/// The [`TOP_VALUES`] most frequent values, ties broken by value so results are stable.
fn top(counts: impl IntoIterator<Item = (Value, usize)>) -> Vec<(Value, usize)> {
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.order(&b.0)));
    v.truncate(TOP_VALUES);
    v
}

impl Value {
    /// A total order: numbers by value, text by bytes, and kinds in a fixed order.
    fn order(&self, other: &Self) -> std::cmp::Ordering {
        use Value::*;
        let rank = |v: &Value| match v {
            Boolean(_) => 0,
            Integer(_) | Float(_) => 1,
            Text(_) => 2,
        };
        match (self, other) {
            (Boolean(a), Boolean(b)) => a.cmp(b),
            (Integer(a), Integer(b)) => a.cmp(b),
            (Text(a), Text(b)) => a.cmp(b),
            (Integer(_) | Float(_), Integer(_) | Float(_)) => {
                as_f64(self).total_cmp(&as_f64(other))
            }
            _ => rank(self).cmp(&rank(other)),
        }
    }
}

fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Integer(i) => *i as f64,
        Value::Float(f) => *f,
        _ => f64::NAN,
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Integer(i) => i.hash(state),
            Value::Float(f) => f.to_bits().hash(state),
            Value::Boolean(b) => b.hash(state),
            Value::Text(s) => s.hash(state),
        }
    }
}

/// Casts a numeric, timestamp or date array to `to`. Timestamps become microseconds since the epoch,
/// and dates become days since the epoch.
fn numeric_as(a: &ArrayRef, kind: Kind, to: &DataType) -> Option<ArrayRef> {
    let a = match (kind, a.data_type()) {
        (Kind::Timestamp, DataType::Timestamp(_, tz)) => {
            let us = cast(a, &DataType::Timestamp(TimeUnit::Microsecond, tz.clone())).ok()?;
            cast(&us, &DataType::Int64).ok()?
        }
        (Kind::Date, DataType::Date64) => {
            let d = cast(a, &DataType::Date32).ok()?;
            cast(&d, &DataType::Int32).ok()?
        }
        (Kind::Date, _) => cast(a, &DataType::Int32).ok()?,
        _ => a.clone(),
    };
    cast(&a, to).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_interpolate_like_numpy() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&v, 0.0), 1.0);
        assert_eq!(quantile(&v, 0.5), 2.5);
        assert_eq!(quantile(&v, 1.0), 4.0);
        assert!((quantile(&v, 0.25) - 1.75).abs() < 1e-12);
    }

    #[test]
    fn histogram_includes_hi_and_counts_outliers() {
        let h = histogram(&[-1.0, 0.0, 0.5, 1.0, 2.0], 0.0, 1.0, 2);
        assert_eq!((h.counts.as_slice(), h.below, h.above), (&[1, 2][..], 1, 1));
        let flat = histogram(&[3.0, 3.0], 3.0, 3.0, 40);
        assert_eq!(flat.counts, vec![2]);
    }

    #[test]
    fn top_is_stable() {
        let t = top([
            (Value::Text("b".into()), 2),
            (Value::Text("a".into()), 2),
            (Value::Text("c".into()), 5),
        ]);
        let names: Vec<_> = t.iter().map(|(v, _)| v.clone()).collect();
        assert_eq!(names, ["c", "a", "b"].map(|s| Value::Text(s.into())));
    }
}
