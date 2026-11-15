//! Selections as DataFusion filter expressions, with the same meaning as
//! [`Selection::mask`](dv_model::selection::Selection::mask): inclusive ranges, and nulls and NaN
//! never match. Filters stay on the column itself where possible, so scans can prune row groups
//! and pages.

use arrow::datatypes::{DataType, Schema, TimeUnit};
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result};
use datafusion::functions::math::expr_fn::isnan;
use datafusion::logical_expr::{Expr, cast, lit};
use datafusion::prelude::ident;
use dv_model::profile::Value;
use dv_model::selection::{Predicate, Selection};

/// The filter for a selection, or `None` when it selects every row.
pub fn selection_expr(selection: &Selection, schema: &Schema) -> Result<Option<Expr>> {
    let mut out: Option<Expr> = None;
    for p in &selection.predicates {
        let e = predicate_expr(p, schema)?;
        out = Some(match out {
            Some(acc) => acc.and(e),
            None => e,
        });
    }
    Ok(out)
}

fn predicate_expr(p: &Predicate, schema: &Schema) -> Result<Expr> {
    let field = schema
        .field_with_name(p.field())
        .map_err(|_| DataFusionError::Plan(format!("no field {:?} to select on", p.field())))?;
    let dt = field.data_type();
    let c = ident(p.field());
    Ok(match p {
        Predicate::IsNull { .. } => c.is_null(),
        Predicate::NotNull { .. } => c.is_not_null(),
        Predicate::Range { min, max, .. } => {
            let bounds: Vec<f64> = min.iter().chain(max).copied().collect();
            // Timestamps compare in the column's unit when every bound is exact in it, so scans
            // can prune; otherwise the column is cast to microseconds rather than rounding a bound.
            let (target, operand) = match dt {
                DataType::Timestamp(unit, tz) if !bounds.iter().all(|b| exact_in(*b, unit)) => {
                    let us = DataType::Timestamp(TimeUnit::Microsecond, tz.clone());
                    (us.clone(), cast(c.clone(), us))
                }
                _ => (dt.clone(), c.clone()),
            };
            let mut e = c.clone().is_not_null();
            if let Some(lo) = min {
                e = e.and(operand.clone().gt_eq(bound(&target, *lo, true)));
            }
            if let Some(hi) = max {
                e = e.and(operand.lt_eq(bound(&target, *hi, false)));
            }
            if matches!(
                dt,
                DataType::Float16 | DataType::Float32 | DataType::Float64
            ) {
                e = e.and(!isnan(c));
            }
            e
        }
        Predicate::OneOf { values, .. } => {
            let list = values
                .iter()
                .map(|v| match v {
                    Value::Integer(i) => lit(*i),
                    Value::Float(f) => lit(*f),
                    Value::Boolean(b) => lit(*b),
                    Value::Text(s) => lit(s.clone()),
                })
                .collect();
            c.in_list(list, false)
        }
    })
}

fn exact_in(us: f64, unit: &TimeUnit) -> bool {
    let us = us.round() as i64;
    match unit {
        TimeUnit::Second => us % 1_000_000 == 0,
        TimeUnit::Millisecond => us % 1000 == 0,
        TimeUnit::Microsecond | TimeUnit::Nanosecond => true,
    }
}

/// A range bound in the column's own type: microseconds for timestamps, days for dates. Date
/// bounds round inward, so whole days compare as the mask does.
fn bound(dt: &DataType, v: f64, lower: bool) -> Expr {
    let days = if lower { v.ceil() } else { v.floor() };
    match dt {
        DataType::Timestamp(unit, tz) => {
            let us = v.round() as i64;
            let tz = tz.clone();
            lit(match unit {
                TimeUnit::Second => {
                    ScalarValue::TimestampSecond(Some(us.div_euclid(1_000_000)), tz)
                }
                TimeUnit::Millisecond => {
                    ScalarValue::TimestampMillisecond(Some(us.div_euclid(1000)), tz)
                }
                TimeUnit::Microsecond => ScalarValue::TimestampMicrosecond(Some(us), tz),
                TimeUnit::Nanosecond => ScalarValue::TimestampNanosecond(Some(us * 1000), tz),
            })
        }
        DataType::Date32 => lit(ScalarValue::Date32(Some(days as i32))),
        DataType::Date64 => lit(ScalarValue::Date64(Some(days as i64 * 86_400_000))),
        _ => lit(v),
    }
}
