//! Selections: what the user brushed, clicked or filtered, as predicates on fields.
//!
//! A selection is a conjunction of predicates. Nulls never match a predicate, as in SQL. A
//! selection carries the generation of the interaction that made it, so results computed for an
//! older selection can be recognized and dropped.

use std::collections::HashSet;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, BooleanBuilder, RecordBatch};
use arrow::compute::kernels::boolean;
use arrow::compute::{cast, is_null};
use arrow::datatypes::{DataType, Float64Type, TimeUnit};
use arrow::error::ArrowError;
use serde::{Deserialize, Serialize};

use crate::profile::{Kind, Value};

/// One condition on one field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Predicate {
    /// `min <= value <= max`. Numbers compare as numbers, timestamps as microseconds since the
    /// epoch (UTC) and dates as days since the epoch. Either bound may be open.
    Range {
        field: String,
        min: Option<f64>,
        max: Option<f64>,
    },
    /// The value is one of these.
    OneOf { field: String, values: Vec<Value> },
    /// The value is null.
    IsNull { field: String },
    /// The value is not null.
    NotNull { field: String },
}

impl Predicate {
    pub fn field(&self) -> &str {
        match self {
            Predicate::Range { field, .. }
            | Predicate::OneOf { field, .. }
            | Predicate::IsNull { field }
            | Predicate::NotNull { field } => field,
        }
    }

    /// Which rows of `column` match.
    pub fn mask(&self, column: &ArrayRef) -> Result<BooleanArray, ArrowError> {
        match self {
            Predicate::Range { min, max, .. } => {
                let values = as_f64(column)?;
                let values = values.as_primitive::<Float64Type>();
                Ok(values
                    .iter()
                    .map(|v| {
                        Some(v.is_some_and(|v| {
                            !v.is_nan()
                                && min.is_none_or(|lo| v >= lo)
                                && max.is_none_or(|hi| v <= hi)
                        }))
                    })
                    .collect())
            }
            Predicate::OneOf { values, .. } => one_of(column, values),
            Predicate::IsNull { .. } => is_null(column),
            Predicate::NotNull { .. } => boolean::not(&is_null(column)?),
        }
    }
}

/// A conjunction of predicates, tagged with the generation that produced it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    pub predicates: Vec<Predicate>,
    pub generation: u64,
}

impl Selection {
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }

    /// Fields the selection depends on.
    pub fn fields(&self) -> impl Iterator<Item = &str> {
        self.predicates.iter().map(Predicate::field)
    }

    /// Which rows of a batch are selected. An empty selection selects every row.
    pub fn mask(&self, batch: &RecordBatch) -> Result<BooleanArray, ArrowError> {
        let mut mask = BooleanArray::from(vec![true; batch.num_rows()]);
        for p in &self.predicates {
            let column = batch.column_by_name(p.field()).ok_or_else(|| {
                ArrowError::SchemaError(format!("no field {:?} to select on", p.field()))
            })?;
            mask = boolean::and(&mask, &p.mask(column)?)?;
        }
        Ok(mask)
    }
}

/// Casts a column to f64 in the units [`Predicate::Range`] compares in.
fn as_f64(column: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    match column.data_type() {
        DataType::Timestamp(..) => {
            let us = cast(column, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
            cast(&cast(&us, &DataType::Int64)?, &DataType::Float64)
        }
        DataType::Date32 | DataType::Date64 => {
            let days = cast(column, &DataType::Date32)?;
            cast(&cast(&days, &DataType::Int32)?, &DataType::Float64)
        }
        _ => cast(column, &DataType::Float64),
    }
}

fn one_of(column: &ArrayRef, values: &[Value]) -> Result<BooleanArray, ArrowError> {
    let kind = Kind::of(column.data_type());
    let mut out = BooleanBuilder::with_capacity(column.len());
    match kind {
        Kind::Text => {
            let set: HashSet<&str> = values
                .iter()
                .filter_map(|v| match v {
                    Value::Text(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            let strings = cast(column, &DataType::Utf8)?;
            for v in strings.as_string::<i32>() {
                out.append_value(v.is_some_and(|s| set.contains(s)));
            }
        }
        Kind::Boolean => {
            let set: HashSet<bool> = values
                .iter()
                .filter_map(|v| match v {
                    Value::Boolean(b) => Some(*b),
                    _ => None,
                })
                .collect();
            for v in column.as_boolean() {
                out.append_value(v.is_some_and(|b| set.contains(&b)));
            }
        }
        _ => {
            let set: Vec<f64> = values
                .iter()
                .filter_map(|v| match v {
                    Value::Integer(i) => Some(*i as f64),
                    Value::Float(f) => Some(*f),
                    _ => None,
                })
                .collect();
            let numbers = as_f64(column)?;
            for v in numbers.as_primitive::<Float64Type>() {
                out.append_value(v.is_some_and(|x| set.contains(&x)));
            }
        }
    }
    Ok(out.finish())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};

    use super::*;

    fn batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("c", DataType::Utf8, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(vec![
                    Some(1.0),
                    None,
                    Some(3.0),
                    Some(f64::NAN),
                ])),
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    None,
                    Some("a"),
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn nulls_and_nan_never_match() {
        let s = Selection {
            predicates: vec![Predicate::Range {
                field: "x".into(),
                min: None,
                max: None,
            }],
            generation: 1,
        };
        let m = s.mask(&batch()).unwrap();
        assert_eq!(
            m.iter().collect::<Vec<_>>(),
            [Some(true), Some(false), Some(true), Some(false)]
        );
    }

    #[test]
    fn predicates_combine_with_and() {
        let s = Selection {
            predicates: vec![
                Predicate::Range {
                    field: "x".into(),
                    min: Some(0.5),
                    max: Some(2.0),
                },
                Predicate::OneOf {
                    field: "c".into(),
                    values: vec![Value::Text("a".into())],
                },
            ],
            generation: 2,
        };
        assert_eq!(s.mask(&batch()).unwrap().true_count(), 1);
        let missing = Selection {
            predicates: vec![Predicate::IsNull {
                field: "nope".into(),
            }],
            generation: 3,
        };
        assert!(missing.mask(&batch()).is_err());
    }
}
