//! Excel workbooks (xlsx, xlsm, xlsb, xls) and OpenDocument spreadsheets, through calamine.
//!
//! A sheet is a table whose first row is the header. Each column's type is inferred from every
//! cell: all booleans, all numbers (integers when every number is whole), all dates, and anything
//! else as text. Error cells such as `#N/A` read as null. Excel stores times as fractions of a
//! day, so they're rounded to the millisecond.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, RecordBatch, StringBuilder,
    TimestampMicrosecondBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use calamine::{Data, Range, Reader, open_workbook_auto};

use crate::{Error, Result};

fn sheet(path: &Path, name: Option<&str>) -> Result<Range<Data>> {
    let err = |e: String| Error::Unsupported(format!("{path:?}: {e}"));
    let mut book = open_workbook_auto(path).map_err(|e| err(e.to_string()))?;
    let name = match name {
        Some(n) => n.to_string(),
        None => book
            .sheet_names()
            .first()
            .cloned()
            .ok_or_else(|| err("the workbook has no sheets".into()))?,
    };
    book.worksheet_range(&name).map_err(|e| err(e.to_string()))
}

/// Column names from the header row: blanks are named `column_N`, and repeats get `_2`, `_3`…
fn header(range: &Range<Data>) -> Vec<String> {
    let mut seen = HashMap::<String, usize>::new();
    let width = range.width();
    let first: Vec<Data> = range.rows().next().map(|r| r.to_vec()).unwrap_or_default();
    (0..width)
        .map(|i| {
            let base = match first.get(i) {
                Some(Data::Empty) | None => format!("column_{}", i + 1),
                Some(d) => d.to_string(),
            };
            let n = seen.entry(base.clone()).or_default();
            *n += 1;
            if *n == 1 { base } else { format!("{base}_{n}") }
        })
        .collect()
}

fn infer(cells: impl Iterator<Item = Option<Data>>) -> DataType {
    let (mut bools, mut ints, mut floats, mut dates, mut other) = (0, 0, 0, 0, 0);
    for c in cells.flatten() {
        match c {
            Data::Bool(_) => bools += 1,
            Data::Int(_) => ints += 1,
            Data::Float(f) if f.fract() == 0.0 && f.abs() < 9.0e15 => ints += 1,
            Data::Float(_) => floats += 1,
            Data::DateTime(_) | Data::DateTimeIso(_) => dates += 1,
            Data::Empty | Data::Error(_) => {}
            _ => other += 1,
        }
    }
    let total = bools + ints + floats + dates + other;
    match total {
        0 => DataType::Utf8,
        t if t == bools => DataType::Boolean,
        t if t == ints => DataType::Int64,
        t if t == ints + floats => DataType::Float64,
        t if t == dates => DataType::Timestamp(TimeUnit::Microsecond, None),
        _ => DataType::Utf8,
    }
}

fn column(range: &Range<Data>, i: usize) -> impl Iterator<Item = Option<Data>> + '_ {
    range.rows().skip(1).map(move |r| r.get(i).cloned())
}

pub(crate) fn schema(path: &Path, sheet_name: Option<&str>) -> Result<SchemaRef> {
    let range = sheet(path, sheet_name)?;
    let fields: Vec<Field> = header(&range)
        .into_iter()
        .enumerate()
        .map(|(i, name)| Field::new(name, infer(column(&range, i)), true))
        .collect();
    Ok(Arc::new(Schema::new(fields)))
}

pub(crate) fn read(
    path: &Path,
    sheet_name: Option<&str>,
    schema: SchemaRef,
) -> Result<Vec<RecordBatch>> {
    let range = sheet(path, sheet_name)?;
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| build(f.data_type(), column(&range, i)))
        .collect();
    Ok(vec![RecordBatch::try_new(schema, columns)?])
}

fn build(dt: &DataType, cells: impl Iterator<Item = Option<Data>>) -> ArrayRef {
    match dt {
        DataType::Boolean => {
            let mut b = BooleanBuilder::new();
            for c in cells {
                b.append_option(match c {
                    Some(Data::Bool(v)) => Some(v),
                    _ => None,
                });
            }
            Arc::new(b.finish())
        }
        DataType::Int64 => {
            let mut b = Int64Builder::new();
            for c in cells {
                b.append_option(match c {
                    Some(Data::Int(v)) => Some(v),
                    Some(Data::Float(v)) => Some(v as i64),
                    _ => None,
                });
            }
            Arc::new(b.finish())
        }
        DataType::Float64 => {
            let mut b = Float64Builder::new();
            for c in cells {
                b.append_option(match c {
                    Some(Data::Int(v)) => Some(v as f64),
                    Some(Data::Float(v)) => Some(v),
                    _ => None,
                });
            }
            Arc::new(b.finish())
        }
        DataType::Timestamp(..) => {
            let mut b = TimestampMicrosecondBuilder::new();
            for c in cells {
                b.append_option(match c {
                    Some(Data::DateTime(d)) => d.as_datetime().map(|t| {
                        let us = t.and_utc().timestamp_micros();
                        (us as f64 / 1000.0).round() as i64 * 1000
                    }),
                    Some(Data::DateTimeIso(s)) => s
                        .parse::<chrono::NaiveDateTime>()
                        .ok()
                        .map(|t| t.and_utc().timestamp_micros()),
                    _ => None,
                });
            }
            Arc::new(b.finish())
        }
        _ => {
            let mut b = StringBuilder::new();
            for c in cells {
                match c {
                    None | Some(Data::Empty | Data::Error(_)) => b.append_null(),
                    Some(d) => b.append_value(d.to_string()),
                }
            }
            Arc::new(b.finish())
        }
    }
}
