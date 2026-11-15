//! SQLite databases, opened read-only.
//!
//! SQLite stores each value with its own type, whatever the column declares. A column's Arrow type
//! comes from the storage classes it actually holds, so no value is dropped for having the "wrong"
//! type: integers only → Int64 (Boolean if declared `BOOL…`), integers and reals → Float64, blobs
//! → Binary, and text → Utf8, or a timestamp when the column is declared as a date or time and
//! every value parses as one. A column that mixes text with numbers is read as text.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryBuilder, BooleanBuilder, Float64Builder, Int64Builder, RecordBatch,
    StringBuilder, TimestampMicrosecondBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};

use crate::table::BATCH_ROWS;
use crate::{Error, Result};

fn err(e: rusqlite::Error) -> Error {
    Error::Unsupported(format!("SQLite: {e}"))
}

fn connect(path: &Path) -> Result<Connection> {
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(err)
}

/// Tables in a database, in name order.
pub fn tables(path: &Path) -> Result<Vec<String>> {
    let conn = connect(path)?;
    let mut stmt = conn
        .prepare("select name from sqlite_schema where type in ('table', 'view') and name not like 'sqlite_%' order by name")
        .map_err(err)?;
    let names = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(err)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(err)?;
    Ok(names)
}

fn pick_table(path: &Path, table: Option<&str>) -> Result<String> {
    if let Some(t) = table {
        return Ok(t.to_string());
    }
    let all = tables(path)?;
    match all.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(Error::Unsupported(format!("{path:?} has no tables"))),
        _ => Err(Error::Unsupported(format!(
            "{path:?} has {} tables; name one of {}",
            all.len(),
            all.join(", ")
        ))),
    }
}

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn is_time_decl(decl: &str) -> bool {
    let d = decl.to_ascii_uppercase();
    d.contains("DATE") || d.contains("TIME")
}

fn parse_time(s: &str) -> Option<i64> {
    const FORMATS: [&str; 2] = ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"];
    FORMATS
        .iter()
        .find_map(|f| chrono::NaiveDateTime::parse_from_str(s, f).ok())
        .or_else(|| {
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .ok()
                .and_then(|d| d.and_hms_opt(0, 0, 0))
        })
        .map(|t| t.and_utc().timestamp_micros())
}

pub(crate) fn schema(path: &Path, table: Option<&str>) -> Result<SchemaRef> {
    let table = pick_table(path, table)?;
    let conn = connect(path)?;
    let stmt = conn
        .prepare(&format!("select * from {}", quote(&table)))
        .map_err(err)?;
    let columns: Vec<(String, String)> = stmt
        .columns()
        .map(|c| {
            (
                c.name().to_string(),
                c.decl_type().unwrap_or("").to_string(),
            )
        })
        .collect();
    drop(stmt);
    let mut fields = Vec::new();
    for (name, decl) in columns {
        let q = quote(&name);
        let mut classes = conn
            .prepare(&format!(
                "select distinct typeof({q}) from {} where {q} is not null",
                quote(&table)
            ))
            .map_err(err)?;
        let classes: Vec<String> = classes
            .query_map([], |r| r.get(0))
            .map_err(err)?
            .collect::<rusqlite::Result<_>>()
            .map_err(err)?;
        let has = |c: &str| classes.iter().any(|x| x == c);
        let upper = decl.to_ascii_uppercase();
        let mixed_text = has("text") && (has("integer") || has("real") || has("blob"));
        let dt = if classes.is_empty() || mixed_text {
            DataType::Utf8
        } else if has("text") {
            let all_times = is_time_decl(&decl) && {
                let mut stmt = conn
                    .prepare(&format!(
                        "select {q} from {} where {q} is not null",
                        quote(&table)
                    ))
                    .map_err(err)?;
                let mut rows = stmt.query([]).map_err(err)?;
                let mut ok = true;
                while let Some(r) = rows.next().map_err(err)? {
                    if let ValueRef::Text(t) = r.get_ref(0).map_err(err)?
                        && parse_time(&String::from_utf8_lossy(t)).is_none()
                    {
                        ok = false;
                        break;
                    }
                }
                ok
            };
            if all_times {
                DataType::Timestamp(TimeUnit::Microsecond, None)
            } else {
                DataType::Utf8
            }
        } else if has("blob") {
            if classes.len() == 1 {
                DataType::Binary
            } else {
                DataType::Utf8
            }
        } else if has("real") {
            DataType::Float64
        } else if upper.starts_with("BOOL") {
            DataType::Boolean
        } else {
            DataType::Int64
        };
        fields.push(Field::new(name, dt, true));
    }
    Ok(Arc::new(Schema::new(fields)))
}

enum Builder {
    Bool(BooleanBuilder),
    Int(Int64Builder),
    Float(Float64Builder),
    Time(TimestampMicrosecondBuilder),
    Text(StringBuilder),
    Blob(BinaryBuilder),
}

impl Builder {
    fn new(dt: &DataType) -> Self {
        match dt {
            DataType::Boolean => Builder::Bool(BooleanBuilder::new()),
            DataType::Int64 => Builder::Int(Int64Builder::new()),
            DataType::Float64 => Builder::Float(Float64Builder::new()),
            DataType::Timestamp(..) => Builder::Time(TimestampMicrosecondBuilder::new()),
            DataType::Binary => Builder::Blob(BinaryBuilder::new()),
            _ => Builder::Text(StringBuilder::new()),
        }
    }

    fn push(&mut self, v: ValueRef<'_>) {
        use ValueRef::*;
        match (self, v) {
            (Builder::Bool(b), Integer(i)) => b.append_value(i != 0),
            (Builder::Int(b), Integer(i)) => b.append_value(i),
            (Builder::Float(b), Integer(i)) => b.append_value(i as f64),
            (Builder::Float(b), Real(f)) => b.append_value(f),
            (Builder::Time(b), Text(t)) => b.append_option(parse_time(&String::from_utf8_lossy(t))),
            (Builder::Blob(b), Blob(x)) => b.append_value(x),
            (Builder::Text(b), Text(t)) => b.append_value(String::from_utf8_lossy(t)),
            (Builder::Text(b), Integer(i)) => b.append_value(i.to_string()),
            (Builder::Text(b), Real(f)) => b.append_value(f.to_string()),
            (Builder::Text(b), Blob(x)) => b.append_value(String::from_utf8_lossy(x)),
            (b, _) => b.null(),
        }
    }

    fn null(&mut self) {
        match self {
            Builder::Bool(b) => b.append_null(),
            Builder::Int(b) => b.append_null(),
            Builder::Float(b) => b.append_null(),
            Builder::Time(b) => b.append_null(),
            Builder::Text(b) => b.append_null(),
            Builder::Blob(b) => b.append_null(),
        }
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            Builder::Bool(b) => Arc::new(b.finish()),
            Builder::Int(b) => Arc::new(b.finish()),
            Builder::Float(b) => Arc::new(b.finish()),
            Builder::Time(b) => Arc::new(b.finish()),
            Builder::Text(b) => Arc::new(b.finish()),
            Builder::Blob(b) => Arc::new(b.finish()),
        }
    }
}

pub(crate) fn read(
    path: &Path,
    table: Option<&str>,
    schema: SchemaRef,
) -> Result<Vec<RecordBatch>> {
    let table = pick_table(path, table)?;
    let conn = connect(path)?;
    let cols: Vec<String> = schema.fields().iter().map(|f| quote(f.name())).collect();
    let mut stmt = conn
        .prepare(&format!(
            "select {} from {}",
            cols.join(", "),
            quote(&table)
        ))
        .map_err(err)?;
    let mut rows = stmt.query([]).map_err(err)?;
    let mut builders: Vec<Builder> = schema
        .fields()
        .iter()
        .map(|f| Builder::new(f.data_type()))
        .collect();
    let mut batches = Vec::new();
    let mut n = 0;
    while let Some(r) = rows.next().map_err(err)? {
        for (i, b) in builders.iter_mut().enumerate() {
            b.push(r.get_ref(i).map_err(err)?);
        }
        n += 1;
        if n == BATCH_ROWS {
            let columns = builders.iter_mut().map(Builder::finish).collect();
            batches.push(RecordBatch::try_new(schema.clone(), columns)?);
            n = 0;
        }
    }
    if n > 0 || batches.is_empty() {
        let columns = builders.iter_mut().map(Builder::finish).collect();
        batches.push(RecordBatch::try_new(schema.clone(), columns)?);
    }
    Ok(batches)
}
