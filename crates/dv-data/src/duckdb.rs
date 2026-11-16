//! DuckDB database files (the `duckdb` feature), opened read-only.
//!
//! Tables and views read with the types DuckDB gives them in Arrow, as DuckDB's own Arrow export
//! converts them: `DECIMAL` stays decimal, `TIMESTAMP` becomes a timestamp, `ENUM` a dictionary,
//! and so on. A table outside the `main` schema is named `schema.table`.
//!
//! Each database opens with external access and extension autoloading turned off, so a read can
//! only touch the file itself, and never installs anything.

use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;

use ::duckdb::{AccessMode, Config, Connection};
use arrow::array::{RecordBatch, StructArray};
use arrow::compute::concat_batches;
use arrow::datatypes::{Schema, SchemaRef};

use crate::table::BATCH_ROWS;
use crate::{Error, Result};

/// The arrow version DuckDB's crate builds on, which can differ from the engine's.
use ::duckdb::arrow as darrow;
use darrow::array::Array as _;

fn err(e: ::duckdb::Error) -> Error {
    Error::Unsupported(format!("DuckDB: {e}"))
}

fn connect(path: &Path) -> Result<Connection> {
    let config = Config::default()
        .access_mode(AccessMode::ReadOnly)
        .and_then(|c| c.enable_autoload_extension(false))
        .and_then(|c| c.with("autoinstall_known_extensions", "false"))
        .and_then(|c| c.enable_external_access(false))
        .map_err(err)?;
    Connection::open_with_flags(path, config).map_err(err)
}

/// A table or view, shown as `schema.name` outside the `main` schema.
struct Entry {
    schema: String,
    name: String,
}

impl Entry {
    fn display(&self) -> String {
        if self.schema == "main" {
            self.name.clone()
        } else {
            format!("{}.{}", self.schema, self.name)
        }
    }

    fn sql(&self) -> String {
        format!("{}.{}", quote(&self.schema), quote(&self.name))
    }
}

fn entries(conn: &Connection) -> Result<Vec<Entry>> {
    let mut stmt = conn
        .prepare(
            "select schema_name, name from (
               select schema_name, table_name as name from duckdb_tables()
                 where database_name = current_database() and not internal and not temporary
               union all
               select schema_name, view_name from duckdb_views()
                 where database_name = current_database() and not internal and not temporary)
             order by schema_name = 'main' desc, schema_name, name",
        )
        .map_err(err)?;
    let rows = stmt
        .query_map([], |r| {
            Ok(Entry {
                schema: r.get(0)?,
                name: r.get(1)?,
            })
        })
        .map_err(err)?;
    rows.collect::<std::result::Result<_, _>>().map_err(err)
}

/// Tables and views in a database: those in the `main` schema first, then `schema.name`, each
/// in name order.
pub fn tables(path: &Path) -> Result<Vec<String>> {
    Ok(entries(&connect(path)?)?
        .iter()
        .map(Entry::display)
        .collect())
}

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// The SQL name of a table, which must be one the database lists.
fn pick_table(path: &Path, conn: &Connection, table: Option<&str>) -> Result<String> {
    let all = entries(conn)?;
    let names = || {
        all.iter()
            .map(Entry::display)
            .collect::<Vec<_>>()
            .join(", ")
    };
    match table {
        Some(t) => all
            .iter()
            .find(|e| e.display() == t)
            .map(Entry::sql)
            .ok_or_else(|| {
                Error::Unsupported(format!("{path:?} has no table {t:?}; it has {}", names()))
            }),
        None => match all.as_slice() {
            [one] => Ok(one.sql()),
            [] => Err(Error::Unsupported(format!("{path:?} has no tables"))),
            _ => Err(Error::Unsupported(format!(
                "{path:?} has {} tables; name one of {}",
                all.len(),
                names()
            ))),
        },
    }
}

/// Moves a batch from DuckDB's arrow to the engine's through the Arrow C data interface, without
/// copying its buffers.
fn convert(batch: darrow::record_batch::RecordBatch) -> Result<RecordBatch> {
    let data = darrow::array::StructArray::from(batch).into_data();
    let (array, schema) = darrow::ffi::to_ffi(&data)
        .map_err(|e| Error::Unsupported(format!("DuckDB: Arrow export: {e}")))?;
    // SAFETY: both are the Arrow C data interface's `ArrowArray` and `ArrowSchema` structs, which
    // every arrow-rs version declares `#[repr(C)]` with the same layout; ownership passes on to
    // `from_ffi`, which calls the release callbacks.
    let (array, schema) = unsafe {
        (
            std::mem::transmute::<darrow::ffi::FFI_ArrowArray, arrow::ffi::FFI_ArrowArray>(array),
            std::mem::transmute::<darrow::ffi::FFI_ArrowSchema, arrow::ffi::FFI_ArrowSchema>(
                schema,
            ),
        )
    };
    // SAFETY: the array and schema were just exported together, and are valid.
    let data = unsafe { arrow::ffi::from_ffi(array, &schema) }?;
    Ok(RecordBatch::from(StructArray::from(data)))
}

pub(crate) fn schema(path: &Path, table: Option<&str>) -> Result<SchemaRef> {
    let conn = connect(path)?;
    let table = pick_table(path, &conn, table)?;
    let mut stmt = conn
        .prepare(&format!("select * from {table} limit 0"))
        .map_err(err)?;
    // Run the query for its result schema, then hand that across on its own.
    stmt.query_arrow([]).map_err(err)?.for_each(drop);
    let ffi = darrow::ffi::FFI_ArrowSchema::try_from(stmt.schema().as_ref())
        .map_err(|e| Error::Unsupported(format!("DuckDB: Arrow export: {e}")))?;
    // SAFETY: as in `convert`.
    let ffi = unsafe {
        std::mem::transmute::<darrow::ffi::FFI_ArrowSchema, arrow::ffi::FFI_ArrowSchema>(ffi)
    };
    Ok(Arc::new(Schema::try_from(&ffi)?))
}

/// Reads a table in batches of about [`BATCH_ROWS`] rows. A thread runs the query and keeps at
/// most two batches ahead of the reader; dropping the iterator stops it.
pub(crate) fn read(
    path: &Path,
    table: Option<&str>,
    schema: SchemaRef,
) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + Send>> {
    let conn = connect(path)?;
    let table = pick_table(path, &conn, table)?;
    let cols: Vec<String> = schema.fields().iter().map(|f| quote(f.name())).collect();
    let sql = format!("select {} from {table}", cols.join(", "));
    let (tx, rx) = mpsc::sync_channel::<Result<RecordBatch>>(2);
    std::thread::Builder::new()
        .name("duckdb-read".into())
        .spawn(move || {
            let run = || -> Result<()> {
                let mut stmt = conn.prepare(&sql).map_err(err)?;
                let mut pending: Vec<RecordBatch> = Vec::new();
                let mut rows = 0;
                let mut sent = false;
                let flush = |pending: &mut Vec<RecordBatch>| -> Result<Option<RecordBatch>> {
                    if pending.is_empty() {
                        return Ok(None);
                    }
                    let b = concat_batches(&schema, pending.iter())?;
                    pending.clear();
                    Ok(Some(b))
                };
                for b in stmt.query_arrow([]).map_err(err)? {
                    let b = convert(b)?;
                    // Batches carry the table's schema, with its field names and metadata.
                    let b = RecordBatch::try_new(schema.clone(), b.columns().to_vec())?;
                    rows += b.num_rows();
                    pending.push(b);
                    if rows >= BATCH_ROWS {
                        rows = 0;
                        if let Some(b) = flush(&mut pending)? {
                            if tx.send(Ok(b)).is_err() {
                                return Ok(());
                            }
                            sent = true;
                        }
                    }
                }
                match flush(&mut pending)? {
                    Some(b) => _ = tx.send(Ok(b)),
                    // An empty table still reads as one empty batch, as the other formats do.
                    None if !sent => _ = tx.send(Ok(RecordBatch::new_empty(schema.clone()))),
                    None => {}
                }
                Ok(())
            };
            if let Err(e) = run() {
                let _ = tx.send(Err(e));
            }
        })?;
    Ok(Box::new(rx.into_iter()))
}
