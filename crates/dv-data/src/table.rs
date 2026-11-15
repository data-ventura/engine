//! Tables in any supported format, opened the same way: Parquet, CSV/TSV, JSON/NDJSON, Arrow
//! IPC (Feather v2), Excel and SQLite, one file or a folder or glob of files with one schema.
//!
//! Opening a table finds its format and schema. Text formats infer the schema from their first
//! rows; JSON strings that all parse as ISO-8601 timestamps become timestamps, since JSON has no
//! timestamp type. Reading is read-only: nothing here writes to a source.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchReader};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};

use crate::parquet::{ParquetTable, Sample};
use crate::{Error, Result, excel, sqlite};

/// Rows read to infer a text format's schema.
const INFER_ROWS: usize = 10_000;
/// Rows per batch when reading text formats.
pub const BATCH_ROWS: usize = 64 * 1024;

/// A table format, with the options that pick what to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Format {
    Parquet,
    /// Delimited text with a header row.
    Csv {
        delimiter: u8,
    },
    /// One JSON object per line.
    Ndjson,
    /// A JSON array of objects.
    Json,
    /// Arrow IPC file (Feather v2) or stream.
    Arrow,
    /// An Excel workbook; the first sheet unless one is named.
    Excel {
        sheet: Option<String>,
    },
    /// A SQLite database; its only table unless one is named.
    Sqlite {
        table: Option<String>,
    },
    GeoJson,
    FlatGeobuf,
    /// A Shapefile's `.shp` (with its `.dbf` and `.prj` beside it).
    Shapefile,
    /// A GeoPackage; its only feature layer unless one is named.
    GeoPackage {
        layer: Option<String>,
    },
}

impl Format {
    /// The format of a file, from its extension or, failing that, its first bytes.
    pub fn detect(path: &Path) -> Result<Format> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        let by_ext = match ext.as_str() {
            "parquet" | "pq" => Some(Format::Parquet),
            "csv" => Some(Format::Csv { delimiter: b',' }),
            "tsv" | "tab" => Some(Format::Csv { delimiter: b'\t' }),
            "ndjson" | "jsonl" => Some(Format::Ndjson),
            "arrow" | "feather" | "ipc" | "arrows" => Some(Format::Arrow),
            "xlsx" | "xlsm" | "xlsb" | "xls" | "ods" => Some(Format::Excel { sheet: None }),
            "sqlite" | "sqlite3" | "db" => Some(Format::Sqlite { table: None }),
            "geojson" => Some(Format::GeoJson),
            "fgb" => Some(Format::FlatGeobuf),
            "shp" => Some(Format::Shapefile),
            "gpkg" => Some(Format::GeoPackage { layer: None }),
            _ => None,
        };
        if let Some(f) = by_ext {
            return Ok(f);
        }
        let mut head = [0u8; 16];
        let n = File::open(path)?.read(&mut head)?;
        let head = &head[..n];
        let first = head.iter().find(|b| !b.is_ascii_whitespace()).copied();
        Ok(match head {
            h if h.starts_with(b"PAR1") => Format::Parquet,
            h if h.starts_with(b"ARROW1") || h.starts_with(&[0xff, 0xff, 0xff, 0xff]) => {
                Format::Arrow
            }
            h if h.starts_with(b"SQLite format 3\0") => Format::Sqlite { table: None },
            h if h.starts_with(b"PK\x03\x04") => Format::Excel { sheet: None },
            _ if first == Some(b'[') => Format::Json,
            _ if first == Some(b'{') => Format::Ndjson,
            _ if ext == "json" => Format::Json,
            _ => {
                return Err(Error::Unsupported(format!(
                    "can't tell the format of {path:?}"
                )));
            }
        })
    }
}

/// An open table: one file, or several files with the same schema read as one.
#[derive(Debug, Clone)]
pub struct Table {
    files: Vec<PathBuf>,
    format: Format,
    schema: SchemaRef,
    parquet: Vec<ParquetTable>,
}

impl Table {
    /// Opens a file, a folder (every file with the most common table extension) or a glob such as
    /// `data/*.csv`, detecting the format.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let files = expand(path.as_ref())?;
        let format = Format::detect(&files[0])?;
        Self::open_files(files, format)
    }

    /// Opens a file, folder or glob in a given format.
    pub fn open_as(path: impl AsRef<Path>, format: Format) -> Result<Self> {
        Self::open_files(expand(path.as_ref())?, format)
    }

    fn open_files(files: Vec<PathBuf>, format: Format) -> Result<Self> {
        let mut parquet = Vec::new();
        let mut schema = None;
        for f in &files {
            let s = match &format {
                Format::Parquet => {
                    let t = ParquetTable::open(f)?;
                    let s = crate::geo::annotate_geoparquet(
                        t.schema(),
                        t.schema().metadata().get("geo"),
                    );
                    parquet.push(t);
                    s
                }
                _ => infer_schema(f, &format)?,
            };
            match &schema {
                None => schema = Some(s),
                Some(first) if !same_fields(first, &s) => {
                    return Err(Error::Unsupported(format!(
                        "{f:?} has a different schema from {:?}",
                        files[0]
                    )));
                }
                Some(_) => {}
            }
        }
        Ok(Table {
            schema: schema.expect("at least one file"),
            files,
            format,
            parquet,
        })
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn format(&self) -> &Format {
        &self.format
    }

    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    /// The row count, when the format records it without a scan.
    pub fn num_rows(&self) -> Option<u64> {
        match self.format {
            Format::Parquet => Some(self.parquet.iter().map(|p| p.num_rows()).sum()),
            _ => None,
        }
    }

    /// Reads every row, file by file, in batches.
    pub fn batches(&self) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + Send>> {
        let mut readers = Vec::new();
        for (i, f) in self.files.iter().enumerate() {
            readers.push(match &self.format {
                Format::Parquet => {
                    let names: Vec<&str> = self
                        .schema
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .collect();
                    let t = &self.parquet[i];
                    let batches: Vec<_> = (0..t.group_rows().len())
                        .map(|g| t.batches(g, &names))
                        .collect::<Result<_>>()?;
                    // Batches carry the table's schema, with GeoParquet's geometry marked.
                    let schema = self.schema.clone();
                    Box::new(batches.into_iter().flatten().map(move |b| {
                        Ok(RecordBatch::try_new(schema.clone(), b?.columns().to_vec())?)
                    })) as Box<dyn Iterator<Item = Result<RecordBatch>> + Send>
                }
                format => read_file(f, format, self.schema.clone())?,
            });
        }
        Ok(Box::new(readers.into_iter().flatten()))
    }

    /// Reads the whole table into one batch.
    pub fn read_all(&self) -> Result<RecordBatch> {
        let batches = self.batches()?.collect::<Result<Vec<_>>>()?;
        Ok(concat_batches(&self.schema, &batches)?)
    }

    /// About `target_rows` rows for a profile. Parquet samples are spread across the file. Other
    /// formats can't seek to a row, so their samples are the first rows, and the rest of the
    /// table is read only to count it.
    pub fn sample(&self, target_rows: usize) -> Result<Sample> {
        if self.format == Format::Parquet && self.files.len() == 1 {
            return self.parquet[0].sample(target_rows);
        }
        let mut batches = Vec::new();
        let mut rows = 0;
        let mut of_rows = 0u64;
        for b in self.batches()? {
            let b = b?;
            of_rows += b.num_rows() as u64;
            let take = target_rows.saturating_sub(rows).min(b.num_rows());
            if take > 0 {
                rows += take;
                batches.push(b.slice(0, take));
            }
        }
        Ok(Sample {
            batch: concat_batches(&self.schema, &batches)?,
            of_rows,
        })
    }
}

/// The files a path names: itself, a folder's table files, or a glob's matches, sorted.
fn expand(path: &Path) -> Result<Vec<PathBuf>> {
    let text = path.to_string_lossy();
    let mut files: Vec<PathBuf> = if text.contains(['*', '?', '[']) {
        glob::glob(&text)
            .map_err(|e| Error::Unsupported(format!("bad glob {text:?}: {e}")))?
            .filter_map(|p| p.ok())
            .filter(|p| p.is_file())
            .collect()
    } else if path.is_dir() {
        let entries: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && is_table_file(p))
            .collect();
        let mut counts = std::collections::HashMap::<String, usize>::new();
        for p in &entries {
            *counts.entry(extension(p)).or_default() += 1;
        }
        let common = counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)));
        match common {
            Some((ext, _)) => entries
                .into_iter()
                .filter(|p| extension(p) == ext)
                .collect(),
            None => Vec::new(),
        }
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    if files.is_empty() {
        return Err(Error::Unsupported(format!("no table files in {path:?}")));
    }
    Ok(files)
}

fn extension(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

/// Extensions a folder's table files may have.
fn is_table_file(p: &Path) -> bool {
    matches!(
        extension(p).as_str(),
        "parquet"
            | "pq"
            | "csv"
            | "tsv"
            | "tab"
            | "ndjson"
            | "jsonl"
            | "json"
            | "arrow"
            | "feather"
            | "ipc"
            | "arrows"
    )
}

fn same_fields(a: &Schema, b: &Schema) -> bool {
    a.fields().len() == b.fields().len()
        && a.fields()
            .iter()
            .zip(b.fields())
            .all(|(x, y)| x.name() == y.name() && x.data_type() == y.data_type())
}

fn infer_schema(path: &Path, format: &Format) -> Result<SchemaRef> {
    match format {
        Format::Parquet => Ok(ParquetTable::open(path)?.schema().clone()),
        Format::Csv { delimiter } => {
            let mut file = File::open(path)?;
            let (schema, _) = csv_format(*delimiter).infer_schema(&mut file, Some(INFER_ROWS))?;
            Ok(Arc::new(schema))
        }
        Format::Ndjson => {
            let mut reader = BufReader::new(File::open(path)?);
            let (schema, _) = arrow::json::reader::infer_json_schema_from_seekable(
                &mut reader,
                Some(INFER_ROWS),
            )?;
            reader.seek(SeekFrom::Start(0))?;
            let sample = arrow::json::ReaderBuilder::new(Arc::new(schema.clone()))
                .with_batch_size(INFER_ROWS)
                .build(reader)?
                .next()
                .transpose()?;
            Ok(refine_timestamps(schema, sample.as_ref()))
        }
        Format::Json => {
            let values = read_json_array(path)?;
            let n = values.len().min(INFER_ROWS);
            let schema = arrow::json::reader::infer_json_schema_from_iterator(
                values[..n].iter().map(|v| Ok(v.clone())),
            )?;
            let mut decoder = arrow::json::ReaderBuilder::new(Arc::new(schema.clone()))
                .with_batch_size(n.max(1))
                .build_decoder()?;
            decoder.serialize(&values[..n])?;
            let sample = decoder.flush()?;
            Ok(refine_timestamps(schema, sample.as_ref()))
        }
        Format::Arrow => Ok(open_ipc(path)?.schema()),
        Format::Excel { sheet } => excel::schema(path, sheet.as_deref()),
        Format::Sqlite { table } => sqlite::schema(path, table.as_deref()),
        geo => Ok(read_geo(path, geo)?.0),
    }
}

fn read_file(
    path: &Path,
    format: &Format,
    schema: SchemaRef,
) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + Send>> {
    Ok(match format {
        Format::Parquet => unreachable!("Parquet is read through ParquetTable"),
        Format::Csv { delimiter } => {
            let reader = arrow::csv::ReaderBuilder::new(schema)
                .with_format(csv_format(*delimiter))
                .with_batch_size(BATCH_ROWS)
                .build(File::open(path)?)?;
            Box::new(reader.map(|b| Ok(b?)))
        }
        Format::Ndjson => {
            let reader = arrow::json::ReaderBuilder::new(schema)
                .with_batch_size(BATCH_ROWS)
                .build(BufReader::new(File::open(path)?))?;
            Box::new(reader.map(|b| Ok(b?)))
        }
        Format::Json => {
            let values = read_json_array(path)?;
            let mut out = Vec::new();
            for chunk in values.chunks(BATCH_ROWS) {
                let mut decoder = arrow::json::ReaderBuilder::new(schema.clone())
                    .with_batch_size(BATCH_ROWS)
                    .build_decoder()?;
                decoder.serialize(chunk)?;
                if let Some(b) = decoder.flush()? {
                    out.push(Ok(b));
                }
            }
            Box::new(out.into_iter())
        }
        Format::Arrow => Box::new(open_ipc(path)?.map(|b| Ok(b?))),
        Format::Excel { sheet } => Box::new(
            excel::read(path, sheet.as_deref(), schema)?
                .into_iter()
                .map(Ok),
        ),
        Format::Sqlite { table } => Box::new(
            sqlite::read(path, table.as_deref(), schema)?
                .into_iter()
                .map(Ok),
        ),
        geo => Box::new(read_geo(path, geo)?.1.into_iter().map(Ok)),
    })
}

/// Reads a vector geospatial file whole.
fn read_geo(path: &Path, format: &Format) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    match format {
        Format::GeoJson => crate::geo::read_geojson(path),
        Format::FlatGeobuf => crate::geo::read_flatgeobuf(path),
        Format::Shapefile => crate::geo::read_shapefile(path),
        Format::GeoPackage { layer } => crate::geo::read_gpkg(path, layer.as_deref()),
        other => unreachable!("{other:?} isn't a vector format"),
    }
}

fn csv_format(delimiter: u8) -> arrow::csv::reader::Format {
    arrow::csv::reader::Format::default()
        .with_header(true)
        .with_delimiter(delimiter)
}

fn read_json_array(path: &Path) -> Result<Vec<serde_json::Value>> {
    let reader = BufReader::new(File::open(path)?);
    serde_json::from_reader(reader)
        .map_err(|e| Error::Unsupported(format!("{path:?} isn't a JSON array of objects: {e}")))
}

fn open_ipc(path: &Path) -> Result<Box<dyn RecordBatchReader + Send>> {
    let mut magic = [0u8; 6];
    let n = File::open(path)?.read(&mut magic)?;
    let file = BufReader::new(File::open(path)?);
    if &magic[..n] == b"ARROW1" {
        Ok(Box::new(arrow::ipc::reader::FileReader::try_new(
            file, None,
        )?))
    } else {
        Ok(Box::new(arrow::ipc::reader::StreamReader::try_new(
            file, None,
        )?))
    }
}

/// Text columns whose sampled values all parse as timestamps become timestamps (microseconds,
/// with no time zone).
fn refine_timestamps(schema: Schema, sample: Option<&RecordBatch>) -> SchemaRef {
    let Some(sample) = sample else {
        return Arc::new(schema);
    };
    let ts = DataType::Timestamp(TimeUnit::Microsecond, None);
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let col = sample.column(i);
            let parses = f.data_type() == &DataType::Utf8
                && col.null_count() < col.len()
                && looks_like_timestamps(col)
                && arrow::compute::cast_with_options(
                    col,
                    &ts,
                    &arrow::compute::CastOptions {
                        safe: false,
                        ..Default::default()
                    },
                )
                .is_ok();
            if parses {
                f.as_ref().clone().with_data_type(ts.clone())
            } else {
                f.as_ref().clone()
            }
        })
        .collect();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// Cheap guard before a cast: the first value looks like `YYYY-MM-DD…`.
fn looks_like_timestamps(col: &arrow::array::ArrayRef) -> bool {
    use arrow::array::AsArray;
    col.as_string::<i32>()
        .iter()
        .flatten()
        .next()
        .is_some_and(|s| {
            let b = s.as_bytes();
            b.len() >= 10 && b[4] == b'-' && b[7] == b'-' && b[..4].iter().all(u8::is_ascii_digit)
        })
}
