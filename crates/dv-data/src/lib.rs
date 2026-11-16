//! Readers for tables, arrays and geospatial data, plus object storage and SFTP access.
//!
//! Part of the Data Ventura engine. See the repository README for the crate map.

#[cfg(unix)]
pub mod alloc;
pub mod array;
pub mod budget;
#[cfg(feature = "duckdb")]
pub mod duckdb;
mod excel;
pub mod geo;
#[cfg(feature = "grib")]
pub mod grib;
pub mod parquet;
pub mod prefetch;
pub mod pressure;
pub mod raster;
pub mod request;
pub mod sqlite;
pub mod store;
pub mod table;
pub mod zarr;

/// Errors from opening or reading a source.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parquet error: {0}")]
    Parquet(#[from] ::parquet::errors::ParquetError),
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("Zarr error: {0}")]
    Zarr(String),
    #[error("{0}")]
    Unsupported(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// DuckDB files in a build without the `duckdb` feature: every call says so.
#[cfg(not(feature = "duckdb"))]
pub mod duckdb {
    use std::path::Path;

    use arrow::array::RecordBatch;
    use arrow::datatypes::SchemaRef;

    use crate::{Error, Result};

    fn off(path: &Path) -> Error {
        Error::Unsupported(format!(
            "{path:?} needs DuckDB support, which this build doesn't include (the `duckdb` feature)"
        ))
    }

    pub fn tables(path: &Path) -> Result<Vec<String>> {
        Err(off(path))
    }

    pub(crate) fn schema(path: &Path, _table: Option<&str>) -> Result<SchemaRef> {
        Err(off(path))
    }

    pub(crate) fn read(
        path: &Path,
        _table: Option<&str>,
        _schema: SchemaRef,
    ) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + Send>> {
        Err(off(path))
    }
}
