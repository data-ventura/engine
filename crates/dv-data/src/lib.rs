//! Readers for tables, arrays and geospatial data, plus object storage and SFTP access.
//!
//! Part of the Data Ventura engine. See the repository README for the crate map.

#[cfg(unix)]
pub mod alloc;
pub mod budget;
pub mod parquet;
pub mod prefetch;
pub mod request;
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
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
