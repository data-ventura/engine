//! Lazy table queries on Apache DataFusion: scans, filters, aggregates and joins by key, time and place.
//!
//! Part of the Data Ventura engine. See the repository README for the crate map.
//!
//! An [`Engine`] holds the registered tables. Parquet, CSV, NDJSON and Arrow files are scanned
//! lazily by DataFusion, with filters and projections pushed into the scan; other formats are read
//! by `dv-data` into memory first. Query memory reserves from the shared budget. SQL is read-only.
//! Results are collected under a [`Generation`], and a query whose generation has passed stops at
//! its next batch.

use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::file_format::arrow::ArrowFormat;
use datafusion::datasource::file_format::csv::CsvFormat;
use datafusion::datasource::file_format::json::JsonFormat;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{DataFrame, SQLOptions, SessionConfig, SessionContext};
use dv_data::budget::Budget;
use dv_data::request::{Cancelled, Generation};
use dv_data::table::{Format, Table};
use futures_util::StreamExt;

pub mod join;
pub mod pool;
pub mod select;
pub mod time;

pub use select::selection_expr;

/// Errors from planning or running a query.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    DataFusion(#[from] datafusion::error::DataFusionError),
    #[error(transparent)]
    Data(#[from] dv_data::Error),
    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),
    #[error(transparent)]
    Cancelled(#[from] Cancelled),
    #[error(transparent)]
    Join(#[from] join::JoinError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Registered tables and the session that queries them.
#[derive(Clone)]
pub struct Engine {
    ctx: SessionContext,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("session", &self.ctx.session_id())
            .finish()
    }
}

impl Engine {
    /// An engine whose query memory comes from `budget`.
    pub fn new(budget: Budget) -> Result<Self> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(pool::BudgetPool::new(budget)))
            .build_arc()?;
        let config = SessionConfig::new().with_information_schema(false);
        Ok(Self {
            ctx: SessionContext::new_with_config_rt(config, runtime),
        })
    }

    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }

    /// Registers a table under `name`, replacing any table of that name.
    pub async fn register(&self, name: &str, table: &Table) -> Result<()> {
        let format: Option<Arc<dyn FileFormat>> = match table.format() {
            Format::Parquet => Some(Arc::new(ParquetFormat::default())),
            Format::Csv { delimiter } => Some(Arc::new(
                CsvFormat::default()
                    .with_has_header(true)
                    .with_delimiter(*delimiter),
            )),
            Format::Ndjson => Some(Arc::new(JsonFormat::default())),
            Format::Arrow => Some(Arc::new(ArrowFormat)),
            Format::Json | Format::Excel { .. } | Format::Sqlite { .. } => None,
        };
        let provider: Arc<dyn datafusion::catalog::TableProvider> = match format {
            Some(format) => {
                let urls = table
                    .files()
                    .iter()
                    .map(|f| {
                        let abs = std::path::absolute(f).map_err(dv_data::Error::from)?;
                        Ok(ListingTableUrl::parse(abs.to_string_lossy())?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let options = ListingOptions::new(format).with_file_extension("");
                let config = ListingTableConfig::new_with_multi_paths(urls)
                    .with_listing_options(options)
                    .with_schema(table.schema().clone());
                Arc::new(ListingTable::try_new(config)?)
            }
            None => {
                let batches = table.batches()?.collect::<dv_data::Result<Vec<_>>>()?;
                Arc::new(MemTable::try_new(table.schema().clone(), vec![batches])?)
            }
        };
        self.ctx.deregister_table(name)?;
        self.ctx.register_table(name, provider)?;
        Ok(())
    }

    /// A registered table, as a lazy DataFrame.
    pub async fn table(&self, name: &str) -> Result<DataFrame> {
        Ok(self.ctx.table(name).await?)
    }

    /// Plans a read-only SQL query. Statements that create, change or drop anything are refused.
    pub async fn sql(&self, sql: &str) -> Result<DataFrame> {
        let options = SQLOptions::new()
            .with_allow_ddl(false)
            .with_allow_dml(false)
            .with_allow_statements(false);
        Ok(self.ctx.sql_with_options(sql, options).await?)
    }

    /// Runs a query to completion, unless `generation` passes first: then it stops at the next
    /// batch and returns [`Cancelled`], and nothing partial is returned.
    pub async fn collect(
        &self,
        df: DataFrame,
        generation: &Generation,
    ) -> Result<Vec<RecordBatch>> {
        generation.check()?;
        let mut stream = df.execute_stream().await?;
        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            generation.check()?;
            out.push(batch?);
        }
        generation.check()?;
        Ok(out)
    }
}
