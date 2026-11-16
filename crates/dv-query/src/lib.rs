//! Lazy table queries on Apache DataFusion: scans, filters, aggregates and joins by key, time and place.
//!
//! Part of the Data Ventura engine. See the repository README for the crate map.
//!
//! An [`Engine`] holds the registered tables. Parquet, CSV, NDJSON and Arrow files are scanned
//! lazily by DataFusion, with filters and projections pushed into the scan; other formats are read
//! by `dv-data` into memory first. Query memory reserves from the shared budget, and queries that
//! outgrow it spill to disk (see [`spill`]). SQL is read-only.
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
use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{DataFrame, SQLOptions, SessionConfig, SessionContext};
use dv_data::budget::Budget;
use dv_data::request::{Cancelled, Generation};
use dv_data::table::{Format, Table};

pub mod join;
pub mod pool;
pub mod select;
pub mod spill;
pub mod time;

pub use select::selection_expr;
pub use spill::{Config, OutOfMemory};

/// The least query memory per partition: room for the sort, aggregate and exchange of one
/// partition to each hold a few batches.
const PARTITION_MEMORY: u64 = 16 << 20;

/// The most spill files one merge pass reads at once.
const MERGE_FAN_IN: usize = 16;

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
    #[error(transparent)]
    OutOfMemory(#[from] OutOfMemory),
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
    /// An engine whose query memory comes from `budget`, spilling to the system's temporary
    /// directory.
    pub fn new(budget: Budget) -> Result<Self> {
        Self::with_config(Config::new(budget))
    }

    /// An engine that uses memory and disk as `config` says. Creates the spill directory if it's
    /// missing.
    pub fn with_config(config: Config) -> Result<Self> {
        let pool = pool::BudgetPool::with_limit(config.budget, config.query_memory);
        let limit = pool.limit();
        let mode = match (&config.spill_dir, config.max_spill) {
            (_, 0) => DiskManagerMode::Disabled,
            (None, _) => DiskManagerMode::OsTmpDirectory,
            (Some(dir), _) => {
                std::fs::create_dir_all(dir).map_err(dv_data::Error::from)?;
                DiskManagerMode::Directories(vec![dir.clone()])
            }
        };
        // A merge holds a batch or two from each spill file it reads, and takes that memory
        // greedily, without being able to spill it; several sorts merge at once. Capping how many
        // files each merge reads keeps them from starving each other under a small limit, and
        // within the open-file limit (256 by default on macOS) under any, at the cost of more
        // merge passes.
        let fan_in = (limit / (2 << 20)).clamp(2, MERGE_FAN_IN as u64) as usize;
        let mut disk = DiskManagerBuilder::default()
            .with_mode(mode)
            .with_max_spill_merge_fan_in(fan_in);
        if config.max_spill > 0 {
            disk = disk.with_max_temp_directory_size(config.max_spill);
        }
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(pool))
            .with_disk_manager_builder(disk)
            .build_arc()?;
        let session = SessionConfig::new().with_information_schema(false);
        // Spilling operators share the limit equally, a few per partition, and each needs room
        // for at least a batch or two. Under a small limit, fewer partitions keep those shares
        // workable; under the usual budget this changes nothing.
        let partitions = session
            .target_partitions()
            .min((limit / PARTITION_MEMORY).max(1) as usize);
        // A sort holds this much back for merging what it spilled, in every partition, and
        // can't spill it. DataFusion's 10 MB would use up a small limit before anything sorts.
        // A sort-merge join sorts both sides, over at least two partitions, and all of them
        // together should leave most of the quarter kept for what can't spill.
        let sorts = 2 * partitions.max(2) as u64;
        let merge = (limit / 16 / sorts).clamp(64 << 10, 10 << 20);
        // Each operator holds at least a batch; under a small limit, smaller batches let it make
        // progress on its share.
        let batch = (limit / 4096).clamp(1024, 8192) as usize;
        let session = session
            .with_target_partitions(partitions)
            .with_batch_size(batch)
            .with_sort_spill_reservation_bytes(merge as usize)
            .with_extension(Arc::new(spill::Tally::default()));
        Ok(Self {
            ctx: SessionContext::new_with_config_rt(session, runtime),
        })
    }

    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }

    /// The directory this engine's queries spill into, once one has spilled.
    pub fn spill_dir(&self) -> Option<std::path::PathBuf> {
        self.ctx
            .runtime_env()
            .disk_manager
            .temp_dir_paths()
            .into_iter()
            .next()
    }

    /// Bytes held in spill files now.
    pub fn spilling(&self) -> u64 {
        self.ctx.runtime_env().disk_manager.used_disk_space()
    }

    /// Bytes this engine's queries have spilled to disk so far, counted when each query ends.
    pub fn spilled(&self) -> u64 {
        self.ctx
            .state()
            .config()
            .get_extension::<spill::Tally>()
            .map_or(0, |t| t.spilled.load(std::sync::atomic::Ordering::Relaxed))
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
            _ => None,
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
    /// batch and returns [`Cancelled`], and nothing partial is returned. A query that outgrows
    /// its memory spills to disk; if that isn't enough, it fails with [`OutOfMemory`].
    pub async fn collect(
        &self,
        df: DataFrame,
        generation: &Generation,
    ) -> Result<Vec<RecordBatch>> {
        spill::collect(df, Some(generation)).await
    }
}
