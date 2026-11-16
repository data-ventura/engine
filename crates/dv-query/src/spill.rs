//! Spilling to disk when a query outgrows its memory.
//!
//! Sorts, aggregates and sort-merge joins write what doesn't fit to temporary files and read it
//! back. DataFusion's hash join can't spill, so a query whose hash join runs out of memory is run
//! again with sort-merge joins, which can. Spill files go in a directory of the engine's own,
//! under [`Config::spill_dir`]: each file is deleted as soon as its operator is done with it, and
//! the directory when the engine is dropped. When even spilling can't make a query fit, it fails
//! with [`OutOfMemory`], which says why.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Poll;
use std::time::{Duration, Instant};

use arrow::array::RecordBatch;
use datafusion::error::DataFusionError;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::{ExecutionPlan, execute_stream};
use datafusion::prelude::DataFrame;
use dv_data::budget::Budget;
use dv_data::request::Generation;
use futures_util::StreamExt;
use futures_util::future::poll_fn;

use crate::Result;

/// The most bytes spill files may take at once, unless [`Config::max_spill`] says otherwise.
pub const DEFAULT_MAX_SPILL: u64 = 100 << 30;

/// How an [`Engine`](crate::Engine) uses memory and disk.
#[derive(Debug, Clone)]
pub struct Config {
    /// The shared memory budget queries reserve from.
    pub budget: Budget,
    /// The most memory queries may hold at once, below the shared budget. `None` lets them use
    /// the whole budget.
    pub query_memory: Option<u64>,
    /// Where queries spill. `None` uses the system's temporary directory. Either way the engine
    /// spills into a new directory of its own inside it.
    pub spill_dir: Option<PathBuf>,
    /// The most bytes spill files may take at once. `0` turns spilling off, so a query that
    /// outgrows its memory fails instead.
    pub max_spill: u64,
}

impl Config {
    /// Queries use the whole budget and spill up to [`DEFAULT_MAX_SPILL`] to the system's
    /// temporary directory.
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            query_memory: None,
            spill_dir: None,
            max_spill: DEFAULT_MAX_SPILL,
        }
    }
}

/// A query that needed more memory than it had, even after spilling what it could.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{}", self.message())]
pub struct OutOfMemory {
    /// The query memory limit, in bytes.
    pub limit: u64,
    /// The spill limit, in bytes; `0` when spilling is off.
    pub max_spill: u64,
    pub cause: Cause,
    /// What the operator that failed reported.
    pub detail: String,
}

/// Why spilling didn't save a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cause {
    /// Spilling is off.
    SpillOff,
    /// The spill files reached [`Config::max_spill`].
    SpillLimit,
    /// The disk the spill files are on is full.
    DiskFull,
    /// The query spilled to more files than the process may have open at once.
    OpenFiles,
    /// An operator that can't spill, or a single batch, needed more than the limit.
    Unspillable,
}

impl OutOfMemory {
    fn message(&self) -> String {
        let limit = size(self.limit);
        match self.cause {
            Cause::SpillOff => format!(
                "the query needs more than its {limit} of memory, and spilling to disk is off \
                 ({})",
                self.detail
            ),
            Cause::SpillLimit => format!(
                "the query needs more than its {limit} of memory, and its spill files reached \
                 their {} limit",
                size(self.max_spill)
            ),
            Cause::DiskFull => format!(
                "the query needs more than its {limit} of memory, and the disk it spills to is \
                 full"
            ),
            Cause::OpenFiles => format!(
                "the query needs more than its {limit} of memory, and spilled to more files than \
                 the system lets it open at once; a larger memory limit makes fewer, larger files"
            ),
            Cause::Unspillable => format!(
                "the query needs more than its {limit} of memory, even after spilling to disk \
                 ({})",
                self.detail
            ),
        }
    }
}

/// A size for messages: whole MB, or KB below 1 MB.
pub(crate) fn size(bytes: u64) -> String {
    if bytes >= 1_000_000 {
        format!("{} MB", bytes / 1_000_000)
    } else {
        format!("{} KB", bytes.div_ceil(1000))
    }
}

/// Bytes spilled by an engine's queries, kept in its session so every query plan can add to it.
#[derive(Debug, Default)]
pub(crate) struct Tally {
    pub(crate) spilled: AtomicU64,
}

/// Runs a query to completion, again with sort-merge joins if a hash join runs out of memory.
/// When a generation is given and passes, the query stops at its next batch.
pub(crate) async fn collect(
    df: DataFrame,
    generation: Option<&Generation>,
) -> Result<Vec<RecordBatch>> {
    let runtime = df.task_ctx().runtime_env();
    let before = runtime.memory_pool.reserved();
    let retry = df.clone();
    let err = match attempt(df, generation).await {
        Ok(Ok(batches)) => return Ok(batches),
        Ok(Err(Failed { err, hash_join })) if hash_join && spills(&runtime) && exhausted(&err) => {
            settle(&runtime, before).await;
            match attempt(sort_merge(retry), generation).await? {
                Ok(batches) => return Ok(batches),
                Err(Failed { err, .. }) => err,
            }
        }
        Ok(Err(Failed { err, .. })) => err,
        Err(other) => return Err(other),
    };
    Err(out_of_memory(&err, &runtime)
        .map(crate::Error::from)
        .unwrap_or_else(|| err.into()))
}

struct Failed {
    err: DataFusionError,
    hash_join: bool,
}

/// One run of a query. The outer error stops the query (cancelled); the inner one may be retried.
async fn attempt(
    df: DataFrame,
    generation: Option<&Generation>,
) -> Result<Result<Vec<RecordBatch>, Failed>> {
    let check = || generation.map_or(Ok(()), Generation::check);
    check()?;
    let task = Arc::new(df.task_ctx());
    let tally = task.session_config().get_extension::<Tally>();
    let plan = match df.create_physical_plan().await {
        Ok(plan) => plan,
        Err(err) => return Err(err.into()),
    };
    let hash_join = has_hash_join(plan.as_ref());
    let mut out = Vec::new();
    let result = async {
        let mut stream = execute_stream(Arc::clone(&plan), task)?;
        while let Some(batch) = stream.next().await {
            if check().is_err() {
                break;
            }
            out.push(batch?);
        }
        Ok::<_, DataFusionError>(())
    }
    .await;
    if let Some(tally) = tally {
        tally
            .spilled
            .fetch_add(spilled_bytes(plan.as_ref()), Ordering::Relaxed);
    }
    check()?;
    Ok(match result {
        Ok(()) => Ok(out),
        Err(err) => Err(Failed { err, hash_join }),
    })
}

/// Waits, up to a second, for a failed query's memory to go back to the pool. Dropping its stream
/// aborts its partitions' tasks, but they let go of what they hold only when they next run.
async fn settle(runtime: &RuntimeEnv, before: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.memory_pool.reserved() > before && Instant::now() < deadline {
        let mut yielded = false;
        poll_fn(|cx| {
            if yielded {
                Poll::Ready(())
            } else {
                yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
        .await;
    }
}

/// The same query, planned with sort-merge joins in place of hash joins.
fn sort_merge(df: DataFrame) -> DataFrame {
    let (state, plan) = df.into_parts();
    let mut config = state.config().clone();
    let options = config.options_mut();
    options.optimizer.prefer_hash_join = false;
    // DataFusion only plans a sort-merge join over several partitions.
    options.execution.target_partitions = options.execution.target_partitions.max(2);
    options.optimizer.repartition_joins = true;
    let state = SessionStateBuilder::new_from_existing(state)
        .with_config(config)
        .build();
    DataFrame::new(state, plan)
}

fn has_hash_join(plan: &dyn ExecutionPlan) -> bool {
    plan.is::<HashJoinExec>() || plan.children().iter().any(|c| has_hash_join(c.as_ref()))
}

fn spilled_bytes(plan: &dyn ExecutionPlan) -> u64 {
    let own = plan.metrics().and_then(|m| m.spilled_bytes()).unwrap_or(0) as u64;
    own + plan
        .children()
        .iter()
        .map(|c| spilled_bytes(c.as_ref()))
        .sum::<u64>()
}

fn spills(runtime: &RuntimeEnv) -> bool {
    runtime.disk_manager.tmp_files_enabled()
}

fn exhausted(err: &DataFusionError) -> bool {
    matches!(err.find_root(), DataFusionError::ResourcesExhausted(_))
}

/// Says why a query ran out of memory, if that's why it failed.
fn out_of_memory(err: &DataFusionError, runtime: &RuntimeEnv) -> Option<OutOfMemory> {
    let limit = match runtime.memory_pool.memory_limit() {
        datafusion::execution::memory_pool::MemoryLimit::Finite(n) => n as u64,
        _ => u64::MAX,
    };
    let max_spill = if spills(runtime) {
        runtime.disk_manager.max_temp_directory_size()
    } else {
        0
    };
    let io = io_cause(err);
    let cause = if let Some(cause) = io {
        cause
    } else if !exhausted(err) {
        return None;
    } else if max_spill == 0 {
        Cause::SpillOff
    } else {
        Cause::Unspillable
    };
    let detail = match err.find_root() {
        DataFusionError::ResourcesExhausted(m) => m.clone(),
        root => root.to_string(),
    };
    Some(OutOfMemory {
        limit,
        max_spill,
        cause,
        detail,
    })
}

/// A spill that failed because the spill limit, the disk or the open-file table was full.
fn io_cause(err: &DataFusionError) -> Option<Cause> {
    let mut source: Option<&dyn std::error::Error> = Some(err);
    while let Some(e) = source {
        if let Some(io) = e.downcast_ref::<std::io::Error>() {
            if io.kind() == std::io::ErrorKind::StorageFull {
                return Some(Cause::DiskFull);
            }
            // `ErrorKind::TooManyOpenFiles` isn't stable yet, but its name is.
            if format!("{:?}", io.kind()) == "TooManyOpenFiles" {
                return Some(Cause::OpenFiles);
            }
            // DataFusion reports its spill limit as an untyped I/O error; its message is the only
            // way to tell it apart.
            if io.to_string().contains("exceeded the allowable limit") {
                return Some(Cause::SpillLimit);
            }
            // An I/O error's `source` skips the error it wraps; look at that one too.
            if let Some(inner) = io.get_ref() {
                source = Some(inner);
                continue;
            }
        }
        source = e.source();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spill_failures_say_why() {
        let open_files = DataFusionError::Shared(Arc::new(DataFusionError::IoError(
            std::io::Error::other(std::io::Error::from_raw_os_error(24)),
        )));
        #[cfg(unix)]
        assert_eq!(io_cause(&open_files), Some(Cause::OpenFiles));
        let limit = DataFusionError::External(Box::new(std::io::Error::other(
            "The used disk space during the spilling process has exceeded the allowable limit of 1.0 MB.",
        )));
        assert_eq!(io_cause(&limit), Some(Cause::SpillLimit));
        let full = DataFusionError::IoError(std::io::ErrorKind::StorageFull.into());
        assert_eq!(io_cause(&full), Some(Cause::DiskFull));
        assert_eq!(io_cause(&DataFusionError::Plan("no".into())), None);
    }

    #[test]
    fn sizes_read_well() {
        assert_eq!(size(16 << 20), "16 MB");
        assert_eq!(size(64 << 10), "66 KB");
        assert_eq!(size(1), "1 KB");
    }
}
