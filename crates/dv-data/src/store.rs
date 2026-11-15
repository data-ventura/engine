//! Object-store layers for reads: retries that make failed reads recover, and fault injection to
//! prove they do.
//!
//! [`RetryStore`] wraps any store. Each read gets a limited time to its first byte and may not stall
//! longer than a limit between body chunks. A read is retried, with exponential backoff and jitter,
//! when it fails for a reason that can pass (server errors, timeouts, dropped connections, a body
//! shorter than the response promised). It isn't retried when the answer is definite: not found
//! (which Zarr reads as a missing chunk, drawn as the fill value), permission denied, or a bad path.
//! Each read's body is fetched whole and its length checked against the range the store said it
//! returned, so a truncated chunk can never be decoded as if it were complete.
//!
//! [`FaultStore`] sits under it in tests and the S12 spike. It fails a seeded fraction of reads
//! with a 5xx error, cuts bodies short, hangs, and can drop the "network" on demand, failing new
//! reads and the bodies of reads in flight.
//!
//! The HTTP client's own retries are turned off (see [`http_store`]): they cover only the request,
//! not the body, and their defaults wait up to three minutes, which no view can.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures_util::stream::{self, BoxStream, StreamExt};
use zarrs_object_store::object_store::http::HttpBuilder;
use zarrs_object_store::object_store::path::Path;
use zarrs_object_store::object_store::{
    self, ClientOptions, CopyOptions, Error as StoreError, GetOptions, GetResult, GetResultPayload,
    ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions,
    PutPayload, PutResult, RetryConfig,
};

type StoreResult<T> = object_store::Result<T>;

/// An HTTP(S) store with the client's own retries off, for wrapping in a [`RetryStore`]. HTTP/2
/// measured about twice as fast as HTTP/1.1 per chunk against Google Cloud Storage.
pub fn http_store(url: &str) -> StoreResult<Box<dyn ObjectStore>> {
    let options = ClientOptions::new().with_allow_http2();
    let retry = RetryConfig {
        max_retries: 0,
        ..RetryConfig::default()
    };
    Ok(Box::new(
        HttpBuilder::new()
            .with_url(url)
            .with_client_options(options)
            .with_retry(retry)
            .build()?,
    ))
}

/// How [`RetryStore`] retries.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Attempts per read, including the first.
    pub attempts: u32,
    /// Wait before the first retry; doubled each time, with ±50% jitter.
    pub backoff: Duration,
    /// Time allowed until the response headers arrive.
    pub first_byte: Duration,
    /// Longest pause allowed between body chunks.
    pub stall: Duration,
}

impl Default for RetryPolicy {
    /// Four attempts, 100/200/400 ms apart. A never-read ERA5 chunk takes p95 ≤ 300 ms here, so
    /// 2 s to the first byte and 2 s per stall are well clear of a slow but healthy read, even with
    /// 32 reads sharing a connection (S9). A read that keeps failing gives up after about 0.7 s
    /// when failures are immediate (a dropped network), and within about 9 s when every attempt
    /// hangs.
    fn default() -> Self {
        Self {
            attempts: 4,
            backoff: Duration::from_millis(100),
            first_byte: Duration::from_secs(2),
            stall: Duration::from_secs(2),
        }
    }
}

/// Counts for reports and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RetryStats {
    pub reads: u64,
    /// Attempts after the first.
    pub retries: u64,
    /// Reads that failed after their last attempt.
    pub gave_up: u64,
    /// Attempts whose body came up short.
    pub truncated: u64,
    /// Attempts that timed out (first byte or stall).
    pub timeouts: u64,
}

#[derive(Debug, Default)]
struct Counters {
    reads: AtomicU64,
    retries: AtomicU64,
    gave_up: AtomicU64,
    truncated: AtomicU64,
    timeouts: AtomicU64,
}

/// Retries reads of the store it wraps. See the module docs. Writes pass straight through: the
/// engine never writes to a source.
#[derive(Debug, Clone)]
pub struct RetryStore {
    inner: Arc<dyn ObjectStore>,
    policy: RetryPolicy,
    counters: Arc<Counters>,
    seed: Arc<AtomicU64>,
}

impl fmt::Display for RetryStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RetryStore({})", self.inner)
    }
}

impl RetryStore {
    pub fn new(inner: impl Into<Arc<dyn ObjectStore>>, policy: RetryPolicy) -> Self {
        Self {
            inner: inner.into(),
            policy,
            counters: Arc::default(),
            seed: Arc::new(AtomicU64::new(0x9e37_79b9_7f4a_7c15)),
        }
    }

    pub fn stats(&self) -> RetryStats {
        let c = &self.counters;
        RetryStats {
            reads: c.reads.load(Ordering::Relaxed),
            retries: c.retries.load(Ordering::Relaxed),
            gave_up: c.gave_up.load(Ordering::Relaxed),
            truncated: c.truncated.load(Ordering::Relaxed),
            timeouts: c.timeouts.load(Ordering::Relaxed),
        }
    }

    /// One attempt: headers within `first_byte`, then the whole body with no stall over `stall`,
    /// checked against the length the response promised.
    async fn attempt(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
        let p = self.policy;
        let head = options.head;
        let result = tokio::time::timeout(p.first_byte, self.inner.get_opts(location, options))
            .await
            .map_err(|_| {
                self.counters.timeouts.fetch_add(1, Ordering::Relaxed);
                transient(format!("no response within {:?}", p.first_byte))
            })??;
        if head {
            return Ok(result);
        }
        let expected = result.range.end - result.range.start;
        let (meta, range, attributes, extensions) = (
            result.meta.clone(),
            result.range.clone(),
            result.attributes.clone(),
            result.extensions.clone(),
        );
        let mut body = result.into_stream();
        let mut buf = BytesMut::with_capacity(expected as usize);
        loop {
            match tokio::time::timeout(p.stall, body.next()).await {
                Err(_) => {
                    self.counters.timeouts.fetch_add(1, Ordering::Relaxed);
                    return Err(transient(format!("body stalled for {:?}", p.stall)));
                }
                Ok(None) => break,
                Ok(Some(chunk)) => buf.extend_from_slice(&chunk?),
            }
        }
        if buf.len() as u64 != expected {
            self.counters.truncated.fetch_add(1, Ordering::Relaxed);
            return Err(transient(format!(
                "body was {} bytes, not the {expected} the response promised",
                buf.len()
            )));
        }
        let bytes: Bytes = buf.freeze();
        Ok(GetResult {
            payload: GetResultPayload::Stream(stream::once(async move { Ok(bytes) }).boxed()),
            meta,
            range,
            attributes,
            extensions,
        })
    }

    fn jitter(&self, d: Duration) -> Duration {
        // A small xorshift, so retries from many reads don't line up; not for anything secret.
        let mut x = self.seed.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.seed.store(x, Ordering::Relaxed);
        let f = 0.5 + (x >> 11) as f64 / (1u64 << 53) as f64; // 0.5..1.5
        d.mul_f64(f)
    }
}

/// Whether a failed read could succeed if tried again.
fn retryable(e: &StoreError) -> bool {
    !matches!(
        e,
        StoreError::NotFound { .. }
            | StoreError::InvalidPath { .. }
            | StoreError::NotSupported { .. }
            | StoreError::Precondition { .. }
            | StoreError::NotModified { .. }
            | StoreError::NotImplemented { .. }
            | StoreError::PermissionDenied { .. }
            | StoreError::Unauthenticated { .. }
    )
}

fn transient(msg: String) -> StoreError {
    StoreError::Generic {
        store: "retry",
        source: msg.into(),
    }
}

#[async_trait]
impl ObjectStore for RetryStore {
    async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
        self.counters.reads.fetch_add(1, Ordering::Relaxed);
        let mut wait = self.policy.backoff;
        let mut attempt = 1;
        loop {
            match self.attempt(location, options.clone()).await {
                Ok(r) => return Ok(r),
                Err(e) if retryable(&e) && attempt < self.policy.attempts => {
                    self.counters.retries.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(self.jitter(wait)).await;
                    wait *= 2;
                    attempt += 1;
                }
                Err(e) => {
                    if retryable(&e) {
                        self.counters.gave_up.fetch_add(1, Ordering::Relaxed);
                    }
                    return Err(e);
                }
            }
        }
    }

    async fn put_opts(&self, l: &Path, p: PutPayload, o: PutOptions) -> StoreResult<PutResult> {
        self.inner.put_opts(l, p, o).await
    }

    async fn put_multipart_opts(
        &self,
        l: &Path,
        o: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(l, o).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, o: CopyOptions) -> StoreResult<()> {
        self.inner.copy_opts(from, to, o).await
    }
}

/// What [`FaultStore`] does to reads, as a fraction of attempts each.
#[derive(Debug, Clone, Copy, Default)]
pub struct FaultRates {
    /// Answered with a 5xx-style server error.
    pub server_error: f64,
    /// The body is cut short (after the response promised the full length).
    pub truncate: f64,
    /// No response until the caller gives up.
    pub hang: f64,
}

/// Switches and counters shared between a [`FaultStore`] and the test driving it.
#[derive(Debug, Default)]
pub struct FaultControl {
    down: AtomicBool,
    seed: AtomicU64,
    /// Faults injected so far, by kind.
    pub server_errors: AtomicU64,
    pub truncations: AtomicU64,
    pub hangs: AtomicU64,
    pub dropped: AtomicU64,
}

impl FaultControl {
    pub fn new(seed: u64) -> Arc<Self> {
        Arc::new(Self {
            seed: AtomicU64::new(seed | 1),
            ..Self::default()
        })
    }

    /// Drops or restores the network: while down, new reads fail at once and bodies in flight
    /// fail at their next chunk.
    pub fn set_down(&self, down: bool) {
        self.down.store(down, Ordering::Release);
    }

    pub fn is_down(&self) -> bool {
        self.down.load(Ordering::Acquire)
    }

    /// A uniform number in `0..1` from a seeded xorshift, so a run's faults are reproducible
    /// given the same order of reads.
    fn roll(&self) -> f64 {
        let x = self
            .seed
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |mut x| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                Some(x)
            })
            .expect("always Some");
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn injected(&self) -> u64 {
        self.server_errors.load(Ordering::Relaxed)
            + self.truncations.load(Ordering::Relaxed)
            + self.hangs.load(Ordering::Relaxed)
            + self.dropped.load(Ordering::Relaxed)
    }
}

/// Injects faults into the reads of the store it wraps. For tests and the S12 spike.
#[derive(Debug, Clone)]
pub struct FaultStore {
    inner: Arc<dyn ObjectStore>,
    rates: FaultRates,
    control: Arc<FaultControl>,
}

impl fmt::Display for FaultStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FaultStore({})", self.inner)
    }
}

impl FaultStore {
    pub fn new(
        inner: impl Into<Arc<dyn ObjectStore>>,
        rates: FaultRates,
        control: Arc<FaultControl>,
    ) -> Self {
        Self {
            inner: inner.into(),
            rates,
            control,
        }
    }
}

fn dropped() -> StoreError {
    StoreError::Generic {
        store: "fault",
        source: "connection reset: network unreachable (injected)".into(),
    }
}

#[async_trait]
impl ObjectStore for FaultStore {
    async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
        let c = &self.control;
        if c.is_down() {
            c.dropped.fetch_add(1, Ordering::Relaxed);
            return Err(dropped());
        }
        let r = self.rates;
        let roll = c.roll();
        if roll < r.server_error {
            c.server_errors.fetch_add(1, Ordering::Relaxed);
            return Err(StoreError::Generic {
                store: "fault",
                source: "HTTP 503 Service Unavailable (injected)".into(),
            });
        }
        if roll < r.server_error + r.hang {
            c.hangs.fetch_add(1, Ordering::Relaxed);
            std::future::pending::<()>().await;
        }
        let truncate = roll < r.server_error + r.hang + r.truncate;
        let result = self.inner.get_opts(location, options).await?;
        if c.is_down() {
            c.dropped.fetch_add(1, Ordering::Relaxed);
            return Err(dropped());
        }
        let expected = result.range.end - result.range.start;
        let (meta, range, attributes, extensions) = (
            result.meta.clone(),
            result.range.clone(),
            result.attributes.clone(),
            result.extensions.clone(),
        );
        let control = self.control.clone();
        let mut sent = 0u64;
        let cut = if truncate {
            c.truncations.fetch_add(1, Ordering::Relaxed);
            Some(expected / 2)
        } else {
            None
        };
        let body = result
            .into_stream()
            .map(move |chunk| {
                if control.is_down() {
                    control.dropped.fetch_add(1, Ordering::Relaxed);
                    return Some(Err(dropped()));
                }
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => return Some(Err(e)),
                };
                match cut {
                    Some(cut) if sent >= cut => None,
                    Some(cut) => {
                        let keep = (cut - sent).min(chunk.len() as u64) as usize;
                        sent += keep as u64;
                        Some(Ok(chunk.slice(..keep)))
                    }
                    None => Some(Ok(chunk)),
                }
            })
            .take_while(|c| std::future::ready(c.is_some()))
            .map(|c| c.expect("taken while Some"))
            .boxed();
        Ok(GetResult {
            payload: GetResultPayload::Stream(body),
            meta,
            range,
            attributes,
            extensions,
        })
    }

    async fn put_opts(&self, l: &Path, p: PutPayload, o: PutOptions) -> StoreResult<PutResult> {
        self.inner.put_opts(l, p, o).await
    }

    async fn put_multipart_opts(
        &self,
        l: &Path,
        o: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(l, o).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, o: CopyOptions) -> StoreResult<()> {
        self.inner.copy_opts(from, to, o).await
    }
}
