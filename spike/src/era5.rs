//! ARCO-ERA5 on Google Cloud Storage, and a measure of the connection to it.
//!
//! S4 and S11 are judged on a stated connection (D10): each run records how long this connection
//! takes to fetch chunks nobody has read lately, and a limit on drawing a never-read step applies
//! only when that p95 is at most [`CONNECTION_P95_MS`].

use std::time::{Duration, Instant};

use zarrs_object_store::object_store::http::HttpBuilder;
use zarrs_object_store::object_store::path::Path;
use zarrs_object_store::object_store::{ClientOptions, ObjectStoreExt};

use crate::{Report, Summary};

pub const URL: &str = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3";
/// The connection the never-read limits assume: p95 of a raw GET of one never-read 2 m temperature
/// chunk (about 2.35 MB).
pub const CONNECTION_P95_MS: f64 = 200.0;
/// Hours since 1900-01-01T00 for 1979-01-01T00 and 2022-01-01T00: the span runs draw from.
/// Chunks fetched to warm the probe's connection before it measures.
pub const WARM_UP: u64 = 2;
const FIRST: u64 = 692_352;
const LAST: u64 = 1_069_440;

/// A first hour, different on every run, with `span` hours after it inside the span runs draw
/// from. Steps there are very unlikely to be in Google's edge cache.
pub fn fresh_hour(span: u64) -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // SplitMix64, so runs a second apart land far apart.
    let mut z = nanos.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    FIRST + z % (LAST - FIRST - span)
}

/// A warm connection for timing raw GETs of never-read chunks.
#[derive(Debug)]
pub struct Probe {
    store: zarrs_object_store::object_store::http::HttpStore,
}

impl Probe {
    /// Opens a connection and warms it with [`WARM_UP`] unmeasured chunks from `first_hour` on,
    /// including its transfer rate: the first full chunks on a new connection measured up to 1.5×
    /// slower than later ones.
    pub async fn new(first_hour: u64) -> Self {
        let store = HttpBuilder::new()
            .with_url(URL)
            .with_client_options(ClientOptions::new().with_allow_http2())
            .build()
            .expect("store");
        let probe = Self { store };
        for hour in first_hour..first_hour + WARM_UP {
            probe.time(hour).await;
        }
        probe
    }

    /// The time to GET the raw bytes of the chunk at `hour`.
    pub async fn time(&self, hour: u64) -> Duration {
        let key = Path::from(format!("2m_temperature/{hour}.0.0"));
        let t = Instant::now();
        let bytes = self
            .store
            .get(&key)
            .await
            .expect("get")
            .bytes()
            .await
            .expect("bytes");
        let elapsed = t.elapsed();
        assert!(bytes.len() > 1_000_000, "{key}: {} bytes", bytes.len());
        elapsed
    }
}

/// Times `n` never-read chunks after warming up, one after another, from `first_hour` on. Uses
/// `WARM_UP + n` hours.
pub async fn probe(first_hour: u64, n: u64) -> Vec<Duration> {
    let probe = Probe::new(first_hour).await;
    let mut times = Vec::new();
    for hour in first_hour + WARM_UP..first_hour + WARM_UP + n {
        times.push(probe.time(hour).await);
    }
    times
}

/// Records the connection's p95 and returns the limit to apply to a never-read step: `limit_ms`
/// if the connection is as fast as assumed, or `None` (recorded, not judged) if it isn't.
pub fn judge_on_connection(report: &mut Report, probe: &[Duration], limit_ms: f64) -> Option<f64> {
    let s = Summary::of(probe);
    let ok = s.p95 <= CONNECTION_P95_MS;
    report.metric(
        "connection.never_read_chunk.p95",
        s.p95,
        "ms",
        None,
        &format!(
            "raw GET, n={} p50={:.0} max={:.0}; {}",
            s.n,
            s.p50,
            s.max,
            if ok {
                format!("within the assumed {CONNECTION_P95_MS} ms, so never-read steps are judged")
            } else {
                format!("slower than the assumed {CONNECTION_P95_MS} ms, so never-read steps are recorded, not judged")
            }
        ),
    );
    ok.then_some(limit_ms)
}
