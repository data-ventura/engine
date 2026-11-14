//! Probes where time goes reading ARCO-ERA5 chunks: a raw HTTPS GET of the chunk's bytes versus a
//! full zarrs read (fetch and decode), on one warm connection pool.

use std::time::Instant;

use dv_data::zarr::ZarrArray;
use zarrs_object_store::object_store::{ObjectStoreExt, http::HttpBuilder, path::Path};

const URL: &str = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3";

#[tokio::main]
async fn main() {
    let a = ZarrArray::open_http(URL, "/2m_temperature")
        .await
        .expect("open");
    let store = HttpBuilder::new()
        .with_url(URL)
        .with_client_options(
            zarrs_object_store::object_store::ClientOptions::new().with_allow_http2(),
        )
        .build()
        .expect("store");
    let base = 1_081_900u64;
    store
        .get(&Path::from("2m_temperature/.zarray"))
        .await
        .expect("warm")
        .bytes()
        .await
        .expect("warm");
    for i in 0..6 {
        let key = Path::from(format!("2m_temperature/{}.0.0", base + i));
        let t = Instant::now();
        let r = store.get(&key).await.expect("get");
        let head = t.elapsed();
        let bytes = r.bytes().await.expect("bytes");
        println!(
            "raw get {}: first byte {head:?}, all {:?}, {} B",
            base + i,
            t.elapsed(),
            bytes.len()
        );
    }
    // Keys never read before (pass a fresh START as the first argument): one GET per chunk
    // against the same chunk split into N parallel range GETs, alternating, 300 ms apart.
    let start: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1_060_000);
    let size = |key: u64| {
        let store = &store;
        async move {
            store
                .head(&Path::from(format!("2m_temperature/{key}.0.0")))
                .await
                .expect("head")
                .size
        }
    };
    let shared = std::sync::Arc::new(
        HttpBuilder::new()
            .with_url(URL)
            .with_client_options(
                zarrs_object_store::object_store::ClientOptions::new().with_allow_http2(),
            )
            .build()
            .expect("store"),
    );
    let mut by_split: std::collections::BTreeMap<u64, Vec<u128>> = Default::default();
    for (i, key) in (start..start + 60).enumerate() {
        let n = [1u64, 2, 4][i % 3];
        let len = size(key).await;
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let path = Path::from(format!("2m_temperature/{key}.0.0"));
        let ranges: Vec<_> = (0..n).map(|j| len * j / n..len * (j + 1) / n).collect();
        let t = Instant::now();
        let mut set = tokio::task::JoinSet::new();
        for (j, r) in ranges.into_iter().enumerate() {
            let (store, path) = (shared.clone(), path.clone());
            set.spawn(async move { (j, store.get_range(&path, r).await.expect("range")) });
        }
        let parts: Vec<_> = set.join_all().await.into_iter().map(|(_, b)| b).collect();
        let elapsed = t.elapsed().as_millis();
        assert_eq!(parts.iter().map(|p| p.len() as u64).sum::<u64>(), len);
        by_split.entry(n).or_default().push(elapsed);
    }
    for (n, mut v) in by_split {
        v.sort_unstable();
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        println!(
            "split {n}: p50 {} p95 {} max {} (n={})",
            q(0.5),
            q(0.95),
            v[v.len() - 1],
            v.len()
        );
    }
    a.chunk_f32(&[base - 1, 0, 0]).await.expect("warm");
    for i in 10..16 {
        let t = Instant::now();
        let v = a.chunk_f32(&[base + i, 0, 0]).await.expect("chunk");
        println!(
            "zarrs read {}: {:?}, {} values",
            base + i,
            t.elapsed(),
            v.len()
        );
    }
}
