//! Failed reads (S12) on the ERA5-like fixture: with faults injected under the retry layer, every
//! read either returns exactly the values a clean read does or fails with an error. Never a wrong
//! value.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use dv_data::store::{FaultControl, FaultRates, FaultStore, RetryPolicy, RetryStore};
use dv_data::zarr::ZarrArray;
use zarrs_object_store::object_store::local::LocalFileSystem;
use zarrs_object_store::object_store::{ObjectStore, ObjectStoreExt};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/arco-like.zarr")
}

fn local() -> Box<dyn ObjectStore> {
    Box::new(LocalFileSystem::new_with_prefix(fixture()).unwrap())
}

/// Quick timeouts, so hung reads cost tens of milliseconds in tests.
fn quick(attempts: u32) -> RetryPolicy {
    RetryPolicy {
        attempts,
        backoff: Duration::from_millis(5),
        first_byte: Duration::from_millis(200),
        stall: Duration::from_millis(200),
    }
}

async fn reference() -> Vec<Vec<f32>> {
    let a = ZarrArray::open_dir(fixture(), "/t2m").await.unwrap();
    let mut steps = Vec::new();
    for t in 0..4 {
        steps.push(a.chunk_f32(&[t, 0, 0]).await.unwrap());
    }
    steps
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[tokio::test]
async fn faults_are_retried_and_never_give_wrong_values() {
    let want = reference().await;
    let control = FaultControl::new(42);
    let rates = FaultRates {
        server_error: 0.2,
        truncate: 0.2,
        hang: 0.05,
    };
    let retry = RetryStore::new(
        Box::new(FaultStore::new(local(), rates, control.clone())) as Box<dyn ObjectStore>,
        quick(8),
    );
    // Opening reads metadata through the same faulty path.
    let a = ZarrArray::open_store(Box::new(retry.clone()), "/t2m")
        .await
        .unwrap();
    let (mut ok, mut failed) = (0, 0);
    for i in 0..200 {
        let t = i % 4;
        match a.chunk_f32(&[t, 0, 0]).await {
            Ok(v) => {
                assert!(same(&v, &want[t as usize]), "step {t}: wrong values");
                ok += 1;
            }
            Err(_) => failed += 1,
        }
    }
    let s = retry.stats();
    assert!(control.truncations.load(Ordering::Relaxed) > 10);
    assert!(control.server_errors.load(Ordering::Relaxed) > 10);
    assert!(control.hangs.load(Ordering::Relaxed) > 0);
    assert!(
        s.retries > 20 && s.truncated > 10 && s.timeouts > 0,
        "{s:?}"
    );
    // With 8 attempts and a 45% fault rate, a read fails outright about 0.2% of the time.
    assert!(ok >= 195, "{ok} ok, {failed} failed, {s:?}");
}

#[tokio::test]
async fn a_truncated_body_is_an_error_not_short_data() {
    let control = FaultControl::new(7);
    let all_truncated = FaultRates {
        truncate: 1.0,
        ..FaultRates::default()
    };
    let retry = RetryStore::new(
        Box::new(FaultStore::new(local(), all_truncated, control.clone())) as Box<dyn ObjectStore>,
        quick(3),
    );
    let err = retry
        .get(&"t2m/0.0.0".into())
        .await
        .expect_err("a truncated read fails");
    assert!(err.to_string().contains("not the"), "{err}");
    assert_eq!(retry.stats().truncated, 3);
    assert_eq!(retry.stats().gave_up, 1);
}

#[tokio::test]
async fn a_missing_chunk_is_the_fill_value_and_not_retried() {
    let control = FaultControl::new(1);
    let retry = RetryStore::new(
        Box::new(FaultStore::new(local(), FaultRates::default(), control)) as Box<dyn ObjectStore>,
        quick(4),
    );
    let a = ZarrArray::open_store(Box::new(retry.clone()), "/t2m")
        .await
        .unwrap();
    // The fixture has no chunk 2: Zarr's fill value, NaN, is the right answer.
    let v = a.chunk_f32(&[2, 0, 0]).await.unwrap();
    assert!(v.iter().all(|x| x.is_nan()));
    assert_eq!(retry.stats().retries, 0);
}

#[tokio::test]
async fn a_network_drop_fails_fast_and_recovers() {
    let want = reference().await;
    let control = FaultControl::new(3);
    let retry = RetryStore::new(
        Box::new(FaultStore::new(
            local(),
            FaultRates::default(),
            control.clone(),
        )) as Box<dyn ObjectStore>,
        RetryPolicy::default(),
    );
    let a = ZarrArray::open_store(Box::new(retry.clone()), "/t2m")
        .await
        .unwrap();
    control.set_down(true);
    let t0 = Instant::now();
    let r = a.chunk_f32(&[1, 0, 0]).await;
    let took = t0.elapsed();
    assert!(r.is_err());
    // Four immediate failures, 100 + 200 + 400 ms of backoff with ±50% jitter.
    assert!(took < Duration::from_millis(1200), "{took:?}");
    control.set_down(false);
    let v = a.chunk_f32(&[1, 0, 0]).await.unwrap();
    assert!(same(&v, &want[1]));
}
