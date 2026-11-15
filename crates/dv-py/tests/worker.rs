//! The worker against a real Python: data round trips exactly, every input type works, user errors
//! and prints are handled, and the memory cap, cancellation and crash recovery behave as SPEC §4.9
//! says. Skipped (with a message) where neither `DV_PYTHON` nor uv is available.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use dv_data::budget::{Budget, Consumer};
use dv_data::request;
use dv_py::shm::Registry;
use dv_py::testing::{python, repo_root};
use dv_py::{Error, Worker, WorkerConfig};

fn setup(tag: &str, cap: Option<u64>) -> Option<(Worker, Budget)> {
    let Some(py) = python() else {
        eprintln!("skipped: no DV_PYTHON and no uv to build a test environment");
        return None;
    };
    let dir = std::env::temp_dir().join(format!("dv-py-it-{}-{tag}", std::process::id()));
    let (registry, _) = Registry::open(&dir).expect("registry");
    let mut config = WorkerConfig::python(py);
    if let Some(cap) = cap {
        config.memory_cap = cap;
    }
    config.cancel_grace = Duration::from_millis(500);
    let budget = Budget::new(4 << 30);
    let mut w = Worker::start(config, budget.clone(), Arc::new(registry)).expect("start");
    w.load(&fixture()).expect("load");
    Some((w, budget))
}

fn fixture() -> PathBuf {
    repo_root().join("fixtures/python/transforms.py")
}

fn table(n: usize) -> (SchemaRef, Vec<RecordBatch>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("i", DataType::Int64, false),
        Field::new("s", DataType::Utf8, true),
    ]));
    let x: ArrayRef = Arc::new(Float64Array::from_iter(
        (0..n).map(|i| (i % 7 != 3).then_some(i as f64 * 0.1 - 3.25)),
    ));
    let i: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..n as i64).map(|i| i * 31 - 7),
    ));
    let s: ArrayRef = Arc::new(StringArray::from_iter(
        (0..n).map(|i| (i % 5 != 0).then(|| format!("row {i} ✓"))),
    ));
    let b = RecordBatch::try_new(schema.clone(), vec![x, i, s]).unwrap();
    // Two batches, so multi-batch streams are covered.
    (schema, vec![b.slice(0, n / 3), b.slice(n / 3, n - n / 3)])
}

fn concat(schema: &SchemaRef, batches: &[RecordBatch]) -> RecordBatch {
    arrow::compute::concat_batches(schema, batches).unwrap()
}

#[test]
fn identity_round_trips_exactly() {
    let Some((mut w, budget)) = setup("identity", None) else {
        return;
    };
    let (schema, batches) = table(10_000);
    let out = w.transform("identity", &schema, &batches, None).unwrap();
    assert_eq!(out.schema.fields(), schema.fields());
    assert_eq!(concat(&out.schema, &out.batches), concat(&schema, &batches));
    // The result is held in the budget while it's alive, and released with it.
    assert!(budget.used_by(Consumer::Python) > 0);
    drop(out);
    assert_eq!(budget.used_by(Consumer::Python), 0);
}

#[test]
fn every_input_type() {
    let Some((mut w, _)) = setup("types", None) else {
        return;
    };
    let (schema, batches) = table(1000);
    let all = concat(&schema, &batches);
    let x = all
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    // polars, given only the declared column.
    let out = w.transform("doubled", &schema, &batches, None).unwrap();
    let got = concat(&out.schema, &out.batches);
    let x2 = got
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(got.num_columns(), 1);
    for i in 0..x.len() {
        assert_eq!(x2.is_null(i), x.is_null(i));
        if !x.is_null(i) {
            assert_eq!(x2.value(i), x.value(i) * 2.0);
        }
    }

    // pandas: the result keeps the input's columns and adds one.
    let out = w.transform("via_pandas", &schema, &batches, None).unwrap();
    let got = concat(&out.schema, &out.batches);
    assert_eq!(got.num_rows(), 1000);
    let plus = got
        .column_by_name("x_plus_one")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..x.len() {
        if !x.is_null(i) {
            assert_eq!(plus.value(i), x.value(i) + 1.0);
        }
    }

    // numpy, on one column. pandas/NumPy turn nulls into NaN; the sum over NaN is NaN.
    let out = w.transform("total", &schema, &batches, None).unwrap();
    let got = concat(&out.schema, &out.batches);
    assert_eq!(got.schema().field(0).name(), "total");
    assert_eq!(got.num_rows(), 1);
}

#[test]
fn user_errors_and_prints() {
    let Some((mut w, _)) = setup("errors", None) else {
        return;
    };
    let (schema, batches) = table(100);
    match w.transform("boom", &schema, &batches, None) {
        Err(Error::Python { message, traceback }) => {
            assert_eq!(message, "ValueError: bad input");
            assert!(traceback.unwrap().contains("transforms.py"));
        }
        other => panic!("expected a Python error, got {other:?}"),
    }
    let out = w.transform("chatty", &schema, &batches, None).unwrap();
    assert_eq!(
        out.batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        1
    );
    // Give the stderr thread a moment.
    std::thread::sleep(Duration::from_millis(100));
    assert!(w.stderr_tail().contains("hello from user code"));
    // A kept reference to the input doesn't break anything.
    w.transform("keeps_input", &schema, &batches, None).unwrap();
    w.transform("identity", &schema, &batches, None).unwrap();
    assert_eq!(w.starts(), 1, "the worker stayed warm throughout");
    assert!(matches!(
        w.transform("missing", &schema, &batches, None),
        Err(Error::Python { .. })
    ));
}

#[test]
fn memory_cap_stops_the_worker_and_it_restarts() {
    let Some((mut w, _)) = setup("cap", Some(768 << 20)) else {
        return;
    };
    let (schema, batches) = table(100);
    match w.transform("hog", &schema, &batches, None) {
        Err(Error::MemoryCap { used, cap }) => {
            assert!(used > cap);
            let msg = Error::MemoryCap { used, cap }.to_string();
            assert!(msg.contains("over its 805 MB memory cap"), "{msg}");
        }
        other => panic!("expected the memory cap, got {other:?}"),
    }
    // The next request starts a fresh worker, with the fixture loaded again.
    let out = w.transform("identity", &schema, &batches, None).unwrap();
    assert_eq!(
        out.batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        100
    );
    assert_eq!(w.starts(), 2);
}

#[test]
fn cancellation_interrupts_python_and_stops_stubborn_code() {
    let Some((mut w, _)) = setup("cancel", None) else {
        return;
    };
    let (schema, batches) = table(100);
    let (submit, _requests) = request::channel::<()>();

    // Interruptible code stops at the cancel message; the worker stays warm.
    let g = submit.submit(());
    let t0 = Instant::now();
    let r = supersede_after(&submit, Duration::from_millis(200), || {
        w.transform("slow", &schema, &batches, Some(&g))
    });
    assert!(
        matches!(r, Err(Error::Cancelled { stopped: false })),
        "{r:?}"
    );
    assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
    assert_eq!(w.starts(), 1);

    // Code that ignores it is stopped after the grace period (500 ms here).
    let g = submit.submit(());
    let t0 = Instant::now();
    let r = supersede_after(&submit, Duration::from_millis(200), || {
        w.transform("stubborn", &schema, &batches, Some(&g))
    });
    assert!(
        matches!(r, Err(Error::Cancelled { stopped: true })),
        "{r:?}"
    );
    let took = t0.elapsed();
    assert!(
        took >= Duration::from_millis(700) && took < Duration::from_secs(3),
        "{took:?}"
    );

    // And the next request gets a fresh worker.
    w.transform("identity", &schema, &batches, None).unwrap();
    assert_eq!(w.starts(), 2);
}

/// Runs `f` while another thread makes a newer request after `delay`.
fn supersede_after<T>(
    submit: &request::Submitter<()>,
    delay: Duration,
    f: impl FnOnce() -> T,
) -> T {
    std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(delay);
            submit.submit(());
        });
        f()
    })
}

#[test]
fn a_crash_is_reported_and_recovered() {
    let Some((mut w, _)) = setup("crash", None) else {
        return;
    };
    let (schema, batches) = table(10);
    let pid = w.info().unwrap()["pid"].as_i64().unwrap();
    // SAFETY: signals the worker process this test started.
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
    std::thread::sleep(Duration::from_millis(100));
    // The dead worker is noticed and replaced before the request is sent.
    w.transform("identity", &schema, &batches, None).unwrap();
    assert_eq!(w.starts(), 2);
}
