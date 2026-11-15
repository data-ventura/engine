//! S6: a Python `@dv.transform` on 1M rows adds under 50 ms of overhead (warm worker, Arrow shared
//! memory).
//!
//! The input is the taxi file's first row group: 1,000,000 trips. The transform computes each
//! trip's great-circle distance and speed, a typical derived field, and declares the six columns it
//! reads, so the engine sends only those. Overhead is everything the call costs except the user's
//! function: encoding the input into shared memory, the control messages, decoding in the worker,
//! conversion to the function's input type and back, writing the result, and mapping and decoding
//! it in the host. It's judged for the two zero-copy input types (pyarrow and polars); the same
//! transform given all 21 columns, and the worker's cold start, are recorded.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Array, Float64Array, RecordBatch, TimestampMicrosecondArray};
use arrow::datatypes::SchemaRef;
use dv_data::budget::Budget;
use dv_data::parquet::ParquetTable;
use dv_py::shm::Registry;
use dv_py::{Worker, WorkerConfig};
use dv_spike::{Caches, Report, spike_data, workspace_root};

const WARMUP: usize = 5;
const RUNS: usize = 30;
const LIMIT_MS: f64 = 50.0;

fn main() {
    let mut report = Report::new("S6", Caches::Warm);
    let py = dv_py::testing::python().expect("a Python environment: set DV_PYTHON or install uv");

    let table = ParquetTable::open(spike_data("nyc-taxi-2015q1.parquet")).expect("open");
    let names: Vec<String> = table
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let batches: Vec<RecordBatch> = table
        .batches(0, &names)
        .expect("row group 0")
        .collect::<Result<_, _>>()
        .expect("read");
    let schema: SchemaRef = batches[0].schema();
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, 1_000_000);

    let dir = std::env::temp_dir().join("dv-spike-s6");
    let (registry, released) = Registry::open(&dir).expect("registry");
    let budget = Budget::for_this_machine();
    let t0 = Instant::now();
    let mut worker = Worker::start(
        WorkerConfig::python(&py),
        budget.clone(),
        Arc::new(registry),
    )
    .expect("start worker");
    let ready = t0.elapsed();
    worker
        .load(&workspace_root().join("fixtures/python/transforms.py"))
        .expect("load");
    let loaded = t0.elapsed();
    let info = worker.info().cloned().unwrap_or_default();
    report.metric(
        "cold_start",
        ms(ready),
        "ms",
        None,
        &format!(
            "spawn to ready; Python {} pyarrow {}; {released} stale segments released",
            info["python"], info["pyarrow"]
        ),
    );
    report.metric(
        "cold_start_and_load",
        ms(loaded),
        "ms",
        None,
        "plus importing the transforms file (numpy, pandas, polars)",
    );

    let mut speeds = Vec::new();
    for (name, judged) in [
        ("trip_speed", true),
        ("trip_speed_polars", true),
        ("trip_speed_wide", false),
    ] {
        let mut overhead = Vec::new();
        let mut total = Vec::new();
        let mut user = Vec::new();
        let mut phases: Vec<(String, Vec<f64>)> = Vec::new();
        let mut host = (Vec::new(), Vec::new());
        let mut last = None;
        for i in 0..WARMUP + RUNS {
            let out = worker
                .transform(name, &schema, &batches, None)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            if i >= WARMUP {
                let t = &out.timings;
                overhead.push(Duration::from_secs_f64(t.overhead() / 1e3));
                total.push(Duration::from_secs_f64(t.total / 1e3));
                user.push(Duration::from_secs_f64(t.user() / 1e3));
                host.0.push(t.host_encode);
                host.1.push(t.host_decode);
                for (k, v) in &t.worker {
                    match phases.iter_mut().find(|(n, _)| n == k) {
                        Some((_, xs)) => xs.push(*v),
                        None => phases.push((k.clone(), vec![*v])),
                    }
                }
            }
            last = Some(out);
        }
        let limit = judged.then_some(LIMIT_MS);
        report.samples(&format!("{name}.overhead"), &overhead, limit);
        report.samples(&format!("{name}.total"), &total, None);
        report.samples(&format!("{name}.user_function"), &user, None);
        let median = |xs: &mut Vec<f64>| {
            xs.sort_by(f64::total_cmp);
            xs[xs.len() / 2]
        };
        let mut breakdown = vec![
            format!("host_encode {:.2}", median(&mut host.0)),
            format!("host_decode {:.2}", median(&mut host.1)),
        ];
        for (k, mut xs) in phases {
            if k != "user" {
                breakdown.push(format!("{k} {:.2}", median(&mut xs)));
            }
        }
        let in_cols = worker
            .transforms()
            .iter()
            .find(|t| t.name == name)
            .and_then(|t| t.columns.as_ref().map(Vec::len))
            .unwrap_or(schema.fields().len());
        report.metric(
            &format!("{name}.breakdown"),
            0.0,
            "",
            None,
            &format!(
                "{in_cols} input columns; median ms: {}",
                breakdown.join(", ")
            ),
        );
        let out = last.expect("ran");
        let all = arrow::compute::concat_batches(&out.schema, &out.batches).expect("concat");
        speeds.push((name, all));
    }

    // The transforms' numbers, checked against the same formula in Rust, and against each other.
    let (worst_km, worst_kmh) = check(&batches, &speeds[0].1);
    report.metric(
        "trip_speed.max_rel_error_km",
        worst_km,
        "",
        Some(1e-9),
        &format!("{worst_km:e} vs a Rust haversine on the same rows"),
    );
    report.metric(
        "trip_speed.max_rel_error_kmh",
        worst_kmh,
        "",
        Some(1e-9),
        &format!(
            "{worst_kmh:e} vs Rust; trips with zero or negative duration are null/NaN in both"
        ),
    );
    for (name, got) in &speeds[1..] {
        let (km, kmh) = check(&batches, got);
        report.metric(
            &format!("{name}.max_rel_error"),
            km.max(kmh),
            "",
            Some(1e-9),
            &format!("{:e} vs Rust", km.max(kmh)),
        );
    }
    report.metric(
        "worker_peak_footprint",
        worker.peak_footprint() as f64 / 1e6,
        "MB",
        None,
        &format!(
            "sampled every 5 ms; cap {} MB",
            WorkerConfig::python(&py).memory_cap / 1_000_000
        ),
    );
    report.metric(
        "worker_starts",
        worker.starts() as f64,
        "",
        Some(1.5),
        "the worker stayed warm for every run",
    );
    drop(speeds);
    report.metric(
        "budget_python_after",
        budget.used_by(dv_data::budget::Consumer::Python) as f64 / 1e6,
        "MB",
        Some(0.5),
        "shared memory returned to the budget once results were dropped",
    );
    std::process::exit(if report.finish() { 0 } else { 1 });
}

/// Largest relative errors of the result's `km` and `kmh` against a Rust haversine.
fn check(input: &[RecordBatch], out: &RecordBatch) -> (f64, f64) {
    let km = col::<Float64Array>(out, "km");
    let kmh = out.column_by_name("kmh").expect("kmh");
    let kmh = kmh.as_any().downcast_ref::<Float64Array>().expect("f64");
    let (mut worst_km, mut worst_kmh, mut row) = (0.0_f64, 0.0_f64, 0);
    let rel = |a: f64, b: f64| {
        if a == b {
            0.0
        } else {
            (a - b).abs() / b.abs().max(1e-300)
        }
    };
    for b in input {
        let (x1, y1) = (
            col::<Float64Array>(b, "startLon"),
            col::<Float64Array>(b, "startLat"),
        );
        let (x2, y2) = (
            col::<Float64Array>(b, "endLon"),
            col::<Float64Array>(b, "endLat"),
        );
        let t1 = col::<TimestampMicrosecondArray>(b, "tpepPickupDateTime");
        let t2 = col::<TimestampMicrosecondArray>(b, "tpepDropoffDateTime");
        for i in 0..b.num_rows() {
            let r = std::f64::consts::PI / 180.0;
            let (la1, la2) = (y1.value(i) * r, y2.value(i) * r);
            let a = ((la2 - la1) / 2.0).sin().powi(2)
                + la1.cos() * la2.cos() * ((x2.value(i) - x1.value(i)) * r / 2.0).sin().powi(2);
            let want_km = 2.0 * 6371.0088 * a.sqrt().asin();
            worst_km = worst_km.max(rel(km.value(row), want_km));
            let hours = (t2.value(i) - t1.value(i)) as f64 / 3.6e9;
            let got = (!kmh.is_null(row))
                .then(|| kmh.value(row))
                .filter(|v| !v.is_nan());
            match (hours > 0.0, got) {
                (true, Some(v)) => worst_kmh = worst_kmh.max(rel(v, want_km / hours)),
                (false, None) => {}
                _ => worst_kmh = f64::INFINITY,
            }
            row += 1;
        }
    }
    (worst_km, worst_kmh)
}

fn col<'a, T: 'static>(b: &'a RecordBatch, name: &str) -> &'a T {
    b.column_by_name(name)
        .unwrap_or_else(|| panic!("column {name}"))
        .as_any()
        .downcast_ref::<T>()
        .unwrap_or_else(|| panic!("type of {name}"))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
