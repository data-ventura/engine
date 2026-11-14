//! S9: a one-year hourly time series at one point, from ERA5 chunked by time (one global chunk per
//! hourly step), shows progress within 300 ms, streams in as it's read, and keeps peak memory
//! inside the budget.
//!
//! Every value costs a whole 2.35 MB chunk, so the year is about 20 GB downloaded to keep 35 KB.
//! Progress is the view showing the series' axis and a step counter, which it can as soon as the
//! store's metadata is open. Reads then run with a bounded number in flight, ramping up from one so
//! the first value arrives after a single chunk. Each read holds a budget reservation for its
//! compressed and decoded chunk until it has taken its one value; values are delivered as they
//! arrive. A read that fails is retried up to [`ATTEMPTS`] times.
//!
//! Arguments, all optional: steps (default 8760), reads in flight (default 16), first hour.

use std::time::{Duration, Instant};

use dv_data::budget::{Budget, Consumer};
use dv_data::zarr::ZarrArray;
use dv_spike::{Caches, Report};
use tokio::task::JoinSet;

const URL: &str = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3";
/// Hours since 1900-01-01T00 for 2019-01-01T00.
const START: u64 = 1_043_136;
const STEPS: u64 = 8760;
/// New York City, 40.75° N 74.0° W, on the 0.25° grid that starts at 90° N and 0° E.
const ROW: u64 = 197;
const COL: u64 = 1144;
/// A read's share of the budget: the compressed chunk plus its decoded 721 × 1440 floats.
const READ_BYTES: u64 = (3 << 20) + 721 * 1440 * 4;
const ATTEMPTS: u32 = 3;
const PROGRESS_LIMIT_MS: f64 = 300.0;

#[tokio::main]
async fn main() {
    let arg = |i: usize| std::env::args().nth(i).and_then(|a| a.parse::<u64>().ok());
    let steps = arg(1).unwrap_or(STEPS);
    let in_flight = arg(2).unwrap_or(16) as usize;
    let start = arg(3).unwrap_or(START);
    let mut report = Report::new(
        if steps == STEPS { "S9" } else { "S9-partial" },
        Caches::Warm,
    );
    let budget = Budget::for_this_machine();
    report.memory("start");

    // The request starts when the user asks for the series.
    let t0 = Instant::now();
    let array = ZarrArray::open_http(URL, "/2m_temperature")
        .await
        .expect("open");
    let opened = t0.elapsed();
    report.metric(
        "progress_shown",
        ms(opened),
        "ms",
        Some(PROGRESS_LIMIT_MS),
        "store metadata open on a cold connection: the view shows the axis and 0 of N steps",
    );

    let mut series = vec![f32::NAN; steps as usize];
    let mut reads = JoinSet::new();
    let (mut next, mut done, mut retries, mut refused) = (0u64, 0u64, 0u32, 0u64);
    let mut first_value = None;
    let mut gaps = Vec::new();
    let mut last = Instant::now();
    let mut reported = 0;
    let mut peak_in_flight = 0;
    while done < steps {
        // Ramp up from one read, so the first value doesn't share the connection with the rest.
        let allowed = in_flight.min(1 + 2 * done as usize);
        while next < steps && reads.len() < allowed {
            // The budget, not just the count, bounds the reads in flight.
            let Ok(reservation) = budget.reserve(Consumer::Arrays, READ_BYTES) else {
                refused += 1;
                break;
            };
            let (array, i) = (array.clone(), next);
            reads.spawn(async move {
                let mut tries = 0;
                loop {
                    tries += 1;
                    let r = array
                        .chunk_subset_f32(&[start + i, 0, 0], &[0..1, ROW..ROW + 1, COL..COL + 1])
                        .await;
                    match r {
                        Ok(v) => {
                            drop(reservation);
                            return (i, v[0], tries - 1);
                        }
                        Err(e) if tries < ATTEMPTS => {
                            eprintln!("step {}: {e}; retrying", start + i);
                            tokio::time::sleep(Duration::from_millis(500 * u64::from(tries))).await;
                        }
                        Err(e) => panic!("step {} failed {ATTEMPTS} times: {e}", start + i),
                    }
                }
            });
            next += 1;
        }
        peak_in_flight = peak_in_flight.max(reads.len());
        let (i, value, r) = reads.join_next().await.expect("reads").expect("read task");
        // Delivered: the view draws this point now.
        series[i as usize] = value;
        retries += r;
        done += 1;
        let now = Instant::now();
        if first_value.is_none() {
            first_value = Some(t0.elapsed());
        } else {
            gaps.push(now - last);
        }
        last = now;
        if done * 20 / steps > reported {
            reported = done * 20 / steps;
            println!(
                "{done}/{steps} after {:.1} s ({:.0} MB/s of chunks)",
                t0.elapsed().as_secs_f64(),
                (done * 2_350_000) as f64 / 1e6 / t0.elapsed().as_secs_f64()
            );
        }
    }
    let total = t0.elapsed();

    let first_value = first_value.expect("a value");
    report.metric(
        "first_value",
        ms(first_value),
        "ms",
        None,
        "from the request, including opening the store: the first point drawn",
    );
    report.metric(
        "first_value_after_open",
        ms(first_value - opened),
        "ms",
        None,
        "one never-read chunk, as when the store is already open",
    );
    report.samples("gap_between_values", &gaps, None);
    let max_gap = gaps.iter().max().copied().unwrap_or_default();
    report.metric(
        "longest_gap",
        ms(max_gap),
        "ms",
        None,
        "the longest the series went without a new point",
    );
    report.metric(
        "total",
        total.as_secs_f64(),
        "s",
        None,
        "recorded, no limit",
    );
    report.metric(
        "throughput",
        (steps * 2_350_000) as f64 / 1e6 / total.as_secs_f64(),
        "MB/s",
        None,
        "compressed chunk bytes, about 2.35 MB per step",
    );
    report.metric("steps", steps as f64, "steps", None, "");
    report.metric(
        "peak_in_flight",
        peak_in_flight as f64,
        "reads",
        None,
        &format!("cap {in_flight}; {refused} times the budget held a read back"),
    );
    report.metric("retries", f64::from(retries), "reads", None, "");
    let missing = series.iter().filter(|v| !v.is_finite()).count();
    report.metric(
        "missing_values",
        missing as f64,
        "values",
        Some(0.5),
        "every step has a value",
    );
    let plausible = series
        .iter()
        .filter(|v| (220.0..330.0).contains(*v))
        .count();
    report.metric(
        "implausible_values",
        (series.len() - plausible) as f64,
        "values",
        Some(0.5),
        "2 m temperature in New York between 220 and 330 K",
    );
    report.memory("after_series");
    let (_, peak) = dv_spike::footprint();
    report.metric(
        "peak_footprint_vs_budget",
        peak as f64 / budget.limit() as f64,
        "ratio",
        Some(1.0),
        &format!(
            "peak {} MB, budget {} MB (60% of RAM), reads reserved at most {} MB",
            peak / 1_000_000,
            budget.limit() / 1_000_000,
            budget.peak() / 1_000_000
        ),
    );

    let out = dv_spike::workspace_root().join("target/spike/s9-series.json");
    std::fs::create_dir_all(out.parent().expect("parent")).expect("create target/spike");
    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({
            "first_hour_since_1900": start,
            "point": { "lat": 40.75, "lon": -74.0 },
            "kelvin": series,
        }))
        .expect("json"),
    )
    .expect("write series");
    println!("series written to {}", out.display());
    std::process::exit(if report.finish() { 0 } else { 1 });
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
