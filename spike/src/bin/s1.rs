//! S1: a 38M-row, 1.1 GB Parquet file shows a sampled profile of all columns in under 0.7 s.
//!
//! Each run opens the file from scratch (footer and page index), reads the exact footer statistics,
//! samples rows from every row group, and profiles every column. Warm means the OS file cache is warm;
//! nothing is kept between runs inside the process.

use std::time::Duration;

use dv_data::parquet::ParquetTable;
use dv_model::profile::{ExactStats, Profile, Value, profile};
use dv_spike::{Caches, Report, spike_data, time};

const SAMPLE_ROWS: usize = 200_000;
const RUNS: usize = 20;
const LIMIT_MS: f64 = 700.0;

fn run() -> (Profile, [Duration; 4]) {
    let path = spike_data("nyc-taxi-2015q1.parquet");
    let (table, open) = time(|| ParquetTable::open(&path).expect("open"));
    let (exact, stats) = time(|| {
        table
            .column_stats()
            .expect("stats")
            .into_iter()
            .map(|s| ExactStats {
                nulls: s.nulls,
                min: s.min.as_ref().and_then(Value::from_array),
                max: s.max.as_ref().and_then(Value::from_array),
            })
            .collect::<Vec<_>>()
    });
    let (sample, read) = time(|| table.sample(SAMPLE_ROWS).expect("sample"));
    let (p, prof) = time(|| profile(&sample.batch, sample.of_rows, exact));
    (p, [open.wall, stats.wall, read.wall, prof.wall])
}

fn main() {
    let mut report = Report::new("S1", Caches::Warm);
    let ((first, _), first_span) = time(run);
    report.span("first_run", first_span, None, "first run in this process");

    let mut totals = Vec::new();
    let mut parts: [Vec<Duration>; 4] = Default::default();
    for _ in 0..RUNS {
        let ((_, p), span) = time(run);
        totals.push(span.wall);
        for (v, d) in parts.iter_mut().zip(p) {
            v.push(d);
        }
    }
    report.samples("profile_total", &totals, Some(LIMIT_MS));
    for (name, v) in [
        "open_footer",
        "footer_stats",
        "read_sample",
        "compute_profile",
    ]
    .iter()
    .zip(&parts)
    {
        report.samples(name, v, None);
    }
    report.metric("rows", first.rows as f64, "rows", None, "");
    report.metric("sample_rows", first.sample_rows as f64, "rows", None, "");
    report.metric("columns", first.columns.len() as f64, "cols", None, "");

    let out = dv_spike::workspace_root().join("target/spike/s1-profile.json");
    std::fs::create_dir_all(out.parent().expect("parent")).expect("create target/spike");
    std::fs::write(&out, serde_json::to_string_pretty(&first).expect("json"))
        .expect("write profile");
    println!("profile written to {}", out.display());
    std::process::exit(if report.finish() { 0 } else { 1 });
}
