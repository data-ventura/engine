//! S8: cold caches. S1 and S4 repeated after `purge`, with the app's disk cache empty and a fresh
//! connection. Progress or a coarse preview within 300 ms; the time until the result is complete is
//! recorded.
//!
//! Run right after `sudo purge`, which empties the OS file cache. The run checks that S1's file
//! isn't cached before it starts. The engine has no disk cache of its own yet, so that's empty by
//! construction; the process is new, so the connection is too.
//!
//! - **S1 cold:** the coarse preview is the footer: row count, schema, and exact nulls, minimum and
//!   maximum for every column. Complete is the sampled profile.
//! - **S4 cold:** progress is the store's metadata, which gives the grid, the time axis and the
//!   value range, so the view can draw its frame and a progress indicator. Complete is the first
//!   never-read step drawn.

use std::time::Instant;

use dv_data::parquet::ParquetTable;
use dv_data::zarr::ZarrArray;
use dv_model::profile::{ExactStats, Value, profile};
use dv_render::field::{FieldMap, FieldView, Grid};
use dv_render::gpu::Gpu;
use dv_spike::era5::{self, URL};
use dv_spike::{Caches, Report, cached_fraction, spike_data};

const SAMPLE_ROWS: usize = 200_000;
const PREVIEW_LIMIT_MS: f64 = 300.0;
/// Above this share of S1's file in the OS cache, the run isn't cold.
const COLD_MAX: f64 = 0.02;
const SIZE: [u32; 2] = [3456, 2234];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let mut report = Report::new("S8", Caches::Cold);
    let path = spike_data("nyc-taxi-2015q1.parquet");
    let before = cached_fraction(&path);
    report.metric(
        "s1_file_cached_before",
        before * 100.0,
        "%",
        Some(COLD_MAX * 100.0),
        "share of the Parquet file in the OS cache at the start; run right after `sudo purge`",
    );

    // S1 cold.
    let t0 = Instant::now();
    let table = ParquetTable::open(&path).expect("open");
    let exact: Vec<ExactStats> = table
        .column_stats()
        .expect("stats")
        .into_iter()
        .map(|s| ExactStats {
            nulls: s.nulls,
            min: s.min.as_ref().and_then(Value::from_array),
            max: s.max.as_ref().and_then(Value::from_array),
        })
        .collect();
    report.metric(
        "s1.preview_footer",
        ms(t0),
        "ms",
        Some(PREVIEW_LIMIT_MS),
        &format!("rows, schema and exact stats for {} columns", exact.len()),
    );
    let sample = table.sample(SAMPLE_ROWS).expect("sample");
    let p = profile(&sample.batch, sample.of_rows, exact);
    report.metric(
        "s1.complete_profile",
        ms(t0),
        "ms",
        None,
        &format!(
            "{} sampled rows of {}; recorded, no limit (warm S1: p50 65 ms)",
            p.sample_rows, p.rows
        ),
    );
    report.metric(
        "s1_file_cached_after",
        cached_fraction(&path) * 100.0,
        "%",
        None,
        "share the profile read",
    );

    // S4 cold, on a fresh connection.
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let hour = era5::fresh_hour(10);
    let gpu = Gpu::headless().expect("GPU");
    rt.block_on(async {
        let t0 = Instant::now();
        let array = ZarrArray::open_http(URL, "/2m_temperature")
            .await
            .expect("open");
        report.metric(
            "s4.progress_metadata",
            ms(t0),
            "ms",
            Some(PREVIEW_LIMIT_MS),
            &format!(
                "grid {}×{}, {} hourly steps: the view's frame and progress indicator",
                array.shape()[2],
                array.shape()[1],
                array.shape()[0]
            ),
        );
        let grid = Grid {
            origin: [0.0, 90.0],
            step: [0.25, -0.25],
            size: [1440, 721],
        };
        let map = FieldMap::new(&gpu, grid, 1, SIZE, [220.0, 320.0], FORMAT);
        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: SIZE[0],
                height: SIZE[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target = target.create_view(&Default::default());
        let values = array.chunk_f32(&[hour, 0, 0]).await.expect("chunk");
        map.upload(&gpu, 0, &values);
        let view = FieldView {
            center: [0.0, 0.0],
            scale: SIZE[0] as f32 / 360.0,
        };
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        map.encode(&gpu, &mut enc, &target, &view, 0, None);
        gpu.queue.submit([enc.finish()]);
        gpu.wait();
        report.metric(
            "s4.complete_first_step",
            ms(t0),
            "ms",
            None,
            &format!("hour {hour}, never read; recorded, no limit"),
        );
    });
    std::process::exit(if report.finish() { 0 } else { 1 });
}
