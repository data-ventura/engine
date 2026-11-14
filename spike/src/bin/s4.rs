//! S4: one ERA5 step from public ARCO Zarr draws in under 300 ms with a warm connection, the next
//! step in the direction of travel draws from RAM, and cached steps scrub at 120 fps.
//!
//! Reads 2 m temperature from Google's public ARCO-ERA5 store (one 721 × 1440 chunk per hourly
//! step, about 2.35 MB compressed), at hours chosen afresh on every run so Google's edge cache
//! hasn't seen them. "Warm connection" means the HTTPS connection is open. The never-read limit is
//! judged only when the connection fetches such a chunk in p95 ≤ 200 ms, measured in the same run
//! (D10). A step's time runs from the request to the frame finishing on the GPU at full-screen
//! 16-inch size. Stepping forward reads ahead under the memory budget. Scrubbing switches between
//! steps resident in a GPU texture array; a frame's time is wall time from encoding to the GPU
//! finishing.
//!
//! Argument, optional: the first hour (since 1900), to repeat a run.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dv_data::budget::Budget;
use dv_data::prefetch::{Fetch, Prefetch};
use dv_data::zarr::ZarrArray;
use dv_render::field::{FieldMap, FieldView, Grid};
use dv_render::gpu::{Gpu, GpuTimer};
use dv_spike::era5::{self, URL};
use dv_spike::{Caches, Report, save_png};

/// GPU layers, and steps in the stepping and scrubbing phases.
const STEPS: u32 = 24;
const PROBES: u64 = 12;
const JUMPS: u64 = 12;
/// Hours between jumps, so read-ahead never covers the next one.
const JUMP_GAP: u64 = 5;
const READ_AHEAD: u64 = 4;
const READ_PARALLEL: usize = 3;
/// Holding an arrow key down in a player steps a few times a second.
const STEP_EVERY: Duration = Duration::from_millis(250);
const SCRUB_FRAMES: usize = 600;
const SIZE: [u32; 2] = [3456, 2234];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const STEP_LIMIT_MS: f64 = 300.0;
const SCRUB_LIMIT_MS: f64 = 1000.0 / 120.0;

#[tokio::main]
async fn main() {
    let start = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or_else(|| era5::fresh_hour(200));
    let mut report = Report::new("S4", Caches::Warm);
    println!("first hour {start} (pass it as the argument to repeat this run)");
    let t = Instant::now();
    let array = ZarrArray::open_http(URL, "/2m_temperature")
        .await
        .expect("open");
    report.metric(
        "open_metadata",
        t.elapsed().as_secs_f64() * 1e3,
        "ms",
        None,
        "cold connection, not part of S4",
    );
    let probe = era5::probe(start, PROBES).await;
    let jump_limit = era5::judge_on_connection(&mut report, &probe, STEP_LIMIT_MS);

    let gpu = Gpu::headless().expect("GPU");
    let timer = GpuTimer::new(&gpu, 2).expect("timestamps");
    let grid = Grid {
        origin: [0.0, 90.0],
        step: [0.25, -0.25],
        size: [1440, 721],
    };
    let map = FieldMap::new(&gpu, grid, STEPS, SIZE, [220.0, 320.0], FORMAT);
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    // The whole globe, fitted to the width.
    let view = FieldView {
        center: [0.0, 0.0],
        scale: SIZE[0] as f32 / 360.0,
    };
    let draw = |layer: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        map.encode(&gpu, &mut enc, &target_view, &view, layer, Some(&timer));
        gpu.queue.submit([enc.finish()]);
        timer.read_ms(&gpu)[1]
    };

    // Warm the connection with a step that isn't measured.
    let t = Instant::now();
    array
        .chunk_f32(&[start + era5::WARM_UP + PROBES, 0, 0])
        .await
        .expect("warm-up chunk");
    report.metric(
        "warmup_step_fetch",
        t.elapsed().as_secs_f64() * 1e3,
        "ms",
        None,
        "not part of S4",
    );

    // Jumps to never-read steps, with nothing read ahead.
    let (mut step_ms, mut fetch_ms) = (Vec::new(), Vec::new());
    for i in 0..JUMPS {
        let t0 = Instant::now();
        let values = array
            .chunk_f32(&[start + 20 + i * JUMP_GAP, 0, 0])
            .await
            .expect("chunk");
        fetch_ms.push(t0.elapsed());
        let layer = i as u32 % STEPS;
        map.upload(&gpu, layer, &values);
        draw(layer);
        step_ms.push(t0.elapsed());
        if i == 0 {
            save_png(&gpu, &target, SIZE, "s4-era5-t2m.png");
        }
    }
    report.samples("new_step_to_frame", &step_ms, jump_limit);
    report.samples("new_step_fetch_decode", &fetch_ms, None);

    // Stepping forward through never-read steps, reading ahead.
    let first = start + 20 + JUMPS * JUMP_GAP + 10;
    let fetch: Fetch = {
        let array = array.clone();
        Arc::new(move |step| {
            let array = array.clone();
            Box::pin(async move { array.chunk_f32(&[step, 0, 0]).await })
        })
    };
    let steps = Prefetch::new(
        Budget::for_this_machine(),
        array.shape()[0],
        READ_AHEAD,
        READ_PARALLEL,
        fetch,
    );
    let mut forward_ms = Vec::new();
    for i in 0..u64::from(STEPS) {
        let t0 = Instant::now();
        steps.go_to(first + i, 1);
        let values = steps.get(first + i).await.expect("step");
        map.upload(&gpu, i as u32, &values);
        draw(i as u32);
        forward_ms.push(t0.elapsed());
        tokio::time::sleep(STEP_EVERY.saturating_sub(t0.elapsed())).await;
    }
    // The first step is a jump: nothing could be read ahead of it.
    report.samples(
        "step_forward_to_frame",
        &forward_ms[1..],
        Some(STEP_LIMIT_MS),
    );
    let stats = steps.stats();
    report.metric(
        "step_forward_from_ram",
        stats.hits as f64,
        "steps",
        None,
        &format!(
            "of {} after the first, one every {} ms, reading {READ_AHEAD} ahead, {READ_PARALLEL} at once",
            STEPS - 1,
            STEP_EVERY.as_millis()
        ),
    );
    drop(steps);

    let (mut frame_ms, mut gpu_ms) = (Vec::new(), Vec::new());
    for f in 0..SCRUB_FRAMES {
        // Back and forth across the resident steps, one step per frame.
        let k = (f as u32) % (2 * STEPS - 2);
        let layer = if k < STEPS { k } else { 2 * STEPS - 2 - k };
        let t0 = Instant::now();
        let g = draw(layer);
        frame_ms.push(t0.elapsed());
        gpu_ms.push(Duration::from_secs_f64(g / 1e3));
    }
    report.samples("scrub_frame", &frame_ms, Some(SCRUB_LIMIT_MS));
    report.samples("scrub_gpu", &gpu_ms, None);
    std::process::exit(if report.finish() { 0 } else { 1 });
}
