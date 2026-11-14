//! S3: brushing a histogram re-filters a map and a bar chart over 38M rows within 60 ms per brush
//! update.
//!
//! The histogram is trip distance (64 bins of a quarter mile). Each brush update filters the
//! pickup density map and the dropoff dots (full-screen 16-inch, whole-city view, the most work)
//! and recounts a bar chart of payment type over all rows. Latency runs from setting the brush to
//! the bar counts being back on the CPU, which is after the map frame has finished on the GPU.
//! Every 20th update's bars are checked against a CPU count.

use std::time::{Duration, Instant};

use dv_render::density::{DensityMap, FRAME_TIMESTAMPS, Filter, LayerData, View};
use dv_render::density_index::DensityIndex;
use dv_render::gpu::{Gpu, GpuTimer};
use dv_spike::taxi::{DIST_BINS, DIST_SHIFT, PAY_SHIFT, TaxiPoints};
use dv_spike::{Caches, Report, time};
use rayon::prelude::*;

const LIMIT_MS: f64 = 60.0;
const UPDATES: usize = 200;
const DOT_STRIDE: usize = 38;
const SIZE: [u32; 2] = [3456, 2234];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn main() {
    let mut report = Report::new("S3", Caches::Warm);
    let (taxi, load) = time(TaxiPoints::load);
    report.span("load", load, None, "not part of S3");
    report.memory("after_load");
    let gpu = Gpu::headless().expect("GPU");
    let timer = GpuTimer::new(&gpu, FRAME_TIMESTAMPS).expect("timestamps");

    let [x0, y0, x1, y1] = taxi.extent;
    let o = taxi.origin;
    let rel = [
        (x0 - o[0]) as f32,
        (y0 - o[1]) as f32,
        (x1 - o[0]) as f32,
        (y1 - o[1]) as f32,
    ];
    let dots: Vec<[f32; 2]> = taxi.dropoffs.iter().step_by(DOT_STRIDE).copied().collect();
    let dot_attrs: Vec<u32> = taxi.attrs.iter().step_by(DOT_STRIDE).copied().collect();
    let ((density, dot_layer), build) = time(|| {
        let d = DensityIndex::build(&taxi.pickups, rel);
        let d_attrs = d.permute(&taxi.attrs);
        let k = DensityIndex::build(&dots, rel);
        let k_attrs = k.permute(&dot_attrs);
        (
            LayerData {
                index: d,
                attrs: d_attrs,
            },
            LayerData {
                index: k,
                attrs: k_attrs,
            },
        )
    });
    report.span("index_build", build, None, "not part of S3");
    report.memory("after_index_build");
    // The indexes hold their own sorted copies. The attributes stay for the CPU check.
    let attrs = taxi.attrs;
    drop((taxi.pickups, taxi.dropoffs, dots, dot_attrs));
    report.memory("after_freeing_sources");
    let (histogram, hist_time) = time(|| {
        attrs
            .par_chunks(1 << 16)
            .map(|chunk| {
                let mut h = [0u64; 256];
                for a in chunk {
                    h[((a >> DIST_SHIFT) & 0xFF) as usize] += 1;
                }
                h
            })
            .reduce(
                || [0u64; 256],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    });
    report.span(
        "histogram_once",
        hist_time,
        None,
        "the brushed view; its own brush doesn't filter it",
    );
    let invalid = histogram[255];
    report.metric(
        "dist_invalid_rows",
        invalid as f64,
        "rows",
        None,
        "negative distance; excluded by any brush",
    );

    let mut map = DensityMap::new(&gpu, o, density, dot_layer, SIZE, FORMAT);
    map.set_bars(Some(PAY_SHIFT));
    report.memory("after_upload");
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
    let fit = (f64::from(SIZE[0]) / (x1 - x0)).min(f64::from(SIZE[1]) / (y1 - y0));
    let city = View {
        center: [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
        scale: fit,
    };

    let frame = |map: &mut DensityMap| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        map.encode(&gpu, &mut enc, &target, &city, Some(&timer));
        gpu.queue.submit([enc.finish()]);
        let bars = map.read_bars(&gpu);
        let stamps = timer.read_ms(&gpu);
        (bars, stamps[stamps.len() - 1])
    };
    for _ in 0..5 {
        frame(&mut map);
    }
    let (all_bars, unfiltered_ms) = frame(&mut map);
    report.memory("after_first_frames");
    report.metric(
        "unfiltered_gpu_frame",
        unfiltered_ms,
        "ms",
        None,
        "cells, for comparison",
    );
    let total: u64 = all_bars.iter().map(|&b| u64::from(b)).sum();
    assert_eq!(total, attrs.len() as u64, "unfiltered bars count every row");

    let (mut latency, mut gpu_ms) = (Vec::new(), Vec::new());
    let mut checked = 0;
    for i in 0..UPDATES {
        let width = 2 + (i as u32 % 12);
        let lo = (i as u32 * 3) % (DIST_BINS - width);
        let filter = Filter {
            shift: DIST_SHIFT,
            lo,
            hi: lo + width,
        };
        let t0 = Instant::now();
        map.set_filter(Some(filter));
        let (bars, ms) = frame(&mut map);
        latency.push(t0.elapsed());
        gpu_ms.push(Duration::from_secs_f64(ms / 1e3));
        if i % 20 == 0 {
            let mut want = [0u32; 16];
            for a in attrs.iter() {
                let v = (a >> DIST_SHIFT) & 0xFF;
                if v >= filter.lo && v <= filter.hi {
                    want[((a >> PAY_SHIFT) & 15) as usize] += 1;
                }
            }
            assert_eq!(bars, want, "bars for brush {lo}..={}", lo + width);
            checked += 1;
        }
    }
    report.samples("brush_update_latency", &latency, Some(LIMIT_MS));
    report.samples("brush_gpu_frame", &gpu_ms, None);
    report.metric(
        "bars_checked_against_cpu",
        checked as f64,
        "updates",
        None,
        "all matched",
    );
    std::process::exit(if report.finish() { 0 } else { 1 });
}
