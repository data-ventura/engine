//! S2: 38M points as density plus 1M individual points pan and zoom with p95 frame time under 8.3 ms.
//!
//! Renders offscreen at the size of a full-screen 16-inch MacBook Pro (3456 × 2234 pixels), the
//! largest built-in display, along a scripted path: zoom from the whole city into Midtown, pan, and
//! zoom back out. Every frame re-bins the 38M pickups at screen resolution (pre-counted sub-pixel
//! cells when zoomed out, raw points in view when zoomed in) and draws ~1M dropoffs as dots. Frame
//! time is GPU time from the first compute pass to the end of the render pass, from GPU timestamps.

use std::time::Instant;

use dv_render::density::{DensityMap, FRAME_TIMESTAMPS, View};
use dv_render::density_index::{DensityIndex, Plan};
use dv_render::gpu::{Gpu, GpuTimer};
use dv_spike::taxi::TaxiPoints;
use dv_spike::{Caches, Report, save_png, time, workspace_root};

const LIMIT_P95_MS: f64 = 8.3;
const FRAMES_PER_LEG: usize = 200;
const DOT_STRIDE: usize = 38;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn main() {
    let mut report = Report::new("S2", Caches::Warm);
    let (taxi, load) = time(TaxiPoints::load);
    report.memory("after_load");
    report.span(
        "load_and_project",
        load,
        None,
        "read 4 columns, project, not part of S2",
    );

    let gpu = Gpu::headless().expect("GPU");
    println!("adapter: {} ({:?})", gpu.adapter.name, gpu.adapter.backend);
    let timer = GpuTimer::new(&gpu, FRAME_TIMESTAMPS).expect("this GPU has no timestamp queries");

    let dots: Vec<[f32; 2]> = taxi.dropoffs.iter().step_by(DOT_STRIDE).copied().collect();
    let [x0, y0, x1, y1] = taxi.extent;
    let rel = [
        (x0 - taxi.origin[0]) as f32,
        (y0 - taxi.origin[1]) as f32,
        (x1 - taxi.origin[0]) as f32,
        (y1 - taxi.origin[1]) as f32,
    ];
    let ((index, dot_index), build) = time(|| {
        (
            DensityIndex::build(&taxi.pickups, rel),
            DensityIndex::build(&dots, rel),
        )
    });
    let (extent, origin) = (taxi.extent, taxi.origin);
    // The indexes hold their own sorted copies.
    drop((taxi, dots));
    report.span(
        "index_build",
        build,
        None,
        "once per dataset, not part of S2",
    );
    report.memory("after_index_build");
    report.metric("density_points", index.len as f64, "pts", None, "pickups");
    report.metric(
        "outside_extent",
        (index.len - index.inside) as f64,
        "pts",
        None,
        "binned raw every frame",
    );
    for l in &index.levels {
        report.metric(
            &format!("cells_level_{}", l.bits),
            l.len as f64,
            "cells",
            None,
            "",
        );
    }
    report.metric(
        "dot_points",
        dot_index.len as f64,
        "pts",
        None,
        "every 38th dropoff",
    );

    let mut map: Option<DensityMap> = None;
    let mut indexes = Some((index, dot_index));
    for (label, size, limit) in [
        ("16in_fullscreen", [3456, 2234], Some(LIMIT_P95_MS)),
        ("window_1728x1117", [1728, 1117], None),
    ] {
        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
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
        // One map for both sizes: the indexes are uploaded once and the map is resized.
        let map = match &mut map {
            Some(m) => {
                m.resize(&gpu, size);
                m
            }
            None => {
                let (d, o) = indexes.take().expect("indexes");
                map.insert(DensityMap::new(
                    &gpu,
                    origin,
                    d.into(),
                    o.into(),
                    size,
                    FORMAT,
                ))
            }
        };
        let path = camera_path(extent, size);

        let (mut gpu_ms, mut cpu_ms, mut modes) = (Vec::new(), Vec::new(), Vec::new());
        let mut phases: Vec<Vec<f64>> = Vec::new();
        for (i, view) in path.iter().enumerate() {
            let t0 = Instant::now();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            let binning = map.encode(&gpu, &mut enc, &target_view, view, Some(&timer));
            gpu.queue.submit([enc.finish()]);
            cpu_ms.push(t0.elapsed());
            let stamps = timer.read_ms(&gpu);
            let ms = stamps[stamps.len() - 1];
            // The first few frames include pipeline warm-up.
            if i >= 5 {
                gpu_ms.push(std::time::Duration::from_secs_f64(ms / 1e3));
                phases.push(stamps.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>());
                let name = |p: Plan| match p {
                    Plan::Cells { bits, items } => format!("L{bits}:{items}"),
                    Plan::Points { items } => format!("pts:{items}"),
                };
                modes.push(format!("{}|{}", name(binning.density), name(binning.dots)));
            }
            if i == 0 || i == FRAMES_PER_LEG {
                let name = format!("s2-{label}-{}.png", if i == 0 { "city" } else { "midtown" });
                save_png(&gpu, &target, size, &name);
            }
        }
        let series: Vec<String> = gpu_ms
            .iter()
            .zip(&modes)
            .zip(&phases)
            .map(|((d, m), p)| {
                let p: Vec<String> = p.iter().map(|x| format!("{x:.2}")).collect();
                format!("{:.3} {m} {}", d.as_secs_f64() * 1e3, p.join(" "))
            })
            .collect();
        let series_path = workspace_root().join(format!("target/spike/s2-{label}-frames.txt"));
        std::fs::write(series_path, series.join("\n")).expect("write frame series");
        report.samples(&format!("{label}.gpu_frame"), &gpu_ms, limit);
        report.samples(&format!("{label}.cpu_encode_submit"), &cpu_ms, None);
        // Phase times stay in the frame series only: Metal overlaps passes that touch different
        // buffers, so phases don't add up to the frame time.
    }
    std::process::exit(if report.finish() { 0 } else { 1 });
}

/// Zoom 32× into Midtown, pan 6 km east to west, then zoom back out.
fn camera_path(extent: [f64; 4], size: [u32; 2]) -> Vec<View> {
    let [x0, y0, x1, y1] = extent;
    let fit = (f64::from(size[0]) / (x1 - x0)).min(f64::from(size[1]) / (y1 - y0));
    let city = [(x0 + x1) / 2.0, (y0 + y1) / 2.0];
    let (mx, my) = dv_grid::mercator::forward(-73.9855, 40.7580); // Times Square
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let n = FRAMES_PER_LEG;
    let mut path = Vec::with_capacity(3 * n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let c = [lerp(city[0], mx, t), lerp(city[1], my, t)];
        path.push(View {
            center: c,
            scale: fit * 32f64.powf(t),
        });
    }
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        path.push(View {
            center: [mx + 3000.0 - 6000.0 * t, my],
            scale: fit * 32.0,
        });
    }
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let c = [lerp(mx - 3000.0, city[0], t), lerp(my, city[1], t)];
        path.push(View {
            center: c,
            scale: fit * 32f64.powf(1.0 - t),
        });
    }
    path
}
