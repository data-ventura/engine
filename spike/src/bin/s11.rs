//! S11: rapid interaction. Drag the time slider across 500 steps in 5 s, and brush at 50 updates/s
//! for 10 s. At most 2 requests in flight per view, the final position draws within 300 ms after
//! the input stops, no stale frame is ever drawn, and memory returns to within 10% of where it
//! started.
//!
//! Both views run their requests through `dv_data::request`: one running, at most one waiting, a
//! superseded running request cancelled by dropping it, and results checked against the view's
//! current generation before they're drawn.
//!
//! - **Slider:** 20 drags, each over 500 ERA5 steps not read before (at hours chosen afresh on
//!   every run), one every 10 ms. The final-step limit is judged only when the connection fetches a
//!   never-read chunk in p95 ≤ 200 ms, measured in the same run (D10). A step that's resident on
//!   the GPU draws at once; otherwise the nearest resident step draws at once (the coarse preview)
//!   and the step is requested once the slider rests for 30 ms.
//! - **Brush:** the S3 setup, with the map and bar chart redrawn on a render thread for each brush
//!   update, one every 20 ms. A frame is presented only if its brush is still the latest.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use dv_data::request::{self, Generation};
use dv_data::zarr::ZarrArray;
use dv_render::density::{DensityMap, Filter, LayerData, View};
use dv_render::density_index::DensityIndex;
use dv_render::field::{FieldMap, FieldView, Grid};
use dv_render::gpu::Gpu;
use dv_spike::era5::{self, URL};
use dv_spike::taxi::{DIST_BINS, DIST_SHIFT, PAY_SHIFT, TaxiPoints};
use dv_spike::{Caches, Report, footprint};

const SLIDER_STEPS: u64 = 500;
const SLIDER_TICK: Duration = Duration::from_millis(10);
/// Drags per run, each over 500 steps no drag has read, so the final-step time is a distribution.
/// With fewer than 20, the p95 is the slowest drag.
const DRAGS: u64 = 20;
/// Requests wait until the slider has rested this long. Measured: cancelling on every step instead
/// starts a fetch per step, wastes about 1.2 GB of downloads per drag, and wasn't faster.
const SLIDER_SETTLE: Duration = Duration::from_millis(30);
const BRUSH_UPDATES: usize = 500;
const BRUSH_TICK: Duration = Duration::from_millis(20);
const LAYERS: u32 = 24;
const DOT_STRIDE: usize = 38;
const SIZE: [u32; 2] = [3456, 2234];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const FINAL_LIMIT_MS: f64 = 300.0;

fn main() {
    let mut report = Report::new("S11", Caches::Warm);
    // Probes, a warm-up step, then the drags, 1000 hours apart.
    let first = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or_else(|| era5::fresh_hour(DRAGS * 1000 + 20));
    println!("first hour {first} (pass it as the argument to repeat this run)");
    let gpu = Arc::new(Gpu::headless().expect("GPU"));
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(slider(&mut report, gpu.clone(), first));
    brush(&mut report, gpu);
    std::process::exit(if report.finish() { 0 } else { 1 });
}

/// Counts requests running at once, and the most ever.
#[derive(Debug, Default)]
struct InFlight {
    now: AtomicUsize,
    most: AtomicUsize,
    started: AtomicUsize,
    finished: AtomicUsize,
}

/// Held while a request runs; dropped when it finishes or is cancelled.
struct Running(Arc<InFlight>);

impl Running {
    fn start(f: &Arc<InFlight>) -> Self {
        let n = f.now.fetch_add(1, Ordering::SeqCst) + 1;
        f.most.fetch_max(n, Ordering::SeqCst);
        f.started.fetch_add(1, Ordering::SeqCst);
        Self(f.clone())
    }

    fn finish(self) {
        self.0.finished.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.0.now.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Counts of what a view presented.
#[derive(Debug, Default)]
struct Presented {
    fresh: usize,
    stale_discarded: usize,
    stale_drawn: usize,
}

async fn slider(report: &mut Report, gpu: Arc<Gpu>, first: u64) {
    // The connection is timed before the drags and again between them, as it drifts.
    let probe = era5::Probe::new(first).await;
    let mut probes = Vec::new();
    for hour in first + era5::WARM_UP..first + era5::WARM_UP + 12 {
        probes.push(probe.time(hour).await);
    }
    let slider_start = first + era5::WARM_UP + 20;
    let array = ZarrArray::open_http(URL, "/2m_temperature")
        .await
        .expect("open");
    let grid = Grid {
        origin: [0.0, 90.0],
        step: [0.25, -0.25],
        size: [1440, 721],
    };
    let map = FieldMap::new(&gpu, grid, LAYERS, SIZE, [220.0, 320.0], FORMAT);
    let target = target(&gpu);
    let view = FieldView {
        center: [0.0, 0.0],
        scale: SIZE[0] as f32 / 360.0,
    };
    let draw = |layer: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        map.encode(&gpu, &mut enc, &target, &view, layer, None);
        gpu.queue.submit([enc.finish()]);
        gpu.wait();
    };
    // The step before the drag is resident, as it would be after opening the view.
    let first = array
        .chunk_f32(&[slider_start - 1, 0, 0])
        .await
        .expect("chunk");
    map.upload(&gpu, 0, &first);
    drop(first);
    draw(0);
    // Resident steps by layer, least recently used first.
    let mut resident: Vec<(u64, u32)> = vec![(slider_start - 1, 0)];
    let before = footprint().0;

    let inflight = Arc::new(InFlight::default());
    let (submit, requests) = request::channel::<u64>();
    let (done_tx, mut done) = tokio::sync::mpsc::unbounded_channel();
    let worker = tokio::spawn({
        let (array, inflight) = (array.clone(), inflight.clone());
        request::serve_with(
            requests,
            request::Policy {
                settle: SLIDER_SETTLE,
                cancel_running: true,
            },
            move |_g: Generation, step: u64| {
                let (array, inflight) = (array.clone(), inflight.clone());
                async move {
                    let running = Running::start(&inflight);
                    let t = Instant::now();
                    let values = array.chunk_f32(&[step, 0, 0]).await;
                    running.finish();
                    (step, values, t.elapsed())
                }
            },
            move |g, out| {
                let _ = done_tx.send((g, out));
            },
        )
    });

    let mut presented = Presented::default();
    let (mut cached_draws, mut preview_draws) = (0, 0);
    let mut present = |g: Generation,
                       (step, values, _): (u64, dv_data::Result<Vec<f32>>, Duration),
                       resident: &mut Vec<(u64, u32)>,
                       current: u64| {
        if g.value() != current || !g.is_current() {
            presented.stale_discarded += 1;
            return None;
        }
        let values = values.expect("chunk");
        let layer = if resident.len() < LAYERS as usize {
            resident.len() as u32
        } else {
            resident.remove(0).1
        };
        map.upload(&gpu, layer, &values);
        resident.push((step, layer));
        // Checked again right before drawing: nothing newer may have been asked for.
        if g.value() == current {
            draw(layer);
            presented.fresh += 1;
        } else {
            presented.stale_drawn += 1;
        }
        Some(step)
    };

    let (mut finals, mut final_fetches, mut input_ms) = (Vec::new(), Vec::new(), 0.0);
    for drag in 0..DRAGS {
        let start = slider_start + drag * 1000;
        let t0 = tokio::time::Instant::now();
        let mut last_step = 0;
        for i in 0..SLIDER_STEPS {
            let step = start + i;
            last_step = step;
            if let Some(k) = resident.iter().position(|r| r.0 == step) {
                let r = resident.remove(k);
                resident.push(r);
                draw(r.1);
                cached_draws += 1;
            } else {
                // The coarse preview: the nearest resident step, at once.
                let nearest = resident
                    .iter()
                    .min_by_key(|r| r.0.abs_diff(step))
                    .expect("one resident");
                draw(nearest.1);
                preview_draws += 1;
                submit.submit(step);
            }
            if i + 1 == SLIDER_STEPS {
                break;
            }
            // Present results as they arrive until the next input.
            let next = t0 + SLIDER_TICK * (i as u32 + 1);
            loop {
                tokio::select! {
                    Some((g, out)) = done.recv() => {
                        present(g, out, &mut resident, submit.current());
                    }
                    _ = tokio::time::sleep_until(next) => break,
                }
            }
        }
        // The input stops at the last step.
        let stopped = Instant::now();
        input_ms = t0.elapsed().as_secs_f64() * 1e3;
        loop {
            let (g, out) = done.recv().await.expect("worker running");
            let fetch = out.2;
            if present(g, out, &mut resident, submit.current()) == Some(last_step) {
                finals.push(stopped.elapsed());
                final_fetches.push(fetch);
                break;
            }
        }
        // A pause between drags, as between gestures, in which the connection is timed again.
        let pause = Instant::now();
        probes.push(probe.time(start + SLIDER_STEPS + 100).await);
        tokio::time::sleep(Duration::from_secs(1).saturating_sub(pause.elapsed())).await;
    }
    // Measured with the view still open, after its last frame.
    let after = footprint().0;
    drop(submit);
    worker.await.expect("worker");

    report.metric(
        "slider.input_duration",
        input_ms,
        "ms",
        None,
        "each drag: 500 steps, one per 10 ms; first to last input",
    );
    let final_limit = era5::judge_on_connection(report, &probes, FINAL_LIMIT_MS);
    report.samples("slider.final_step_drawn_after_stop", &finals, final_limit);
    let ours: Vec<Duration> = finals
        .iter()
        .zip(&final_fetches)
        .map(|(f, x)| f.saturating_sub(*x))
        .collect();
    report.samples("slider.final_step_minus_its_fetch", &ours, None);
    let each: Vec<String> = finals
        .iter()
        .map(|d| format!("{:.0}", d.as_secs_f64() * 1e3))
        .collect();
    println!("slider final step per drag (ms): {}", each.join(", "));
    in_flight_metrics(report, "slider", &inflight);
    presented_metrics(report, "slider", &presented);
    report.metric(
        "slider.preview_draws",
        f64::from(preview_draws),
        "frames",
        None,
        &format!("nearest resident step drawn at once; {cached_draws} steps were resident"),
    );
    memory_return(report, "slider", before, after);
}

fn brush(report: &mut Report, gpu: Arc<Gpu>) {
    let taxi = TaxiPoints::load();
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
    let d = DensityIndex::build(&taxi.pickups, rel);
    let d_attrs = d.permute(&taxi.attrs);
    let k = DensityIndex::build(&dots, rel);
    let k_attrs = k.permute(&dot_attrs);
    let attrs = taxi.attrs;
    drop((taxi.pickups, taxi.dropoffs, dots, dot_attrs));
    let mut map = DensityMap::new(
        &gpu,
        o,
        LayerData {
            index: d,
            attrs: d_attrs,
        },
        LayerData {
            index: k,
            attrs: k_attrs,
        },
        SIZE,
        FORMAT,
    );
    map.set_bars(Some(PAY_SHIFT));
    let fit = (f64::from(SIZE[0]) / (x1 - x0)).min(f64::from(SIZE[1]) / (y1 - y0));
    let city = View {
        center: [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
        scale: fit,
    };
    let filter_for = |i: usize| {
        let width = 2 + (i as u32 % 12);
        let lo = (i as u32 * 7) % (DIST_BINS - width);
        Filter {
            shift: DIST_SHIFT,
            lo,
            hi: lo + width,
        }
    };

    // The render thread owns the map and draws the latest brush, one frame at a time.
    let inflight = Arc::new(InFlight::default());
    let (submit, requests) = request::channel::<Filter>();
    let (done_tx, done) = mpsc::channel();
    let render = std::thread::spawn({
        let (gpu, inflight) = (gpu.clone(), inflight.clone());
        move || {
            let target = target(&gpu);
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("runtime");
            // A warm-up frame, not measured.
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            map.encode(&gpu, &mut enc, &target, &city, None);
            gpu.queue.submit([enc.finish()]);
            gpu.wait();
            let map = std::cell::RefCell::new(map);
            rt.block_on(request::serve(
                requests,
                |_g, filter: Filter| {
                    let running = Running::start(&inflight);
                    let mut map = map.borrow_mut();
                    map.set_filter(Some(filter));
                    let mut enc = gpu.device.create_command_encoder(&Default::default());
                    map.encode(&gpu, &mut enc, &target, &city, None);
                    gpu.queue.submit([enc.finish()]);
                    let bars = map.read_bars(&gpu);
                    running.finish();
                    std::future::ready((filter, bars))
                },
                move |g, out| {
                    let _ = done_tx.send((g, out));
                },
            ));
        }
    });
    // Let the render thread finish its warm-up.
    std::thread::sleep(Duration::from_millis(500));
    let before = footprint().0;

    let mut presented = Presented::default();
    let mut present = |g: Generation, (filter, bars): (Filter, [u32; 16]), current: u64| {
        if g.value() != current {
            presented.stale_discarded += 1;
            return None;
        }
        presented.fresh += 1;
        Some((filter, bars))
    };
    let t0 = Instant::now();
    let mut last = None;
    for i in 0..BRUSH_UPDATES {
        let filter = filter_for(i);
        submit.submit(filter);
        last = Some(filter);
        if i + 1 == BRUSH_UPDATES {
            break;
        }
        // Present frames as they arrive until the next input.
        let next = t0 + BRUSH_TICK * (i as u32 + 1);
        while let Ok((g, out)) = done.recv_timeout(next.saturating_duration_since(Instant::now())) {
            present(g, out, submit.current());
        }
    }
    // The input stops at the last update.
    let stopped = Instant::now();
    let input_ms = t0.elapsed().as_secs_f64() * 1e3;
    let last = last.expect("brushed");
    let (final_ms, bars) = loop {
        let (g, out) = done.recv().expect("render thread running");
        if let Some((filter, bars)) = present(g, out, submit.current())
            && filter == last
        {
            break (stopped.elapsed().as_secs_f64() * 1e3, bars);
        }
    };
    // Measured with the view still open, after its last frame.
    let after = footprint().0;
    drop(submit);
    render.join().expect("render thread");

    let mut want = [0u32; 16];
    for a in &attrs {
        let v = (a >> DIST_SHIFT) & 0xFF;
        if v >= last.lo && v <= last.hi {
            want[((a >> PAY_SHIFT) & 15) as usize] += 1;
        }
    }
    assert_eq!(bars, want, "final bars match a CPU count");

    report.metric(
        "brush.input_duration",
        input_ms,
        "ms",
        None,
        "500 updates, one per 20 ms; first to last input",
    );
    report.metric(
        "brush.final_drawn_after_stop",
        final_ms,
        "ms",
        Some(FINAL_LIMIT_MS),
        "final bars match a CPU count",
    );
    in_flight_metrics(report, "brush", &inflight);
    presented_metrics(report, "brush", &presented);
    memory_return(report, "brush", before, after);
}

fn in_flight_metrics(report: &mut Report, view: &str, f: &InFlight) {
    let most = f.most.load(Ordering::SeqCst);
    let (started, finished) = (
        f.started.load(Ordering::SeqCst),
        f.finished.load(Ordering::SeqCst),
    );
    // One waiting request at most, by construction: the channel holds only the latest.
    report.metric(
        &format!("{view}.max_in_flight"),
        (most + 1) as f64,
        "requests",
        Some(2.5),
        &format!(
            "{most} running at most, plus 1 waiting; {started} started, {} cancelled",
            started - finished
        ),
    );
}

fn presented_metrics(report: &mut Report, view: &str, p: &Presented) {
    report.metric(
        &format!("{view}.stale_frames_drawn"),
        p.stale_drawn as f64,
        "frames",
        Some(0.5),
        &format!(
            "{} fresh results drawn, {} stale results discarded",
            p.fresh, p.stale_discarded
        ),
    );
}

fn memory_return(report: &mut Report, view: &str, before: u64, after: u64) {
    // Only growth counts: memory below where it started has returned.
    let growth = (after as f64 - before as f64).max(0.0) / before as f64 * 100.0;
    report.metric(
        &format!("{view}.memory_growth"),
        growth,
        "%",
        Some(10.0),
        &format!(
            "footprint {:.0} MB before, {:.0} MB after",
            before as f64 / 1e6,
            after as f64 / 1e6
        ),
    );
}

fn target(gpu: &Gpu) -> wgpu::TextureView {
    gpu.device
        .create_texture(&wgpu::TextureDescriptor {
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
        })
        .create_view(&Default::default())
}
