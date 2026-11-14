//! S10: memory pressure. S3 and S4 together in one process, with the memory budget (SPEC §4.9)
//! capped at 4 GB: no crash, detail is reduced and the view says so, and peak memory stays below
//! the cap plus 10%.
//!
//! Usage: `s10 [cap in GB, default 4]`. At 4 GB both workloads fit at full detail, so the spike
//! also runs a tight cap to exercise the degrade path.
//!
//! Every large allocation reserves its bytes first:
//!
//! - **Baseline** (process, GPU device, render targets): a fixed, measured reservation.
//! - **Taxi load and index build** (arrays): the measured peak per row. If the budget can't hold
//!   every row, the layer loads every k-th row instead, the coarsest level of detail that fits, and
//!   the view says so. After upload, the reservation moves to what the GPU holds.
//! - **ERA5 steps on the GPU** (24 layers) and **decoded steps in RAM** (an LRU cache that the
//!   budget can evict).
//!
//! The workload interleaves the two: 48 new ERA5 steps (24 resident on the GPU, the rest in the RAM
//! cache), then 100 brush updates alternating with scrubbing frames that re-upload cached steps.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use dv_data::budget::{Budget, Consumer, Evict, Reservation};
use dv_data::zarr::ZarrArray;
use dv_render::density::{DensityMap, Filter, View};
use dv_render::field::{FieldMap, FieldView, Grid};
use dv_render::gpu::Gpu;
use dv_spike::taxi::{DIST_BINS, DIST_SHIFT, PAY_SHIFT, TaxiLayers, TaxiPoints};
use dv_spike::{Caches, Report, footprint};

const URL: &str = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3";
/// 2023-06-01T00, hours since 1900: S4's steps and the 24 after them.
const START: u64 = 1_081_824;
const NEW_STEPS: u64 = 48;
const LAYERS: u32 = 24;
const STEP_BYTES: u64 = 721 * 1440 * 4;
const BRUSH_UPDATES: usize = 100;
const DOT_STRIDE: usize = 38;
const SIZE: [u32; 2] = [3456, 2234];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Measured footprint outside the tracked consumers: the process, the GPU device, pipelines and
/// both views' render targets at full-screen 16-inch size.
const BASELINE: u64 = 450 << 20;
/// Measured peak bytes per loaded row from reading to the end of the index build (S3: 1.76 GB
/// above baseline for 38.5M rows), with margin.
const BUILD_BYTES_PER_ROW: u64 = 52;
/// Rows in the file.
const ROWS: u64 = 38_549_374;

fn main() {
    let cap_gb: f64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4.0);
    let cap = (cap_gb * 1e9) as u64;
    let mut report = Report::new(&format!("S10@{cap_gb}GB"), Caches::Warm);
    let budget = Budget::new(cap);
    let _baseline = budget
        .reserve(Consumer::Gpu, BASELINE)
        .expect("the baseline fits any cap the spike runs");
    let gpu = Gpu::headless().expect("GPU");
    let rt = tokio::runtime::Runtime::new().expect("runtime");

    // ERA5 first: the field view's GPU layers, fewer if they don't fit.
    let layers = (1..=LAYERS)
        .rev()
        .find_map(|n| {
            budget
                .reserve(Consumer::Gpu, u64::from(n) * STEP_BYTES)
                .ok()
                .map(|r| (n, r))
        })
        .expect("at least one layer fits");
    let (layers, _layers_reservation) = layers;
    let cache = Arc::new(StepCache::default());
    budget.register(Arc::downgrade(&cache) as Weak<dyn Evict>);

    // The taxi layer at the finest detail the budget can hold.
    let free = budget.limit() - budget.used();
    let stride = (ROWS * BUILD_BYTES_PER_ROW).div_ceil(free).max(1) as usize;
    let build = budget
        .reserve(
            Consumer::Arrays,
            ROWS.div_ceil(stride as u64) * BUILD_BYTES_PER_ROW,
        )
        .expect("the stride was chosen to fit");
    let t = Instant::now();
    let taxi = TaxiLayers::build(TaxiPoints::load_every(stride), DOT_STRIDE);
    report.metric(
        "taxi_load_and_build",
        t.elapsed().as_secs_f64() * 1e3,
        "ms",
        None,
        "",
    );
    let detail = if stride == 1 {
        "all trips".to_string()
    } else {
        format!(
            "Showing 1 in {stride} trips: the {cap_gb} GB memory budget can't hold all {ROWS} rows."
        )
    };
    report.metric(
        "detail.taxi_stride",
        stride as f64,
        "rows",
        None,
        &format!("view says: {detail}"),
    );
    report.metric(
        "detail.era5_gpu_layers",
        f64::from(layers),
        "steps",
        None,
        "resident on the GPU at once",
    );
    // From here the layers live on the GPU and the CPU keeps the attributes for checking. The
    // build reservation is swapped for those; nothing else allocates in between.
    drop(build);
    let gpu_res = budget
        .reserve(Consumer::Gpu, taxi.gpu_bytes())
        .expect("smaller than the build reservation it replaces");
    let attrs_res = budget
        .reserve(Consumer::Arrays, taxi.attrs.len() as u64 * 4)
        .expect("smaller than the build reservation it replaces");
    let TaxiLayers {
        origin,
        extent,
        density,
        dots,
        attrs,
    } = taxi;
    let mut map = DensityMap::new(&gpu, origin, density, dots, SIZE, FORMAT);
    map.set_bars(Some(PAY_SHIFT));
    report.memory("after_taxi_upload");

    let field = FieldMap::new(
        &gpu,
        Grid {
            origin: [0.0, 90.0],
            step: [0.25, -0.25],
            size: [1440, 721],
        },
        layers,
        SIZE,
        [220.0, 320.0],
        FORMAT,
    );
    let target = target(&gpu);
    let globe = FieldView {
        center: [0.0, 0.0],
        scale: SIZE[0] as f32 / 360.0,
    };
    let [x0, y0, x1, y1] = extent;
    let city = View {
        center: [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
        scale: (f64::from(SIZE[0]) / (x1 - x0)).min(f64::from(SIZE[1]) / (y1 - y0)),
    };
    let draw_field = |layer: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        field.encode(&gpu, &mut enc, &target, &globe, layer, None);
        gpu.queue.submit([enc.finish()]);
        gpu.wait();
    };

    // 48 new steps: each goes to the RAM cache (if the budget allows) and to a GPU layer.
    let array = rt
        .block_on(ZarrArray::open_http(URL, "/2m_temperature"))
        .expect("open");
    let mut on_gpu: VecDeque<(u64, u32)> = VecDeque::new();
    let mut not_cached = 0;
    for i in 0..NEW_STEPS {
        let step = START + i;
        let values = rt.block_on(array.chunk_f32(&[step, 0, 0])).expect("chunk");
        let layer = place(&mut on_gpu, step, layers);
        field.upload(&gpu, layer, &values);
        draw_field(layer);
        if !cache.insert(&budget, step, values) {
            not_cached += 1;
        }
    }
    report.memory("after_era5_steps");

    // Brushing alternates with scrubbing back over all 48 steps.
    let (mut brush_ms, mut scrub_ms) = (Vec::new(), Vec::new());
    let (mut refetched, mut checked) = (0, 0);
    for i in 0..BRUSH_UPDATES {
        let width = 2 + (i as u32 % 12);
        let lo = (i as u32 * 5) % (DIST_BINS - width);
        let filter = Filter {
            shift: DIST_SHIFT,
            lo,
            hi: lo + width,
        };
        let t0 = Instant::now();
        map.set_filter(Some(filter));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        map.encode(&gpu, &mut enc, &target, &city, None);
        gpu.queue.submit([enc.finish()]);
        let bars = map.read_bars(&gpu);
        brush_ms.push(t0.elapsed());
        if i % 20 == 0 {
            let mut want = [0u32; 16];
            for a in &attrs {
                let v = (a >> DIST_SHIFT) & 0xFF;
                if v >= lo && v <= lo + width {
                    want[((a >> PAY_SHIFT) & 15) as usize] += 1;
                }
            }
            assert_eq!(bars, want, "bars for brush {lo}..={}", lo + width);
            checked += 1;
        }

        let step = START + (NEW_STEPS - 1) - (i as u64 % NEW_STEPS);
        let t0 = Instant::now();
        let layer = match on_gpu.iter().position(|s| s.0 == step) {
            Some(k) => on_gpu[k].1,
            None => {
                let layer = place(&mut on_gpu, step, layers);
                match cache.get(step) {
                    Some(v) => field.upload(&gpu, layer, &v),
                    None => {
                        refetched += 1;
                        let v = rt.block_on(array.chunk_f32(&[step, 0, 0])).expect("chunk");
                        field.upload(&gpu, layer, &v);
                        cache.insert(&budget, step, v);
                    }
                }
                layer
            }
        };
        draw_field(layer);
        scrub_ms.push(t0.elapsed());
    }
    report.memory("after_interaction");
    report.samples("brush_update", &brush_ms, None);
    report.samples("scrub_step", &scrub_ms, None);
    report.metric(
        "bars_checked_against_cpu",
        f64::from(checked),
        "updates",
        None,
        "all matched (on the rows loaded)",
    );
    report.metric(
        "era5_cache",
        cache.len() as f64,
        "steps",
        None,
        &format!(
            "{} evicted by the budget, {not_cached} not cached, {refetched} refetched while scrubbing",
            cache.evicted()
        ),
    );
    report.metric(
        "budget_peak_reserved",
        budget.peak() as f64 / 1e6,
        "MB",
        None,
        &format!("of {} MB", cap / 1_000_000),
    );
    let peak = footprint().1;
    report.metric(
        "peak_footprint_vs_cap",
        peak as f64 / 1e6,
        "MB",
        Some(cap as f64 * 1.1 / 1e6),
        "the cap plus 10%",
    );
    drop((gpu_res, attrs_res));
    std::process::exit(if report.finish() { 0 } else { 1 });
}

/// The GPU layer for `step`: a free one, or the least recently placed step's.
fn place(on_gpu: &mut VecDeque<(u64, u32)>, step: u64, layers: u32) -> u32 {
    let layer = if on_gpu.len() < layers as usize {
        on_gpu.len() as u32
    } else {
        on_gpu.pop_front().expect("full").1
    };
    on_gpu.push_back((step, layer));
    layer
}

/// One cached step: its index, its values, and the memory they hold under the budget.
type Entry = (u64, Arc<Vec<f32>>, Reservation);

/// Decoded ERA5 steps in RAM, least recently inserted first, each held under the budget.
#[derive(Default)]
struct StepCache {
    steps: Mutex<VecDeque<Entry>>,
    evicted: std::sync::atomic::AtomicUsize,
}

impl StepCache {
    /// Caches `values` if the budget has room, evicting this cache's oldest steps if needed.
    /// Returns false if it didn't fit.
    fn insert(&self, budget: &Budget, step: u64, values: Vec<f32>) -> bool {
        let bytes = values.len() as u64 * 4;
        let r = budget.try_reserve(Consumer::Cache, bytes).or_else(|| {
            self.evict(bytes);
            budget.try_reserve(Consumer::Cache, bytes)
        });
        match r {
            Some(r) => {
                let mut steps = self.steps.lock().expect("cache");
                steps.push_back((step, Arc::new(values), r));
                true
            }
            None => false,
        }
    }

    fn get(&self, step: u64) -> Option<Arc<Vec<f32>>> {
        let steps = self.steps.lock().expect("cache");
        steps.iter().find(|s| s.0 == step).map(|s| s.1.clone())
    }

    fn len(&self) -> usize {
        self.steps.lock().expect("cache").len()
    }

    fn evicted(&self) -> usize {
        self.evicted.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Evict for StepCache {
    fn evict(&self, bytes: u64) -> u64 {
        let mut steps = self.steps.lock().expect("cache");
        let mut freed = 0;
        while freed < bytes {
            let Some((_, _, r)) = steps.pop_front() else {
                break;
            };
            freed += r.bytes();
            self.evicted
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        freed
    }
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
