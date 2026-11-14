//! S7's scene in a window: S2's map (38M pickups as density, 1M dropoffs as dots) drawn to a
//! `CAMetalLayer` owned by a native host, on a loop that zooms into Midtown, pans and zooms out.
//!
//! The host calls [`LiveMap::new`] with its layer, [`LiveMap::render`] once per display refresh,
//! and passes the layer's size in pixels with every call, so a frame always matches the layer. The
//! points load on a background thread; until they're in, frames are cleared to the background
//! color, so the window appears at once.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use dv_render::density::{DensityMap, LayerData, View};
use dv_render::density_index::DensityIndex;
use dv_render::gpu::Gpu;

use crate::taxi::TaxiPoints;
use crate::{Caches, Report, Summary};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const DOT_STRIDE: usize = 38;
/// Seconds per leg of the loop: zoom in, pan, zoom out.
const LEG_SECONDS: f64 = 4.0;
/// The map's background, drawn while the points load.
const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.02,
    b: 0.03,
    a: 1.0,
};

/// What the host needs to know about one frame.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Frame {
    /// Whether the map was drawn; false while the points load.
    pub ready: bool,
    /// Whether a frame was presented at all (false if the window is hidden).
    pub presented: bool,
    /// CPU time from asking for a drawable to presenting it, in milliseconds. Includes waiting for
    /// a free drawable, which is how a GPU that falls behind shows up.
    pub cpu_ms: f64,
    pub width: u32,
    pub height: u32,
}

struct Loaded {
    origin: [f64; 2],
    extent: [f64; 4],
    density: LayerData,
    dots: LayerData,
}

/// The map, drawn to a host's layer.
pub struct LiveMap {
    gpu: Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    loading: Option<mpsc::Receiver<Loaded>>,
    map: Option<(DensityMap, [f64; 4])>,
    /// While the host is resizing: the surface stays at least this size, and frames draw at its
    /// top left (see [`Self::begin_resize`]).
    resizing: Option<[u32; 2]>,
}

impl std::fmt::Debug for LiveMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveMap")
            .field("size", &[self.config.width, self.config.height])
            .field("ready", &self.map.is_some())
            .finish()
    }
}

impl LiveMap {
    /// Opens the GPU for `layer` and starts loading the points.
    ///
    /// # Safety
    ///
    /// `layer` must point to a valid `CAMetalLayer` that outlives the map.
    #[cfg(target_os = "macos")]
    pub unsafe fn new(layer: *mut std::ffi::c_void, size: [u32; 2]) -> Result<Self, String> {
        // SAFETY: passed on from the caller.
        let (gpu, surface) = unsafe { Gpu::for_metal_layer(layer)? };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: FORMAT,
            width: size[0].max(1),
            height: size[1].max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: Vec::new(),
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        surface.configure(&gpu.device, &config);
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let taxi = TaxiPoints::load();
            let dots: Vec<[f32; 2]> = taxi.dropoffs.iter().step_by(DOT_STRIDE).copied().collect();
            let [x0, y0, x1, y1] = taxi.extent;
            let o = taxi.origin;
            let rel = [
                (x0 - o[0]) as f32,
                (y0 - o[1]) as f32,
                (x1 - o[0]) as f32,
                (y1 - o[1]) as f32,
            ];
            let density = DensityIndex::build(&taxi.pickups, rel).into();
            let dots = DensityIndex::build(&dots, rel).into();
            let _ = tx.send(Loaded {
                origin: o,
                extent: taxi.extent,
                density,
                dots,
            });
        });
        Ok(Self {
            gpu,
            surface,
            config,
            loading: Some(rx),
            map: None,
            resizing: None,
        })
    }

    /// Starts a live resize. Reconfiguring a surface waits for the GPU to go idle, and a resize
    /// changes the size every frame, so until [`Self::end_resize`] the surface is sized once to
    /// `max` pixels (the screen) and each frame draws at its top left. The host's layer must show
    /// its contents at the top left, unscaled.
    pub fn begin_resize(&mut self, max: [u32; 2]) {
        self.resizing = Some(max);
    }

    /// Ends a live resize; the next frame sizes the surface to the view again.
    pub fn end_resize(&mut self) {
        self.resizing = None;
    }

    /// Draws and presents the frame at `seconds` into the loop, at `size` pixels.
    pub fn render(&mut self, seconds: f64, size: [u32; 2]) -> Frame {
        let size = [size[0].max(1), size[1].max(1)];
        let start = Instant::now();
        let surface = match self.resizing {
            // Grow only: shrinking would reconfigure on every frame of the resize.
            Some(max) => [
                max[0].max(size[0]).max(self.config.width),
                max[1].max(size[1]).max(self.config.height),
            ],
            None => size,
        };
        let resized = [self.config.width, self.config.height] != surface;
        if resized {
            self.config.width = surface[0];
            self.config.height = surface[1];
            self.surface.configure(&self.gpu.device, &self.config);
        }
        if let Some(rx) = &self.loading
            && let Ok(l) = rx.try_recv()
        {
            let map = DensityMap::new(&self.gpu, l.origin, l.density, l.dots, size, FORMAT);
            self.map = Some((map, l.extent));
            self.loading = None;
        }
        let configured = start.elapsed();
        let t0 = Instant::now();
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return Frame::default();
            }
            other => {
                if std::env::var_os("DV_TRACE_FRAMES").is_some() {
                    eprintln!("no drawable: {other:?}");
                }
                return Frame::default();
            }
        };
        let acquired = t0.elapsed();
        let target = texture.texture.create_view(&Default::default());
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let mut map_resize = Duration::ZERO;
        let ready = match &mut self.map {
            Some((map, extent)) => {
                let r = Instant::now();
                map.resize(&self.gpu, size);
                map_resize = r.elapsed();
                map.encode(
                    &self.gpu,
                    &mut enc,
                    &target,
                    &tour(*extent, size, seconds),
                    None,
                );
                true
            }
            None => {
                enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("loading"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(BACKGROUND),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                false
            }
        };
        let encoded = t0.elapsed();
        self.gpu.queue.submit([enc.finish()]);
        self.gpu.queue.present(texture);
        if std::env::var_os("DV_TRACE_FRAMES").is_some()
            && (start.elapsed().as_secs_f64() > 0.0125 || resized)
        {
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            eprintln!(
                "slow frame {:.1} ms{}: configure {:.1}, acquire {:.1}, map resize {:.1}, encode {:.1}, submit+present {:.1}",
                ms(start.elapsed()),
                if resized { " (resized)" } else { "" },
                ms(configured),
                ms(acquired),
                ms(map_resize),
                ms(encoded - acquired - map_resize),
                ms(t0.elapsed() - encoded),
            );
        }
        Frame {
            ready,
            presented: true,
            cpu_ms: t0.elapsed().as_secs_f64() * 1e3,
            width: size[0],
            height: size[1],
        }
    }
}

/// Where the camera is `seconds` into the loop: zoom 32× from the whole city into Midtown, pan
/// 6 km east to west, zoom back out, with smooth starts and stops.
pub fn tour(extent: [f64; 4], size: [u32; 2], seconds: f64) -> View {
    let [x0, y0, x1, y1] = extent;
    let fit = (f64::from(size[0]) / (x1 - x0)).min(f64::from(size[1]) / (y1 - y0));
    let city = [(x0 + x1) / 2.0, (y0 + y1) / 2.0];
    let (mx, my) = dv_grid::mercator::forward(-73.9855, 40.7580); // Times Square
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let ease = |t: f64| t * t * (3.0 - 2.0 * t);
    let at = seconds.rem_euclid(3.0 * LEG_SECONDS) / LEG_SECONDS;
    let (leg, t) = (at.floor() as u32, ease(at.fract()));
    match leg {
        0 => View {
            center: [lerp(city[0], mx + 3000.0, t), lerp(city[1], my, t)],
            scale: fit * 32f64.powf(t),
        },
        1 => View {
            center: [mx + 3000.0 - 6000.0 * t, my],
            scale: fit * 32.0,
        },
        _ => View {
            center: [lerp(mx - 3000.0, city[0], t), lerp(my, city[1], t)],
            scale: fit * 32f64.powf(1.0 - t),
        },
    }
}

/// What the host measured while the map ran, for S7.
#[derive(Debug, Clone, Copy)]
pub struct S7<'a> {
    /// The display's highest refresh rate.
    pub max_fps: f64,
    /// Time between successive display-link callbacks while the map animated at a fixed size.
    pub intervals_ms: &'a [f64],
    /// CPU time per frame (see [`Frame::cpu_ms`]).
    pub frame_cpu_ms: &'a [f64],
    /// Resizes of the view during the scripted resize, and how many of them drew a frame of the new
    /// size in the same Core Animation transaction as the resize.
    pub resizes: u32,
    pub resizes_drawn_in_step: u32,
    /// Time between display-link callbacks during the scripted resize.
    pub resize_intervals_ms: &'a [f64],
}

/// Records S7's metrics and returns whether they pass.
pub fn report_s7(s: &S7<'_>) -> bool {
    let mut report = Report::new("S7", Caches::Warm);
    let period = 1000.0 / s.max_fps;
    let dur = |v: &[f64]| -> Vec<Duration> {
        v.iter()
            .map(|ms| Duration::from_secs_f64(ms / 1e3))
            .collect()
    };
    report.metric(
        "display_max_fps",
        s.max_fps,
        "fps",
        None,
        "built-in display",
    );
    let n = s.intervals_ms.len().max(1) as f64;
    let mean_fps = 1000.0 / (s.intervals_ms.iter().sum::<f64>() / n);
    report.metric(
        "fps_below_max",
        s.max_fps - mean_fps,
        "fps",
        Some(2.0),
        &format!(
            "mean {mean_fps:.1} fps over {} frames while the map animates",
            s.intervals_ms.len()
        ),
    );
    let late = s
        .intervals_ms
        .iter()
        .filter(|&&ms| ms > 1.5 * period)
        .count();
    report.metric(
        "frames_missed",
        late as f64 / n * 100.0,
        "%",
        Some(1.0),
        &format!(
            "{late} intervals longer than 1.5 refreshes ({:.1} ms)",
            1.5 * period
        ),
    );
    report.samples("frame_interval", &dur(s.intervals_ms), None);
    report.samples("frame_cpu", &dur(s.frame_cpu_ms), None);
    let summary = Summary::of(&dur(s.resize_intervals_ms));
    report.metric(
        "resize_frames_not_in_step",
        f64::from(s.resizes - s.resizes_drawn_in_step),
        "frames",
        Some(0.5),
        &format!(
            "{} of {} resizes drew the new size in the resize's own transaction; frame interval while resizing p50 {:.1} p95 {:.1} ms",
            s.resizes_drawn_in_step, s.resizes, summary.p50, summary.p95
        ),
    );
    let late = s
        .resize_intervals_ms
        .iter()
        .filter(|&&ms| ms > 1.5 * period)
        .count();
    report.metric(
        "resize_frames_missed",
        late as f64 / s.resize_intervals_ms.len().max(1) as f64 * 100.0,
        "%",
        None,
        &format!(
            "{late} of {} intervals while resizing longer than 1.5 refreshes; recorded, no limit",
            s.resize_intervals_ms.len()
        ),
    );
    report.memory("end");
    report.finish()
}
