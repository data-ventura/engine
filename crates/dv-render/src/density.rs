//! A density map of many points, recomputed at screen resolution every frame, with a smaller set of
//! points drawn on top as dots.
//!
//! Points are projected coordinates stored as `f32` relative to an `f64` origin, so a city-sized
//! extent keeps centimeter precision. Every frame re-bins at screen resolution from a
//! [`DensityIndex`] for each layer: pre-counted cells smaller than half a pixel or raw points,
//! whichever is less work for the view. Nothing is resampled from an earlier frame.
//!
//! Dots share one color and opacity, so they're composited order-independently from per-pixel
//! optical depth. The result doesn't depend on draw order, and pixels stop accepting dots once
//! they're opaque, so dense areas stay cheap.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::colormaps::VIRIDIS;
use crate::density_index::{CHUNK, Chunk, DensityIndex, Plan};
use crate::gpu::{Gpu, GpuTimer};

/// Timestamps a frame writes: the start, then the end of each of its five phases.
pub const FRAME_TIMESTAMPS: u32 = 6;
/// Phase names, in timestamp order after the start. Metal may overlap `bin` and `dots`, which
/// touch different buffers, so only the whole-frame time is exact.
pub const PHASES: [&str; 5] = ["clear", "bin", "dots", "max", "draw"];

/// Threads launched for each grid-stride compute dispatch.
const WORKGROUPS: u32 = 8192;
const WORKGROUP_SIZE: u32 = 256;
/// The most workgroups one dimension of a dispatch allows.
const MAX_WORKGROUPS: u32 = 65535;
/// Size of the stats buffer: max, landed, 32 histogram bins, color-scale top.
const STATS_BYTES: u64 = 35 * 4;
/// Dot radius in pixels.
const DOT_RADIUS: f32 = 1.5;

/// What to show: a center in projected meters and a zoom in pixels per meter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    pub center: [f64; 2],
    pub scale: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    center: [f32; 2],
    scale: f32,
    radius: f32,
    size: [u32; 2],
    dot_chunk_count: u32,
    threads: u32,
    grid_min: [f32; 2],
    side: f32,
    chunk_count: u32,
    filter_on: u32,
    filter_shift: u32,
    filter_lo: u32,
    filter_hi: u32,
    bars_on: u32,
    bar_shift: u32,
    rows: u32,
    _pad: u32,
}

/// Keeps rows whose attribute byte at bit `shift` is in `lo..=hi`. Every layer and the bar
/// counts use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Filter {
    pub shift: u32,
    pub lo: u32,
    pub hi: u32,
}

/// How a frame binned each layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binning {
    pub density: Plan,
    pub dots: Plan,
}

/// A layer to draw: its index, and optionally one attribute word per row in the index's point
/// order (see [`DensityIndex::permute`]) for filtering and bar counts.
#[derive(Debug, Clone)]
pub struct LayerData {
    pub index: DensityIndex,
    pub attrs: Vec<u32>,
}

impl From<DensityIndex> for LayerData {
    fn from(index: DensityIndex) -> Self {
        Self {
            index,
            attrs: Vec::new(),
        }
    }
}

/// One indexed layer on the GPU.
#[derive(Debug)]
struct Layer {
    index: DensityIndex,
    points: wgpu::Buffer,
    cells: wgpu::Buffer,
    chunks: wgpu::Buffer,
    attrs: wgpu::Buffer,
    scratch: Vec<Chunk>,
}

impl Layer {
    /// Uploads the index's points, cells and attributes, and frees the CPU copies.
    fn upload(gpu: &Gpu, data: LayerData, label: &str) -> Self {
        let LayerData { mut index, attrs } = data;
        assert!(
            attrs.is_empty() || attrs.len() == index.len as usize,
            "one attribute per row"
        );
        let storage =
            |contents: &[u8], what: &str| gpu.storage_buffer(contents, &format!("{label} {what}"));
        let (pts, cls) = index.take_data();
        let points = storage(bytemuck::cast_slice(&pts), "points");
        let cells = storage(bytemuck::cast_slice(&cls), "cells");
        let attrs = storage(bytemuck::cast_slice(&attrs), "attributes");
        drop((pts, cls));
        // Worst case: every point in one chunk each per CHUNK, every tile run split, the outside.
        let max_chunks = u64::from(index.len) / u64::from(CHUNK) + 2 * 4096 + 2;
        let chunks = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} chunks")),
            size: max_chunks * std::mem::size_of::<Chunk>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            index,
            points,
            cells,
            chunks,
            attrs,
            scratch: Vec::new(),
        }
    }

    fn plan(&mut self, gpu: &Gpu, rect: [f32; 4], scale: f64, allow_cells: bool) -> Plan {
        let plan = self.index.plan(rect, scale, allow_cells, &mut self.scratch);
        gpu.queue
            .write_buffer(&self.chunks, 0, bytemuck::cast_slice(&self.scratch));
        plan
    }
}

/// Per-pixel buffers and the bind groups that use them. Rebuilt when the target is resized.
#[derive(Debug)]
struct Screen {
    counts: wgpu::Buffer,
    compute_group: wgpu::BindGroup,
    draw_group: wgpu::BindGroup,
}

/// Renders a density map and dots into a target of a given size.
#[derive(Debug)]
pub struct DensityMap {
    size: [u32; 2],
    origin: [f64; 2],
    density: Layer,
    dots: Layer,
    filter: Option<Filter>,
    bar_shift: Option<u32>,
    bars: wgpu::Buffer,
    uniform: wgpu::Buffer,
    stats: wgpu::Buffer,
    lut: wgpu::Buffer,
    compute_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
    screen: Screen,
    clear: wgpu::ComputePipeline,
    bin: wgpu::ComputePipeline,
    splat: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    normalize: wgpu::ComputePipeline,
    bar_counts: wgpu::ComputePipeline,
    draw: wgpu::RenderPipeline,
}

impl DensityMap {
    /// Sets up a map of `size` pixels drawing one indexed layer as density and another as dots.
    /// Both indexes must be relative to `origin` and share an extent.
    pub fn new(
        gpu: &Gpu,
        origin: [f64; 2],
        density: LayerData,
        dots: LayerData,
        size: [u32; 2],
        format: wgpu::TextureFormat,
    ) -> Self {
        assert_eq!(
            (density.index.min, density.index.side),
            (dots.index.min, dots.index.side),
            "layers must share an extent"
        );
        let device = &gpu.device;
        let density = Layer::upload(gpu, density, "density");
        let dots = Layer::upload(gpu, dots, "dots");
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let uniform = buffer(
            "density view",
            std::mem::size_of::<ViewUniform>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let bars = buffer(
            "bars",
            16 * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let stats = buffer(
            "density stats",
            STATS_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let lut: Vec<[f32; 4]> = VIRIDIS
            .iter()
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    1.0,
                ]
            })
            .collect();
        let lut = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colormap"),
            contents: bytemuck::cast_slice(&lut),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let layout = |label, stages, kinds: &[Option<bool>]| {
            let entries: Vec<_> = kinds
                .iter()
                .enumerate()
                .map(|(i, kind)| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: stages,
                    ty: wgpu::BindingType::Buffer {
                        ty: match kind {
                            None => wgpu::BufferBindingType::Uniform,
                            Some(read_only) => wgpu::BufferBindingType::Storage {
                                read_only: *read_only,
                            },
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            })
        };
        let (ro, rw) = (Some(true), Some(false));
        let compute_layout = layout(
            "density compute",
            wgpu::ShaderStages::COMPUTE,
            &[None, ro, rw, rw, ro, ro, ro, rw, ro, ro, ro, ro, rw],
        );
        let draw_layout = layout(
            "density draw",
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            &[None, ro, ro, ro, ro],
        );
        let module = |label, source: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        };
        let compute_module = module("density.wgsl", include_str!("density.wgsl"));
        let draw_module = module("density_draw.wgsl", include_str!("density_draw.wgsl"));
        let pipeline_layout = |bgl| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            })
        };
        let compute_pl = pipeline_layout(&compute_layout);
        let compute = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&compute_pl),
                module: &compute_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let draw = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("density draw"),
            layout: Some(&pipeline_layout(&draw_layout)),
            vertex: wgpu::VertexState {
                module: &draw_module,
                entry_point: Some("fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &draw_module,
                entry_point: Some("density"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let screen = Screen::new(
            gpu,
            size,
            &compute_layout,
            &draw_layout,
            &uniform,
            &stats,
            &lut,
            &bars,
            &density,
            &dots,
        );
        Self {
            size,
            origin,
            density,
            dots,
            filter: None,
            bar_shift: None,
            bars,
            uniform,
            stats,
            lut,
            compute_layout,
            draw_layout,
            screen,
            clear: compute("clear"),
            bin: compute("bin_chunks"),
            splat: compute("splat_chunks"),
            reduce: compute("reduce_max"),
            normalize: compute("normalize"),
            bar_counts: compute("bar_counts"),
            draw,
        }
    }

    /// The buffer holding, after a frame: the largest count, the points in view, 32 bins of pixels
    /// by floor(log2(count)), and the top of the color scale (all `u32`).
    pub fn stats_buffer(&self) -> &wgpu::Buffer {
        &self.stats
    }

    /// Filters every layer and the bar counts, or clears the filter with `None`. Filtered frames
    /// bin raw points, since pre-counted cells carry no attributes.
    pub fn set_filter(&mut self, filter: Option<Filter>) {
        self.filter = filter;
    }

    /// Counts passing density rows by the 4 attribute bits at `shift` every frame, or stops with
    /// `None`. Read the counts with [`Self::read_bars`].
    pub fn set_bars(&mut self, shift: Option<u32>) {
        self.bar_shift = shift;
    }

    /// The last frame's bar counts. Waits for the GPU.
    pub fn read_bars(&self, gpu: &Gpu) -> [u32; 16] {
        let v: Vec<u32> = bytemuck::cast_slice(&read_buffer(gpu, &self.bars)).to_vec();
        v.try_into().expect("16 bars")
    }

    /// Changes the target size. Per-pixel buffers are reallocated; the indexes stay on the GPU.
    pub fn resize(&mut self, gpu: &Gpu, size: [u32; 2]) {
        if size == self.size {
            return;
        }
        self.size = size;
        self.screen = Screen::new(
            gpu,
            size,
            &self.compute_layout,
            &self.draw_layout,
            &self.uniform,
            &self.stats,
            &self.lut,
            &self.bars,
            &self.density,
            &self.dots,
        );
    }

    /// Reads the per-pixel counts of the last frame, row by row from the top. For tests and
    /// diagnostics: it waits for the GPU.
    pub fn read_counts(&self, gpu: &Gpu) -> Vec<u32> {
        bytemuck::cast_slice(&read_buffer(gpu, &self.screen.counts)).to_vec()
    }

    /// Reads the stats buffer (see [`Self::stats_buffer`]). Waits for the GPU.
    pub fn read_stats(&self, gpu: &Gpu) -> Vec<u32> {
        bytemuck::cast_slice(&read_buffer(gpu, &self.stats)).to_vec()
    }

    /// Records one frame into `encoder`, drawn at the top left of `target`, which must be at least
    /// the map's size. With a timer of [`FRAME_TIMESTAMPS`] slots, it records the start, then the
    /// end of each of [`PHASES`].
    pub fn encode(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        view: &View,
        timer: Option<&GpuTimer>,
    ) -> Binning {
        let center = [
            (view.center[0] - self.origin[0]) as f32,
            (view.center[1] - self.origin[1]) as f32,
        ];
        // Pad by a dot radius so dots just outside the edge still draw their visible part.
        let pad = f64::from(DOT_RADIUS + 1.0);
        let half = [
            ((f64::from(self.size[0]) / 2.0 + pad) / view.scale) as f32,
            ((f64::from(self.size[1]) / 2.0 + pad) / view.scale) as f32,
        ];
        let rect = [
            center[0] - half[0],
            center[1] - half[1],
            center[0] + half[0],
            center[1] + half[1],
        ];
        let binning = Binning {
            density: self
                .density
                .plan(gpu, rect, view.scale, self.filter.is_none()),
            dots: self.dots.plan(gpu, rect, view.scale, self.filter.is_none()),
        };
        let chunk_count = self.density.scratch.len() as u32;
        let dot_chunk_count = self.dots.scratch.len() as u32;
        let u = ViewUniform {
            center,
            scale: view.scale as f32,
            radius: DOT_RADIUS,
            size: self.size,
            dot_chunk_count,
            threads: WORKGROUPS * WORKGROUP_SIZE,
            grid_min: self.density.index.min,
            side: self.density.index.side,
            chunk_count,
            filter_on: u32::from(self.filter.is_some()),
            filter_shift: self.filter.map_or(0, |f| f.shift),
            filter_lo: self.filter.map_or(0, |f| f.lo),
            filter_hi: self.filter.map_or(0, |f| f.hi),
            bars_on: u32::from(self.bar_shift.is_some()),
            bar_shift: self.bar_shift.unwrap_or(0),
            rows: self.density.index.len,
            _pad: 0,
        };
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));

        fn compute_pass<'e>(
            encoder: &'e mut wgpu::CommandEncoder,
            group: &wgpu::BindGroup,
            timer: Option<&GpuTimer>,
            label: &str,
            stamps: (Option<u32>, Option<u32>),
        ) -> wgpu::ComputePass<'e> {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: timer.map(|t| t.compute(stamps.0, stamps.1)),
            });
            pass.set_bind_group(0, group, &[]);
            pass
        }
        let group = &self.screen.compute_group;
        let work = [
            ("clear", &self.clear, WORKGROUPS, (Some(0), Some(1))),
            (
                "bin",
                &self.bin,
                chunk_count.min(MAX_WORKGROUPS),
                (None, Some(2)),
            ),
            (
                "dots",
                &self.splat,
                dot_chunk_count.min(MAX_WORKGROUPS),
                (None, Some(3)),
            ),
            ("max", &self.reduce, WORKGROUPS, (None, Some(4))),
        ];
        for (label, pipeline, workgroups, stamps) in work {
            let mut pass = compute_pass(encoder, group, timer, label, stamps);
            if workgroups > 0 {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            if label == "bin" && self.bar_shift.is_some() {
                pass.set_pipeline(&self.bar_counts);
                pass.dispatch_workgroups(WORKGROUPS, 1, 1);
            }
            if label == "max" {
                pass.set_pipeline(&self.normalize);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timer.map(|t| t.render(None, Some(5))),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // The map's own size, at the target's top left: a window's surface can be larger than
            // the map while it's being resized.
            pass.set_viewport(0.0, 0.0, self.size[0] as f32, self.size[1] as f32, 0.0, 1.0);
            pass.set_bind_group(0, &self.screen.draw_group, &[]);
            pass.set_pipeline(&self.draw);
            pass.draw(0..3, 0..1);
        }
        if let Some(t) = timer {
            t.resolve(encoder);
        }
        binning
    }
}

/// Copies a buffer to the CPU. Waits for the GPU.
fn read_buffer(gpu: &Gpu, src: &wgpu::Buffer) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: src.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, src.size());
    gpu.queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    gpu.wait();
    rx.recv().expect("map callback").expect("map readback");
    let out = staging.get_mapped_range(..).expect("mapped").to_vec();
    staging.unmap();
    out
}

impl Screen {
    #[allow(clippy::too_many_arguments)]
    fn new(
        gpu: &Gpu,
        size: [u32; 2],
        compute_layout: &wgpu::BindGroupLayout,
        draw_layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        stats: &wgpu::Buffer,
        lut: &wgpu::Buffer,
        bars: &wgpu::Buffer,
        density: &Layer,
        dots: &Layer,
    ) -> Self {
        let device = &gpu.device;
        let pixels = u64::from(size[0]) * u64::from(size[1]);
        let per_pixel = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: pixels * 4,
                usage,
                mapped_at_creation: false,
            })
        };
        let counts = per_pixel(
            "density counts",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let cover = per_pixel("dot cover", wgpu::BufferUsages::STORAGE);
        let bind = |layout, buffers: &[&wgpu::Buffer]| {
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &entries,
            })
        };
        let compute_group = bind(
            compute_layout,
            &[
                uniform,
                &density.points,
                &counts,
                stats,
                &density.cells,
                &density.chunks,
                &dots.points,
                &cover,
                &dots.cells,
                &dots.chunks,
                &density.attrs,
                &dots.attrs,
                bars,
            ],
        );
        let draw_group = bind(draw_layout, &[uniform, &counts, stats, &cover, lut]);
        Self {
            counts,
            compute_group,
            draw_group,
        }
    }
}
