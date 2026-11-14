//! Gridded fields on a regular longitude-latitude grid, such as one time step of a reanalysis.
//!
//! Steps live in a texture array on the GPU. Scrubbing between resident steps only changes which
//! layer is drawn; a new step is one texture upload into a free layer.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::colormaps::VIRIDIS;
use crate::gpu::{Gpu, GpuTimer};

/// A regular grid: the first point's longitude and latitude and the step between points, in
/// degrees, and the number of points along each axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub origin: [f32; 2],
    pub step: [f32; 2],
    pub size: [u32; 2],
}

/// What to show: the center in degrees and the zoom in pixels per degree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldView {
    pub center: [f32; 2],
    pub scale: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    center: [f32; 2],
    scale: f32,
    _pad0: f32,
    size: [f32; 2],
    origin: [f32; 2],
    step: [f32; 2],
    grid: [u32; 2],
    range: [f32; 2],
    layer: u32,
    _pad1: u32,
}

/// Draws one resident step of a field into a target.
#[derive(Debug)]
pub struct FieldMap {
    grid: Grid,
    size: [u32; 2],
    range: [f32; 2],
    layers: u32,
    texture: wgpu::Texture,
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl FieldMap {
    /// A map of `size` pixels with room for `layers` resident steps, colored over `range`.
    pub fn new(
        gpu: &Gpu,
        grid: Grid,
        layers: u32,
        size: [u32; 2],
        range: [f32; 2],
        format: wgpu::TextureFormat,
    ) -> Self {
        let device = &gpu.device;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field steps"),
            size: wgpu::Extent3d {
                width: grid.size[0],
                height: grid.size[1],
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field view"),
            size: std::mem::size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        let fs = wgpu::ShaderStages::FRAGMENT;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: fs,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: fs,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: fs,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lut.as_entire_binding(),
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("field.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("field"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("draw"),
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
        Self {
            grid,
            size,
            range,
            layers,
            texture,
            uniform,
            group,
            pipeline,
        }
    }

    /// How many steps can be resident at once.
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Uploads one step's values, row by row from the first latitude, into `layer`.
    pub fn upload(&self, gpu: &Gpu, layer: u32, values: &[f32]) {
        let [w, h] = self.grid.size;
        assert_eq!(values.len(), (w * h) as usize, "one value per grid point");
        assert!(layer < self.layers, "layer {layer} of {}", self.layers);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Records drawing resident `layer`. With a two-slot timer, it measures the pass.
    pub fn encode(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        view: &FieldView,
        layer: u32,
        timer: Option<&GpuTimer>,
    ) {
        let u = ViewUniform {
            center: view.center,
            scale: view.scale,
            _pad0: 0.0,
            size: [self.size[0] as f32, self.size[1] as f32],
            origin: self.grid.origin,
            step: self.grid.step,
            grid: self.grid.size,
            range: self.range,
            layer,
            _pad1: 0,
        };
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("field"),
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
            timestamp_writes: timer.map(|t| t.render(Some(0), Some(1))),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.group, &[]);
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
        drop(pass);
        if let Some(t) = timer {
            t.resolve(encoder);
        }
    }
}
