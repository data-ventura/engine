//! The GPU device and GPU-side timing.

use std::sync::mpsc;

/// A wgpu device and queue, plus whether GPU timestamps are available.
#[derive(Debug)]
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::AdapterInfo,
    pub timestamps: bool,
}

impl Gpu {
    /// Opens the high-performance adapter without a window, with the adapter's full limits so large
    /// point buffers fit in one binding.
    pub fn headless() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        Self::open(&instance, None)
    }

    /// Opens the adapter that can present to a `CAMetalLayer`, and a surface on that layer. The
    /// host owns the layer's frame and scale; the surface sets its drawable size when configured.
    ///
    /// # Safety
    ///
    /// `layer` must point to a valid `CAMetalLayer` that outlives the returned surface.
    #[cfg(target_os = "macos")]
    pub unsafe fn for_metal_layer(
        layer: *mut std::ffi::c_void,
    ) -> Result<(Self, wgpu::Surface<'static>), String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        // SAFETY: the caller guarantees a live CAMetalLayer that outlives the surface.
        let surface = unsafe {
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(layer))
        }
        .map_err(|e| format!("no surface on the layer: {e}"))?;
        let gpu = Self::open(&instance, Some(&surface))?;
        Ok((gpu, surface))
    }

    fn open(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, String> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("dv-render"),
            required_features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU device: {e}"))?;
        Ok(Self {
            device,
            queue,
            adapter: adapter.get_info(),
            timestamps,
        })
    }

    /// Blocks until all submitted work has finished.
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Creates a storage buffer holding `contents`, copied in slices of [`UPLOAD_SLICE`] bytes
    /// with a wait after each, so the staging memory wgpu needs stays at one slice instead of a
    /// second copy of the whole buffer. Empty contents get an 8-byte buffer, since bindings can't
    /// be empty.
    pub fn storage_buffer(&self, contents: &[u8], label: &str) -> wgpu::Buffer {
        let size = contents.len().max(8) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for (i, slice) in contents.chunks(UPLOAD_SLICE).enumerate() {
            self.queue
                .write_buffer(&buffer, (i * UPLOAD_SLICE) as u64, slice);
            self.queue.submit([]);
            self.wait();
        }
        buffer
    }
}

/// Bytes copied to the GPU at a time by [`Gpu::storage_buffer`].
pub const UPLOAD_SLICE: usize = 32 << 20;

/// GPU timestamps written at pass boundaries within one submission.
#[derive(Debug)]
pub struct GpuTimer {
    set: wgpu::QuerySet,
    slots: u32,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f64,
}

impl GpuTimer {
    /// A timer with `slots` timestamps. `None` if the device has no timestamp queries.
    pub fn new(gpu: &Gpu, slots: u32) -> Option<Self> {
        if !gpu.timestamps {
            return None;
        }
        let set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame timer"),
            ty: wgpu::QueryType::Timestamp,
            count: slots,
        });
        let size = u64::from(slots) * 8;
        let buffer = |usage, label| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Some(Self {
            set,
            slots,
            resolve: buffer(
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                "timer resolve",
            ),
            readback: buffer(
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                "timer readback",
            ),
            period_ns: f64::from(gpu.queue.get_timestamp_period()),
        })
    }

    /// Timestamp writes for a compute pass: slot indices for its beginning and end.
    pub fn compute(
        &self,
        begin: Option<u32>,
        end: Option<u32>,
    ) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: begin,
            end_of_pass_write_index: end,
        }
    }

    /// Timestamp writes for a render pass: slot indices for its beginning and end.
    pub fn render(
        &self,
        begin: Option<u32>,
        end: Option<u32>,
    ) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: begin,
            end_of_pass_write_index: end,
        }
    }

    /// Copies the timestamps out. Call after the measured passes, in the same encoder.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let size = u64::from(self.slots) * 8;
        encoder.resolve_query_set(&self.set, 0..self.slots, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, size);
    }

    /// Waits for the submission and returns each slot's time in milliseconds after slot 0.
    pub fn read_ms(&self, gpu: &Gpu) -> Vec<f64> {
        let (tx, rx) = mpsc::channel();
        self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        gpu.wait();
        rx.recv().expect("map callback").expect("map timer buffer");
        let ticks: Vec<u64> = {
            let view = self
                .readback
                .get_mapped_range(..)
                .expect("mapped timer buffer");
            view.as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_ne_bytes(*b))
                .collect()
        };
        self.readback.unmap();
        ticks
            .iter()
            .map(|t| t.saturating_sub(ticks[0]) as f64 * self.period_ns / 1e6)
            .collect()
    }
}
