//! Checks GPU density binning against binning each point on the CPU by the same definition: a
//! point at `p` lands in pixel `floor((p - center) * scale + size / 2)`, with y pointing down.
//! Skipped when no GPU is available.

use dv_render::density::{DensityMap, Filter, LayerData, View};
use dv_render::density_index::{DensityIndex, Plan};
use dv_render::gpu::Gpu;

const SIZE: [u32; 2] = [320, 200];

/// Clustered points from a fixed-seed generator, plus a few far outside the extent.
fn points() -> Vec<[f32; 2]> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut pts = Vec::new();
    for i in 0..60_000 {
        // A sum of uniforms clusters points toward a few centers.
        let c = [
            (i % 3) as f32 * 200.0 - 200.0,
            (i % 2) as f32 * 150.0 - 75.0,
        ];
        let dx = (next() + next() + next() - 1.5) * 120.0;
        let dy = (next() + next() + next() - 1.5) * 120.0;
        pts.push([c[0] + dx, c[1] + dy]);
    }
    pts.extend([[1e6, 1e6], [-5e5, 3.0], [0.0, -9e5]]);
    pts
}

fn pixel_of(p: [f32; 2], view: &View) -> Option<usize> {
    let (c, s) = (
        [view.center[0] as f32, view.center[1] as f32],
        view.scale as f32,
    );
    let x = SIZE[0] as f32 * 0.5 + (p[0] - c[0]) * s;
    let y = SIZE[1] as f32 * 0.5 - (p[1] - c[1]) * s;
    (x >= 0.0 && y >= 0.0 && x < SIZE[0] as f32 && y < SIZE[1] as f32)
        .then(|| y as usize * SIZE[0] as usize + x as usize)
}

fn cpu_counts(pts: &[[f32; 2]], view: &View) -> Vec<u32> {
    let mut out = vec![0; (SIZE[0] * SIZE[1]) as usize];
    for i in pts.iter().filter_map(|p| pixel_of(*p, view)) {
        out[i] += 1;
    }
    out
}

fn render(gpu: &Gpu, map: &mut DensityMap, view: &View) -> (Plan, Vec<u32>, Vec<u32>) {
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
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    let b = map.encode(
        gpu,
        &mut enc,
        &target.create_view(&Default::default()),
        view,
        None,
    );
    gpu.queue.submit([enc.finish()]);
    (b.density, map.read_counts(gpu), map.read_stats(gpu))
}

#[test]
fn gpu_binning_matches_cpu() {
    let Ok(gpu) = Gpu::headless() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let pts = points();
    let extent = [-500.0, -400.0, 500.0, 400.0];
    let index = DensityIndex::build(&pts, extent);
    let dots = DensityIndex::build(&[], extent);
    let mut map = DensityMap::new(
        &gpu,
        [0.0, 0.0],
        index.clone().into(),
        dots.into(),
        SIZE,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    // Zoomed in: raw points, so every pixel matches except where f32 rounding differs exactly at
    // a pixel edge.
    let zoomed = View {
        center: [-195.0, 72.0],
        scale: 50.0,
    };
    let (plan, gpu_counts, stats) = render(&gpu, &mut map, &zoomed);
    assert!(matches!(plan, Plan::Points { .. }), "{plan:?}");
    let want = cpu_counts(&pts, &zoomed);
    assert_eq!(
        gpu_counts.iter().sum::<u32>(),
        want.iter().sum::<u32>(),
        "points in view"
    );
    let differ = gpu_counts.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert!(
        differ * 1000 <= want.len(),
        "{differ} of {} pixels differ",
        want.len()
    );
    assert_eq!(stats[0], *want.iter().max().unwrap(), "largest count");
    assert_eq!(stats[1], want.iter().sum::<u32>(), "landed");

    // Zoomed out: pre-counted cells. The GPU bins each cell's count at its center, exactly like
    // the CPU does here; and every point is within a quarter pixel of its cell's center.
    let wide = View {
        center: [0.0, 0.0],
        scale: 0.3,
    };
    let (plan, gpu_counts, _) = render(&gpu, &mut map, &wide);
    let Plan::Cells { bits, .. } = plan else {
        panic!("{plan:?}")
    };
    let level = index.levels.iter().find(|l| l.bits == bits).unwrap();
    let cell = index.side / (1u32 << bits) as f32;
    let mut want = vec![0u32; (SIZE[0] * SIZE[1]) as usize];
    for c in &index.cells[level.offset as usize..(level.offset + level.len) as usize] {
        let xy = [(c[0] >> 16) as f32, (c[0] & 0xFFFF) as f32];
        let center = [
            index.min[0] + (xy[0] + 0.5) * cell,
            index.min[1] + (xy[1] + 0.5) * cell,
        ];
        if let Some(i) = pixel_of(center, &wide) {
            want[i] += c[1];
        }
    }
    let outside = &index.points[index.inside as usize..];
    for (w, n) in want.iter_mut().zip(cpu_counts(outside, &wide)) {
        *w += n;
    }
    assert_eq!(
        gpu_counts.iter().sum::<u32>(),
        cpu_counts(&pts, &wide).iter().sum::<u32>()
    );
    let differ = gpu_counts.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert!(
        differ * 1000 <= want.len(),
        "{differ} of {} pixels differ",
        want.len()
    );
    let pixel = 1.0 / wide.scale as f32;
    for p in &index.points[..index.inside as usize] {
        let cx = index.min[0] + (((p[0] - index.min[0]) / cell).floor() + 0.5) * cell;
        let cy = index.min[1] + (((p[1] - index.min[1]) / cell).floor() + 0.5) * cell;
        assert!((p[0] - cx).abs() <= pixel / 4.0 && (p[1] - cy).abs() <= pixel / 4.0);
    }
}

#[test]
fn gpu_filter_and_bars_match_cpu() {
    let Ok(gpu) = Gpu::headless() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let pts = points();
    // Byte 0: a bin from position; byte 1: a category from row number.
    let attrs: Vec<u32> = pts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let bin = ((p[0].abs() + p[1].abs()) as u32 / 7) % 64;
            bin | ((i as u32 % 6 + 1) << 8)
        })
        .collect();
    let extent = [-500.0, -400.0, 500.0, 400.0];
    let index = DensityIndex::build(&pts, extent);
    let sorted_attrs = index.permute(&attrs);
    let dots = DensityIndex::build(&[], extent);
    let mut map = DensityMap::new(
        &gpu,
        [0.0, 0.0],
        LayerData {
            index,
            attrs: sorted_attrs,
        },
        dots.into(),
        SIZE,
        wgpu::TextureFormat::Rgba8Unorm,
    );
    let filter = Filter {
        shift: 0,
        lo: 10,
        hi: 25,
    };
    map.set_filter(Some(filter));
    map.set_bars(Some(8));
    let kept: Vec<[f32; 2]> = pts
        .iter()
        .zip(&attrs)
        .filter(|(_, a)| (10..=25).contains(&(*a & 0xFF)))
        .map(|(p, _)| *p)
        .collect();
    let mut want_bars = [0u32; 16];
    for a in attrs.iter().filter(|a| (10..=25).contains(&(*a & 0xFF))) {
        want_bars[((a >> 8) & 15) as usize] += 1;
    }

    // Zoomed out would use cells, but a filter forces raw points, so pixels match the CPU.
    for view in [
        View {
            center: [0.0, 0.0],
            scale: 0.3,
        },
        View {
            center: [-195.0, 72.0],
            scale: 50.0,
        },
    ] {
        let (plan, gpu_counts, _) = render(&gpu, &mut map, &view);
        assert!(matches!(plan, Plan::Points { .. }), "{plan:?}");
        let want = cpu_counts(&kept, &view);
        assert_eq!(
            gpu_counts.iter().sum::<u32>(),
            want.iter().sum::<u32>(),
            "kept points in view"
        );
        let differ = gpu_counts.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert!(differ * 1000 <= want.len(), "{differ} pixels differ");
        assert_eq!(map.read_bars(&gpu), want_bars, "bars count every kept row");
    }
}
