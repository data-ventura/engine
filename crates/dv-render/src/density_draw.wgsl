// Draws the density counts as a colormapped image, with the splatted dots composited on top.

struct View {
    // View center relative to the points' origin, in projected meters.
    center: vec2<f32>,
    // Pixels per projected meter.
    scale: f32,
    // Dot radius in pixels.
    radius: f32,
    size: vec2<u32>,
    dot_chunk_count: u32,
    threads: u32,
    // Corner and side of the indexes' square extent, relative to the origin.
    grid_min: vec2<f32>,
    side: f32,
    chunk_count: u32,
    // A row passes when byte `filter_shift / 8` of its attribute word is in [filter_lo, filter_hi],
    // or always when filter_on is 0.
    filter_on: u32,
    filter_shift: u32,
    filter_lo: u32,
    filter_hi: u32,
    // Bars count passing rows by the 4 bits at bar_shift, over all `rows` density rows.
    bars_on: u32,
    bar_shift: u32,
    rows: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> stats: array<u32>;
@group(0) @binding(3) var<storage, read> cover: array<u32>;
@group(0) @binding(4) var<storage, read> lut: array<vec4<f32>, 256>;

const BACKGROUND = vec3<f32>(0.055, 0.102, 0.169); // #0E1A2B
const DOT = vec3<f32>(0.922, 0.408, 0.204); // #EB6834
const DEPTH_SCALE = 1024.0;

// A 256-step sRGB colormap, indexed by t in [0, 1].
fn colormap(t: f32) -> vec3<f32> {
    return lut[u32(clamp(t, 0.0, 1.0) * 255.0 + 0.5)].rgb;
}

@vertex
fn fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn density(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let i = u32(pos.y) * view.size.x + u32(pos.x);
    let c = counts[i];
    var color = BACKGROUND;
    if (c > 0u) {
        color = colormap(log2(1.0 + f32(c)) / log2(1.0 + f32(stats[34])));
    }
    let dot_alpha = 1.0 - exp(-f32(cover[i]) / DEPTH_SCALE);
    return vec4<f32>(mix(color, DOT, dot_alpha), 1.0);
}
