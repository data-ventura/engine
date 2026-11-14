// A gridded field on a regular longitude-latitude grid, drawn in plate carree. Each pixel shows the
// nearest grid point's value (no smoothing, so cells stay honest), colored over a fixed range.
// Missing values (NaN) show the background.

struct View {
    // Longitude and latitude at the center of the target, in degrees.
    center: vec2<f32>,
    // Pixels per degree.
    scale: f32,
    _pad0: f32,
    size: vec2<f32>,
    // Grid: first longitude and latitude, and the step between points, in degrees.
    origin: vec2<f32>,
    step: vec2<f32>,
    grid: vec2<u32>,
    // Values mapped to the ends of the colormap.
    range: vec2<f32>,
    // Which layer of the texture array to draw.
    layer: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var field: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<vec4<f32>, 256>;

const BACKGROUND = vec3<f32>(0.055, 0.102, 0.169); // #0E1A2B

@vertex
fn fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn draw(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let lon = view.center.x + (pos.x - view.size.x * 0.5) / view.scale;
    let lat = view.center.y - (pos.y - view.size.y * 0.5) / view.scale;
    if (abs(lat) > 90.0) {
        return vec4<f32>(BACKGROUND, 1.0);
    }
    let nx = i32(view.grid.x);
    let ny = i32(view.grid.y);
    let fi = round((lon - view.origin.x) / view.step.x);
    let i = ((i32(fi) % nx) + nx) % nx;
    let j = clamp(i32(round((lat - view.origin.y) / view.step.y)), 0, ny - 1);
    let v = textureLoad(field, vec2<i32>(i, j), i32(view.layer), 0).r;
    if (v != v) {
        return vec4<f32>(BACKGROUND, 1.0);
    }
    let t = clamp((v - view.range.x) / (view.range.y - view.range.x), 0.0, 1.0);
    return vec4<f32>(lut[u32(t * 255.0 + 0.5)].rgb, 1.0);
}
