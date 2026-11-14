// Point density at screen resolution, recomputed every frame.
//
// Compute: clear the per-pixel counts; bin pre-counted cells (zoomed out) and raw points from tiles in
// view (zoomed in, and points outside the index) with atomic adds; splat dots; then find the largest
// count and a histogram of log2 counts. The color scale tops out at the 99.5th percentile of non-empty
// pixels, so a few pixels with thousands of duplicate coordinates don't darken everything else; busier
// pixels saturate. The draw pass (density_draw.wgsl) colors each pixel and composites the dots.
//
// Dots all share one color and opacity, so blending them is order-independent: a pixel's dot opacity
// is 1 - exp(-sum of -ln(1 - alpha_i)). Each pixel sums that optical depth in fixed point and stops
// accepting dots once it's visually opaque, so hot spots don't serialize.

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
@group(0) @binding(1) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
// [0] the largest count, [1] points that landed in view, [2..34) pixels by floor(log2(count)),
// [34] the top of the color scale.
@group(0) @binding(3) var<storage, read_write> stats: array<atomic<u32>>;
// Pre-counted cells of every level: [x << 16 | y, count].
// Pre-counted cells of every level: [x << 16 | y, count].
@group(0) @binding(4) var<storage, read> cells: array<vec2<u32>>;
// Work items: [first item, count, level bits or 0 for raw points, 0].
@group(0) @binding(5) var<storage, read> chunks: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> dots: array<vec2<f32>>;
// Per-pixel dot optical depth, in units of 1/DEPTH_SCALE.
@group(0) @binding(7) var<storage, read_write> cover: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> dot_cells: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read> dot_chunks: array<vec4<u32>>;
// Per-row attribute words, in each layer's point order.
@group(0) @binding(10) var<storage, read> attrs: array<u32>;
@group(0) @binding(11) var<storage, read> dot_attrs: array<u32>;
// Passing rows by category.
@group(0) @binding(12) var<storage, read_write> bars: array<atomic<u32>, 16>;

const DOT_ALPHA = 0.35;
const HIST = 2u;
const TOP = 34u;
const TOP_QUANTILE = 0.995;
const DEPTH_SCALE = 1024.0;
// Optical depth 7 is 99.9% opaque: further dots change nothing visible.
const DEPTH_OPAQUE = 7168u;

fn to_pixel(p: vec2<f32>) -> vec2<f32> {
    let s = (p - view.center) * view.scale;
    return vec2<f32>(f32(view.size.x) * 0.5 + s.x, f32(view.size.y) * 0.5 - s.y);
}

@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = view.size.x * view.size.y;
    for (var i = gid.x; i < n; i += view.threads) {
        atomicStore(&counts[i], 0u);
        atomicStore(&cover[i], 0u);
    }
    if (gid.x < 35u) {
        atomicStore(&stats[gid.x], 0u);
    }
    if (gid.x < 16u) {
        atomicStore(&bars[gid.x], 0u);
    }
}

fn add(px: vec2<f32>, n: u32) -> u32 {
    if (px.x >= 0.0 && px.y >= 0.0 && px.x < f32(view.size.x) && px.y < f32(view.size.y)) {
        atomicAdd(&counts[u32(px.y) * view.size.x + u32(px.x)], n);
        return n;
    }
    return 0u;
}

fn passes(attr: u32) -> bool {
    let v = (attr >> view.filter_shift) & 0xFFu;
    return view.filter_on == 0u || (v >= view.filter_lo && v <= view.filter_hi);
}

fn cell_center(c: vec2<u32>, bits: u32) -> vec2<f32> {
    let xy = vec2<f32>(f32(c.x >> 16u), f32(c.x & 0xFFFFu));
    return view.grid_min + (xy + 0.5) * (view.side / f32(1u << bits));
}

@compute @workgroup_size(256)
fn bin_chunks(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    var landed = 0u;
    for (var w = wid.x; w < view.chunk_count; w += nwg.x) {
        let chunk = chunks[w];
        for (var j = lid; j < chunk.y; j += 256u) {
            if (chunk.z == 0u) {
                if (passes(attrs[chunk.x + j])) {
                    landed += add(to_pixel(points[chunk.x + j]), 1u);
                }
            } else {
                let c = cells[chunk.x + j];
                landed += add(to_pixel(cell_center(c, chunk.z)), c.y);
            }
        }
    }
    if (landed > 0u) {
        atomicAdd(&stats[1], landed);
    }
}

// Adds `n` coincident dots centered at pixel position `p`.
fn splat(p: vec2<f32>, n: u32) {
    let r = view.radius;
    let w = i32(view.size.x);
    let x0 = max(i32(floor(p.x - r - 0.5)), 0);
    let x1 = min(i32(floor(p.x + r + 0.5)), w - 1);
    let y0 = max(i32(floor(p.y - r - 0.5)), 0);
    let y1 = min(i32(floor(p.y + r + 0.5)), i32(view.size.y) - 1);
    for (var y = y0; y <= y1; y++) {
        for (var x = x0; x <= x1; x++) {
            let d = distance(vec2<f32>(f32(x), f32(y)) + 0.5, p);
            let a = DOT_ALPHA * clamp(r + 0.5 - d, 0.0, 1.0);
            if (a <= 0.0) {
                continue;
            }
            let idx = u32(y * w + x);
            if (atomicLoad(&cover[idx]) < DEPTH_OPAQUE) {
                let depth = min(-log(1.0 - a) * f32(n), 8.0);
                atomicAdd(&cover[idx], u32(depth * DEPTH_SCALE + 0.5));
            }
        }
    }
}

@compute @workgroup_size(256)
fn splat_chunks(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    for (var w = wid.x; w < view.dot_chunk_count; w += nwg.x) {
        let chunk = dot_chunks[w];
        for (var j = lid; j < chunk.y; j += 256u) {
            if (chunk.z == 0u) {
                if (passes(dot_attrs[chunk.x + j])) {
                    splat(to_pixel(dots[chunk.x + j]), 1u);
                }
            } else {
                let c = dot_cells[chunk.x + j];
                splat(to_pixel(cell_center(c, chunk.z)), c.y);
            }
        }
    }
}

var<workgroup> local_bars: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn bar_counts(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if (lid < 16u) {
        atomicStore(&local_bars[lid], 0u);
    }
    workgroupBarrier();
    for (var i = gid.x; i < view.rows; i += view.threads) {
        let a = attrs[i];
        if (passes(a)) {
            atomicAdd(&local_bars[(a >> view.bar_shift) & 15u], 1u);
        }
    }
    workgroupBarrier();
    if (lid < 16u) {
        let n = atomicLoad(&local_bars[lid]);
        if (n > 0u) {
            atomicAdd(&bars[lid], n);
        }
    }
}

var<workgroup> partial: array<u32, 256>;
var<workgroup> hist: array<atomic<u32>, 32>;

@compute @workgroup_size(256)
fn reduce_max(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if (lid < 32u) {
        atomicStore(&hist[lid], 0u);
    }
    workgroupBarrier();
    let n = view.size.x * view.size.y;
    var m = 0u;
    for (var i = gid.x; i < n; i += view.threads) {
        let c = atomicLoad(&counts[i]);
        if (c > 0u) {
            m = max(m, c);
            atomicAdd(&hist[firstLeadingBit(c)], 1u);
        }
    }
    partial[lid] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            partial[lid] = max(partial[lid], partial[lid + s]);
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        atomicMax(&stats[0], partial[0]);
    }
    if (lid < 32u) {
        let h = atomicLoad(&hist[lid]);
        if (h > 0u) {
            atomicAdd(&stats[HIST + lid], h);
        }
    }
}

// One thread: the top of the color scale is the upper edge of the log2 bin holding the
// TOP_QUANTILE pixel, capped at the largest count.
@compute @workgroup_size(1)
fn normalize() {
    var total = 0u;
    for (var b = 0u; b < 32u; b++) {
        total += atomicLoad(&stats[HIST + b]);
    }
    let wanted = u32(ceil(f32(total) * TOP_QUANTILE));
    var seen = 0u;
    var top = atomicLoad(&stats[0]);
    for (var b = 0u; b < 32u; b++) {
        seen += atomicLoad(&stats[HIST + b]);
        if (seen >= wanted) {
            top = min(top, (2u << b) - 1u);
            break;
        }
    }
    atomicStore(&stats[TOP], max(top, 1u));
}
