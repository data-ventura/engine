//! The spike's taxi points: pickups and dropoffs projected to Web Mercator, with a packed
//! attribute word per trip for brushing.

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type, Int64Type, TimeUnit};
use dv_data::parquet::ParquetTable;
use dv_grid::mercator;
use rayon::prelude::*;

use crate::spike_data;

/// Bits 0–7 of the attribute word: trip distance in [`DIST_BIN_MILES`] bins, the last bin open
/// ended, and [`DIST_INVALID`] for negative or missing distances.
pub const DIST_SHIFT: u32 = 0;
pub const DIST_BIN_MILES: f64 = 0.25;
pub const DIST_BINS: u32 = 64;
pub const DIST_INVALID: u32 = 255;
/// Bits 8–11: the TLC payment type code (1 card, 2 cash, 3 no charge, 4 dispute, 5 unknown,
/// 6 voided), or 0 if missing.
pub const PAY_SHIFT: u32 = 8;
/// Bits 16–20: pickup hour of day, local time.
pub const HOUR_SHIFT: u32 = 16;

const COLUMNS: [&str; 7] = [
    "startLon",
    "startLat",
    "endLon",
    "endLat",
    "tripDistance",
    "paymentType",
    "tpepPickupDateTime",
];

/// Pickup and dropoff locations as `f32` meters relative to `origin`.
#[derive(Debug)]
pub struct TaxiPoints {
    pub origin: [f64; 2],
    pub pickups: Vec<[f32; 2]>,
    pub dropoffs: Vec<[f32; 2]>,
    /// One packed attribute word per trip (see the `*_SHIFT` constants).
    pub attrs: Vec<u32>,
    /// A robust extent of the pickups, `[min_x, min_y, max_x, max_y]` in absolute meters: the
    /// 5th to 95th percentile on each axis of a 200,000-row sample, widened by half its size on
    /// every side, so zero and garbage coordinates don't stretch the first view.
    pub extent: [f64; 4],
}

impl TaxiPoints {
    /// Rows whose pickup and dropoff are both finite. Coordinates of 0 are kept: they're in the
    /// data, and they land far off-screen near (0°, 0°).
    ///
    /// The output arrays are allocated once at the file's row count. Row groups are decoded in
    /// parallel, a batch at a time, straight into their own part of the arrays; dropped rows leave
    /// gaps that are closed at the end. Only the output and one batch per thread are in memory.
    pub fn load() -> Self {
        Self::load_every(1)
    }

    /// Like [`Self::load`], keeping only every `stride`-th row of the file: a coarser level of
    /// detail for when the memory budget can't hold every row.
    pub fn load_every(stride: usize) -> Self {
        assert!(stride >= 1);
        let table = ParquetTable::open(spike_data("nyc-taxi-2015q1.parquet")).expect("open");
        // The extent comes from a sample, so the full data can go straight to f32.
        let mut sampled = Vec::new();
        for_each_row(
            &table.sample(200_000).expect("sample").batch,
            |_, p, _, _| sampled.push(p),
        );
        let axis = |i: usize| {
            let mut v: Vec<f64> = sampled.iter().map(|p| p[i]).collect();
            v.sort_by(f64::total_cmp);
            let (lo, hi) = (v[v.len() / 20], v[v.len() * 19 / 20]);
            let pad = (hi - lo) / 2.0;
            (lo - pad, hi + pad)
        };
        let ((x0, x1), (y0, y1)) = (axis(0), axis(1));
        let origin = [(x0 + x1) / 2.0, (y0 + y1) / 2.0];
        let rel = |p: [f64; 2]| [(p[0] - origin[0]) as f32, (p[1] - origin[1]) as f32];

        let rows = table.group_rows();
        // Each group's first row in the file, and how many of its rows the stride keeps.
        let firsts: Vec<usize> = rows
            .iter()
            .scan(0, |at, &r| {
                let first = *at;
                *at += r;
                Some(first)
            })
            .collect();
        let sizes: Vec<usize> = rows
            .iter()
            .zip(&firsts)
            .map(|(&r, &f)| (f + r).div_ceil(stride) - f.div_ceil(stride))
            .collect();
        let n: usize = sizes.iter().sum();
        let mut pickups = vec![[0f32; 2]; n];
        let mut dropoffs = vec![[0f32; 2]; n];
        let mut attrs = vec![0u32; n];
        let mut parts = Vec::with_capacity(sizes.len());
        let (mut p, mut d, mut a) = (&mut pickups[..], &mut dropoffs[..], &mut attrs[..]);
        for (g, &len) in sizes.iter().enumerate() {
            let (p0, p1) = p.split_at_mut(len);
            let (d0, d1) = d.split_at_mut(len);
            let (a0, a1) = a.split_at_mut(len);
            parts.push((g, p0, d0, a0));
            (p, d, a) = (p1, d1, a1);
        }
        let kept: Vec<usize> = parts
            .into_par_iter()
            .map(|(g, p, d, a)| {
                let (mut k, mut row) = (0, firsts[g]);
                for batch in table.batches(g, &COLUMNS).expect("read") {
                    let batch = batch.expect("decode");
                    for_each_row(&batch, |i, pick, drop, attr| {
                        if (row + i).is_multiple_of(stride) {
                            (p[k], d[k], a[k]) = (rel(pick), rel(drop), attr);
                            k += 1;
                        }
                    });
                    row += batch.num_rows();
                }
                k
            })
            .collect();
        // Close the gaps left by dropped rows.
        let (mut to, mut from) = (0, 0);
        for (&len, &k) in sizes.iter().zip(&kept) {
            pickups.copy_within(from..from + k, to);
            dropoffs.copy_within(from..from + k, to);
            attrs.copy_within(from..from + k, to);
            (to, from) = (to + k, from + len);
        }
        pickups.truncate(to);
        dropoffs.truncate(to);
        attrs.truncate(to);
        Self {
            origin,
            pickups,
            dropoffs,
            attrs,
            extent: [x0, y0, x1, y1],
        }
    }
}

/// The trip distance bin of a distance in miles.
pub fn dist_bin(miles: f64) -> u32 {
    if miles.is_finite() && miles >= 0.0 {
        ((miles / DIST_BIN_MILES) as u32).min(DIST_BINS - 1)
    } else {
        DIST_INVALID
    }
}

/// Calls `f(row in batch, pickup, dropoff, attribute word)` for each row of `b` whose pickup and
/// dropoff are both finite, with positions in absolute Web Mercator meters.
fn for_each_row(b: &RecordBatch, mut f: impl FnMut(usize, [f64; 2], [f64; 2], u32)) {
    let col = |name: &str| b.column_by_name(name).expect(name);
    let f64s = |name: &str| col(name).as_primitive::<Float64Type>().clone();
    let (slon, slat, elon, elat) = (
        f64s("startLon"),
        f64s("startLat"),
        f64s("endLon"),
        f64s("endLat"),
    );
    let dist = f64s("tripDistance");
    let pay = cast(col("paymentType"), &DataType::Utf8).expect("payment as text");
    let pay = pay.as_string::<i32>();
    let us = cast(
        col("tpepPickupDateTime"),
        &DataType::Timestamp(TimeUnit::Microsecond, None),
    )
    .and_then(|a| cast(&a, &DataType::Int64))
    .expect("pickup time");
    let us = us.as_primitive::<Int64Type>();

    for i in 0..b.num_rows() {
        let finite = [&slon, &slat, &elon, &elat]
            .iter()
            .all(|c| c.is_valid(i) && c.value(i).is_finite());
        if !finite {
            continue;
        }
        let (sx, sy) = mercator::forward(slon.value(i), slat.value(i));
        let (ex, ey) = mercator::forward(elon.value(i), elat.value(i));
        let d = if dist.is_valid(i) {
            dist_bin(dist.value(i))
        } else {
            DIST_INVALID
        };
        let p = if pay.is_valid(i) {
            pay.value(i).parse::<u32>().unwrap_or(0).min(15)
        } else {
            0
        };
        let h = if us.is_valid(i) {
            us.value(i).div_euclid(3_600_000_000).rem_euclid(24) as u32
        } else {
            0
        };
        let attr = (d << DIST_SHIFT) | (p << PAY_SHIFT) | (h << HOUR_SHIFT);
        f(i, [sx, sy], [ex, ey], attr);
    }
}

/// The two map layers built from the taxi points, ready for [`dv_render::density::DensityMap`].
#[derive(Debug)]
pub struct TaxiLayers {
    pub origin: [f64; 2],
    pub extent: [f64; 4],
    /// Pickups as density, with their attribute words.
    pub density: dv_render::density::LayerData,
    /// Every `dot_stride`-th dropoff as dots, with their attribute words.
    pub dots: dv_render::density::LayerData,
    /// Every row's attribute word in load order, for checking GPU counts.
    pub attrs: Vec<u32>,
}

impl TaxiLayers {
    /// Indexes the pickups and every `dot_stride`-th dropoff, then frees the source points: the
    /// indexes hold their own sorted copies.
    pub fn build(taxi: TaxiPoints, dot_stride: usize) -> Self {
        use dv_render::density::LayerData;
        use dv_render::density_index::DensityIndex;

        let [x0, y0, x1, y1] = taxi.extent;
        let o = taxi.origin;
        let rel = [
            (x0 - o[0]) as f32,
            (y0 - o[1]) as f32,
            (x1 - o[0]) as f32,
            (y1 - o[1]) as f32,
        ];
        let dots: Vec<[f32; 2]> = taxi.dropoffs.iter().step_by(dot_stride).copied().collect();
        let dot_attrs: Vec<u32> = taxi.attrs.iter().step_by(dot_stride).copied().collect();
        drop(taxi.dropoffs);
        let d = DensityIndex::build(&taxi.pickups, rel);
        drop(taxi.pickups);
        let d_attrs = d.permute(&taxi.attrs);
        let k = DensityIndex::build(&dots, rel);
        let k_attrs = k.permute(&dot_attrs);
        Self {
            origin: o,
            extent: taxi.extent,
            density: LayerData {
                index: d,
                attrs: d_attrs,
            },
            dots: LayerData {
                index: k,
                attrs: k_attrs,
            },
            attrs: taxi.attrs,
        }
    }

    /// Bytes the layers will take on the GPU: points, cells and attribute words.
    pub fn gpu_bytes(&self) -> u64 {
        [&self.density, &self.dots]
            .iter()
            .map(|l| {
                (l.index.points.len() * 8 + l.index.cells.len() * 8 + l.attrs.len() * 4) as u64
            })
            .sum()
    }
}
