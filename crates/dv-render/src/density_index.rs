//! A spatial index that keeps density maps fast at every zoom.
//!
//! Points inside a square extent are sorted by their Morton key on a 65,536 × 65,536 grid, so every
//! 64 × 64 tile of the extent is one contiguous range, and so is every cell at any coarser level.
//! Each frame binds one of two kinds of work, whichever is smaller:
//!
//! - **Pre-counted cells** in view, at the coarsest level whose cells are at most half a pixel: one
//!   atomic add per non-empty cell, so a point lands at most a quarter pixel from where it would be
//!   binned on its own.
//! - **Raw points** in view.
//!
//! Cells at each level are in Morton order too, so both kinds are culled by tile.
//!
//! Points outside the extent (bad coordinates, mostly) are kept at the end and binned raw every
//! frame, so nothing is dropped.

use rayon::prelude::*;

/// Bits per axis of the finest grid.
pub const BITS: u32 = 16;
/// Bits per axis of the culling tiles.
const TILE_BITS: u32 = 6;
/// Pre-counted levels, in bits per axis. Coarsest first.
pub const LEVELS: [u32; 4] = [10, 12, 14, 16];
/// Items per work item.
pub const CHUNK: u32 = 4096;
/// A work item: `[first item, count, level bits or 0 for raw points, 0]`.
pub type Chunk = [u32; 4];

/// One pre-counted level: non-empty cells as `[x << 16 | y, count]`.
#[derive(Debug, Clone)]
pub struct Level {
    pub bits: u32,
    /// Offset of this level's first cell in [`DensityIndex::cells`].
    pub offset: u32,
    pub len: u32,
    /// Start of each tile's cells in [`DensityIndex::cells`], plus a final end offset.
    pub tile_starts: Vec<u32>,
}

/// What a frame binned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// Pre-counted cells with this many bits per axis.
    Cells { bits: u32, items: u64 },
    /// Raw points.
    Points { items: u64 },
}

#[derive(Debug, Clone)]
pub struct DensityIndex {
    /// Points inside the extent in Morton order, then the points outside it.
    pub points: Vec<[f32; 2]>,
    /// For each sorted point, its position in the input, so per-row attributes can be put in the
    /// same order (see [`Self::permute`]).
    pub order: Vec<u32>,
    /// How many of `points` are inside the extent.
    pub inside: u32,
    /// How many points there are in total. Stays valid after [`Self::take_data`].
    pub len: u32,
    /// `[min_x, min_y]` of the square extent, relative to the points' origin.
    pub min: [f32; 2],
    /// Side of the square extent.
    pub side: f32,
    /// Start of each tile's range in `points`, plus a final end offset. Indexed by Morton tile index.
    pub tile_starts: Vec<u32>,
    /// Every level's cells, concatenated.
    pub cells: Vec<[u32; 2]>,
    pub levels: Vec<Level>,
}

impl DensityIndex {
    /// Builds the index over `extent` = `[min_x, min_y, max_x, max_y]`, relative to the points'
    /// origin. The extent is made square around its center.
    pub fn build(points: &[[f32; 2]], extent: [f32; 4]) -> Self {
        let side = (extent[2] - extent[0]).max(extent[3] - extent[1]);
        let cx = (extent[0] + extent[2]) / 2.0;
        let cy = (extent[1] + extent[3]) / 2.0;
        let min = [cx - side / 2.0, cy - side / 2.0];
        let n = (1u32 << BITS) as f32;

        // Inside points as one u64 each, the Morton key above the row, so sorting puts them in
        // Morton order with ties in row order and the result doesn't depend on the sort. Outside
        // points keep row order.
        let cell = |p: [f32; 2]| {
            let fx = (p[0] - min[0]) / side * n;
            let fy = (p[1] - min[1]) / side * n;
            (!outside(p, min, side) && fx < n && fy < n).then(|| morton(fx as u32, fy as u32))
        };
        let mut keyed: Vec<u64> = points
            .par_iter()
            .enumerate()
            .filter_map(|(i, &p)| cell(p).map(|k| u64::from(k) << 32 | i as u64))
            .collect();
        let outside_rows: Vec<u32> = points
            .par_iter()
            .enumerate()
            .filter(|(_, p)| cell(**p).is_none())
            .map(|(i, _)| i as u32)
            .collect();
        keyed.par_sort_unstable();
        let inside = keyed.len() as u32;
        let key = |k: u64| (k >> 32) as u32;
        let sorted = &keyed[..];

        let tiles = 1u32 << (2 * TILE_BITS);
        let tile_shift = 2 * (BITS - TILE_BITS);
        let tile_starts = tile_offsets(sorted, |k| key(k) >> tile_shift, tiles, 0);

        // Cells never cross a tile, so each level is counted tile by tile in parallel.
        let mut cells = Vec::new();
        let mut levels = Vec::new();
        for bits in LEVELS {
            let shift = 2 * (BITS - bits);
            let per_tile: Vec<Vec<[u32; 2]>> = tile_starts
                .par_windows(2)
                .map(|w| {
                    let run = &sorted[w[0] as usize..w[1] as usize];
                    let mut out = Vec::new();
                    let mut i = 0;
                    while i < run.len() {
                        let cell = key(run[i]) >> shift;
                        let len = run[i..].partition_point(|&k| key(k) >> shift == cell);
                        let (x, y) = demorton(cell);
                        out.push([x << 16 | y, len as u32]);
                        i += len;
                    }
                    out
                })
                .collect();
            let offset = cells.len() as u32;
            let mut starts = Vec::with_capacity(per_tile.len() + 1);
            for t in per_tile {
                starts.push(cells.len() as u32);
                cells.extend_from_slice(&t);
            }
            starts.push(cells.len() as u32);
            levels.push(Level {
                bits,
                offset,
                len: cells.len() as u32 - offset,
                tile_starts: starts,
            });
        }

        let mut order = Vec::with_capacity(points.len());
        keyed
            .par_iter()
            .map(|&k| k as u32)
            .collect_into_vec(&mut order);
        drop(keyed);
        order.extend_from_slice(&outside_rows);
        Self {
            len: order.len() as u32,
            points: order.par_iter().map(|&i| points[i as usize]).collect(),
            order,
            inside,
            min,
            side,
            tile_starts,
            cells,
            levels,
        }
    }

    /// Puts per-row values in the index's point order.
    pub fn permute<T: Copy + Send + Sync>(&self, values: &[T]) -> Vec<T> {
        self.order.par_iter().map(|&i| values[i as usize]).collect()
    }

    /// Moves the points and cells out, for upload. Planning still works afterwards: it only needs
    /// the offsets.
    pub fn take_data(&mut self) -> (Vec<[f32; 2]>, Vec<[u32; 2]>) {
        self.order = Vec::new();
        (
            std::mem::take(&mut self.points),
            std::mem::take(&mut self.cells),
        )
    }

    /// The coarsest level whose cells are at most half a pixel, if any.
    pub fn level_for(&self, pixels_per_unit: f64) -> Option<&Level> {
        let pixel = 1.0 / pixels_per_unit;
        self.levels
            .iter()
            .find(|l| f64::from(self.side) / f64::from(1u32 << l.bits) <= pixel / 2.0)
    }

    /// Plans a frame showing `view` (`[min_x, min_y, max_x, max_y]`, relative to the origin) at
    /// `pixels_per_unit`: work items for whichever is smaller of the cells in view (at the level
    /// from [`Self::level_for`]) and the raw points in view, then every outside point.
    /// With `allow_cells` false (when rows are filtered, since cells carry no attributes), only
    /// raw points are planned.
    pub fn plan(
        &self,
        view: [f32; 4],
        pixels_per_unit: f64,
        allow_cells: bool,
        out: &mut Vec<Chunk>,
    ) -> Plan {
        out.clear();
        let tiles = self.tiles_in_view(view);
        let points = runs(&tiles, &self.tile_starts);
        let n_points: u64 = points.iter().map(|r| u64::from(r.1 - r.0)).sum();
        let mut plan = Plan::Points { items: n_points };
        let mut chosen = (points, 0);
        if let Some(level) = self.level_for(pixels_per_unit).filter(|_| allow_cells) {
            let cells = runs(&tiles, &level.tile_starts);
            let n_cells: u64 = cells.iter().map(|r| u64::from(r.1 - r.0)).sum();
            if n_cells < n_points {
                plan = Plan::Cells {
                    bits: level.bits,
                    items: n_cells,
                };
                chosen = (cells, level.bits);
            }
        }
        for (start, end) in chosen.0 {
            push_chunks(start, end, chosen.1, out);
        }
        push_chunks(self.inside, self.len, 0, out);
        plan
    }

    /// Tiles overlapping `view`, sorted by Morton index.
    fn tiles_in_view(&self, view: [f32; 4]) -> Vec<u32> {
        let t = 1u32 << TILE_BITS;
        let tile = self.side / t as f32;
        let overlaps = view[2] >= self.min[0]
            && view[3] >= self.min[1]
            && view[0] <= self.min[0] + self.side
            && view[1] <= self.min[1] + self.side;
        if !overlaps {
            return Vec::new();
        }
        let to_tile = |v: f32, m: f32| ((v - m) / tile).floor().clamp(0.0, (t - 1) as f32) as u32;
        let (x0, x1) = (to_tile(view[0], self.min[0]), to_tile(view[2], self.min[0]));
        let (y0, y1) = (to_tile(view[1], self.min[1]), to_tile(view[3], self.min[1]));
        let mut tiles: Vec<u32> = (y0..=y1)
            .flat_map(|y| (x0..=x1).map(move |x| morton(x, y)))
            .collect();
        tiles.sort_unstable();
        tiles
    }
}

/// Start offsets of each tile's range in `keys`, sorted by `tile`, shifted by `base`, plus a final
/// end.
fn tile_offsets(keys: &[u64], tile: impl Fn(u64) -> u32 + Sync, tiles: u32, base: u32) -> Vec<u32> {
    let mut starts: Vec<u32> = (0..tiles)
        .into_par_iter()
        .map(|t| base + keys.partition_point(|&k| tile(k) < t) as u32)
        .collect();
    starts.push(base + keys.len() as u32);
    starts
}

/// Merges sorted tiles into contiguous `(start, end)` ranges of `starts`.
fn runs(tiles: &[u32], starts: &[u32]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < tiles.len() {
        let mut j = i;
        // Adjacent tiles in Morton order are one contiguous range.
        while j + 1 < tiles.len() && tiles[j + 1] == tiles[j] + 1 {
            j += 1;
        }
        let (start, end) = (starts[tiles[i] as usize], starts[tiles[j] as usize + 1]);
        if end > start {
            out.push((start, end));
        }
        i = j + 1;
    }
    out
}

fn outside(p: [f32; 2], min: [f32; 2], side: f32) -> bool {
    !(p[0] >= min[0] && p[1] >= min[1] && p[0] < min[0] + side && p[1] < min[1] + side)
}

fn push_chunks(start: u32, end: u32, bits: u32, out: &mut Vec<Chunk>) {
    let mut s = start;
    while s < end {
        let len = CHUNK.min(end - s);
        out.push([s, len, bits, 0]);
        s += len;
    }
}

/// Interleaves the low 16 bits of `x` and `y`, `x` in the even bits.
pub fn morton(x: u32, y: u32) -> u32 {
    spread(x) | (spread(y) << 1)
}

fn spread(v: u32) -> u32 {
    let mut v = v & 0xFFFF;
    v = (v | (v << 8)) & 0x00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333;
    (v | (v << 1)) & 0x5555_5555
}

fn compact(v: u32) -> u32 {
    let mut v = v & 0x5555_5555;
    v = (v | (v >> 1)) & 0x3333_3333;
    v = (v | (v >> 2)) & 0x0F0F_0F0F;
    v = (v | (v >> 4)) & 0x00FF_00FF;
    (v | (v >> 8)) & 0xFFFF
}

pub fn demorton(k: u32) -> (u32, u32) {
    (compact(k), compact(k >> 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_round_trips() {
        for (x, y) in [(0, 0), (1, 0), (0, 1), (65535, 65535), (12345, 54321)] {
            assert_eq!(demorton(morton(x, y)), (x, y));
        }
        assert_eq!(morton(1, 0), 1);
        assert_eq!(morton(0, 1), 2);
    }

    #[test]
    fn index_keeps_every_point_and_counts_cells() {
        let mut pts = Vec::new();
        // 50 distinct places, 20 points each, so cells are cheaper than points when zoomed out.
        for i in 0..1000 {
            let f = (i % 50) as f32;
            pts.push([(f * 7.3) % 100.0, (f * 3.1) % 100.0]);
        }
        pts.push([500.0, 500.0]);
        pts.push([-1e7, 0.0]);
        let idx = DensityIndex::build(&pts, [0.0, 0.0, 100.0, 100.0]);
        assert_eq!(idx.points.len(), pts.len());
        assert_eq!(idx.inside, 1000);
        for l in &idx.levels {
            let total: u32 = idx.cells[l.offset as usize..(l.offset + l.len) as usize]
                .iter()
                .map(|c| c[1])
                .sum();
            assert_eq!(total, 1000, "level {} counts every inside point", l.bits);
        }
        // At a scale where no level is fine enough, a view covering everything yields every point
        // exactly once.
        let mut chunks = Vec::new();
        let plan = idx.plan([-1e9, -1e9, 1e9, 1e9], 1e9, true, &mut chunks);
        assert!(matches!(plan, Plan::Points { .. }));
        assert_eq!(chunks.iter().map(|c| c[1]).sum::<u32>(), pts.len() as u32);
        // A view of one corner yields its points and the outside ones, and fewer than all.
        idx.plan([0.0, 0.0, 10.0, 10.0], 1e9, true, &mut chunks);
        let n = chunks.iter().map(|c| c[1]).sum::<u32>();
        let corner = pts
            .iter()
            .filter(|p| (0.0..10.0).contains(&p[0]) && (0.0..10.0).contains(&p[1]))
            .count() as u32;
        assert!(
            n >= corner + 2 && n < pts.len() as u32,
            "{n} corner={corner}"
        );
        // Zoomed out, cells carry every inside point's weight.
        let plan = idx.plan([-1e9, -1e9, 1e9, 1e9], 0.001, true, &mut chunks);
        assert!(matches!(plan, Plan::Cells { .. }), "{plan:?}");
        let weight: u32 = chunks
            .iter()
            .filter(|c| c[2] > 0)
            .flat_map(|c| &idx.cells[c[0] as usize..(c[0] + c[1]) as usize])
            .map(|cell| cell[1])
            .sum();
        assert_eq!(weight, 1000);
        // Without cells, the same view plans raw points.
        let plan = idx.plan([-1e9, -1e9, 1e9, 1e9], 0.001, false, &mut chunks);
        assert!(matches!(plan, Plan::Points { .. }), "{plan:?}");
        // The order maps every sorted point back to its row.
        assert!(
            idx.order
                .iter()
                .zip(&idx.points)
                .all(|(&i, p)| pts[i as usize] == *p)
        );
    }
}
