//! Conservative (first-order, area-weighted) regridding.
//!
//! Each target cell takes the mean of the source cells it overlaps, weighted by overlap area, so
//! a field's area-weighted integral is preserved: wherever the target grid covers the source,
//! the sum of target value × valid area equals the sum of source value × area.
//!
//! **Missing values.** Missing (NaN) source values are left out and the rest renormalised by the
//! area they cover, as xESMF and xarray-regrid do with `skipna`. A target cell is missing when no
//! valid source value overlaps it, or when valid values cover less than `min_valid` of its area
//! (xESMF's and xarray-regrid's `1 - na_thres`). Parts of a target cell outside the source grid
//! count as not covered, so values are never extrapolated. To mask source cells (land, say), set
//! them to NaN.
//!
//! **Geometry.** Rectilinear grids in longitude and latitude have exact cell areas: their edges
//! are meridians and parallels, so an overlap's area is R² Δλ Δ(sin φ). A global longitude axis
//! wraps. Planar rectilinear grids (in a projected CRS) use plane areas in CRS units, which are
//! true areas only in an equal-area projection. Curvilinear cells have great-circle edges, as in
//! ESMF (see [`curvilinear`](crate::curvilinear)); a rectilinear grid given as one
//! ([`Rectilinear::to_curvilinear`]) then gets great-circle edges too, as ESMF gives it.
//! Overlaps smaller than 1e-10 of either cell are rounding along shared edges and are dropped.

use crate::crs::CrsError;
use crate::curvilinear::{Curvilinear, clip, polygon_area};
use crate::mesh::EARTH_RADIUS;

/// Overlaps smaller than this fraction of either cell (or of either interval, per axis) are
/// dropped as rounding.
const SLIVER: f64 = 1e-10;

/// A rectilinear grid given by its cell edges: `x.len() - 1` by `y.len() - 1` cells, row-major
/// with `y` as the slow axis like [`Grid`](crate::regular::Grid). Edges may be unevenly spaced
/// (Gaussian latitudes) and run in either direction.
#[derive(Debug, Clone, PartialEq)]
pub struct Rectilinear {
    x: Vec<f64>,
    y: Vec<f64>,
    geographic: bool,
}

impl Rectilinear {
    /// A grid from its edges: longitude and latitude in degrees when `geographic`, otherwise
    /// planar coordinates in CRS units.
    pub fn from_edges(x: Vec<f64>, y: Vec<f64>, geographic: bool) -> Result<Self, CrsError> {
        for (name, e) in [("x", &x), ("y", &y)] {
            if e.len() < 2 {
                return Err(CrsError(format!("{name} needs at least two edges")));
            }
            if e.iter().any(|v| !v.is_finite()) {
                return Err(CrsError(format!("{name} edges must be finite")));
            }
            let up = e[1] > e[0];
            if e.windows(2).any(|w| (w[1] > w[0]) != up || w[1] == w[0]) {
                return Err(CrsError(format!("{name} edges must be strictly monotonic")));
            }
        }
        if geographic {
            if y.iter().any(|v| v.abs() > 90.0) {
                return Err(CrsError("latitude edges must be within ±90°".into()));
            }
            if (x[x.len() - 1] - x[0]).abs() > 360.0 + 1e-9 {
                return Err(CrsError("longitude edges span more than 360°".into()));
            }
        }
        Ok(Self { x, y, geographic })
    }

    /// A grid from cell centers, with edges halfway between them and the outer edges as far
    /// out again, as xarray-regrid and CDO infer them. Latitude edges stop at the poles.
    pub fn from_centers(x: &[f64], y: &[f64], geographic: bool) -> Result<Self, CrsError> {
        let edges = |c: &[f64]| -> Result<Vec<f64>, CrsError> {
            if c.len() < 2 {
                return Err(CrsError("edges need at least two centers".into()));
            }
            let n = c.len();
            let mut e = Vec::with_capacity(n + 1);
            e.push(c[0] - (c[1] - c[0]) / 2.0);
            e.extend(c.windows(2).map(|w| (w[0] + w[1]) / 2.0));
            e.push(c[n - 1] + (c[n - 1] - c[n - 2]) / 2.0);
            Ok(e)
        };
        let mut ye = edges(y)?;
        if geographic {
            ye.iter_mut().for_each(|v| *v = v.clamp(-90.0, 90.0));
        }
        Self::from_edges(edges(x)?, ye, geographic)
    }

    pub fn nx(&self) -> usize {
        self.x.len() - 1
    }

    pub fn ny(&self) -> usize {
        self.y.len() - 1
    }

    pub fn x_edges(&self) -> &[f64] {
        &self.x
    }

    pub fn y_edges(&self) -> &[f64] {
        &self.y
    }

    pub fn is_geographic(&self) -> bool {
        self.geographic
    }

    /// Cell areas: square meters on a sphere of [`EARTH_RADIUS`] for a geographic grid,
    /// otherwise squared CRS units.
    pub fn areas(&self) -> Vec<f64> {
        let (xs, ys) = (self.widths(&self.x, false), self.widths(&self.y, true));
        let r2 = if self.geographic {
            EARTH_RADIUS * EARTH_RADIUS
        } else {
            1.0
        };
        ys.iter()
            .flat_map(|h| xs.iter().map(move |w| w * h * r2))
            .collect()
    }

    /// Cell widths along one axis, in the measure areas multiply: radians of longitude, the
    /// difference in sin(latitude), or plain coordinate differences.
    fn widths(&self, e: &[f64], is_y: bool) -> Vec<f64> {
        e.windows(2)
            .map(|w| {
                let (lo, hi) = (w[0].min(w[1]), w[0].max(w[1]));
                match (self.geographic, is_y) {
                    (true, true) => hi.to_radians().sin() - lo.to_radians().sin(),
                    (true, false) => (hi - lo).to_radians(),
                    (false, _) => hi - lo,
                }
            })
            .collect()
    }

    /// The grid as a curvilinear one, with great-circle edges between its corners, to regrid
    /// from or to a curvilinear grid. Only for geographic grids with cells under 180° wide.
    pub fn to_curvilinear(&self) -> Result<Curvilinear, CrsError> {
        if !self.geographic {
            return Err(CrsError(
                "only a longitude-latitude grid can be curvilinear".into(),
            ));
        }
        if self.x.windows(2).any(|w| (w[1] - w[0]).abs() >= 180.0) {
            return Err(CrsError("cells must be under 180° wide".into()));
        }
        let (lon, lat): (Vec<f64>, Vec<f64>) = self
            .y
            .iter()
            .flat_map(|y| self.x.iter().map(move |x| (*x, *y)))
            .unzip();
        Curvilinear::from_corners(self.nx(), self.ny(), &lon, &lat)
    }
}

/// Overlaps along one axis: for each target interval, the source intervals it meets and the
/// overlap in the measure of [`Rectilinear::widths`].
fn axis_overlaps(src: &[f64], dst: &[f64], geographic: bool, is_y: bool) -> Vec<Vec<(u32, f64)>> {
    let span = |e: &[f64], i: usize| (e[i].min(e[i + 1]), e[i].max(e[i + 1]));
    let measure = |lo: f64, hi: f64| match (geographic, is_y) {
        (true, true) => hi.to_radians().sin() - lo.to_radians().sin(),
        (true, false) => (hi - lo).to_radians(),
        (false, _) => hi - lo,
    };
    // A longitude axis wraps: compare against the source moved by whole turns.
    let wraps = geographic && !is_y;
    (0..dst.len() - 1)
        .map(|i| {
            let (dlo, dhi) = span(dst, i);
            let mut row = Vec::new();
            for k in 0..src.len() - 1 {
                let (slo, shi) = span(src, k);
                let turns = if wraps {
                    ((dlo - shi) / 360.0).floor() as i64..=((dhi - slo) / 360.0).ceil() as i64
                } else {
                    0..=0
                };
                let mut w = 0.0;
                for t in turns {
                    let shift = t as f64 * 360.0;
                    let (lo, hi) = (dlo.max(slo + shift), dhi.min(shi + shift));
                    if hi > lo {
                        w += measure(lo, hi);
                    }
                }
                if w > SLIVER * measure(dlo, dhi).min(measure(slo, shi)) {
                    row.push((k as u32, w));
                }
            }
            row
        })
        .collect()
}

/// Conservative regridding weights from a source grid to a target grid: for each target cell,
/// the source cells it overlaps and the overlap areas.
#[derive(Debug, Clone)]
pub struct Conservative {
    /// Each target cell's first entry, and one past the last cell's.
    start: Vec<usize>,
    src: Vec<u32>,
    overlap: Vec<f64>,
    src_area: Vec<f64>,
    dst_area: Vec<f64>,
}

impl Conservative {
    /// Weights between rectilinear grids, both geographic or both planar.
    pub fn rectilinear(src: &Rectilinear, dst: &Rectilinear) -> Result<Self, CrsError> {
        if src.geographic != dst.geographic {
            return Err(CrsError(
                "can't regrid between a geographic and a planar grid".into(),
            ));
        }
        let g = src.geographic;
        let xo = axis_overlaps(&src.x, &dst.x, g, false);
        let yo = axis_overlaps(&src.y, &dst.y, g, true);
        let r2 = if g { EARTH_RADIUS * EARTH_RADIUS } else { 1.0 };
        let snx = src.nx();
        let mut w = Self::empty(src.areas(), dst.areas());
        for row in &yo {
            for col in &xo {
                for (l, wy) in row {
                    for (k, wx) in col {
                        w.src.push(*l * snx as u32 + k);
                        w.overlap.push(wx * wy * r2);
                    }
                }
                w.start.push(w.src.len());
            }
        }
        Ok(w)
    }

    /// Weights from a curvilinear grid to another, or to a rectilinear one through
    /// [`Rectilinear::to_curvilinear`]. Target cells must be convex, as in ESMF; source cells
    /// needn't be.
    pub fn curvilinear(src: &Curvilinear, dst: &Curvilinear) -> Self {
        let r2 = EARTH_RADIUS * EARTH_RADIUS;
        let mut w = Self::empty(src.areas(), dst.areas());
        let mut found = Vec::new();
        for d in 0..dst.cells.len() {
            found.clear();
            src.index
                .query(&dst.index.boxes[d], &mut |s| found.push(s as u32));
            found.sort_unstable();
            let dst_area = w.dst_area[d] / r2;
            for &s in &found {
                let mut a = 0.0;
                for sp in src.cells.parts(s as usize) {
                    for dp in dst.cells.parts(d) {
                        a += polygon_area(&clip(sp, dp));
                    }
                }
                if a > SLIVER * dst_area.min(w.src_area[s as usize] / r2) {
                    w.src.push(s);
                    w.overlap.push(a * r2);
                }
            }
            w.start.push(w.src.len());
        }
        w
    }

    fn empty(src_area: Vec<f64>, dst_area: Vec<f64>) -> Self {
        Self {
            start: vec![0],
            src: Vec::new(),
            overlap: Vec::new(),
            src_area,
            dst_area,
        }
    }

    /// The number of source cells.
    pub fn sources(&self) -> usize {
        self.src_area.len()
    }

    /// The number of target cells.
    pub fn targets(&self) -> usize {
        self.dst_area.len()
    }

    /// Source cell areas (square meters, or squared CRS units on a planar grid).
    pub fn source_areas(&self) -> &[f64] {
        &self.src_area
    }

    /// Target cell areas, in the same units.
    pub fn target_areas(&self) -> &[f64] {
        &self.dst_area
    }

    /// The source cells a target cell overlaps, and the overlap areas.
    pub fn overlaps(&self, target: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.start[target]..self.start[target + 1];
        self.src[r.clone()]
            .iter()
            .zip(&self.overlap[r])
            .map(|(s, a)| (*s as usize, *a))
    }

    /// Valid (overlap) area and value × area summed over one target cell.
    fn sums(&self, target: usize, values: &[f64]) -> (f64, f64) {
        self.overlaps(target)
            .filter(|(s, _)| values[*s].is_finite())
            .fold((0.0, 0.0), |(a, va), (s, w)| (a + w, va + w * values[s]))
    }

    /// Regrids source values (row-major, NaN for missing) to the target grid. A target cell is
    /// missing unless valid values cover at least `min_valid` (0 to 1) of its area; with 0, any
    /// valid overlap will do. Coverage is compared with a relative tolerance of 1e-9, so a fully
    /// covered cell passes `min_valid` = 1 despite rounding.
    pub fn apply(&self, values: &[f64], min_valid: f64) -> Vec<f64> {
        assert_eq!(values.len(), self.sources(), "one value per source cell");
        assert!(
            (0.0..=1.0).contains(&min_valid),
            "min_valid is a fraction from 0 to 1"
        );
        (0..self.targets())
            .map(|d| {
                let (a, va) = self.sums(d, values);
                if a > 0.0 && a >= min_valid * self.dst_area[d] * (1.0 - 1e-9) {
                    va / a
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// The fraction of each target cell's area covered by valid source values (ESMF's
    /// destination fraction): 1 inside the source grid where nothing is missing. NaN for target
    /// cells with no geometry.
    pub fn valid_fraction(&self, values: &[f64]) -> Vec<f64> {
        assert_eq!(values.len(), self.sources(), "one value per source cell");
        (0..self.targets())
            .map(|d| self.sums(d, values).0 / self.dst_area[d])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_from_centers() {
        let g = Rectilinear::from_centers(&[0.5, 1.5, 2.5], &[89.5, 88.5], true).unwrap();
        assert_eq!(g.x_edges(), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(g.y_edges(), [90.0, 89.0, 88.0]);
        // Latitude edges stop at the poles.
        let g = Rectilinear::from_centers(&[0.0, 1.0], &[90.0, 89.0, 88.0], true).unwrap();
        assert_eq!(g.y_edges()[0], 90.0);
        assert!(Rectilinear::from_centers(&[0.0, 2.0, 1.0], &[0.0, 1.0], false).is_err());
        assert!(Rectilinear::from_edges(vec![0.0, 1.0], vec![0.0, 91.0], true).is_err());
    }

    #[test]
    fn longitude_wraps() {
        // A cell from 350° to 370° on a 0–360 source takes the source's last and first cells.
        let src =
            Rectilinear::from_edges(vec![0.0, 10.0, 350.0, 360.0], vec![0.0, 1.0], true).unwrap();
        let dst = Rectilinear::from_edges(vec![-10.0, 10.0], vec![0.0, 1.0], true).unwrap();
        let w = Conservative::rectilinear(&src, &dst).unwrap();
        assert_eq!(w.apply(&[1.0, 5.0, 3.0], 1.0), [2.0]);
        let dst = Rectilinear::from_edges(vec![350.0, 370.0], vec![0.0, 1.0], true).unwrap();
        let w = Conservative::rectilinear(&src, &dst).unwrap();
        assert_eq!(w.apply(&[1.0, 5.0, f64::NAN], 0.0), [1.0]);
        assert!(w.apply(&[1.0, 5.0, f64::NAN], 0.6)[0].is_nan());
    }

    #[test]
    fn outside_the_source_is_missing() {
        let src = Rectilinear::from_edges(vec![0.0, 1.0, 2.0], vec![0.0, 1.0], false).unwrap();
        let dst = Rectilinear::from_edges(vec![1.5, 2.5, 3.5], vec![0.0, 1.0], false).unwrap();
        let w = Conservative::rectilinear(&src, &dst).unwrap();
        let out = w.apply(&[1.0, 2.0], 0.0);
        assert_eq!(out[0], 2.0);
        assert!(out[1].is_nan());
        assert_eq!(w.valid_fraction(&[1.0, 2.0]), [0.5, 0.0]);
        assert!(w.apply(&[1.0, 2.0], 0.6)[0].is_nan());
    }
}
