//! Regular grids (evenly spaced x and y coordinates) and interpolation on them.
//!
//! Points outside the grid are unmatched (NaN), never clamped to the edge. Bilinear
//! interpolation needs all four surrounding cells; if any is missing the result is missing, so
//! interpolation never runs across a gap. A global longitude axis wraps across the antimeridian.

use crate::crs::CrsError;

/// One evenly spaced axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis {
    pub start: f64,
    /// Signed spacing: negative for descending coordinates (latitude from 90 down).
    pub step: f64,
    pub len: usize,
    /// Whether the axis is a longitude covering all 360°, so it wraps.
    pub periodic: bool,
}

impl Axis {
    /// An axis from its coordinates, which must be evenly spaced (within 1e-6 of a step).
    pub fn from_coords(coords: &[f64]) -> Result<Self, CrsError> {
        if coords.len() < 2 {
            return Err(CrsError("an axis needs at least two coordinates".into()));
        }
        let step = (coords[coords.len() - 1] - coords[0]) / (coords.len() - 1) as f64;
        if step == 0.0 || !step.is_finite() {
            return Err(CrsError("axis coordinates must be distinct".into()));
        }
        for (i, c) in coords.iter().enumerate() {
            if (c - (coords[0] + step * i as f64)).abs() > 1e-6 * step.abs() {
                return Err(CrsError(format!("axis isn't evenly spaced at index {i}")));
            }
        }
        Ok(Self {
            start: coords[0],
            step,
            len: coords.len(),
            periodic: false,
        })
    }

    /// Marks a longitude axis as periodic if its cells cover the full circle.
    pub fn longitude(mut self) -> Self {
        self.periodic = ((self.step.abs() * self.len as f64) - 360.0).abs() < 1e-6;
        self
    }

    pub fn coord(&self, i: usize) -> f64 {
        self.start + self.step * i as f64
    }

    /// The fractional index of `x`, or `None` outside the axis. A periodic axis maps `x` into
    /// `[0, len)` and may fall between the last and first coordinates.
    fn position(&self, x: f64) -> Option<f64> {
        let mut f = (x - self.start) / self.step;
        if self.periodic {
            f = f.rem_euclid(self.len as f64);
        }
        let last = (self.len - 1) as f64;
        let tol = 1e-9;
        if self.periodic {
            Some(f)
        } else if f >= -tol && f <= last + tol {
            Some(f.clamp(0.0, last))
        } else {
            None
        }
    }

    /// The two indices around `x` and the weight of the second. As in scipy, the pair is taken in
    /// ascending coordinate order (`c[i] <= x < c[i + 1]`, the last pair at the top end), so a
    /// descending axis brackets the same cells an ascending copy of it would.
    fn bracket(&self, x: f64) -> Option<(usize, usize, f64)> {
        let f = self.position(x)?;
        let n = self.len;
        if self.periodic {
            let i = (f.floor() as usize).min(n - 1);
            return Some((i, (i + 1) % n, f - i as f64));
        }
        let ascending = if self.step > 0.0 {
            f
        } else {
            (n - 1) as f64 - f
        };
        let i = (ascending.floor() as usize).min(n - 2);
        let w = ascending - i as f64;
        Some(if self.step > 0.0 {
            (i, i + 1, w)
        } else {
            (n - 1 - i, n - 2 - i, w)
        })
    }

    fn nearest(&self, x: f64) -> Option<usize> {
        let f = self.position(x)?;
        let i = f.round() as usize;
        Some(if i >= self.len {
            if self.periodic { 0 } else { self.len - 1 }
        } else {
            i
        })
    }
}

/// A regular grid of values, row-major with `y` as the slow axis: `values[iy * x.len + ix]`.
#[derive(Debug, Clone, Copy)]
pub struct Grid<'a> {
    pub x: Axis,
    pub y: Axis,
    pub values: &'a [f64],
}

impl Grid<'_> {
    fn at(&self, ix: usize, iy: usize) -> f64 {
        self.values[iy * self.x.len + ix]
    }

    /// Bilinear interpolation at one point. NaN outside the grid or next to a missing value.
    pub fn bilinear(&self, x: f64, y: f64) -> f64 {
        let (Some((x0, x1, wx)), Some((y0, y1, wy))) = (self.x.bracket(x), self.y.bracket(y))
        else {
            return f64::NAN;
        };
        let corners = [
            (self.at(x0, y0), (1.0 - wx) * (1.0 - wy)),
            (self.at(x1, y0), wx * (1.0 - wy)),
            (self.at(x0, y1), (1.0 - wx) * wy),
            (self.at(x1, y1), wx * wy),
        ];
        // Any missing corner makes the result missing, even one with no weight because the point
        // is on a grid line, as in xarray and scipy. Python exported from a workspace then gives
        // the same answer.
        if corners.iter().any(|(v, _)| v.is_nan()) {
            return f64::NAN;
        }
        corners.iter().map(|(v, w)| v * w).sum()
    }

    /// The value of the nearest grid point. NaN outside the grid.
    pub fn nearest(&self, x: f64, y: f64) -> f64 {
        match (self.x.nearest(x), self.y.nearest(y)) {
            (Some(ix), Some(iy)) => self.at(ix, iy),
            _ => f64::NAN,
        }
    }

    /// Bilinear regridding onto another regular grid's points, row-major like `values`.
    pub fn regrid_bilinear(&self, x: &Axis, y: &Axis) -> Vec<f64> {
        let mut out = Vec::with_capacity(x.len * y.len);
        for iy in 0..y.len {
            for ix in 0..x.len {
                out.push(self.bilinear(x.coord(ix), y.coord(iy)));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_and_never_clamps() {
        let x = Axis::from_coords(&[0.0, 90.0, 180.0, 270.0])
            .unwrap()
            .longitude();
        let y = Axis::from_coords(&[10.0, 0.0]).unwrap();
        let values = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
        let g = Grid {
            x,
            y,
            values: &values,
        };
        assert!(x.periodic);
        assert_eq!(g.bilinear(315.0, 5.0), 1.5, "between 270 and 360 = 0");
        assert_eq!(g.bilinear(-45.0, 5.0), 1.5);
        assert!(g.bilinear(45.0, 11.0).is_nan(), "outside in y");
        assert_eq!(g.bilinear(90.0, 10.0), 1.0, "on a grid point");
    }

    #[test]
    fn missing_values_stay_missing() {
        let x = Axis::from_coords(&[0.0, 1.0, 2.0]).unwrap();
        let y = Axis::from_coords(&[0.0, 1.0]).unwrap();
        let values = [0.0, f64::NAN, 2.0, 0.0, 1.0, 2.0];
        let g = Grid {
            x,
            y,
            values: &values,
        };
        assert!(g.bilinear(0.5, 0.5).is_nan());
        assert!(
            g.bilinear(2.0, 0.5).is_nan(),
            "a missing corner counts even with no weight"
        );
        assert!(Axis::from_coords(&[0.0, 1.0, 3.0]).is_err());
    }
}
