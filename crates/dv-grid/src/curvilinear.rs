//! Curvilinear grids (rotated-pole, polar stereographic, ocean model grids) and the cell that
//! contains a point.
//!
//! A cell is the quadrilateral between four corners given in degrees. Its edges are great-circle
//! arcs, as in ESMF and S2, so cells that share corners tile the sphere with no gaps or overlaps,
//! across the antimeridian and over the poles alike. A cell with a missing (non-finite) corner has
//! no geometry: it contains no point and takes no part in regridding, as ocean grids with missing
//! coordinates over land expect. A concave cell is split into two triangles; a twisted
//! (self-intersecting) one has no geometry either. Corners repeated at a pole make a triangle.

use crate::crs::CrsError;
use crate::mesh::{EARTH_RADIUS, unit};

pub(crate) type Vec3 = [f64; 3];

fn sub(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: &Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// Corners closer than this (as a chord on the unit sphere, about 6 nm on Earth) are one corner.
const SAME: f64 = 1e-15;

/// Signed area (steradians) of the spherical triangle `abc`, positive when counterclockwise seen
/// from outside the sphere (Van Oosterom and Strackee). Differences from `a` keep the triple
/// product accurate for small triangles.
fn triangle_area(a: &Vec3, b: &Vec3, c: &Vec3) -> f64 {
    let num = dot(a, &cross(&sub(b, a), &sub(c, a)));
    let den = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
    2.0 * num.atan2(den)
}

/// Signed area of a spherical polygon with great-circle edges, as a fan from its first vertex.
pub(crate) fn polygon_area(v: &[Vec3]) -> f64 {
    (1..v.len().saturating_sub(1))
        .map(|i| triangle_area(&v[0], &v[i], &v[i + 1]))
        .sum()
}

/// How far `p` is on the inner (left) side of the great circle through `a` and then `b`, scaled
/// by the edge length. Differences from `a` keep it accurate for short edges.
fn side(a: &Vec3, b: &Vec3, p: &Vec3) -> f64 {
    dot(&sub(p, a), &cross(a, &sub(b, a)))
}

/// Clips a polygon to a convex one (Sutherland–Hodgman on great circles). Both counterclockwise.
pub(crate) fn clip(subject: &[Vec3], convex: &[Vec3]) -> Vec<Vec3> {
    let mut out = subject.to_vec();
    let mut input = Vec::with_capacity(out.len() + convex.len());
    for (i, a) in convex.iter().enumerate() {
        let b = &convex[(i + 1) % convex.len()];
        std::mem::swap(&mut out, &mut input);
        out.clear();
        let Some(mut s) = input.last() else {
            break;
        };
        let mut ds = side(a, b, s);
        for e in &input {
            let de = side(a, b, e);
            if (ds < 0.0) != (de < 0.0) {
                // The arc from s to e crosses the great circle: the crossing is on the chord
                // between them, pushed back out to the sphere.
                let t = ds / (ds - de);
                let p = [
                    s[0] + t * (e[0] - s[0]),
                    s[1] + t * (e[1] - s[1]),
                    s[2] + t * (e[2] - s[2]),
                ];
                let n = norm(&p);
                out.push([p[0] / n, p[1] / n, p[2] / n]);
            }
            if de >= 0.0 {
                out.push(*e);
            }
            (s, ds) = (e, de);
        }
    }
    out
}

/// An axis-aligned box around part of the unit sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    const EMPTY: Self = Self {
        min: [f64::INFINITY; 3],
        max: [f64::NEG_INFINITY; 3],
    };

    fn point(p: &Vec3) -> Self {
        Self { min: *p, max: *p }
    }

    fn union(&self, o: &Self) -> Self {
        Self {
            min: std::array::from_fn(|i| self.min[i].min(o.min[i])),
            max: std::array::from_fn(|i| self.max[i].max(o.max[i])),
        }
    }

    fn meets(&self, o: &Self) -> bool {
        (0..3).all(|i| self.min[i] <= o.max[i] && o.min[i] <= self.max[i])
    }

    fn center(&self, axis: usize) -> f64 {
        (self.min[axis] + self.max[axis]) / 2.0
    }
}

/// Static bounding-box tree: like the k-d tree in `mesh`, each range of `order` is split at its
/// median on axis `depth % 3`, and `ranges[mid]` holds the box around that whole range.
#[derive(Debug, Clone)]
pub(crate) struct BoxTree {
    pub(crate) boxes: Vec<Aabb>,
    order: Vec<u32>,
    ranges: Vec<Aabb>,
}

impl BoxTree {
    /// Indexes boxes; empty ones are left out.
    fn new(boxes: Vec<Aabb>) -> Self {
        let mut order: Vec<u32> = (0..boxes.len() as u32)
            .filter(|&i| boxes[i as usize] != Aabb::EMPTY)
            .collect();
        build(&boxes, &mut order, 0);
        let mut ranges = vec![Aabb::EMPTY; order.len()];
        fill(&boxes, &order, &mut ranges);
        Self {
            boxes,
            order,
            ranges,
        }
    }

    /// Calls `f` with every item whose box meets `q`.
    pub(crate) fn query(&self, q: &Aabb, f: &mut impl FnMut(usize)) {
        self.search(0, self.order.len(), q, f);
    }

    fn search(&self, lo: usize, hi: usize, q: &Aabb, f: &mut impl FnMut(usize)) {
        if lo >= hi {
            return;
        }
        let mid = lo + (hi - lo) / 2;
        if !self.ranges[mid].meets(q) {
            return;
        }
        let item = self.order[mid] as usize;
        if self.boxes[item].meets(q) {
            f(item);
        }
        self.search(lo, mid, q, f);
        self.search(mid + 1, hi, q, f);
    }
}

fn build(boxes: &[Aabb], order: &mut [u32], depth: usize) {
    if order.len() <= 1 {
        return;
    }
    let axis = depth % 3;
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |a, b| {
        boxes[*a as usize]
            .center(axis)
            .total_cmp(&boxes[*b as usize].center(axis))
    });
    let (left, right) = order.split_at_mut(mid);
    build(boxes, left, depth + 1);
    build(boxes, &mut right[1..], depth + 1);
}

fn fill(boxes: &[Aabb], order: &[u32], ranges: &mut [Aabb]) -> Aabb {
    if order.is_empty() {
        return Aabb::EMPTY;
    }
    let mid = order.len() / 2;
    let (left, right) = ranges.split_at_mut(mid);
    let b = boxes[order[mid] as usize]
        .union(&fill(boxes, &order[..mid], left))
        .union(&fill(boxes, &order[mid + 1..], &mut right[1..]));
    ranges[mid] = b;
    b
}

/// Cells as convex spherical polygons ("parts"), counterclockwise seen from outside the sphere.
#[derive(Debug, Clone)]
pub(crate) struct Cells {
    verts: Vec<Vec3>,
    /// Each part's first vertex, and one past the last part's.
    parts: Vec<u32>,
    /// Each cell's first part, and one past the last cell's.
    cells: Vec<u32>,
}

impl Cells {
    fn new() -> Self {
        Self {
            verts: Vec::new(),
            parts: vec![0],
            cells: vec![0],
        }
    }

    fn push_part(&mut self, v: &[Vec3]) {
        self.verts.extend_from_slice(v);
        self.parts.push(self.verts.len() as u32);
    }

    /// Adds a cell from its corners in order (either direction), or with no geometry.
    fn push(&mut self, corners: Option<&[Vec3]>) {
        let mut v: Vec<Vec3> = Vec::with_capacity(4);
        for c in corners.unwrap_or_default() {
            if v.last().is_none_or(|l| norm(&sub(l, c)) > SAME) {
                v.push(*c);
            }
        }
        while v.len() > 1 && norm(&sub(&v[0], &v[v.len() - 1])) <= SAME {
            v.pop();
        }
        let area = polygon_area(&v);
        if v.len() >= 3 && area != 0.0 {
            if area < 0.0 {
                v.reverse();
            }
            let n = v.len();
            // A corner turns right (is reflex) when the turn's sine is clearly negative.
            let reflex: Vec<usize> = (0..n)
                .filter(|&i| {
                    let (a, b, c) = (&v[(i + n - 1) % n], &v[i], &v[(i + 1) % n]);
                    let (ba, bc) = (sub(a, b), sub(c, b));
                    dot(b, &cross(&bc, &ba)) < -1e-12 * norm(&ba) * norm(&bc)
                })
                .collect();
            match reflex[..] {
                [] => self.push_part(&v),
                [i] if n == 4 => {
                    let at = |k: usize| v[(i + k) % 4];
                    self.push_part(&[at(0), at(1), at(2)]);
                    self.push_part(&[at(0), at(2), at(3)]);
                }
                // Twisted: no geometry.
                _ => {}
            }
        }
        self.cells.push(self.parts.len() as u32 - 1);
    }

    pub(crate) fn len(&self) -> usize {
        self.cells.len() - 1
    }

    pub(crate) fn parts(&self, cell: usize) -> impl Iterator<Item = &[Vec3]> {
        (self.cells[cell]..self.cells[cell + 1]).map(|p| {
            &self.verts[self.parts[p as usize] as usize..self.parts[p as usize + 1] as usize]
        })
    }

    /// Area in steradians; NaN for a cell with no geometry.
    pub(crate) fn area(&self, cell: usize) -> f64 {
        if self.cells[cell] == self.cells[cell + 1] {
            return f64::NAN;
        }
        self.parts(cell).map(polygon_area).sum()
    }

    /// A box around the cell: around the spherical cap, centred on the mean of its corners, that
    /// holds every corner. A convex cell smaller than a hemisphere lies inside that cap.
    fn bounds(&self, cell: usize) -> Aabb {
        let verts: Vec<&Vec3> = self.parts(cell).flatten().collect();
        if verts.is_empty() {
            return Aabb::EMPTY;
        }
        let s = verts
            .iter()
            .fold([0.0; 3], |s, v| [s[0] + v[0], s[1] + v[1], s[2] + v[2]]);
        let len = norm(&s);
        if len < 1e-6 * verts.len() as f64 {
            return Aabb {
                min: [-1.0; 3],
                max: [1.0; 3],
            };
        }
        let c = [s[0] / len, s[1] / len, s[2] / len];
        let r = verts.iter().map(|v| norm(&sub(v, &c))).fold(0.0, f64::max) * (1.0 + 1e-9) + 1e-12;
        Aabb {
            min: c.map(|x| x - r),
            max: c.map(|x| x + r),
        }
    }

    /// Whether the cell holds `p`, edges included.
    fn contains(&self, cell: usize, p: &Vec3) -> bool {
        self.parts(cell)
            .any(|v| (0..v.len()).all(|i| side(&v[i], &v[(i + 1) % v.len()], p) >= 0.0))
    }
}

/// A curvilinear grid: `nx` × `ny` cells, row-major with `y` as the slow axis like
/// [`Grid`](crate::regular::Grid), so cell `iy * nx + ix` lies between corners `(ix, iy)` and
/// `(ix + 1, iy + 1)`.
#[derive(Debug, Clone)]
pub struct Curvilinear {
    nx: usize,
    ny: usize,
    pub(crate) cells: Cells,
    pub(crate) index: BoxTree,
}

impl Curvilinear {
    /// A grid from the longitude and latitude of its corners, `(ny + 1) × (nx + 1)` of each in
    /// row-major order, as in ESMF's corner stagger or CF's 2-D bounds collapsed to vertices.
    pub fn from_corners(nx: usize, ny: usize, lon: &[f64], lat: &[f64]) -> Result<Self, CrsError> {
        let n = (nx + 1) * (ny + 1);
        if lon.len() != n || lat.len() != n {
            return Err(CrsError(format!(
                "a {nx} × {ny} grid needs {n} corners, not {} and {}",
                lon.len(),
                lat.len()
            )));
        }
        let at = |ix: usize, iy: usize| iy * (nx + 1) + ix;
        Ok(Self::build(nx, ny, |ix, iy| {
            [
                at(ix, iy),
                at(ix + 1, iy),
                at(ix + 1, iy + 1),
                at(ix, iy + 1),
            ]
            .map(|k| (lon[k], lat[k]))
        }))
    }

    /// A grid from CF cell bounds: four vertices per cell, `ny × nx × 4` of each in row-major
    /// order, going around the cell (CF asks for counterclockwise; either direction works).
    pub fn from_bounds(nx: usize, ny: usize, lon: &[f64], lat: &[f64]) -> Result<Self, CrsError> {
        let n = nx * ny * 4;
        if lon.len() != n || lat.len() != n {
            return Err(CrsError(format!(
                "a {nx} × {ny} grid needs {n} bounds, not {} and {}",
                lon.len(),
                lat.len()
            )));
        }
        Ok(Self::build(nx, ny, |ix, iy| {
            let k = (iy * nx + ix) * 4;
            std::array::from_fn(|i| (lon[k + i], lat[k + i]))
        }))
    }

    fn build(nx: usize, ny: usize, corners: impl Fn(usize, usize) -> [(f64, f64); 4]) -> Self {
        let mut cells = Cells::new();
        for iy in 0..ny {
            for ix in 0..nx {
                let c = corners(ix, iy);
                let finite = c.iter().all(|(x, y)| x.is_finite() && y.is_finite());
                let v = c.map(|(x, y)| unit(x, y));
                cells.push(finite.then_some(&v[..]));
            }
        }
        let boxes = (0..cells.len()).map(|c| cells.bounds(c)).collect();
        Self {
            nx,
            ny,
            cells,
            index: BoxTree::new(boxes),
        }
    }

    pub fn nx(&self) -> usize {
        self.nx
    }

    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Cell areas in square meters on a sphere of [`EARTH_RADIUS`]; NaN for cells with no
    /// geometry.
    pub fn areas(&self) -> Vec<f64> {
        (0..self.cells.len())
            .map(|c| self.cells.area(c) * EARTH_RADIUS * EARTH_RADIUS)
            .collect()
    }

    /// The cell containing a point given in degrees, or `None` outside the grid. A point on an
    /// edge shared by two cells goes to the lower cell index.
    pub fn locate(&self, lon: f64, lat: f64) -> Option<usize> {
        if !lon.is_finite() || !lat.is_finite() {
            return None;
        }
        let p = unit(lon, lat);
        let mut best = None;
        self.index.query(&Aabb::point(&p), &mut |c| {
            if best.is_none_or(|b| c < b) && self.cells.contains(c, &p) {
                best = Some(c);
            }
        });
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(lon: &[f64], lat: &[f64]) -> Curvilinear {
        Curvilinear::from_corners(1, 1, lon, lat).unwrap()
    }

    #[test]
    fn concave_twisted_and_polar_cells() {
        // Corners in either direction give the same cell.
        let a = grid(&[0.0, 1.0, 0.0, 1.0], &[0.0, 0.0, 1.0, 1.0]);
        let b =
            Curvilinear::from_bounds(1, 1, &[0.0, 0.0, 1.0, 1.0], &[0.0, 1.0, 1.0, 0.0]).unwrap();
        assert!((a.areas()[0] / b.areas()[0] - 1.0).abs() < 1e-12);
        assert_eq!(a.locate(0.5, 0.5), Some(0));
        assert_eq!(a.locate(1.5, 0.5), None);

        // A concave (arrowhead) cell: the notch isn't inside, the rest is.
        let c =
            Curvilinear::from_bounds(1, 1, &[0.0, 2.0, 1.0, 2.0], &[0.0, -1.0, 0.0, 1.0]).unwrap();
        assert_eq!(c.cells.parts(0).count(), 2);
        assert_eq!(c.locate(0.5, 0.0), Some(0));
        assert_eq!(c.locate(1.5, 0.0), None);
        assert_eq!(c.locate(1.9, 0.92), Some(0));
        assert_eq!(c.locate(1.9, 0.5), None);

        // A twisted cell, and one with a missing corner, have no geometry.
        let t =
            Curvilinear::from_bounds(1, 1, &[0.0, 1.0, 0.0, 1.0], &[0.0, 1.0, 1.0, 0.0]).unwrap();
        assert!(t.areas()[0].is_nan() && t.locate(0.5, 0.5).is_none());
        let m = Curvilinear::from_bounds(1, 1, &[0.0, 1.0, f64::NAN, 0.0], &[0.0, 0.0, 1.0, 1.0])
            .unwrap();
        assert!(m.areas()[0].is_nan());

        // Corners repeated at the pole make a triangle.
        let p = Curvilinear::from_bounds(1, 1, &[0.0, 90.0, 90.0, 0.0], &[80.0, 80.0, 90.0, 90.0])
            .unwrap();
        assert_eq!(p.cells.parts(0).next().unwrap().len(), 3);
        assert_eq!(p.locate(45.0, 89.0), Some(0));
        assert_eq!(p.locate(30.0, 89.999), Some(0));
        assert_eq!(p.locate(120.0, 89.0), None);
    }

    #[test]
    fn neighbours_split_at_their_shared_edge() {
        let g = Curvilinear::from_corners(
            2,
            1,
            &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(g.locate(1.0 - 1e-9, 0.5), Some(0));
        assert_eq!(g.locate(1.0 + 1e-9, 0.5), Some(1));
        assert_eq!(g.locate(2.0 + 1e-9, 0.5), None);
    }

    #[test]
    fn clipping_matches_known_areas() {
        let sq = |x0: f64, y0: f64, d: f64| {
            [(x0, y0), (x0 + d, y0), (x0 + d, y0 + d), (x0, y0 + d)].map(|(x, y)| unit(x, y))
        };
        let a = sq(0.0, 0.0, 2.0);
        // Two cells that share a meridian split `a` in two.
        let quad = |x0: f64, x1: f64| {
            [(x0, -5.0), (x1, -5.0), (x1, 5.0), (x0, 5.0)].map(|(x, y)| unit(x, y))
        };
        let left = polygon_area(&clip(&a, &quad(-5.0, 1.0)));
        let right = polygon_area(&clip(&a, &quad(1.0, 7.0)));
        assert!((left / right - 1.0).abs() < 1e-9, "symmetric about lon 1");
        assert!(((left + right) / polygon_area(&a) - 1.0).abs() < 1e-12);
        // Overlap is symmetric, and a disjoint cell clips to nothing.
        let b = sq(1.0, 1.0, 2.0);
        assert!((polygon_area(&clip(&a, &b)) / polygon_area(&clip(&b, &a)) - 1.0).abs() < 1e-12);
        assert!(clip(&a, &sq(5.0, 5.0, 1.0)).len() < 3);
        // Clipping to itself keeps the whole cell.
        assert!((polygon_area(&clip(&a, &a)) / polygon_area(&a) - 1.0).abs() < 1e-12);
    }
}
