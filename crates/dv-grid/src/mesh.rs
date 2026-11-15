//! Unstructured meshes (ICON, MPAS, HEALPix and the like): nodes found by nearest distance.
//!
//! Nodes are points on the sphere. A query takes the nearest node within a maximum great-circle
//! distance; farther than that, the point is unmatched. Distances use a sphere of the WGS 84
//! mean radius, as unstructured models do.

/// Mean Earth radius (IUGG), in meters.
pub const EARTH_RADIUS: f64 = 6_371_008.8;

fn unit(lon: f64, lat: f64) -> [f64; 3] {
    let (lon, lat) = (lon.to_radians(), lat.to_radians());
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Great-circle distance in meters from the chord between two unit vectors.
fn chord_to_meters(chord2: f64) -> f64 {
    2.0 * (chord2.sqrt() / 2.0).min(1.0).asin() * EARTH_RADIUS
}

/// Mesh nodes, indexed for nearest-node queries (a static k-d tree on unit vectors).
#[derive(Debug, Clone)]
pub struct Nodes {
    points: Vec<[f64; 3]>,
    /// Node indices in k-d tree order: each range's median splits it on axis `depth % 3`.
    order: Vec<u32>,
}

impl Nodes {
    /// Indexes nodes given in degrees. Nodes with a non-finite coordinate are left out.
    pub fn new(lon: &[f64], lat: &[f64]) -> Self {
        assert_eq!(
            lon.len(),
            lat.len(),
            "longitudes and latitudes differ in length"
        );
        let points: Vec<[f64; 3]> = lon.iter().zip(lat).map(|(x, y)| unit(*x, *y)).collect();
        let mut order: Vec<u32> = (0..points.len() as u32)
            .filter(|&i| points[i as usize].iter().all(|c| c.is_finite()))
            .collect();
        build(&points, &mut order, 0);
        Self { points, order }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The nearest node to a point and its distance in meters, if within `max_meters`.
    pub fn nearest(&self, lon: f64, lat: f64, max_meters: f64) -> Option<(usize, f64)> {
        if !lon.is_finite() || !lat.is_finite() {
            return None;
        }
        let q = unit(lon, lat);
        let mut best = (u32::MAX, f64::INFINITY);
        search(&self.points, &self.order, 0, &q, &mut best);
        let meters = chord_to_meters(best.1);
        (best.0 != u32::MAX && meters <= max_meters).then_some((best.0 as usize, meters))
    }
}

fn build(points: &[[f64; 3]], order: &mut [u32], depth: usize) {
    if order.len() <= 1 {
        return;
    }
    let axis = depth % 3;
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |a, b| {
        points[*a as usize][axis].total_cmp(&points[*b as usize][axis])
    });
    let (left, right) = order.split_at_mut(mid);
    build(points, left, depth + 1);
    build(points, &mut right[1..], depth + 1);
}

fn search(points: &[[f64; 3]], order: &[u32], depth: usize, q: &[f64; 3], best: &mut (u32, f64)) {
    if order.is_empty() {
        return;
    }
    let mid = order.len() / 2;
    let node = order[mid];
    let p = &points[node as usize];
    let d = dist2(p, q);
    // Ties go to the lower node index, so results don't depend on tree layout.
    if d < best.1 || (d == best.1 && node < best.0) {
        *best = (node, d);
    }
    let axis = depth % 3;
    let diff = q[axis] - p[axis];
    let (near, far) = if diff < 0.0 {
        (&order[..mid], &order[mid + 1..])
    } else {
        (&order[mid + 1..], &order[..mid])
    };
    search(points, near, depth + 1, q, best);
    if diff * diff <= best.1 {
        search(points, far, depth + 1, q, best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_brute_force() {
        let mut lon = Vec::new();
        let mut lat = Vec::new();
        let mut s = 12345u64;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..2000 {
            lon.push(rnd() * 360.0 - 180.0);
            lat.push((rnd() * 2.0 - 1.0).asin().to_degrees());
        }
        let nodes = Nodes::new(&lon, &lat);
        for _ in 0..500 {
            let (x, y) = (rnd() * 360.0 - 180.0, rnd() * 180.0 - 90.0);
            let q = unit(x, y);
            let brute = (0..lon.len())
                .min_by(|a, b| {
                    dist2(&unit(lon[*a], lat[*a]), &q)
                        .total_cmp(&dist2(&unit(lon[*b], lat[*b]), &q))
                })
                .unwrap();
            assert_eq!(nodes.nearest(x, y, f64::INFINITY).unwrap().0, brute);
        }
    }
}
