//! Spherical Web Mercator (EPSG:3857), the projection of web map tiles.
//!
//! Coordinates are meters on a sphere of the WGS 84 semi-major axis. Latitudes are clamped to the
//! projection's limit of about ±85.0511°, where the map becomes square.

/// WGS 84 semi-major axis, in meters.
pub const RADIUS: f64 = 6_378_137.0;
/// The latitude at which Web Mercator's extent is square, in degrees.
pub const MAX_LATITUDE: f64 = 85.051_128_779_806_59;

/// Projects longitude and latitude in degrees to Web Mercator meters.
pub fn forward(lon: f64, lat: f64) -> (f64, f64) {
    let lat = lat.clamp(-MAX_LATITUDE, MAX_LATITUDE).to_radians();
    let x = RADIUS * lon.to_radians();
    let y = RADIUS * (std::f64::consts::FRAC_PI_4 + lat / 2.0).tan().ln();
    (x, y)
}

/// Converts Web Mercator meters back to longitude and latitude in degrees.
pub fn inverse(x: f64, y: f64) -> (f64, f64) {
    let lon = (x / RADIUS).to_degrees();
    let lat = (2.0 * (y / RADIUS).exp().atan() - std::f64::consts::FRAC_PI_2).to_degrees();
    (lon, lat)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference from PROJ via pyproj: Transformer.from_crs("EPSG:4326", "EPSG:3857",
    // always_xy=True).transform(-73.9857, 40.7484), the Empire State Building.
    #[test]
    fn matches_proj() {
        let (x, y) = forward(-73.9857, 40.7484);
        assert!((x - -8_236_050.45).abs() < 0.01, "{x}");
        assert!((y - 4_975_301.25).abs() < 0.01, "{y}");
    }

    #[test]
    fn round_trips_and_clamps() {
        let (lon, lat) = inverse(forward(151.2093, -33.8688).0, forward(151.2093, -33.8688).1);
        assert!((lon - 151.2093).abs() < 1e-9 && (lat + 33.8688).abs() < 1e-9);
        let (x, y) = forward(180.0, 90.0);
        assert!(
            (x - y).abs() < 1e-6,
            "the extent is square at the latitude limit"
        );
    }
}
