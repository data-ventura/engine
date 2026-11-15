//! Coordinate reference systems and transforms between them.
//!
//! CRSs come from EPSG codes or PROJ strings, through proj4rs, a pure-Rust port of PROJ's
//! projection math: it runs everywhere the engine does, the browser included. It has no datum
//! grid files, so datum shifts are the 3- and 7-parameter Helmert kind; anything needing a grid
//! shift fails rather than silently being approximated.
//!
//! Coordinates are always (x, y) = (easting, northing) or (longitude, latitude) in degrees,
//! whatever the CRS's official axis order, as with pyproj's `always_xy`.

use std::fmt;

use proj4rs::Proj;
use proj4rs::transform::transform;

/// A CRS that can't be read or a transform that can't be made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct CrsError(pub String);

/// A coordinate reference system.
#[derive(Clone)]
pub struct Crs {
    proj: Proj,
    text: String,
}

impl fmt::Debug for Crs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Crs").field(&self.text).finish()
    }
}

impl fmt::Display for Crs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl Crs {
    /// Reads `EPSG:<code>`, a PROJ string (`+proj=…`) or `WGS84`.
    pub fn parse(text: &str) -> Result<Self, CrsError> {
        let proj = Proj::from_user_string(text.trim())
            .map_err(|e| CrsError(format!("can't read the CRS {text:?}: {e}")))?;
        Ok(Self {
            proj,
            text: text.trim().to_string(),
        })
    }

    pub fn epsg(code: u16) -> Result<Self, CrsError> {
        Self::parse(&format!("EPSG:{code}"))
    }

    /// WGS 84 longitude and latitude.
    pub fn wgs84() -> Self {
        Self::epsg(4326).expect("EPSG:4326 is built in")
    }

    /// Whether coordinates are longitude and latitude.
    pub fn is_geographic(&self) -> bool {
        self.proj.is_latlong()
    }
}

/// Transforms coordinates from one CRS to another.
#[derive(Debug, Clone)]
pub struct Transformer {
    src: Crs,
    dst: Crs,
}

impl Transformer {
    pub fn new(src: &Crs, dst: &Crs) -> Self {
        Self {
            src: src.clone(),
            dst: dst.clone(),
        }
    }

    /// Transforms one point. `None` if it's outside where the projection is defined.
    pub fn point(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let (x, y) = if self.src.is_geographic() {
            (x.to_radians(), y.to_radians())
        } else {
            (x, y)
        };
        let mut p = (x, y);
        transform(&self.src.proj, &self.dst.proj, &mut p).ok()?;
        let p = if self.dst.is_geographic() {
            (p.0.to_degrees(), p.1.to_degrees())
        } else {
            p
        };
        (p.0.is_finite() && p.1.is_finite()).then_some(p)
    }

    /// Transforms points in place. Points that fail become NaN.
    pub fn transform(&self, xs: &mut [f64], ys: &mut [f64]) {
        for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
            (*x, *y) = self.point(*x, *y).unwrap_or((f64::NAN, f64::NAN));
        }
    }
}
