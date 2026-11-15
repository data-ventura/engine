//! GeoTIFF rasters (striped or tiled, so Cloud Optimized GeoTIFFs too) as arrays.
//!
//! A raster opens as an [`ArraySource`] with one variable per band (`band_1`, …) on dimensions
//! `y` and `x`, plus `x` and `y` coordinate variables at pixel centers in the raster's CRS. The
//! CRS (`EPSG:…`) is the `crs` attribute; GDAL's nodata value is each band's `_FillValue`, so
//! [`read`](crate::array::read) masks it. Only the full-resolution image is read; overviews
//! and range requests over HTTP come later.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

use crate::array::{ArraySource, AttrValue, Attrs, Variable};
use crate::{Error, Result};

#[derive(Debug)]
pub struct GeoTiff {
    vars: Vec<Variable>,
    /// Band-major, each row-major.
    bands: Vec<Vec<f64>>,
    x: Vec<f64>,
    y: Vec<f64>,
    attrs: Attrs,
}

fn err(e: tiff::TiffError) -> Error {
    Error::Unsupported(format!("GeoTIFF: {e}"))
}

fn to_f64(r: DecodingResult) -> Result<Vec<f64>> {
    use DecodingResult::*;
    Ok(match r {
        U8(v) => v.into_iter().map(f64::from).collect(),
        U16(v) => v.into_iter().map(f64::from).collect(),
        U32(v) => v.into_iter().map(f64::from).collect(),
        U64(v) => v.into_iter().map(|x| x as f64).collect(),
        I8(v) => v.into_iter().map(f64::from).collect(),
        I16(v) => v.into_iter().map(f64::from).collect(),
        I32(v) => v.into_iter().map(f64::from).collect(),
        I64(v) => v.into_iter().map(|x| x as f64).collect(),
        F32(v) => v.into_iter().map(f64::from).collect(),
        F64(v) => v,
        F16(v) => v.into_iter().map(f64::from).collect(),
    })
}

/// GeoKey IDs used here.
const RASTER_TYPE: u16 = 1025;
const GEOGRAPHIC_TYPE: u16 = 2048;
const PROJECTED_TYPE: u16 = 3072;
const PIXEL_IS_POINT: u16 = 2;

impl GeoTiff {
    pub fn open(path: &Path) -> Result<Self> {
        let mut d = Decoder::new(BufReader::new(File::open(path)?)).map_err(err)?;
        let (w, h) = d.dimensions().map_err(err)?;
        let (w, h) = (w as usize, h as usize);
        let samples = d
            .find_tag_unsigned::<u16>(Tag::SamplesPerPixel)
            .map_err(err)?
            .unwrap_or(1) as usize;
        let planar = d
            .find_tag_unsigned::<u16>(Tag::PlanarConfiguration)
            .map_err(err)?
            .unwrap_or(1);

        // Pixel-to-model transform: x = x0 + col * dx, y = y0 + row * dy, at pixel corners.
        let (x0, dx, y0, dy) = if let Ok(m) = d.get_tag_f64_vec(Tag::ModelTransformationTag) {
            if m.len() < 8 || m[1] != 0.0 || m[4] != 0.0 {
                return Err(Error::Unsupported(
                    "rotated GeoTIFFs aren't supported".into(),
                ));
            }
            (m[3], m[0], m[7], m[5])
        } else {
            let scale = d.get_tag_f64_vec(Tag::ModelPixelScaleTag).map_err(err)?;
            let tie = d.get_tag_f64_vec(Tag::ModelTiepointTag).map_err(err)?;
            if scale.len() < 2 || tie.len() < 6 {
                return Err(Error::Unsupported(
                    "GeoTIFF georeferencing is incomplete".into(),
                ));
            }
            // The tie point maps raster (i, j) to model (x, y).
            (
                tie[3] - tie[0] * scale[0],
                scale[0],
                tie[4] + tie[1] * scale[1],
                -scale[1],
            )
        };

        let keys = d
            .get_tag_u16_vec(Tag::GeoKeyDirectoryTag)
            .unwrap_or_default();
        let key = |id: u16| {
            keys.chunks(4)
                .skip(1)
                .find(|k| k.len() == 4 && k[0] == id && k[1] == 0)
                .map(|k| k[3])
        };
        // Pixel-is-point rasters put coordinates on pixel centers already; GDAL shifts them.
        let half = if key(RASTER_TYPE) == Some(PIXEL_IS_POINT) {
            0.0
        } else {
            0.5
        };
        let crs = key(PROJECTED_TYPE)
            .or_else(|| key(GEOGRAPHIC_TYPE))
            .filter(|c| *c != 0 && *c != 32767)
            .map(|c| format!("EPSG:{c}"));
        let nodata = d
            .get_tag_ascii_string(Tag::GdalNodata)
            .ok()
            .and_then(|s| s.trim().trim_end_matches('\0').parse::<f64>().ok());

        let raw = to_f64(d.read_image().map_err(err)?)?;
        let bands: Vec<Vec<f64>> = if samples == 1 {
            vec![raw]
        } else if planar == 2 {
            raw.chunks(w * h).map(<[f64]>::to_vec).collect()
        } else {
            (0..samples)
                .map(|b| raw.iter().skip(b).step_by(samples).copied().collect())
                .collect()
        };

        let x: Vec<f64> = (0..w).map(|i| x0 + dx * (i as f64 + half)).collect();
        let y: Vec<f64> = (0..h).map(|j| y0 + dy * (j as f64 + half)).collect();
        let mut attrs = Attrs::new();
        if let Some(c) = &crs {
            attrs.insert("crs".into(), AttrValue::Text(c.clone()));
        }
        let mut vars: Vec<Variable> = (0..bands.len())
            .map(|b| {
                let mut a = Attrs::new();
                if let Some(n) = nodata {
                    a.insert("_FillValue".into(), AttrValue::Numbers(vec![n]));
                }
                if let Some(c) = &crs {
                    a.insert("crs".into(), AttrValue::Text(c.clone()));
                }
                Variable {
                    name: format!("band_{}", b + 1),
                    dims: vec!["y".into(), "x".into()],
                    shape: vec![h as u64, w as u64],
                    dtype: "f8".into(),
                    attrs: a,
                }
            })
            .collect();
        for (name, len) in [("x", w), ("y", h)] {
            vars.push(Variable {
                name: name.into(),
                dims: vec![name.into()],
                shape: vec![len as u64],
                dtype: "f8".into(),
                attrs: Attrs::new(),
            });
        }
        Ok(Self {
            vars,
            bands,
            x,
            y,
            attrs,
        })
    }
}

impl ArraySource for GeoTiff {
    fn variables(&self) -> &[Variable] {
        &self.vars
    }

    fn attrs(&self) -> &Attrs {
        &self.attrs
    }

    fn read_raw(&self, var: &str, start: &[u64], count: &[u64]) -> Result<Vec<f64>> {
        let v = self
            .variable(var)
            .ok_or_else(|| Error::Unsupported(format!("no variable {var:?}")))?;
        let data: &[f64] = match var {
            "x" => &self.x,
            "y" => &self.y,
            _ => {
                let b: usize = var.trim_start_matches("band_").parse().unwrap_or(0);
                &self.bands[b - 1]
            }
        };
        crate::array::slab(data, &v.shape, start, count)
    }
}
