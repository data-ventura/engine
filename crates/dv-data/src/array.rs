//! N-dimensional arrays from files: NumPy `.npy`/`.npz`, GeoTIFF rasters, and NetCDF 3/4 and
//! HDF5 (with the `netcdf` feature, through the netCDF-C library).
//!
//! Every format opens as an [`ArraySource`]: named variables on named dimensions, with their
//! attributes. [`read`] returns a hyperslab as `f64`, decoded as xarray decodes CF data: values
//! equal to `_FillValue` or `missing_value` become NaN, `_Unsigned` integers are read as
//! unsigned, and then `scale_factor` and `add_offset` apply. [`read_raw`](ArraySource::read_raw)
//! skips the decoding. Times stay numbers here; `dv_model::time` decodes them by calendar.

use std::collections::BTreeMap;
use std::path::Path;

use crate::{Error, Result};

/// An attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Text(String),
    Numbers(Vec<f64>),
}

impl AttrValue {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            AttrValue::Text(s) => Some(s),
            AttrValue::Numbers(_) => None,
        }
    }

    pub fn as_numbers(&self) -> Option<&[f64]> {
        match self {
            AttrValue::Numbers(v) => Some(v),
            AttrValue::Text(_) => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        self.as_numbers().and_then(|v| v.first().copied())
    }
}

pub type Attrs = BTreeMap<String, AttrValue>;

/// The element type of a variable as stored, in NumPy's notation (`i2`, `u1`, `f4`, `f8`…).
pub type DType = String;

/// A variable: an array on named dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: String,
    pub dims: Vec<String>,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub attrs: Attrs,
}

impl Variable {
    pub fn len(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attrs.get(name)
    }
}

/// An opened array file.
pub trait ArraySource: Send + Sync + std::fmt::Debug {
    fn variables(&self) -> &[Variable];

    /// Attributes of the file itself.
    fn attrs(&self) -> &Attrs;

    /// Reads `count` elements from `start` along each dimension, row-major, without CF decoding.
    fn read_raw(&self, var: &str, start: &[u64], count: &[u64]) -> Result<Vec<f64>>;

    fn variable(&self, name: &str) -> Option<&Variable> {
        self.variables().iter().find(|v| v.name == name)
    }
}

/// Opens an array file by its extension: `.npy`, `.npz`, `.tif` (GeoTIFF), or (with the `netcdf` feature) `.nc`,
/// `.nc4`, `.cdf`, `.h5`, `.hdf5`, `.he5`, `.h5ad`.
pub fn open(path: impl AsRef<Path>) -> Result<Box<dyn ArraySource>> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "npy" => Ok(Box::new(npy::Npy::open(path)?)),
        "npz" => Ok(Box::new(npy::Npy::open_npz(path)?)),
        "tif" | "tiff" => Ok(Box::new(crate::raster::GeoTiff::open(path)?)),
        "nc" | "nc4" | "cdf" | "netcdf" | "h5" | "hdf5" | "he5" | "h5ad" => open_netcdf(path),
        _ => Err(Error::Unsupported(format!(
            "{path:?} isn't a known array format"
        ))),
    }
}

#[cfg(feature = "netcdf")]
fn open_netcdf(path: &Path) -> Result<Box<dyn ArraySource>> {
    Ok(Box::new(netcdf::NetCdf::open(path)?))
}

#[cfg(not(feature = "netcdf"))]
fn open_netcdf(path: &Path) -> Result<Box<dyn ArraySource>> {
    Err(Error::Unsupported(format!(
        "{path:?} needs NetCDF/HDF5 support, which this build doesn't include (the `netcdf` feature)"
    )))
}

/// Reads a hyperslab and decodes it as CF and xarray do. See the module docs.
pub fn read(src: &dyn ArraySource, var: &str, start: &[u64], count: &[u64]) -> Result<Vec<f64>> {
    let v = src
        .variable(var)
        .ok_or_else(|| Error::Unsupported(format!("no variable {var:?}")))?;
    let mut data = src.read_raw(var, start, count)?;
    decode(&mut data, v);
    Ok(data)
}

/// Reads a whole variable, decoded.
pub fn read_all(src: &dyn ArraySource, var: &str) -> Result<Vec<f64>> {
    let v = src
        .variable(var)
        .ok_or_else(|| Error::Unsupported(format!("no variable {var:?}")))?;
    let start = vec![0; v.shape.len()];
    let count = v.shape.clone();
    read(src, var, &start, &count)
}

/// Applies CF decoding in place: `_Unsigned`, then the fill and missing values, then scale and
/// offset.
pub fn decode(data: &mut [f64], v: &Variable) {
    let unsigned_bits = match (
        v.attr("_Unsigned").and_then(AttrValue::as_text),
        v.dtype.as_str(),
    ) {
        (Some(u), "i1" | "i2" | "i4" | "i8") if u.eq_ignore_ascii_case("true") => {
            Some(v.dtype[1..].parse::<i32>().unwrap_or(8) * 8)
        }
        _ => None,
    };
    let as_unsigned = |x: f64| match unsigned_bits {
        Some(bits) if x < 0.0 => x + 2f64.powi(bits),
        _ => x,
    };
    let mut missing: Vec<f64> = Vec::new();
    for key in ["_FillValue", "missing_value"] {
        if let Some(vals) = v.attr(key).and_then(AttrValue::as_numbers) {
            missing.extend(vals.iter().map(|x| as_unsigned(*x)));
        }
    }
    let scale = v
        .attr("scale_factor")
        .and_then(AttrValue::as_f64)
        .unwrap_or(1.0);
    let offset = v
        .attr("add_offset")
        .and_then(AttrValue::as_f64)
        .unwrap_or(0.0);
    for x in data.iter_mut() {
        let raw = as_unsigned(*x);
        *x = if missing
            .iter()
            .any(|m| *m == raw || (m.is_nan() && raw.is_nan()))
        {
            f64::NAN
        } else if scale != 1.0 || offset != 0.0 {
            raw * scale + offset
        } else {
            raw
        };
    }
}

/// Copies a hyperslab out of a whole row-major array.
pub(crate) fn slab(data: &[f64], shape: &[u64], start: &[u64], count: &[u64]) -> Result<Vec<f64>> {
    if start.len() != shape.len() || count.len() != shape.len() {
        return Err(Error::Unsupported(
            "start and count need one entry per dimension".into(),
        ));
    }
    for d in 0..shape.len() {
        if start[d] + count[d] > shape[d] {
            return Err(Error::Unsupported(format!(
                "dimension {d}: {}..{} is outside 0..{}",
                start[d],
                start[d] + count[d],
                shape[d]
            )));
        }
    }
    let total: u64 = count.iter().product();
    let mut out = Vec::with_capacity(total as usize);
    let mut idx = vec![0u64; shape.len()];
    for _ in 0..total {
        let mut flat = 0u64;
        for d in 0..shape.len() {
            flat = flat * shape[d] + start[d] + idx[d];
        }
        out.push(data[flat as usize]);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < count[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(out)
}

mod npy {
    //! NumPy `.npy` and `.npz` files, read whole: they have no chunks to read part of.

    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    use npyz::{DType, NpyFile, Order, TypeChar};

    use super::{Attrs, Variable, slab};
    use crate::{Error, Result};

    #[derive(Debug)]
    pub struct Npy {
        vars: Vec<Variable>,
        data: Vec<Vec<f64>>,
        attrs: Attrs,
    }

    fn to_f64<R: std::io::Read>(f: NpyFile<R>) -> Result<(Variable, Vec<f64>)> {
        let shape = f.shape().to_vec();
        let order = f.order();
        let DType::Plain(ty) = f.dtype() else {
            return Err(Error::Unsupported(
                "only plain numeric .npy arrays are supported".into(),
            ));
        };
        let (kind, size) = (ty.type_char(), ty.size_field());
        let io = |e: std::io::Error| Error::Io(e);
        let data: Vec<f64> = match (kind, size) {
            (TypeChar::Float, 8) => f.into_vec::<f64>().map_err(io)?,
            (TypeChar::Float, 4) => f
                .into_vec::<f32>()
                .map_err(io)?
                .into_iter()
                .map(f64::from)
                .collect(),
            (TypeChar::Int, 8) => f
                .into_vec::<i64>()
                .map_err(io)?
                .into_iter()
                .map(|x| x as f64)
                .collect(),
            (TypeChar::Int, 4) => f
                .into_vec::<i32>()
                .map_err(io)?
                .into_iter()
                .map(f64::from)
                .collect(),
            (TypeChar::Int, 2) => f
                .into_vec::<i16>()
                .map_err(io)?
                .into_iter()
                .map(f64::from)
                .collect(),
            (TypeChar::Int, 1) => f
                .into_vec::<i8>()
                .map_err(io)?
                .into_iter()
                .map(f64::from)
                .collect(),
            (TypeChar::Uint, 8) => f
                .into_vec::<u64>()
                .map_err(io)?
                .into_iter()
                .map(|x| x as f64)
                .collect(),
            (TypeChar::Uint, 4) => f
                .into_vec::<u32>()
                .map_err(io)?
                .into_iter()
                .map(f64::from)
                .collect(),
            (TypeChar::Uint, 2) => f
                .into_vec::<u16>()
                .map_err(io)?
                .into_iter()
                .map(f64::from)
                .collect(),
            (TypeChar::Uint, 1) => f
                .into_vec::<u8>()
                .map_err(io)?
                .into_iter()
                .map(f64::from)
                .collect(),
            (TypeChar::Bool, 1) => f
                .into_vec::<bool>()
                .map_err(io)?
                .into_iter()
                .map(|b| b as u8 as f64)
                .collect(),
            other => {
                return Err(Error::Unsupported(format!(
                    "unsupported .npy element type {other:?}"
                )));
            }
        };
        let data = if order == Order::Fortran && shape.len() > 1 {
            fortran_to_c(&data, &shape)
        } else {
            data
        };
        let prefix = match kind {
            TypeChar::Float => "f",
            TypeChar::Int => "i",
            TypeChar::Uint => "u",
            _ => "b",
        };
        let var = Variable {
            name: String::new(),
            dims: (0..shape.len()).map(|i| format!("dim_{i}")).collect(),
            shape,
            dtype: format!("{prefix}{size}"),
            attrs: Attrs::new(),
        };
        Ok((var, data))
    }

    fn fortran_to_c(data: &[f64], shape: &[u64]) -> Vec<f64> {
        let n = data.len();
        let mut out = vec![0.0; n];
        let mut idx = vec![0u64; shape.len()];
        for x in out.iter_mut() {
            // Fortran offset: the first index varies fastest.
            let mut f = 0u64;
            for d in (0..shape.len()).rev() {
                f = f * shape[d] + idx[d];
            }
            *x = data[f as usize];
            for d in (0..shape.len()).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    }

    impl Npy {
        /// One array, named after the file.
        pub fn open(path: &Path) -> Result<Self> {
            let f = NpyFile::new(BufReader::new(File::open(path)?))?;
            let (mut var, data) = to_f64(f)?;
            var.name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "array".into());
            Ok(Self {
                vars: vec![var],
                data: vec![data],
                attrs: Attrs::new(),
            })
        }

        /// Every array in the archive, by name. Each has its own dimensions (`<name>_dim_N`),
        /// since `.npz` doesn't say which arrays share one.
        pub fn open_npz(path: &Path) -> Result<Self> {
            let bad = |e: zip::result::ZipError| Error::Unsupported(format!(".npz: {e}"));
            let mut zip = zip::ZipArchive::new(BufReader::new(File::open(path)?)).map_err(bad)?;
            let mut vars = Vec::new();
            let mut data = Vec::new();
            for i in 0..zip.len() {
                let entry = zip.by_index(i).map_err(bad)?;
                // numpy names each member `<array>.npy`.
                let Some(name) = entry
                    .name()
                    .map_err(bad)?
                    .strip_suffix(".npy")
                    .map(str::to_string)
                else {
                    continue;
                };
                let f = NpyFile::new(entry)?;
                let (mut var, d) = to_f64(f)?;
                var.dims = var.dims.iter().map(|x| format!("{name}_{x}")).collect();
                var.name = name;
                vars.push(var);
                data.push(d);
            }
            Ok(Self {
                vars,
                data,
                attrs: Attrs::new(),
            })
        }
    }

    impl super::ArraySource for Npy {
        fn variables(&self) -> &[Variable] {
            &self.vars
        }

        fn attrs(&self) -> &Attrs {
            &self.attrs
        }

        fn read_raw(&self, var: &str, start: &[u64], count: &[u64]) -> Result<Vec<f64>> {
            let i = self
                .vars
                .iter()
                .position(|v| v.name == var)
                .ok_or_else(|| Error::Unsupported(format!("no variable {var:?}")))?;
            slab(&self.data[i], &self.vars[i].shape, start, count)
        }
    }
}

#[cfg(feature = "netcdf")]
mod netcdf {
    //! NetCDF 3 and 4 and HDF5 files through netCDF-C. A plain HDF5 file without dimension
    //! scales gets netCDF-C's `phony_dim_N` dimensions.

    use std::path::Path;
    use std::sync::Mutex;

    use netcdf::AttributeValue;
    use netcdf::types::NcVariableType;

    use super::{AttrValue, Attrs, Variable};
    use crate::{Error, Result};

    #[derive(Debug)]
    pub struct NetCdf {
        // netcdf::File isn't Sync. (The netcdf crate also runs every netCDF-C call under one
        // global lock, since netCDF-C and HDF5 aren't thread-safe.)
        file: Mutex<netcdf::File>,
        vars: Vec<Variable>,
        attrs: Attrs,
    }

    fn err(e: netcdf::Error) -> Error {
        Error::Unsupported(format!("NetCDF: {e}"))
    }

    fn attr_value(v: AttributeValue) -> Option<AttrValue> {
        use AttributeValue::*;
        let nums = |v: Vec<f64>| Some(AttrValue::Numbers(v));
        match v {
            Str(s) => Some(AttrValue::Text(s)),
            Strs(s) => Some(AttrValue::Text(s.join("\n"))),
            Uchar(x) => nums(vec![x.into()]),
            Schar(x) => nums(vec![x.into()]),
            Ushort(x) => nums(vec![x.into()]),
            Short(x) => nums(vec![x.into()]),
            Uint(x) => nums(vec![x.into()]),
            Int(x) => nums(vec![x.into()]),
            Ulonglong(x) => nums(vec![x as f64]),
            Longlong(x) => nums(vec![x as f64]),
            Float(x) => nums(vec![x.into()]),
            Double(x) => nums(vec![x]),
            Uchars(x) => nums(x.into_iter().map(f64::from).collect()),
            Schars(x) => nums(x.into_iter().map(f64::from).collect()),
            Ushorts(x) => nums(x.into_iter().map(f64::from).collect()),
            Shorts(x) => nums(x.into_iter().map(f64::from).collect()),
            Uints(x) => nums(x.into_iter().map(f64::from).collect()),
            Ints(x) => nums(x.into_iter().map(f64::from).collect()),
            Ulonglongs(x) => nums(x.into_iter().map(|v| v as f64).collect()),
            Longlongs(x) => nums(x.into_iter().map(|v| v as f64).collect()),
            Floats(x) => nums(x.into_iter().map(f64::from).collect()),
            Doubles(x) => nums(x),
        }
    }

    fn attrs<'a>(it: impl Iterator<Item = netcdf::Attribute<'a>>) -> Attrs {
        it.filter_map(|a| Some((a.name().to_string(), attr_value(a.value().ok()?)?)))
            .collect()
    }

    fn dtype(t: &NcVariableType) -> String {
        use netcdf::types::{FloatType, IntType};
        match t {
            NcVariableType::Int(i) => match i {
                IntType::I8 => "i1",
                IntType::U8 => "u1",
                IntType::I16 => "i2",
                IntType::U16 => "u2",
                IntType::I32 => "i4",
                IntType::U32 => "u4",
                IntType::I64 => "i8",
                IntType::U64 => "u8",
            }
            .into(),
            NcVariableType::Float(FloatType::F32) => "f4".into(),
            NcVariableType::Float(FloatType::F64) => "f8".into(),
            NcVariableType::Char => "S1".into(),
            NcVariableType::String => "str".into(),
            other => format!("{other:?}"),
        }
    }

    impl NetCdf {
        pub fn open(path: &Path) -> Result<Self> {
            let file = netcdf::open(path).map_err(err)?;
            let vars = file
                .variables()
                .map(|v| Variable {
                    name: v.name(),
                    dims: v.dimensions().iter().map(|d| d.name()).collect(),
                    shape: v.dimensions().iter().map(|d| d.len() as u64).collect(),
                    dtype: dtype(&v.vartype()),
                    attrs: attrs(v.attributes()),
                })
                .collect();
            let attrs = attrs(file.attributes());
            Ok(Self {
                file: Mutex::new(file),
                vars,
                attrs,
            })
        }
    }

    impl super::ArraySource for NetCdf {
        fn variables(&self) -> &[Variable] {
            &self.vars
        }

        fn attrs(&self) -> &Attrs {
            &self.attrs
        }

        fn read_raw(&self, var: &str, start: &[u64], count: &[u64]) -> Result<Vec<f64>> {
            let file = self.file.lock().expect("netCDF file");
            let v = file
                .variable(var)
                .ok_or_else(|| Error::Unsupported(format!("no variable {var:?}")))?;
            if v.dimensions().is_empty() {
                return v.get_values::<f64, _>(()).map_err(err);
            }
            let ranges: Vec<std::ops::Range<usize>> = start
                .iter()
                .zip(count)
                .map(|(s, c)| *s as usize..(*s + *c) as usize)
                .collect();
            v.get_values::<f64, _>(ranges).map_err(err)
        }
    }
}
