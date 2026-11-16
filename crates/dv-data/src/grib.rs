//! GRIB 1 and 2 files through ECMWF's ecCodes library (the `grib` feature), read as cfgrib reads
//! them.
//!
//! Opening a file scans its messages once, reading only their keys, and groups them into
//! variables as cfgrib does: one variable per parameter, level type, step type and grid, named by
//! ecCodes' `cfVarName` (`t2m`, `u10`, `t`…). A variable's messages are laid out on the
//! dimensions `number` (ensemble member), `time` (forecast reference time), `step` (forecast
//! period) and the level type (such as `isobaricInhPa`), then the grid: `latitude` and
//! `longitude` for regular grids, `y` and `x` (with two-dimensional `latitude` and `longitude`)
//! for projected ones, or one `values` dimension for others, such as reduced Gaussian grids. A
//! dimension with one value becomes a scalar coordinate, as cfgrib squeezes it. Times are numbers
//! in cfgrib's units: seconds since 1970 for `time` and `valid_time`, hours for `step`.
//!
//! cfgrib returns each GRIB file as several datasets when its variables disagree on a
//! coordinate. Here they share one namespace, so a coordinate that differs between variables
//! (`heightAboveGround` at 2 m for `t2m` and 10 m for `u10`) gets a numbered copy
//! (`heightAboveGround_2`), and each data variable's `coordinates` attribute names the scalar and
//! auxiliary coordinates it uses.
//!
//! Values are what ecCodes decodes, rounded to `f32` as cfgrib returns them, with NaN where the
//! bitmap says a point is missing. Reads decode only the messages a hyperslab touches.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_void};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::array::{ArraySource, AttrValue, Attrs, Variable, check_slab, slab};
use crate::{Error, Result};

#[allow(non_camel_case_types)]
mod ffi {
    //! The parts of ecCodes' C API (`eccodes.h`) used here.

    use std::ffi::{c_char, c_int, c_long, c_void};

    #[repr(C)]
    pub struct codes_handle {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct codes_context {
        _private: [u8; 0],
    }

    /// `PRODUCT_GRIB` in ecCodes' `ProductKind` enum.
    pub const PRODUCT_GRIB: c_int = 1;
    pub const SUCCESS: c_int = 0;

    unsafe extern "C" {
        pub fn codes_handle_new_from_file(
            c: *mut codes_context,
            f: *mut libc::FILE,
            product: c_int,
            error: *mut c_int,
        ) -> *mut codes_handle;
        pub fn codes_handle_new_from_message(
            c: *mut codes_context,
            data: *const c_void,
            data_len: usize,
        ) -> *mut codes_handle;
        pub fn codes_handle_delete(h: *mut codes_handle) -> c_int;
        pub fn codes_is_defined(h: *const codes_handle, key: *const c_char) -> c_int;
        pub fn codes_get_long(
            h: *const codes_handle,
            key: *const c_char,
            value: *mut c_long,
        ) -> c_int;
        pub fn codes_get_double(
            h: *const codes_handle,
            key: *const c_char,
            value: *mut f64,
        ) -> c_int;
        pub fn codes_get_string(
            h: *const codes_handle,
            key: *const c_char,
            value: *mut c_char,
            length: *mut usize,
        ) -> c_int;
        pub fn codes_get_size(
            h: *const codes_handle,
            key: *const c_char,
            size: *mut usize,
        ) -> c_int;
        pub fn codes_get_double_array(
            h: *const codes_handle,
            key: *const c_char,
            vals: *mut f64,
            length: *mut usize,
        ) -> c_int;
        pub fn codes_set_double(h: *mut codes_handle, key: *const c_char, val: f64) -> c_int;
        pub fn codes_get_message_offset(h: *const codes_handle, offset: *mut libc::off_t) -> c_int;
        pub fn codes_get_message_size(h: *const codes_handle, size: *mut usize) -> c_int;
        pub fn codes_get_error_message(code: c_int) -> *const c_char;
    }
}

/// ecCodes is only thread-safe when built with its threads option, so every call holds this.
static ECCODES: Mutex<()> = Mutex::new(());

/// The missing value cfgrib sets before decoding, so missing points can be told apart from data.
const MISSING: f64 = 3.4028234663852886e38;

fn eccodes_err(code: c_int, what: &str) -> Error {
    // SAFETY: ecCodes returns a static, NUL-terminated message for every code.
    let msg = unsafe { CStr::from_ptr(ffi::codes_get_error_message(code)) };
    Error::Unsupported(format!("GRIB: {what}: {}", msg.to_string_lossy()))
}

fn key(name: &str) -> CString {
    CString::new(name).expect("key names have no NUL")
}

/// A decoded message. Callers hold [`ECCODES`] for its whole life.
struct Handle(*mut ffi::codes_handle);

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the handle came from ecCodes and is deleted once.
        unsafe { ffi::codes_handle_delete(self.0) };
    }
}

impl Handle {
    fn defined(&self, name: &str) -> bool {
        // SAFETY: a live handle and a NUL-terminated key.
        unsafe { ffi::codes_is_defined(self.0, key(name).as_ptr()) != 0 }
    }

    fn long(&self, name: &str) -> Result<i64> {
        let mut v: c_long = 0;
        // SAFETY: a live handle, a NUL-terminated key and a valid out pointer.
        let rc = unsafe { ffi::codes_get_long(self.0, key(name).as_ptr(), &mut v) };
        if rc != ffi::SUCCESS {
            return Err(eccodes_err(rc, name));
        }
        // `c_long` is 32 bits on Windows.
        #[allow(clippy::useless_conversion)]
        Ok(v.into())
    }

    fn double(&self, name: &str) -> Result<f64> {
        let mut v = 0.0;
        // SAFETY: as in `long`.
        let rc = unsafe { ffi::codes_get_double(self.0, key(name).as_ptr(), &mut v) };
        if rc != ffi::SUCCESS {
            return Err(eccodes_err(rc, name));
        }
        Ok(v)
    }

    fn string(&self, name: &str) -> Result<String> {
        let mut buf = vec![0 as c_char; 1024];
        let mut len = buf.len();
        // SAFETY: `len` is the buffer's size; ecCodes writes at most that many bytes, NUL included.
        let rc = unsafe {
            ffi::codes_get_string(self.0, key(name).as_ptr(), buf.as_mut_ptr(), &mut len)
        };
        if rc != ffi::SUCCESS {
            return Err(eccodes_err(rc, name));
        }
        // SAFETY: on success the buffer holds a NUL-terminated string.
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(s.to_string_lossy().into_owned())
    }

    /// A string key, or `unknown` (as ecCodes names unknown parameters) when it isn't defined.
    fn text(&self, name: &str) -> String {
        self.string(name).unwrap_or_else(|_| "unknown".into())
    }

    fn doubles(&self, name: &str) -> Result<Vec<f64>> {
        let k = key(name);
        let mut n = 0usize;
        // SAFETY: as in `long`.
        let rc = unsafe { ffi::codes_get_size(self.0, k.as_ptr(), &mut n) };
        if rc != ffi::SUCCESS {
            return Err(eccodes_err(rc, name));
        }
        let mut v = vec![0.0; n];
        // SAFETY: `n` is the buffer's length.
        let rc = unsafe { ffi::codes_get_double_array(self.0, k.as_ptr(), v.as_mut_ptr(), &mut n) };
        if rc != ffi::SUCCESS {
            return Err(eccodes_err(rc, name));
        }
        v.truncate(n);
        Ok(v)
    }

    /// The decoded values as cfgrib returns them: `f32` precision, NaN where missing.
    fn values(&self) -> Result<Vec<f64>> {
        // SAFETY: a live handle. This changes only the in-memory message, never the file.
        let rc = unsafe { ffi::codes_set_double(self.0, key("missingValue").as_ptr(), MISSING) };
        if rc != ffi::SUCCESS {
            return Err(eccodes_err(rc, "missingValue"));
        }
        let mut v = self.doubles("values")?;
        for x in &mut v {
            *x = if *x == MISSING {
                f64::NAN
            } else {
                f64::from(*x as f32)
            };
        }
        Ok(v)
    }
}

/// Where a message is in the file, and its grid.
#[derive(Debug, Clone)]
struct Message {
    offset: u64,
    len: usize,
    grid: usize,
}

/// The keys that place a message: cfgrib's header coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Header {
    number: Option<f64>,
    time: f64,
    step: f64,
    level: f64,
}

/// A grid, shared by every message whose grid section is the same.
#[derive(Debug, Clone)]
struct Grid {
    md5: String,
    kind: GridKind,
    lats: Vec<f64>,
    lons: Vec<f64>,
    /// Names given to its dimensions and coordinates, once placed.
    names: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GridKind {
    /// `regular_ll` and `regular_gg`: dimension coordinates `latitude` and `longitude`.
    Regular { ny: u64, nx: u64 },
    /// Projected grids: dimensions `y` and `x`, with 2-D `latitude` and `longitude`.
    Projected { ny: u64, nx: u64 },
    /// Anything else: one `values` dimension.
    Points { n: u64 },
}

impl GridKind {
    fn shape(&self) -> Vec<u64> {
        match *self {
            GridKind::Regular { ny, nx } | GridKind::Projected { ny, nx } => vec![ny, nx],
            GridKind::Points { n } => vec![n],
        }
    }
}

/// Grids cfgrib gives `y` and `x` dimensions with two-dimensional coordinates.
const PROJECTED: [&str; 6] = [
    "rotated_ll",
    "rotated_gg",
    "lambert",
    "lambert_azimuthal_equal_area",
    "albers",
    "polar_stereographic",
];

/// The messages of one variable before it's laid out.
#[derive(Debug)]
struct Group {
    name: String,
    type_of_level: String,
    step_type: String,
    grid: usize,
    attrs: Attrs,
    messages: Vec<(Header, usize)>,
}

/// One header axis of a data variable: its distinct values, and whether it's a dimension.
#[derive(Debug, Clone)]
struct Axis {
    values: Vec<f64>,
    dim: bool,
}

#[derive(Debug)]
enum Data {
    /// A coordinate, held in memory.
    Coord(Vec<f64>),
    /// A data variable: the message for each combination of its four header axes (number, time,
    /// step, level), row-major, and its grid.
    Field {
        axes: [Axis; 4],
        cells: Vec<Option<usize>>,
        grid: usize,
    },
}

/// A GRIB file opened as an [`ArraySource`].
#[derive(Debug)]
pub struct Grib {
    path: PathBuf,
    file: Mutex<File>,
    messages: Vec<Message>,
    grids: Vec<Grid>,
    vars: Vec<Variable>,
    data: Vec<Data>,
    attrs: Attrs,
}

fn text(s: impl Into<String>) -> AttrValue {
    AttrValue::Text(s.into())
}

fn num(x: f64) -> AttrValue {
    AttrValue::Numbers(vec![x])
}

/// Seconds since 1970 from GRIB's `dataDate` (YYYYMMDD) and `dataTime` (HHMM).
fn seconds(date: i64, time: i64) -> Result<f64> {
    let d = chrono::NaiveDate::from_ymd_opt(
        (date / 10000) as i32,
        (date / 100 % 100) as u32,
        (date % 100) as u32,
    )
    .and_then(|d| d.and_hms_opt((time / 100) as u32, (time % 100) as u32, 0))
    .ok_or_else(|| Error::Unsupported(format!("GRIB: bad date {date} {time:04}")))?;
    Ok(d.and_utc().timestamp() as f64)
}

/// Seconds in one of GRIB's step units (code table 4.4), as cfgrib converts them.
fn step_unit_seconds(unit: i64) -> Option<f64> {
    Some(match unit {
        0 => 60.0,
        1 => 3600.0,
        2 => 86400.0,
        10 => 10800.0,
        11 => 21600.0,
        12 => 43200.0,
        13 => 1.0,
        14 => 900.0,
        15 => 1800.0,
        _ => return None,
    })
}

/// cfgrib's attributes for the level coordinate of a level type.
fn level_attrs(type_of_level: &str) -> Attrs {
    let mut a = Attrs::new();
    let (units, positive, standard, long) = match type_of_level {
        "isobaricInhPa" => ("hPa", Some("down"), "air_pressure", "pressure"),
        "isobaricInPa" => ("Pa", Some("down"), "air_pressure", "pressure"),
        "heightAboveGround" => ("m", Some("up"), "height", "height above the surface"),
        "depthBelowLand" | "depthBelowLandLayer" => ("m", Some("down"), "depth", "soil depth"),
        "hybrid" => (
            "1",
            Some("down"),
            "atmosphere_hybrid_sigma_pressure_coordinate",
            "hybrid level",
        ),
        other => {
            a.insert("units".into(), text("1"));
            a.insert(
                "long_name".into(),
                text(format!("original GRIB coordinate for key: level({other})")),
            );
            return a;
        }
    };
    a.insert("units".into(), text(units));
    if let Some(p) = positive {
        a.insert("positive".into(), text(p));
    }
    if positive == Some("down") && units != "m" {
        a.insert("stored_direction".into(), text("decreasing"));
    }
    a.insert("standard_name".into(), text(standard));
    a.insert("long_name".into(), text(long));
    a
}

fn time_attrs(standard: &str, long: &str) -> Attrs {
    Attrs::from([
        ("units".into(), text("seconds since 1970-01-01T00:00:00")),
        ("calendar".into(), text("proleptic_gregorian")),
        ("standard_name".into(), text(standard)),
        ("long_name".into(), text(long)),
    ])
}

fn coord_attrs(base: &str) -> Attrs {
    let a = |pairs: &[(&str, &str)]| -> Attrs {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), text(*v)))
            .collect()
    };
    match base {
        "number" => a(&[
            ("units", "1"),
            ("standard_name", "realization"),
            ("long_name", "ensemble member numerical id"),
        ]),
        "time" => time_attrs("forecast_reference_time", "initial time of forecast"),
        "valid_time" => time_attrs("time", "time"),
        "step" => a(&[
            ("units", "hours"),
            ("standard_name", "forecast_period"),
            ("long_name", "time since forecast_reference_time"),
        ]),
        "latitude" => a(&[
            ("units", "degrees_north"),
            ("standard_name", "latitude"),
            ("long_name", "latitude"),
        ]),
        "longitude" => a(&[
            ("units", "degrees_east"),
            ("standard_name", "longitude"),
            ("long_name", "longitude"),
        ]),
        level => level_attrs(level),
    }
}

/// Distinct values, sorted as cfgrib sorts each coordinate: pressure levels from the surface up
/// (decreasing), everything else increasing.
fn distinct(mut v: Vec<f64>, decreasing: bool) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v.dedup();
    if decreasing {
        v.reverse();
    }
    v
}

/// Gives coordinates names, reusing a name when the same coordinate appears again and numbering
/// copies that differ (`step`, `step_2`, …).
#[derive(Default)]
struct Namer {
    taken: Vec<String>,
    /// (base name, dims, values) → name.
    coords: Vec<(String, Vec<String>, Vec<f64>, String)>,
}

impl Namer {
    fn fresh(&mut self, base: &str) -> String {
        let mut name = base.to_string();
        let mut n = 1;
        while self.taken.contains(&name) {
            n += 1;
            name = format!("{base}_{n}");
        }
        self.taken.push(name.clone());
        name
    }

    /// The name for a coordinate, and whether it's new.
    fn coord(&mut self, base: &str, dims: &[String], values: &[f64]) -> (String, bool) {
        let same = |v: &[f64]| {
            v.len() == values.len() && v.iter().zip(values).all(|(a, b)| a.total_cmp(b).is_eq())
        };
        if let Some((.., name)) = self
            .coords
            .iter()
            .find(|(b, d, v, _)| b == base && d == dims && same(v))
        {
            return (name.clone(), false);
        }
        let name = self.fresh(base);
        self.coords
            .push((base.into(), dims.to_vec(), values.to_vec(), name.clone()));
        (name, true)
    }
}

impl Grib {
    /// Opens a GRIB 1 or 2 file, scanning its messages' keys (but not their data).
    pub fn open(path: &Path) -> Result<Self> {
        let _lock = ECCODES.lock().expect("ecCodes lock");
        let cpath = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::Unsupported(format!("{path:?} has a NUL in it")))?;
        // SAFETY: a NUL-terminated path and mode.
        let fp = unsafe { libc::fopen(cpath.as_ptr(), c"rb".as_ptr()) };
        if fp.is_null() {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }
        struct Closer(*mut libc::FILE);
        impl Drop for Closer {
            fn drop(&mut self) {
                // SAFETY: opened above, closed once.
                unsafe { libc::fclose(self.0) };
            }
        }
        let fp = Closer(fp);

        let mut messages = Vec::new();
        let mut grids: Vec<Grid> = Vec::new();
        let mut groups: Vec<Group> = Vec::new();
        let mut index: HashMap<(i64, String, String, String, String), usize> = HashMap::new();
        let mut attrs = Attrs::new();
        loop {
            let mut rc: c_int = 0;
            // SAFETY: an open file; ecCodes reads the next GRIB message from it.
            let h = unsafe {
                ffi::codes_handle_new_from_file(
                    std::ptr::null_mut(),
                    fp.0,
                    ffi::PRODUCT_GRIB,
                    &mut rc,
                )
            };
            if h.is_null() {
                if rc == ffi::SUCCESS {
                    break;
                }
                return Err(eccodes_err(rc, &format!("{path:?}")));
            }
            let h = Handle(h);
            let mut offset: libc::off_t = 0;
            let mut len = 0usize;
            // SAFETY: a live handle and valid out pointers.
            let rc = unsafe { ffi::codes_get_message_offset(h.0, &mut offset) };
            if rc != ffi::SUCCESS {
                return Err(eccodes_err(rc, "message offset"));
            }
            // SAFETY: as above.
            let rc = unsafe { ffi::codes_get_message_size(h.0, &mut len) };
            if rc != ffi::SUCCESS {
                return Err(eccodes_err(rc, "message size"));
            }
            if attrs.is_empty() {
                let centre = h.text("centreDescription");
                attrs.insert("GRIB_edition".into(), num(h.long("edition")? as f64));
                attrs.insert("GRIB_centre".into(), text(h.text("centre")));
                attrs.insert("GRIB_centreDescription".into(), text(centre.clone()));
                attrs.insert(
                    "GRIB_subCentre".into(),
                    num(h.long("subCentre").unwrap_or(0) as f64),
                );
                attrs.insert("Conventions".into(), text("CF-1.7"));
                attrs.insert("institution".into(), text(centre));
            }

            let md5 = h.string("md5GridSection")?;
            let grid = match grids.iter().position(|g| g.md5 == md5) {
                Some(g) => g,
                None => {
                    grids.push(read_grid(&h, md5)?);
                    grids.len() - 1
                }
            };
            let msg = messages.len();
            messages.push(Message {
                offset: offset as u64,
                len,
                grid,
            });

            let param = h.long("paramId")?;
            let type_of_level = h.text("typeOfLevel");
            let step_type = h.text("stepType");
            let short = h.text("shortName");
            let mut name = h.text("cfVarName");
            if name == "unknown" {
                name = if short == "unknown" {
                    format!("p{param}")
                } else {
                    short.clone()
                };
            }
            let step_seconds = step_unit_seconds(h.long("stepUnits")?)
                .ok_or_else(|| Error::Unsupported("GRIB: unknown step units".into()))?;
            let header = Header {
                number: if h.defined("number") {
                    Some(h.long("number").unwrap_or(0) as f64)
                } else {
                    None
                },
                time: seconds(h.long("dataDate")?, h.long("dataTime")?)?,
                step: h.long("endStep")? as f64 * step_seconds / 3600.0,
                level: h.double("level").unwrap_or(0.0),
            };
            let group_key = (
                param,
                name.clone(),
                type_of_level.clone(),
                step_type.clone(),
                grids[grid].md5.clone(),
            );
            let g = *index.entry(group_key).or_insert_with(|| {
                let mut a = Attrs::new();
                a.insert("GRIB_paramId".into(), num(param as f64));
                for (attr, k) in [
                    ("GRIB_dataType", "dataType"),
                    ("GRIB_typeOfLevel", "typeOfLevel"),
                    ("GRIB_stepType", "stepType"),
                    ("GRIB_gridType", "gridType"),
                    ("GRIB_name", "name"),
                    ("GRIB_shortName", "shortName"),
                    ("GRIB_units", "units"),
                    ("GRIB_cfName", "cfName"),
                    ("GRIB_cfVarName", "cfVarName"),
                ] {
                    a.insert(attr.into(), text(h.text(k)));
                }
                for (attr, k) in [
                    ("GRIB_numberOfPoints", "numberOfPoints"),
                    ("GRIB_stepUnits", "stepUnits"),
                ] {
                    if let Ok(v) = h.long(k) {
                        a.insert(attr.into(), num(v as f64));
                    }
                }
                a.insert("GRIB_missingValue".into(), num(MISSING));
                a.insert("long_name".into(), text(h.text("name")));
                a.insert("units".into(), text(h.text("units")));
                a.insert("standard_name".into(), text(h.text("cfName")));
                groups.push(Group {
                    name: name.clone(),
                    type_of_level: type_of_level.clone(),
                    step_type: step_type.clone(),
                    grid,
                    attrs: a,
                    messages: Vec::new(),
                });
                groups.len() - 1
            });
            groups[g].messages.push((header, msg));
        }
        drop(fp);
        if messages.is_empty() {
            return Err(Error::Unsupported(format!("{path:?} has no GRIB messages")));
        }
        let (vars, data) = layout(groups, &mut grids);
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(File::open(path)?),
            messages,
            grids,
            vars,
            data,
            attrs,
        })
    }

    /// Decodes one message's values.
    fn field(&self, msg: usize) -> Result<Vec<f64>> {
        let m = &self.messages[msg];
        let mut bytes = vec![0u8; m.len];
        {
            let mut f = self.file.lock().expect("GRIB file");
            f.seek(SeekFrom::Start(m.offset))?;
            f.read_exact(&mut bytes)?;
        }
        let _lock = ECCODES.lock().expect("ecCodes lock");
        // SAFETY: `bytes` outlives the handle, which is dropped at the end of this scope.
        let h = unsafe {
            ffi::codes_handle_new_from_message(
                std::ptr::null_mut(),
                bytes.as_ptr().cast::<c_void>(),
                bytes.len(),
            )
        };
        if h.is_null() {
            return Err(Error::Unsupported(format!(
                "GRIB: can't decode the message at byte {} of {:?}",
                m.offset, self.path
            )));
        }
        let h = Handle(h);
        let values = h.values()?;
        drop(h);
        drop(bytes);
        let want: u64 = self.grids[m.grid].kind.shape().iter().product();
        if values.len() as u64 != want {
            return Err(Error::Unsupported(format!(
                "GRIB: the message at byte {} has {} values, not {want}",
                m.offset,
                values.len()
            )));
        }
        Ok(values)
    }
}

/// Reads a grid's shape and coordinates from its first message.
fn read_grid(h: &Handle, md5: String) -> Result<Grid> {
    let grid_type = h.text("gridType");
    let n = h.long("numberOfPoints")? as u64;
    let kind = if grid_type == "regular_ll" || grid_type == "regular_gg" {
        GridKind::Regular {
            ny: h.long("Nj")? as u64,
            nx: h.long("Ni")? as u64,
        }
    } else if PROJECTED.contains(&grid_type.as_str()) {
        GridKind::Projected {
            ny: h.long("Ny")? as u64,
            nx: h.long("Nx")? as u64,
        }
    } else {
        GridKind::Points { n }
    };
    if kind.shape().iter().product::<u64>() != n {
        return Err(Error::Unsupported(format!(
            "GRIB: a {grid_type} grid of shape {:?} with {n} points",
            kind.shape()
        )));
    }
    let lats = h.doubles("latitudes")?;
    let lons = h.doubles("longitudes")?;
    let (lats, lons) = match kind {
        // Rows of constant latitude: the first point of each row, and the points of the first.
        GridKind::Regular { ny, nx } => (
            (0..ny).map(|j| lats[(j * nx) as usize]).collect(),
            lons[..nx as usize].to_vec(),
        ),
        _ => (lats, lons),
    };
    Ok(Grid {
        md5,
        kind,
        lats,
        lons,
        names: Vec::new(),
    })
}

/// Lays the groups out as variables: data variables in file order, then their coordinates.
fn layout(groups: Vec<Group>, grids: &mut [Grid]) -> (Vec<Variable>, Vec<Data>) {
    let mut namer = Namer::default();
    let mut coords: Vec<(Variable, Data)> = Vec::new();
    let mut fields: Vec<(Variable, Data)> = Vec::new();
    // Data variable names first, so a coordinate never takes one.
    let mut names: Vec<String> = Vec::new();
    for g in &groups {
        let mut name = g.name.clone();
        for extra in [&g.type_of_level, &g.step_type] {
            if names.contains(&name) {
                name = format!("{name}_{extra}");
            }
        }
        let base = name.clone();
        let mut n = 1;
        while names.contains(&name) {
            n += 1;
            name = format!("{base}_{n}");
        }
        names.push(name);
    }
    namer.taken = names.clone();

    let add_coord = |namer: &mut Namer,
                     coords: &mut Vec<(Variable, Data)>,
                     base: &str,
                     dims: Vec<String>,
                     shape: Vec<u64>,
                     values: Vec<f64>,
                     attrs: Attrs|
     -> String {
        let (name, new) = namer.coord(base, &dims, &values);
        if new {
            let dims = if dims.len() == 1 && dims[0] == base {
                vec![name.clone()]
            } else {
                dims
            };
            coords.push((
                Variable {
                    name: name.clone(),
                    dims,
                    shape,
                    dtype: "f8".into(),
                    attrs,
                },
                Data::Coord(values),
            ));
        }
        name
    };

    for (g, name) in groups.into_iter().zip(names) {
        let decreasing = matches!(g.type_of_level.as_str(), "isobaricInhPa" | "isobaricInPa");
        let has_number = g.messages.iter().any(|(h, _)| h.number.is_some());
        let pick = |f: &dyn Fn(&Header) -> f64, decreasing| {
            distinct(g.messages.iter().map(|(h, _)| f(h)).collect(), decreasing)
        };
        let values = [
            if has_number {
                pick(&|h| h.number.unwrap_or(0.0), false)
            } else {
                vec![0.0]
            },
            pick(&|h| h.time, false),
            pick(&|h| h.step, false),
            pick(&|h| h.level, decreasing),
        ];
        let bases = ["number", "time", "step", g.type_of_level.as_str()];
        let mut dims = Vec::new();
        let mut shape = Vec::new();
        let mut aux = Vec::new();
        let mut axis_names = Vec::new();
        for (i, (v, base)) in values.iter().zip(bases).enumerate() {
            if i == 0 && !has_number {
                axis_names.push(String::new());
                continue;
            }
            let dim = v.len() > 1;
            let (cdims, cshape) = if dim {
                (vec![base.to_string()], vec![v.len() as u64])
            } else {
                (Vec::new(), Vec::new())
            };
            let cname = add_coord(
                &mut namer,
                &mut coords,
                base,
                cdims,
                cshape,
                v.clone(),
                coord_attrs(base),
            );
            if dim {
                dims.push(cname.clone());
                shape.push(v.len() as u64);
            } else {
                aux.push(cname.clone());
            }
            axis_names.push(cname);
        }

        // valid_time = time + step, on whichever of the two are dimensions.
        let (times, steps) = (&values[1], &values[2]);
        let mut vt_dims = Vec::new();
        let mut vt_shape = Vec::new();
        for (i, v) in [(1, times), (2, steps)] {
            if v.len() > 1 {
                vt_dims.push(axis_names[i].clone());
                vt_shape.push(v.len() as u64);
            }
        }
        let vt: Vec<f64> = times
            .iter()
            .flat_map(|t| steps.iter().map(move |s| t + s * 3600.0))
            .collect();
        aux.push(add_coord(
            &mut namer,
            &mut coords,
            "valid_time",
            vt_dims,
            vt_shape,
            vt,
            coord_attrs("valid_time"),
        ));

        // The grid's names, given when a variable first uses it.
        let grid = &mut grids[g.grid];
        if grid.names.is_empty() {
            grid.names = match grid.kind {
                GridKind::Regular { ny, nx } => {
                    let lat = add_coord(
                        &mut namer,
                        &mut coords,
                        "latitude",
                        vec!["latitude".into()],
                        vec![ny],
                        grid.lats.clone(),
                        coord_attrs("latitude"),
                    );
                    let lon = add_coord(
                        &mut namer,
                        &mut coords,
                        "longitude",
                        vec!["longitude".into()],
                        vec![nx],
                        grid.lons.clone(),
                        coord_attrs("longitude"),
                    );
                    vec![lat, lon]
                }
                kind => {
                    let shape = kind.shape();
                    let dims: Vec<String> = match kind {
                        GridKind::Projected { .. } => vec![namer.fresh("y"), namer.fresh("x")],
                        _ => vec![namer.fresh("values")],
                    };
                    let lat = add_coord(
                        &mut namer,
                        &mut coords,
                        "latitude",
                        dims.clone(),
                        shape.clone(),
                        grid.lats.clone(),
                        coord_attrs("latitude"),
                    );
                    let lon = add_coord(
                        &mut namer,
                        &mut coords,
                        "longitude",
                        dims.clone(),
                        shape,
                        grid.lons.clone(),
                        coord_attrs("longitude"),
                    );
                    [dims, vec![lat, lon]].concat()
                }
            };
        }
        let grid_dims = match grid.kind {
            GridKind::Regular { .. } => grid.names.clone(),
            _ => {
                let k = grid.names.len() - 2;
                aux.extend(grid.names[k..].iter().cloned());
                grid.names[..k].to_vec()
            }
        };
        dims.extend(grid_dims);
        shape.extend(grid.kind.shape());

        // The message in each cell of the header axes; the first wins when two share one.
        let lens: Vec<usize> = values.iter().map(Vec::len).collect();
        let mut cells = vec![None; lens.iter().product()];
        let pos = |v: &[f64], x: f64| v.iter().position(|y| y.total_cmp(&x).is_eq()).unwrap();
        for (h, msg) in &g.messages {
            let idx = [
                h.number.map_or(0, |n| pos(&values[0], n)),
                pos(&values[1], h.time),
                pos(&values[2], h.step),
                pos(&values[3], h.level),
            ];
            let flat = idx.iter().zip(&lens).fold(0, |acc, (i, n)| acc * n + i);
            cells[flat].get_or_insert(*msg);
        }
        let axes = std::array::from_fn(|i| Axis {
            values: values[i].clone(),
            dim: values[i].len() > 1,
        });

        let mut attrs = g.attrs;
        attrs.insert("coordinates".into(), text(aux.join(" ")));
        fields.push((
            Variable {
                name,
                dims,
                shape,
                dtype: "f4".into(),
                attrs,
            },
            Data::Field {
                axes,
                cells,
                grid: g.grid,
            },
        ));
    }
    fields.into_iter().chain(coords).unzip()
}

impl ArraySource for Grib {
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
        let v = &self.vars[i];
        check_slab(&v.shape, start, count)?;
        let (axes, cells, grid) = match &self.data[i] {
            Data::Coord(values) => {
                if v.shape.is_empty() {
                    return Ok(values.clone());
                }
                return slab(values, &v.shape, start, count);
            }
            Data::Field { axes, cells, grid } => (axes, cells, *grid),
        };
        let grid_shape = self.grids[grid].kind.shape();
        let g = grid_shape.len();
        let h = v.shape.len() - g;
        let (gstart, gcount) = (&start[h..], &count[h..]);
        let block: u64 = gcount.iter().product();
        let lens: Vec<usize> = axes.iter().map(|a| a.values.len()).collect();

        // Walk the header dimensions' part of the hyperslab, one grid block per cell.
        let cells_in_slab: u64 = count[..h].iter().product();
        let mut out = Vec::with_capacity((cells_in_slab * block) as usize);
        let mut idx = vec![0u64; h];
        for _ in 0..cells_in_slab {
            let mut k = 0;
            let mut flat = 0;
            for (a, axis) in axes.iter().enumerate() {
                let at = if axis.dim {
                    let at = start[k] + idx[k];
                    k += 1;
                    at as usize
                } else {
                    0
                };
                flat = flat * lens[a] + at;
            }
            match cells[flat] {
                Some(msg) => {
                    let field = self.field(msg)?;
                    out.extend(slab(&field, &grid_shape, gstart, gcount)?);
                }
                None => out.extend(std::iter::repeat_n(f64::NAN, block as usize)),
            }
            for d in (0..h).rev() {
                idx[d] += 1;
                if idx[d] < count[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(out)
    }
}
