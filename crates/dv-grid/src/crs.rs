//! Coordinate reference systems and transforms between them, through PROJ.
//!
//! The engine uses the same PROJ library and EPSG database as GDAL, QGIS and pyproj, so a
//! transform here gives the numbers those tools give, and code exported to Python reproduces it.
//!
//! **Grids.** The most accurate datum shifts (NADCON, OSTN15, NTv2 and others) need grid files.
//! PROJ uses a grid when it finds one in its search paths and otherwise falls back to a less
//! accurate operation, sometimes a "ballpark" one metres off. The engine never downloads grids;
//! it reports which operation a transform used and its accuracy ([`Transformer::operation_at`]),
//! and which more accurate operations need grids that aren't installed
//! ([`Transformer::missing_grids`]), so the app can say when a result is approximate.
//!
//! Coordinates are always (x, y) = (easting, northing) or (longitude, latitude) in degrees,
//! whatever the CRS's official axis order, as with pyproj's `always_xy`.
//!
//! PROJ objects can't be shared between threads, so each thread keeps its own PROJ context and
//! transforms; [`Crs`] and [`Transformer`] only hold definitions and can be sent anywhere.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int};
use std::fmt;
use std::path::PathBuf;
use std::ptr;

use proj_sys as sys;

/// A CRS that can't be read or a transform that can't be made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct CrsError(pub String);

/// Where PROJ finds its database and grid files.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ProjData {
    /// The only directories searched for grid files, replacing PROJ's defaults (`PROJ_DATA` and
    /// the install's data directory). `None` keeps the defaults; an empty list means no grids.
    pub grid_paths: Option<Vec<PathBuf>>,
    /// The `proj.db` to use. `None` keeps the one PROJ finds by default.
    pub database: Option<PathBuf>,
}

impl ProjData {
    /// PROJ's default database, with grids only from `paths`.
    pub fn grids_in(paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        Self {
            grid_paths: Some(paths.into_iter().map(Into::into).collect()),
            database: None,
        }
    }
}

/// A coordinate reference system.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Crs {
    /// The definition PROJ reads, as given.
    text: String,
    name: String,
    geographic: bool,
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
    /// Reads anything PROJ reads: `EPSG:<code>` (or another authority), WKT, PROJJSON or a PROJ
    /// string (`+proj=…`). `WGS84` means EPSG:4326.
    pub fn parse(text: &str) -> Result<Self, CrsError> {
        let text = text.trim();
        // A bare PROJ string is an operation to PROJ; marking it as a CRS is what pyproj does.
        let text = if text.eq_ignore_ascii_case("wgs84") {
            "EPSG:4326".to_string()
        } else if text.starts_with("+proj=") && !text.contains("+type=crs") {
            format!("{text} +type=crs")
        } else {
            text.to_string()
        };
        let text = text.as_str();
        with_context(&ProjData::default(), |c| {
            let crs = c.create(text)?;
            if !is_crs(crs.0) {
                return Err(CrsError(format!(
                    "{text:?} isn't a coordinate reference system"
                )));
            }
            // SAFETY: crs is a live object; PROJ returns its name or null.
            let name = unsafe { string(sys::proj_get_name(crs.0)) }.unwrap_or_default();
            Ok(Self {
                text: text.to_string(),
                name,
                geographic: is_geographic(c.ctx, crs.0),
            })
        })?
    }

    pub fn epsg(code: u16) -> Result<Self, CrsError> {
        Self::parse(&format!("EPSG:{code}"))
    }

    /// WGS 84 longitude and latitude.
    pub fn wgs84() -> Self {
        Self::epsg(4326).expect("EPSG:4326 is in PROJ's database")
    }

    /// Whether coordinates are longitude and latitude.
    pub fn is_geographic(&self) -> bool {
        self.geographic
    }

    /// The CRS's name, such as "OSGB36 / British National Grid".
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// The operation PROJ used for a transform.
#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    pub name: String,
    /// Accuracy in metres, when the EPSG database gives one.
    pub accuracy_m: Option<f64>,
    /// A "ballpark" operation ignores a datum difference it has no parameters for, and can be
    /// tens or hundreds of metres off.
    pub ballpark: bool,
}

/// A more accurate operation that can't be used because grids aren't installed.
#[derive(Debug, Clone, PartialEq)]
pub struct MissingGrids {
    pub operation: String,
    pub accuracy_m: Option<f64>,
    /// The grid files it needs that weren't found, by file name.
    pub grids: Vec<String>,
    /// Where each grid can be downloaded (PROJ's CDN), in the same order; `None` for grids PROJ
    /// can't distribute.
    pub urls: Vec<Option<String>>,
}

/// Transforms coordinates from one CRS to another.
#[derive(Debug, Clone)]
pub struct Transformer {
    src: Crs,
    dst: Crs,
    data: ProjData,
}

impl Transformer {
    /// A transform with PROJ's default database and grids.
    pub fn new(src: &Crs, dst: &Crs) -> Result<Self, CrsError> {
        Self::with_data(src, dst, &ProjData::default())
    }

    /// A transform using the given database and grids.
    pub fn with_data(src: &Crs, dst: &Crs, data: &ProjData) -> Result<Self, CrsError> {
        let t = Self {
            src: src.clone(),
            dst: dst.clone(),
            data: data.clone(),
        };
        t.with_op(|_, _| ())?;
        Ok(t)
    }

    /// Runs `f` with this thread's PROJ context and this transform's operation.
    fn with_op<R>(
        &self,
        f: impl FnOnce(*mut sys::PJ_CONTEXT, *mut sys::PJ) -> R,
    ) -> Result<R, CrsError> {
        with_context(&self.data, |c| {
            let op = c.operation(&self.src.text, &self.dst.text)?;
            Ok(f(c.ctx, op))
        })?
    }

    /// Transforms one point. `None` if it's outside where the transform is defined.
    pub fn point(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        self.with_op(|_, op| trans(op, x, y)).ok().flatten()
    }

    /// Transforms points in place. Points that fail become NaN.
    pub fn transform(&self, xs: &mut [f64], ys: &mut [f64]) {
        let n = xs.len().min(ys.len());
        let done = self.with_op(|_, op| {
            let stride = std::mem::size_of::<f64>();
            // SAFETY: xs and ys hold at least n f64s each, at a stride of one f64; PROJ writes the
            // results in place and reads nothing else (z and t are null with count 0).
            unsafe {
                sys::proj_trans_generic(
                    op,
                    sys::PJ_DIRECTION_PJ_FWD,
                    xs.as_mut_ptr(),
                    stride,
                    n,
                    ys.as_mut_ptr(),
                    stride,
                    n,
                    ptr::null_mut(),
                    0,
                    0,
                    ptr::null_mut(),
                    0,
                    0,
                );
            }
        });
        for (x, y) in xs[..n].iter_mut().zip(ys[..n].iter_mut()) {
            if done.is_err() || !x.is_finite() || !y.is_finite() {
                (*x, *y) = (f64::NAN, f64::NAN);
            }
        }
    }

    /// The operation PROJ uses at a point: with several candidates (by area of use and grid
    /// availability) it can differ from place to place. `None` if the point fails.
    pub fn operation_at(&self, x: f64, y: f64) -> Option<Operation> {
        self.with_op(|ctx, op| {
            trans(op, x, y)?;
            // SAFETY: op is live and was just used; PROJ returns a new object (or null) that we
            // destroy.
            let used = Owned(unsafe { sys::proj_trans_get_last_used_operation(op) });
            // A single-operation transform reports nothing: it is its own operation.
            let o = if used.0.is_null() { op } else { used.0 };
            Some(describe(ctx, o))
        })
        .ok()
        .flatten()
    }

    /// Operations more accurate than the best usable one that need grid files PROJ didn't find,
    /// most accurate first. Empty when the best operation is already available.
    pub fn missing_grids(&self) -> Result<Vec<MissingGrids>, CrsError> {
        with_context(&self.data, |c| {
            let src = c.create(&self.src.text)?;
            let dst = c.create(&self.dst.text)?;
            let ctx = c.ctx;
            // SAFETY: every object below is created on ctx and destroyed before returning; list
            // indices stay within proj_list_get_count.
            unsafe {
                let factory = sys::proj_create_operation_factory_context(ctx, ptr::null());
                if factory.is_null() {
                    return Err(c.error("can't list transformations"));
                }
                sys::proj_operation_factory_context_set_grid_availability_use(
                    ctx,
                    factory,
                    sys::PROJ_GRID_AVAILABILITY_USE_PROJ_GRID_AVAILABILITY_IGNORED,
                );
                sys::proj_operation_factory_context_set_spatial_criterion(
                    ctx,
                    factory,
                    sys::PROJ_SPATIAL_CRITERION_PROJ_SPATIAL_CRITERION_PARTIAL_INTERSECTION,
                );
                let list = sys::proj_create_operations(ctx, src.0, dst.0, factory);
                sys::proj_operation_factory_context_destroy(factory);
                if list.is_null() {
                    return Err(c.error("can't list transformations"));
                }
                let mut usable_best = f64::INFINITY;
                let mut missing = Vec::new();
                for i in 0..sys::proj_list_get_count(list) {
                    let op = Owned(sys::proj_list_get(ctx, list, i));
                    if op.0.is_null() {
                        continue;
                    }
                    let d = describe(ctx, op.0);
                    let acc = d.accuracy_m.unwrap_or(f64::INFINITY);
                    if sys::proj_coordoperation_is_instantiable(ctx, op.0) != 0 {
                        if !d.ballpark {
                            usable_best = usable_best.min(acc);
                        }
                        continue;
                    }
                    let (mut grids, mut urls) = (Vec::new(), Vec::new());
                    for g in 0..sys::proj_coordoperation_get_grid_used_count(ctx, op.0) {
                        let (mut short, mut full, mut package, mut url) =
                            (ptr::null(), ptr::null(), ptr::null(), ptr::null());
                        let (mut direct, mut open, mut available) = (0, 0, 0);
                        let ok = sys::proj_coordoperation_get_grid_used(
                            ctx,
                            op.0,
                            g,
                            &mut short,
                            &mut full,
                            &mut package,
                            &mut url,
                            &mut direct,
                            &mut open,
                            &mut available,
                        );
                        if ok != 0 && available == 0 {
                            grids.push(string(short).unwrap_or_default());
                            urls.push(string(url).filter(|u| !u.is_empty()));
                        }
                    }
                    if !grids.is_empty() {
                        missing.push(MissingGrids {
                            operation: d.name,
                            accuracy_m: d.accuracy_m,
                            grids,
                            urls,
                        });
                    }
                }
                sys::proj_list_destroy(list);
                missing.retain(|m| m.accuracy_m.is_some_and(|a| a < usable_best));
                missing.sort_by(|a, b| a.accuracy_m.partial_cmp(&b.accuracy_m).expect("finite"));
                missing.dedup_by(|a, b| a.grids == b.grids);
                Ok(missing)
            }
        })?
    }
}

fn trans(op: *mut sys::PJ, x: f64, y: f64) -> Option<(f64, f64)> {
    let coord = sys::PJ_COORD {
        xyzt: sys::PJ_XYZT {
            x,
            y,
            z: 0.0,
            t: f64::INFINITY,
        },
    };
    // SAFETY: op is a live transform owned by this thread's context; every PJ_COORD variant is
    // four f64s, so reading xyzt is always valid.
    let out = unsafe { sys::proj_trans(op, sys::PJ_DIRECTION_PJ_FWD, coord).xyzt };
    (out.x.is_finite() && out.y.is_finite()).then_some((out.x, out.y))
}

fn describe(ctx: *mut sys::PJ_CONTEXT, op: *mut sys::PJ) -> Operation {
    // SAFETY: op is a live coordinate operation on ctx.
    unsafe {
        let accuracy = sys::proj_coordoperation_get_accuracy(ctx, op);
        Operation {
            name: string(sys::proj_get_name(op)).unwrap_or_default(),
            accuracy_m: (accuracy >= 0.0).then_some(accuracy),
            ballpark: sys::proj_coordoperation_has_ballpark_transformation(ctx, op) != 0,
        }
    }
}

fn is_crs(obj: *mut sys::PJ) -> bool {
    // SAFETY: obj is a live object.
    let t = unsafe { sys::proj_get_type(obj) };
    (sys::PJ_TYPE_PJ_TYPE_CRS..=sys::PJ_TYPE_PJ_TYPE_OTHER_CRS).contains(&t)
}

fn is_geographic(ctx: *mut sys::PJ_CONTEXT, obj: *mut sys::PJ) -> bool {
    // SAFETY: obj is a live object on ctx; the source CRS of a bound CRS is destroyed here.
    unsafe {
        match sys::proj_get_type(obj) {
            sys::PJ_TYPE_PJ_TYPE_GEOGRAPHIC_CRS
            | sys::PJ_TYPE_PJ_TYPE_GEOGRAPHIC_2D_CRS
            | sys::PJ_TYPE_PJ_TYPE_GEOGRAPHIC_3D_CRS => true,
            // A PROJ string with +towgs84 makes a CRS bound to WGS 84.
            sys::PJ_TYPE_PJ_TYPE_BOUND_CRS => {
                let base = Owned(sys::proj_get_source_crs(ctx, obj));
                !base.0.is_null() && is_geographic(ctx, base.0)
            }
            _ => false,
        }
    }
}

/// Copies a C string PROJ owns.
///
/// # Safety
/// `s` is null or a valid NUL-terminated string.
unsafe fn string(s: *const c_char) -> Option<String> {
    // SAFETY: the caller guarantees s is null or valid.
    (!s.is_null()).then(|| unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned())
}

/// A PROJ object destroyed on drop.
struct Owned(*mut sys::PJ);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: we own the object, and its context outlives it.
            unsafe { sys::proj_destroy(self.0) };
        }
    }
}

/// One thread's PROJ context, with the transforms made on it.
struct Context {
    ctx: *mut sys::PJ_CONTEXT,
    ops: HashMap<(String, String), Owned>,
}

impl Context {
    fn new(data: &ProjData) -> Result<Self, CrsError> {
        // SAFETY: a new context, configured before any object is created on it.
        let ctx = unsafe { sys::proj_context_create() };
        if ctx.is_null() {
            return Err(CrsError("can't start PROJ".into()));
        }
        let c = Self {
            ctx,
            ops: HashMap::new(),
        };
        // SAFETY: ctx is live; the C strings outlive the calls that read them.
        unsafe {
            sys::proj_log_level(ctx, sys::PJ_LOG_LEVEL_PJ_LOG_NONE);
            // Never download grids: results mustn't depend on the network.
            sys::proj_context_set_enable_network(ctx, 0);
            // Search paths also locate proj.db, so keep the default database explicitly.
            let db = match &data.database {
                Some(p) => Some(cstring(&p.to_string_lossy())?),
                None if data.grid_paths.is_some() => {
                    string(sys::proj_context_get_database_path(ctx))
                        .map(|s| cstring(&s))
                        .transpose()?
                }
                None => None,
            };
            if let Some(paths) = &data.grid_paths {
                let mut owned: Vec<CString> = paths
                    .iter()
                    .map(|p| cstring(&p.to_string_lossy()))
                    .collect::<Result<_, _>>()?;
                // An empty list would restore the defaults; a path that doesn't exist finds none.
                if owned.is_empty() {
                    owned.push(cstring("/nonexistent/dataventura-no-grids")?);
                }
                let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();
                sys::proj_context_set_search_paths(ctx, ptrs.len() as c_int, ptrs.as_ptr());
            }
            if let Some(db) = db
                && sys::proj_context_set_database_path(ctx, db.as_ptr(), ptr::null(), ptr::null())
                    == 0
            {
                return Err(c.error("can't open PROJ's database"));
            }
        }
        Ok(c)
    }

    fn error(&self, what: &str) -> CrsError {
        // SAFETY: ctx is live; PROJ returns a static or context-owned message.
        let msg = unsafe {
            string(sys::proj_context_errno_string(
                self.ctx,
                sys::proj_context_errno(self.ctx),
            ))
        };
        CrsError(match msg {
            Some(m) => format!("{what}: {m}"),
            None => what.to_string(),
        })
    }

    fn create(&self, text: &str) -> Result<Owned, CrsError> {
        let c = cstring(text)?;
        // SAFETY: ctx is live and c is a valid C string.
        let obj = Owned(unsafe { sys::proj_create(self.ctx, c.as_ptr()) });
        if obj.0.is_null() {
            return Err(self.error(&format!("can't read the CRS {text:?}")));
        }
        Ok(obj)
    }

    /// The transform between two CRSs, made once per thread, with x/y axis order.
    fn operation(&mut self, src: &str, dst: &str) -> Result<*mut sys::PJ, CrsError> {
        let key = (src.to_string(), dst.to_string());
        if let Some(op) = self.ops.get(&key) {
            return Ok(op.0);
        }
        let (s, d) = (self.create(src)?, self.create(dst)?);
        // SAFETY: s and d are live CRSs on ctx; the results are owned and destroyed by Owned.
        let op = unsafe {
            let t = Owned(sys::proj_create_crs_to_crs_from_pj(
                self.ctx,
                s.0,
                d.0,
                ptr::null_mut(),
                ptr::null(),
            ));
            if t.0.is_null() {
                return Err(self.error(&format!("can't transform from {src} to {dst}")));
            }
            Owned(sys::proj_normalize_for_visualization(self.ctx, t.0))
        };
        if op.0.is_null() {
            return Err(self.error(&format!("can't transform from {src} to {dst}")));
        }
        let p = op.0;
        self.ops.insert(key, op);
        Ok(p)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.ops.clear();
        // SAFETY: every object made on ctx is gone.
        unsafe { sys::proj_context_destroy(self.ctx) };
    }
}

fn cstring(s: &str) -> Result<CString, CrsError> {
    CString::new(s).map_err(|_| CrsError(format!("{s:?} contains a NUL byte")))
}

thread_local! {
    static CONTEXTS: RefCell<HashMap<ProjData, Context>> = RefCell::new(HashMap::new());
}

/// Runs `f` with this thread's context for `data`, creating it on first use.
fn with_context<R>(data: &ProjData, f: impl FnOnce(&mut Context) -> R) -> Result<R, CrsError> {
    CONTEXTS.with(|m| {
        let mut m = m.borrow_mut();
        if !m.contains_key(data) {
            m.insert(data.clone(), Context::new(data)?);
        }
        Ok(f(m.get_mut(data).expect("just inserted")))
    })
}
