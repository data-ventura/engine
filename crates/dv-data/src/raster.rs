//! GeoTIFF rasters (striped or tiled, so Cloud Optimized GeoTIFFs too) as arrays, on disk or read
//! by HTTP range requests from object storage.
//!
//! A raster opens as an [`ArraySource`] with one variable per band (`band_1`, …) on dimensions
//! `y` and `x`, plus `x` and `y` coordinate variables at pixel centers in the raster's CRS. The
//! CRS (`EPSG:…`) is the `crs` attribute; GDAL's nodata value is each band's `_FillValue`, so
//! [`read`](crate::array::read) masks it.
//!
//! Reads decode only the tiles (or strips) a window touches. Overviews are further
//! [`levels`](GeoTiff::levels), chosen for a requested resolution as GDAL chooses them
//! ([`level_for_shape`](GeoTiff::level_for_shape)) and read with
//! [`read_level`](GeoTiff::read_level).
//!
//! Remote files ([`GeoTiff::open_url`]) are never downloaded whole. Opening fetches the first
//! 16 KiB, which in a COG holds the headers of every level; any header beyond it is fetched in
//! 16 KiB pieces as the decoder reaches it. A read then fetches exactly the byte ranges of the
//! tiles it needs, many at once, merging tiles that are adjacent in the file into one request.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures_util::stream::{self, StreamExt, TryStreamExt};
use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::Tag;
use zarrs_object_store::object_store::path::Path as ObjectPath;
use zarrs_object_store::object_store::{
    GetOptions, GetRange, ObjectStore, ObjectStoreExt, parse_url_opts,
};

use crate::array::{ArraySource, AttrValue, Attrs, Variable, check_slab};
use crate::store::{RetryPolicy, RetryStore, http_store};
use crate::zarr::{block_on, cloud_env};
use crate::{Error, Result};

/// One resolution of a raster: the full image (level 0) or an overview.
#[derive(Debug, Clone, PartialEq)]
pub struct Level {
    pub width: u64,
    pub height: u64,
    /// Pixel-corner geotransform, as GDAL gives it: `x = x0 + col * dx`, `y = y0 + row * dy`.
    pub x0: f64,
    pub dx: f64,
    pub y0: f64,
    pub dy: f64,
    ifd: usize,
    /// Tile (or strip) width and height.
    chunk: (u64, u64),
    offsets: Vec<u64>,
    byte_counts: Vec<u64>,
}

impl Level {
    /// X coordinates of pixel centers.
    pub fn x(&self) -> Vec<f64> {
        (0..self.width)
            .map(|i| self.x0 + self.dx * (i as f64 + 0.5))
            .collect()
    }

    /// Y coordinates of pixel centers.
    pub fn y(&self) -> Vec<f64> {
        (0..self.height)
            .map(|j| self.y0 + self.dy * (j as f64 + 0.5))
            .collect()
    }
}

trait ReadSeek: Read + Seek + Send {}
impl<T: Read + Seek + Send> ReadSeek for T {}

type Tiff = Decoder<Box<dyn ReadSeek>>;

pub struct GeoTiff {
    vars: Vec<Variable>,
    attrs: Attrs,
    levels: Vec<Level>,
    samples: usize,
    planar: bool,
    /// The decoder and the IFD it's on.
    tiff: Mutex<(Tiff, usize)>,
    /// Set when the file is read by ranges from an object store.
    remote: Option<Arc<Remote>>,
}

impl std::fmt::Debug for GeoTiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeoTiff")
            .field("variables", &self.vars)
            .field("levels", &self.levels)
            .finish_non_exhaustive()
    }
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

/// `NewSubfileType` bits: a reduced-resolution image (an overview), and a transparency mask.
const REDUCED: u32 = 1;
const MASK: u32 = 4;

/// GDAL's default oversampling threshold for nearest-neighbour reads
/// (`GDAL_OVERVIEW_OVERSAMPLING_THRESHOLD`): an overview up to 1.2 times coarser than asked for
/// still counts.
const OVERSAMPLING: f64 = 1.2;

/// The level at IFD `ifd`, which the decoder is on, with the full image's pixel-corner transform.
fn level(d: &mut Tiff, ifd: usize, base: Option<&Level>, t: [f64; 4]) -> Result<Level> {
    let (w, h) = d.dimensions().map_err(err)?;
    let (w, h) = (u64::from(w), u64::from(h));
    let (cw, ch) = d.chunk_dimensions();
    let (offsets, counts) = match d.get_chunk_type() {
        ChunkType::Tile => (Tag::TileOffsets, Tag::TileByteCounts),
        ChunkType::Strip => (Tag::StripOffsets, Tag::StripByteCounts),
    };
    // An overview keeps the full image's extent: GDAL scales the pixel size by the size ratio.
    let (sx, sy) = match base {
        Some(b) => (b.width as f64 / w as f64, b.height as f64 / h as f64),
        None => (1.0, 1.0),
    };
    Ok(Level {
        width: w,
        height: h,
        x0: t[0],
        dx: t[1] * sx,
        y0: t[2],
        dy: t[3] * sy,
        ifd,
        chunk: (u64::from(cw), u64::from(ch).min(h)),
        offsets: d.get_tag_u64_vec(offsets).map_err(err)?,
        byte_counts: d.get_tag_u64_vec(counts).map_err(err)?,
    })
}

impl GeoTiff {
    /// Opens a GeoTIFF on disk, reading only its headers.
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_reader(Box::new(BufReader::new(File::open(path)?)), None)
    }

    /// Opens the GeoTIFF at `location`: a path, or an `https://`, `s3://`, `gs://` or `az://`
    /// URL, read by ranges (see the module docs). Cloud credentials and regions come from the
    /// environment as for [`ZarrSource::open`](crate::zarr::ZarrSource::open), and reads over the
    /// network are retried as [`RetryStore`] describes. A URL's query string isn't kept, so
    /// presigned URLs aren't supported.
    pub fn open_url(location: &str) -> Result<Self> {
        let bad = |e: &dyn std::fmt::Display| Error::Unsupported(format!("{location}: {e}"));
        let Some((scheme, _)) = location.split_once("://") else {
            return Self::open(Path::new(location));
        };
        let url = url::Url::parse(location).map_err(|e| bad(&e))?;
        let (store, path): (Box<dyn ObjectStore>, ObjectPath) = match scheme {
            "file" => {
                let path = url
                    .to_file_path()
                    .map_err(|()| bad(&"isn't a local path"))?;
                return Self::open(&path);
            }
            "http" | "https" => {
                let origin = url.origin().ascii_serialization();
                let path = ObjectPath::from_url_path(url.path()).map_err(|e| bad(&e))?;
                (http_store(&origin).map_err(|e| bad(&e))?, path)
            }
            _ => parse_url_opts(&url, cloud_env()).map_err(|e| bad(&e))?,
        };
        let retry = RetryStore::new(store, RetryPolicy::default());
        Self::open_store(Arc::new(retry), path.as_ref())
    }

    /// Opens the GeoTIFF at `path` in any object store, read by ranges.
    pub fn open_store(store: Arc<dyn ObjectStore>, path: &str) -> Result<Self> {
        let path = ObjectPath::from(path);
        let remote = Arc::new(block_on(Remote::open(store, path))?);
        let reader = RangeReader {
            remote: remote.clone(),
            pos: 0,
        };
        Self::from_reader(Box::new(reader), Some(remote))
    }

    fn from_reader(reader: Box<dyn ReadSeek>, remote: Option<Arc<Remote>>) -> Result<Self> {
        let mut d = Decoder::new(reader).map_err(err)?;
        let samples = d
            .find_tag_unsigned::<u16>(Tag::SamplesPerPixel)
            .map_err(err)?
            .unwrap_or(1) as usize;
        let planar = d
            .find_tag_unsigned::<u16>(Tag::PlanarConfiguration)
            .map_err(err)?
            .unwrap_or(1)
            == 2;

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
        // Pixel-is-point rasters put the transform on pixel centers; GDAL shifts it to corners.
        let (x0, y0) = if key(RASTER_TYPE) == Some(PIXEL_IS_POINT) {
            (x0 - 0.5 * dx, y0 - 0.5 * dy)
        } else {
            (x0, y0)
        };
        let crs = key(PROJECTED_TYPE)
            .or_else(|| key(GEOGRAPHIC_TYPE))
            .filter(|c| *c != 0 && *c != 32767)
            .map(|c| format!("EPSG:{c}"));
        let nodata = d
            .get_tag_ascii_string(Tag::GdalNodata)
            .ok()
            .and_then(|s| s.trim().trim_end_matches('\0').parse::<f64>().ok());

        // Overviews are the later IFDs marked reduced-resolution, other than masks, in file order
        // (GDAL's overview order). Anything this decoder can't read ends the scan.
        let t = [x0, dx, y0, dy];
        let mut levels = vec![level(&mut d, 0, None, t)?];
        let mut ifd = 1;
        while d.more_images() && d.seek_to_image(ifd).is_ok() {
            let kind = d
                .find_tag_unsigned::<u32>(Tag::NewSubfileType)
                .ok()
                .flatten()
                .unwrap_or(0);
            let same_samples = d
                .find_tag_unsigned::<u16>(Tag::SamplesPerPixel)
                .ok()
                .flatten()
                .unwrap_or(1) as usize
                == samples;
            if kind & REDUCED != 0 && kind & MASK == 0 && same_samples {
                let l = level(&mut d, ifd, Some(&levels[0]), t)?;
                if l.width <= levels[0].width && l.height <= levels[0].height {
                    levels.push(l);
                }
            }
            ifd += 1;
        }
        d.seek_to_image(0).map_err(err)?;

        let (w, h) = (levels[0].width, levels[0].height);
        let mut attrs = Attrs::new();
        if let Some(c) = &crs {
            attrs.insert("crs".into(), AttrValue::Text(c.clone()));
        }
        let mut vars: Vec<Variable> = (0..samples)
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
                    shape: vec![h, w],
                    dtype: "f8".into(),
                    attrs: a,
                }
            })
            .collect();
        for (name, len) in [("x", w), ("y", h)] {
            vars.push(Variable {
                name: name.into(),
                dims: vec![name.into()],
                shape: vec![len],
                dtype: "f8".into(),
                attrs: Attrs::new(),
            });
        }
        Ok(Self {
            vars,
            attrs,
            levels,
            samples,
            planar,
            tiff: Mutex::new((d, 0)),
            remote,
        })
    }

    /// The full image (level 0), then each overview, finest first.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// The level GDAL reads when `factor` source pixels make one output pixel: the coarsest
    /// whose own factor (the smaller of its width and height ratios to the full image) is under
    /// `factor` times 1.2, GDAL's oversampling threshold for nearest-neighbour reads.
    pub fn level_for_factor(&self, factor: f64) -> usize {
        let base = &self.levels[0];
        let mut best = (0, 0.0);
        for (i, l) in self.levels.iter().enumerate().skip(1) {
            let f = (base.width as f64 / l.width as f64).min(base.height as f64 / l.height as f64);
            if f < factor * OVERSAMPLING && f > best.1 {
                best = (i, f);
            }
        }
        best.0
    }

    /// The level GDAL reads for a window of `count` full-resolution pixels (rows, columns) drawn
    /// into `out` pixels, as `GDALRasterBand::RasterIO` chooses it.
    pub fn level_for_shape(&self, count: [u64; 2], out: [u64; 2]) -> usize {
        let fy = count[0] as f64 / out[0].max(1) as f64;
        let fx = count[1] as f64 / out[1].max(1) as f64;
        // GDAL takes the less reduced axis, except that a one-row buffer goes by its width.
        let factor = if fx < fy || out[0] == 1 { fx } else { fy };
        self.level_for_factor(factor)
    }

    /// The level for drawing at `resolution` CRS units per pixel.
    pub fn level_for_resolution(&self, resolution: f64) -> usize {
        let base = &self.levels[0];
        self.level_for_factor(resolution / base.dx.abs().min(base.dy.abs()))
    }

    /// Reads `count` pixels (rows, columns) from `start` of band `band` (from 1) at `level`, without
    /// masking nodata: [`array::decode`](crate::array::decode) with the band's variable does.
    pub fn read_level(
        &self,
        level: usize,
        band: usize,
        start: [u64; 2],
        count: [u64; 2],
    ) -> Result<Vec<f64>> {
        Ok(self
            .read_bands(level, &[band], start, count)?
            .pop()
            .unwrap_or_default())
    }

    /// Reads the same window of several bands. A pixel-interleaved file's tiles hold every band,
    /// so this fetches and decodes each tile once.
    pub fn read_bands(
        &self,
        level: usize,
        bands: &[usize],
        start: [u64; 2],
        count: [u64; 2],
    ) -> Result<Vec<Vec<f64>>> {
        let l = self
            .levels
            .get(level)
            .ok_or_else(|| Error::Unsupported(format!("no level {level}")))?;
        if let Some(b) = bands.iter().find(|b| **b == 0 || **b > self.samples) {
            return Err(Error::Unsupported(format!("no band {b}")));
        }
        check_slab(&[l.height, l.width], &start, &count)?;
        let mut out = vec![vec![0.0; (count[0] * count[1]) as usize]; bands.len()];
        if count[0] == 0 || count[1] == 0 {
            return Ok(out);
        }

        // Chunks (tiles, or strips as tiles the image's width) the window touches, per plane.
        let (cw, ch) = l.chunk;
        let across = l.width.div_ceil(cw);
        let per_plane = across * l.height.div_ceil(ch);
        let planes: Vec<u64> = if self.planar {
            bands.iter().map(|b| *b as u64 - 1).collect()
        } else {
            vec![0]
        };
        let mut chunks = Vec::new();
        for plane in &planes {
            for r in start[0] / ch..=(start[0] + count[0] - 1) / ch {
                for c in start[1] / cw..=(start[1] + count[1] - 1) / cw {
                    chunks.push((*plane, r, c, plane * per_plane + r * across + c));
                }
            }
        }

        let mut guard = self.tiff.lock().expect("GeoTIFF decoder");
        let (tiff, current) = &mut *guard;
        if let Some(remote) = &self.remote {
            let ranges = chunks.iter().filter_map(|(.., i)| {
                let (off, len) = (
                    *l.offsets.get(*i as usize)?,
                    *l.byte_counts.get(*i as usize)?,
                );
                (len > 0).then_some(off..off + len)
            });
            remote.prefetch(ranges.collect())?;
        }
        let result = (|| {
            if *current != l.ifd {
                tiff.seek_to_image(l.ifd).map_err(err)?;
                *current = l.ifd;
            }
            let (end_r, end_c) = (start[0] + count[0], start[1] + count[1]);
            for (plane, r, c, i) in chunks {
                let data = to_f64(tiff.read_chunk(i as u32).map_err(err)?)?;
                let (dw, _) = tiff.chunk_data_dimensions(i as u32);
                let dw = u64::from(dw);
                let spp = if self.planar { 1 } else { self.samples as u64 };
                let (r0, c0) = (r * ch, c * cw);
                let rows = r0.max(start[0])..(r0 + ch).min(end_r);
                let cols = c0.max(start[1])..(c0 + cw).min(end_c);
                for (o, b) in bands.iter().enumerate() {
                    let sample = match self.planar {
                        true if *b as u64 - 1 != plane => continue,
                        true => 0,
                        false => *b as u64 - 1,
                    };
                    for y in rows.clone() {
                        for x in cols.clone() {
                            let src = ((y - r0) * dw + (x - c0)) * spp + sample;
                            let dst = (y - start[0]) * count[1] + (x - start[1]);
                            out[o][dst as usize] = *data.get(src as usize).ok_or_else(|| {
                                Error::Unsupported(format!("GeoTIFF: chunk {i} is short"))
                            })?;
                        }
                    }
                }
            }
            Ok(out)
        })();
        if let Some(remote) = &self.remote {
            remote.release();
        }
        result
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
        match var {
            "x" => crate::array::slab(&self.levels[0].x(), &v.shape, start, count),
            "y" => crate::array::slab(&self.levels[0].y(), &v.shape, start, count),
            _ => {
                check_slab(&v.shape, start, count)?;
                let b: usize = var.trim_start_matches("band_").parse().unwrap_or(0);
                self.read_level(0, b, [start[0], start[1]], [count[0], count[1]])
            }
        }
    }
}

/// Bytes fetched when a read misses: GDAL's first read of a COG is 16 KiB too.
const READ_AHEAD: u64 = 16 * 1024;

/// Requests in flight at once while fetching tiles.
const CONCURRENCY: usize = 32;

/// A file in an object store, read by ranges, with the ranges fetched so far.
#[derive(Debug)]
struct Remote {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    size: u64,
    cache: Mutex<Cache>,
}

#[derive(Debug, Default)]
struct Cache {
    /// Fetched ranges by start offset.
    spans: BTreeMap<u64, Bytes>,
    /// Starts of spans fetched for one read, dropped when it ends; the rest (headers) are kept.
    tiles: Vec<u64>,
}

fn store_err(e: impl std::fmt::Display) -> Error {
    Error::Io(io::Error::other(e.to_string()))
}

impl Remote {
    /// Fetches the first [`READ_AHEAD`] bytes, which also tell the file's size.
    async fn open(store: Arc<dyn ObjectStore>, path: ObjectPath) -> Result<Self> {
        let options = GetOptions {
            range: Some(GetRange::Bounded(0..READ_AHEAD)),
            ..GetOptions::default()
        };
        let got = store.get_opts(&path, options).await.map_err(store_err)?;
        let size = got.meta.size;
        let head = got.bytes().await.map_err(store_err)?;
        let mut cache = Cache::default();
        cache.spans.insert(0, head);
        Ok(Self {
            store,
            path,
            size,
            cache: Mutex::new(cache),
        })
    }

    /// Fetches `ranges` that aren't cached, merging those that touch, and keeps them until
    /// [`release`](Self::release).
    fn prefetch(&self, mut ranges: Vec<Range<u64>>) -> Result<()> {
        ranges.sort_by_key(|r| r.start);
        let mut wanted: Vec<Range<u64>> = Vec::new();
        {
            let cache = self.cache.lock().expect("range cache");
            for r in ranges {
                if cache.covers(&r) {
                    continue;
                }
                match wanted.last_mut() {
                    Some(last) if last.end >= r.start => last.end = last.end.max(r.end),
                    _ => wanted.push(r),
                }
            }
        }
        if wanted.is_empty() {
            return Ok(());
        }
        let (store, path) = (self.store.clone(), self.path.clone());
        let fetched = block_on(async move {
            stream::iter(wanted)
                .map(|r| {
                    let (store, path) = (store.clone(), path.clone());
                    async move {
                        let bytes = store.get_range(&path, r.clone()).await;
                        bytes.map(|b| (r.start, b)).map_err(store_err)
                    }
                })
                .buffer_unordered(CONCURRENCY)
                .try_collect::<Vec<_>>()
                .await
        })?;
        let mut cache = self.cache.lock().expect("range cache");
        for (start, bytes) in fetched {
            cache.spans.insert(start, bytes);
            cache.tiles.push(start);
        }
        Ok(())
    }

    /// Drops the spans [`prefetch`](Self::prefetch) fetched.
    fn release(&self) {
        let mut cache = self.cache.lock().expect("range cache");
        let Cache { spans, tiles } = &mut *cache;
        for start in tiles.drain(..) {
            spans.remove(&start);
        }
    }

    /// Copies bytes from `pos` into `buf`, fetching [`READ_AHEAD`] bytes (or more, for a larger
    /// read) on a miss. Returns how many were copied.
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> io::Result<usize> {
        if pos >= self.size || buf.is_empty() {
            return Ok(0);
        }
        let next = {
            let cache = self.cache.lock().expect("range cache");
            if let Some(n) = cache.copy(pos, buf) {
                return Ok(n);
            }
            cache.spans.range(pos + 1..).next().map(|(s, _)| *s)
        };
        let end = (pos + READ_AHEAD.max(buf.len() as u64))
            .min(self.size)
            .min(next.unwrap_or(u64::MAX));
        let (store, path) = (self.store.clone(), self.path.clone());
        let bytes =
            block_on(async move { store.get_range(&path, pos..end).await.map_err(store_err) })
                .map_err(io::Error::other)?;
        let mut cache = self.cache.lock().expect("range cache");
        cache.spans.insert(pos, bytes);
        Ok(cache.copy(pos, buf).unwrap_or(0))
    }
}

impl Cache {
    /// The span holding `pos`, if one does.
    fn span(&self, pos: u64) -> Option<(u64, &Bytes)> {
        let (start, bytes) = self.spans.range(..=pos).next_back()?;
        (pos < start + bytes.len() as u64).then_some((*start, bytes))
    }

    fn covers(&self, r: &Range<u64>) -> bool {
        self.span(r.start)
            .is_some_and(|(s, b)| r.end <= s + b.len() as u64)
    }

    fn copy(&self, pos: u64, buf: &mut [u8]) -> Option<usize> {
        let (start, bytes) = self.span(pos)?;
        let from = &bytes[(pos - start) as usize..];
        let n = from.len().min(buf.len());
        buf[..n].copy_from_slice(&from[..n]);
        Some(n)
    }
}

/// A [`Remote`] as a reader for the TIFF decoder.
struct RangeReader {
    remote: Arc<Remote>,
    pos: u64,
}

impl Read for RangeReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.remote.read_at(self.pos, buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for RangeReader {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let pos = match to {
            SeekFrom::Start(p) => Some(p),
            SeekFrom::End(d) => self.remote.size.checked_add_signed(d),
            SeekFrom::Current(d) => self.pos.checked_add_signed(d),
        };
        self.pos = pos.ok_or_else(|| io::Error::other("seek before the start of the file"))?;
        Ok(self.pos)
    }
}
