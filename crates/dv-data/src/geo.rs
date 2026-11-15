//! Vector geospatial files as tables: GeoJSON, FlatGeobuf, Shapefile, GeoPackage and GeoParquet.
//!
//! Each opens as a [`Table`](crate::table::Table) whose geometry is a WKB column marked as
//! GeoArrow (`geoarrow.wkb`) with its CRS, so roles, queries and joins treat it like any other
//! table. Properties are typed the way JSON properties are. Geometries are 2D; Z and M values
//! are dropped. Formats other than GeoParquet are read whole into memory.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryBuilder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use serde_json::{Map, Value};

use crate::{Error, Result};

/// The name of the geometry column the vector readers add.
pub const GEOMETRY: &str = "geometry";

fn bad(msg: impl Into<String>) -> Error {
    Error::Unsupported(msg.into())
}

/// The field for a WKB geometry column, marked as GeoArrow with its CRS (such as `EPSG:4326`).
pub fn geometry_field(name: &str, crs: Option<&str>) -> Field {
    let mut meta = HashMap::from([(
        "ARROW:extension:name".to_string(),
        "geoarrow.wkb".to_string(),
    )]);
    let ext = match crs {
        Some(c) => serde_json::json!({"crs": c, "crs_type": "authority_code"}).to_string(),
        None => "{}".to_string(),
    };
    meta.insert("ARROW:extension:metadata".into(), ext);
    Field::new(name, DataType::Binary, true).with_metadata(meta)
}

/// The CRS recorded on a GeoArrow field, if any.
pub fn field_crs(field: &Field) -> Option<String> {
    let ext = field.metadata().get("ARROW:extension:metadata")?;
    let v: Value = serde_json::from_str(ext).ok()?;
    v.get("crs")?.as_str().map(str::to_string)
}

// ---------------------------------------------------------------------------------------------
// WKB

const POINT: u32 = 1;
const LINESTRING: u32 = 2;
const POLYGON: u32 = 3;
const MULTIPOINT: u32 = 4;
const MULTILINESTRING: u32 = 5;
const MULTIPOLYGON: u32 = 6;
const COLLECTION: u32 = 7;

struct WkbWriter(Vec<u8>);

impl WkbWriter {
    fn header(&mut self, kind: u32) {
        self.0.push(1); // little-endian
        self.0.extend_from_slice(&kind.to_le_bytes());
    }

    fn count(&mut self, n: usize) {
        self.0.extend_from_slice(&(n as u32).to_le_bytes());
    }

    fn xy(&mut self, x: f64, y: f64) {
        self.0.extend_from_slice(&x.to_le_bytes());
        self.0.extend_from_slice(&y.to_le_bytes());
    }
}

fn position(v: &Value) -> Result<(f64, f64)> {
    let a = v
        .as_array()
        .filter(|a| a.len() >= 2)
        .ok_or_else(|| bad("a GeoJSON position needs x and y"))?;
    match (a[0].as_f64(), a[1].as_f64()) {
        (Some(x), Some(y)) => Ok((x, y)),
        _ => Err(bad("GeoJSON positions must be numbers")),
    }
}

fn positions(w: &mut WkbWriter, v: &Value) -> Result<()> {
    let a = v
        .as_array()
        .ok_or_else(|| bad("expected a list of positions"))?;
    w.count(a.len());
    for p in a {
        let (x, y) = position(p)?;
        w.xy(x, y);
    }
    Ok(())
}

fn rings(w: &mut WkbWriter, v: &Value) -> Result<()> {
    let a = v
        .as_array()
        .ok_or_else(|| bad("expected a list of rings"))?;
    w.count(a.len());
    a.iter().try_for_each(|r| positions(w, r))
}

/// A GeoJSON geometry object as 2D WKB.
pub fn geojson_to_wkb(g: &Value) -> Result<Vec<u8>> {
    let mut w = WkbWriter(Vec::new());
    write_geojson(&mut w, g)?;
    Ok(w.0)
}

fn write_geojson(w: &mut WkbWriter, g: &Value) -> Result<()> {
    let kind = g.get("type").and_then(Value::as_str).unwrap_or_default();
    let coords = g.get("coordinates").unwrap_or(&Value::Null);
    let list = || {
        coords
            .as_array()
            .ok_or_else(|| bad(format!("{kind} needs coordinates")))
    };
    match kind {
        "Point" => {
            w.header(POINT);
            let (x, y) = position(coords)?;
            w.xy(x, y);
        }
        "LineString" => {
            w.header(LINESTRING);
            positions(w, coords)?;
        }
        "Polygon" => {
            w.header(POLYGON);
            rings(w, coords)?;
        }
        "MultiPoint" => {
            w.header(MULTIPOINT);
            let a = list()?;
            w.count(a.len());
            for p in a {
                w.header(POINT);
                let (x, y) = position(p)?;
                w.xy(x, y);
            }
        }
        "MultiLineString" => {
            w.header(MULTILINESTRING);
            let a = list()?;
            w.count(a.len());
            for l in a {
                w.header(LINESTRING);
                positions(w, l)?;
            }
        }
        "MultiPolygon" => {
            w.header(MULTIPOLYGON);
            let a = list()?;
            w.count(a.len());
            for p in a {
                w.header(POLYGON);
                rings(w, p)?;
            }
        }
        "GeometryCollection" => {
            w.header(COLLECTION);
            let a = g
                .get("geometries")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("GeometryCollection needs geometries"))?;
            w.count(a.len());
            a.iter().try_for_each(|x| write_geojson(w, x))?;
        }
        other => return Err(bad(format!("unknown GeoJSON geometry type {other:?}"))),
    }
    Ok(())
}

/// A geometry's type name (`Polygon`, `MultiPolygon`…) and every vertex, in order, read from
/// WKB or EWKB of any dimension (only x and y are kept).
pub fn wkb_vertices(wkb: &[u8]) -> Result<(&'static str, Vec<[f64; 2]>)> {
    let mut r = WkbReader { b: wkb, pos: 0 };
    let mut out = Vec::new();
    let kind = r.geometry(&mut out)?;
    let name = match kind {
        POINT => "Point",
        LINESTRING => "LineString",
        POLYGON => "Polygon",
        MULTIPOINT => "MultiPoint",
        MULTILINESTRING => "MultiLineString",
        MULTIPOLYGON => "MultiPolygon",
        _ => "GeometryCollection",
    };
    Ok((name, out))
}

struct WkbReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl WkbReader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let s = self
            .b
            .get(self.pos..self.pos + N)
            .ok_or_else(|| bad("truncated WKB"))?;
        self.pos += N;
        Ok(s.try_into().expect("length checked"))
    }

    fn geometry(&mut self, out: &mut Vec<[f64; 2]>) -> Result<u32> {
        let le = self.take::<1>()?[0] == 1;
        let u32_ = |b: [u8; 4]| {
            if le {
                u32::from_le_bytes(b)
            } else {
                u32::from_be_bytes(b)
            }
        };
        let raw = u32_(self.take::<4>()?);
        // EWKB flags in the high bits; ISO dimensions as thousands.
        let (has_z, has_m, has_srid) = (
            raw & 0x8000_0000 != 0,
            raw & 0x4000_0000 != 0,
            raw & 0x2000_0000 != 0,
        );
        let base = raw & 0x0fff_ffff;
        let (kind, iso) = (base % 1000, base / 1000);
        let dims = 2
            + usize::from(has_z || iso == 1 || iso == 3)
            + usize::from(has_m || iso == 2 || iso == 3);
        if has_srid {
            self.take::<4>()?;
        }
        let f64_ = |b: [u8; 8]| {
            if le {
                f64::from_le_bytes(b)
            } else {
                f64::from_be_bytes(b)
            }
        };
        let point = |r: &mut Self, out: &mut Vec<[f64; 2]>| -> Result<()> {
            let x = f64_(r.take::<8>()?);
            let y = f64_(r.take::<8>()?);
            for _ in 2..dims {
                r.take::<8>()?;
            }
            out.push([x, y]);
            Ok(())
        };
        match kind {
            POINT => point(self, out)?,
            LINESTRING => {
                for _ in 0..u32_(self.take::<4>()?) {
                    point(self, out)?;
                }
            }
            POLYGON => {
                for _ in 0..u32_(self.take::<4>()?) {
                    for _ in 0..u32_(self.take::<4>()?) {
                        point(self, out)?;
                    }
                }
            }
            MULTIPOINT | MULTILINESTRING | MULTIPOLYGON | COLLECTION => {
                for _ in 0..u32_(self.take::<4>()?) {
                    self.geometry(out)?;
                }
            }
            other => return Err(bad(format!("unsupported WKB geometry type {other}"))),
        }
        Ok(kind)
    }
}

// ---------------------------------------------------------------------------------------------
// Building tables

/// Properties as JSON objects, typed as JSON is, plus a geometry column.
fn build(
    props: Vec<Map<String, Value>>,
    geoms: Vec<Option<Vec<u8>>>,
    crs: Option<&str>,
) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    let values: Vec<Value> = props.into_iter().map(Value::Object).collect();
    let prop_schema =
        arrow::json::reader::infer_json_schema_from_iterator(values.iter().map(|v| Ok(v.clone())))?;
    let prop_schema = Arc::new(prop_schema);
    let mut decoder = arrow::json::ReaderBuilder::new(prop_schema.clone())
        .with_batch_size(values.len().max(1))
        .build_decoder()?;
    decoder.serialize(&values)?;
    let props = decoder
        .flush()?
        .unwrap_or_else(|| RecordBatch::new_empty(prop_schema.clone()));

    let mut b = BinaryBuilder::new();
    for g in &geoms {
        b.append_option(g.as_deref());
    }
    let mut fields: Vec<Field> = prop_schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let mut columns: Vec<ArrayRef> = props.columns().to_vec();
    let name = if fields.iter().any(|f| f.name() == GEOMETRY) {
        "geometry_"
    } else {
        GEOMETRY
    };
    fields.push(geometry_field(name, crs));
    columns.push(Arc::new(b.finish()));
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    Ok((schema, vec![batch]))
}

/// A GeoJSON FeatureCollection (or a single Feature), always in WGS 84 (RFC 7946).
pub(crate) fn read_geojson(path: &Path) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    let v: Value = serde_json::from_reader(BufReader::new(File::open(path)?))
        .map_err(|e| bad(format!("{path:?} isn't GeoJSON: {e}")))?;
    let features = match v.get("type").and_then(Value::as_str) {
        Some("FeatureCollection") => v["features"].as_array().cloned().unwrap_or_default(),
        Some("Feature") => vec![v.clone()],
        _ => {
            return Err(bad(format!(
                "{path:?} isn't a GeoJSON FeatureCollection or Feature"
            )));
        }
    };
    let mut props = Vec::with_capacity(features.len());
    let mut geoms = Vec::with_capacity(features.len());
    for f in &features {
        props.push(
            f.get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default(),
        );
        geoms.push(match f.get("geometry") {
            Some(Value::Null) | None => None,
            Some(g) => Some(geojson_to_wkb(g)?),
        });
    }
    build(props, geoms, Some("EPSG:4326"))
}

/// A FlatGeobuf file. Its CRS comes from the header's organization and code.
pub(crate) fn read_flatgeobuf(path: &Path) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    use flatgeobuf::FallibleStreamingIterator;
    use geozero::{ColumnValue, CoordDimensions, FeatureProperties, PropertyProcessor, ToWkb};

    struct Props(Map<String, Value>);
    impl PropertyProcessor for Props {
        fn property(
            &mut self,
            _i: usize,
            name: &str,
            v: &ColumnValue,
        ) -> geozero::error::Result<bool> {
            let value = match v {
                ColumnValue::Byte(x) => Value::from(*x),
                ColumnValue::UByte(x) => Value::from(*x),
                ColumnValue::Bool(x) => Value::from(*x),
                ColumnValue::Short(x) => Value::from(*x),
                ColumnValue::UShort(x) => Value::from(*x),
                ColumnValue::Int(x) => Value::from(*x),
                ColumnValue::UInt(x) => Value::from(*x),
                ColumnValue::Long(x) => Value::from(*x),
                ColumnValue::ULong(x) => Value::from(*x),
                ColumnValue::Float(x) => Value::from(*x),
                ColumnValue::Double(x) => Value::from(*x),
                ColumnValue::String(s) | ColumnValue::Json(s) | ColumnValue::DateTime(s) => {
                    Value::from(*s)
                }
                ColumnValue::Binary(_) => Value::Null,
            };
            self.0.insert(name.to_string(), value);
            Ok(false)
        }
    }

    let err = |e: flatgeobuf::Error| bad(format!("FlatGeobuf: {e}"));
    let reader = flatgeobuf::FgbReader::open(BufReader::new(File::open(path)?)).map_err(err)?;
    let header = reader.header();
    let crs = header.crs().and_then(|c| {
        let org = c.org().unwrap_or("EPSG");
        (c.code() > 0).then(|| format!("{org}:{}", c.code()))
    });
    let columns: Vec<String> = header
        .columns()
        .map(|cs| cs.iter().map(|c| c.name().to_string()).collect())
        .unwrap_or_default();
    let mut it = reader.select_all().map_err(err)?;
    let mut props = Vec::new();
    let mut geoms = Vec::new();
    while let Some(f) = it.next().map_err(err)? {
        let mut p = Props(Map::new());
        f.process_properties(&mut p)
            .map_err(|e| bad(format!("FlatGeobuf: {e}")))?;
        // Absent properties are nulls, so every column appears in the schema.
        for c in &columns {
            p.0.entry(c.clone()).or_insert(Value::Null);
        }
        props.push(p.0);
        geoms.push(f.to_wkb(CoordDimensions::xy()).ok());
    }
    build(props, geoms, crs.as_deref())
}

/// A Shapefile (`.shp` with its `.dbf`, and `.prj` for the CRS when it names an EPSG code).
///
/// Rings are grouped as GDAL groups them: each outer ring starts a polygon and inner rings join
/// the polygon before them; a shape with several outer rings is a MultiPolygon.
pub(crate) fn read_shapefile(path: &Path) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    use shapefile::dbase::FieldValue;
    use shapefile::{PolygonRing, Shape};

    let err = |e: shapefile::Error| bad(format!("Shapefile: {e}"));
    // Records are maps; columns follow the DBF's field order instead.
    let order: Vec<String> = shapefile::dbase::Reader::from_path(path.with_extension("dbf"))
        .map_err(|e| bad(format!("Shapefile: {e}")))?
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    let mut reader = shapefile::Reader::from_path(path).map_err(err)?;
    let mut props = Vec::new();
    let mut geoms = Vec::new();
    for item in reader.iter_shapes_and_records() {
        let (shape, record) = item.map_err(err)?;
        let mut w = WkbWriter(Vec::new());
        let pts = |w: &mut WkbWriter, ps: &[(f64, f64)]| {
            w.count(ps.len());
            for (x, y) in ps {
                w.xy(*x, *y);
            }
        };
        let geom = match shape {
            Shape::NullShape => None,
            Shape::Point(p) => {
                w.header(POINT);
                w.xy(p.x, p.y);
                Some(w.0)
            }
            Shape::Multipoint(m) => {
                w.header(MULTIPOINT);
                w.count(m.points().len());
                for p in m.points() {
                    w.header(POINT);
                    w.xy(p.x, p.y);
                }
                Some(w.0)
            }
            Shape::Polyline(l) => {
                let parts: Vec<Vec<(f64, f64)>> = l
                    .parts()
                    .iter()
                    .map(|p| p.iter().map(|q| (q.x, q.y)).collect())
                    .collect();
                if parts.len() == 1 {
                    w.header(LINESTRING);
                    pts(&mut w, &parts[0]);
                } else {
                    w.header(MULTILINESTRING);
                    w.count(parts.len());
                    for p in &parts {
                        w.header(LINESTRING);
                        pts(&mut w, p);
                    }
                }
                Some(w.0)
            }
            Shape::Polygon(poly) => {
                let mut polys: Vec<Vec<Vec<(f64, f64)>>> = Vec::new();
                for ring in poly.rings() {
                    let (outer, ps) = match ring {
                        PolygonRing::Outer(ps) => (true, ps),
                        PolygonRing::Inner(ps) => (false, ps),
                    };
                    let ps: Vec<(f64, f64)> = ps.iter().map(|q| (q.x, q.y)).collect();
                    match polys.last_mut() {
                        Some(last) if !outer => last.push(ps),
                        _ => polys.push(vec![ps]),
                    }
                }
                let poly_wkb = |w: &mut WkbWriter, p: &Vec<Vec<(f64, f64)>>| {
                    w.header(POLYGON);
                    w.count(p.len());
                    for r in p {
                        pts(w, r);
                    }
                };
                if polys.len() == 1 {
                    poly_wkb(&mut w, &polys[0]);
                } else {
                    w.header(MULTIPOLYGON);
                    w.count(polys.len());
                    for p in &polys {
                        poly_wkb(&mut w, p);
                    }
                }
                Some(w.0)
            }
            other => {
                return Err(bad(format!(
                    "unsupported Shapefile shape {}",
                    other.shapetype()
                )));
            }
        };
        let mut fields: HashMap<String, Value> = HashMap::new();
        for (name, value) in record {
            let v = match value {
                FieldValue::Character(s) => s.map(Value::from).unwrap_or(Value::Null),
                FieldValue::Numeric(n) => n.map(num).unwrap_or(Value::Null),
                FieldValue::Logical(b) => b.map(Value::from).unwrap_or(Value::Null),
                FieldValue::Float(f) => f.map(|x| num(x as f64)).unwrap_or(Value::Null),
                FieldValue::Integer(i) => Value::from(i),
                FieldValue::Double(d) | FieldValue::Currency(d) => num(d),
                FieldValue::Date(d) => d
                    .map(|d| {
                        Value::from(format!("{:04}-{:02}-{:02}", d.year(), d.month(), d.day()))
                    })
                    .unwrap_or(Value::Null),
                FieldValue::Memo(s) => Value::from(s),
                _ => Value::Null,
            };
            fields.insert(name, v);
        }
        let p: Map<String, Value> = order
            .iter()
            .map(|n| (n.clone(), fields.remove(n).unwrap_or(Value::Null)))
            .collect();
        props.push(p);
        geoms.push(geom);
    }
    let crs = std::fs::read_to_string(path.with_extension("prj"))
        .ok()
        .and_then(|wkt| epsg_from_wkt(&wkt));
    build(props, geoms, crs.as_deref())
}

/// DBF numbers are decimal text; whole ones read as integers, as GDAL types integer fields.
fn num(x: f64) -> Value {
    if x.fract() == 0.0 && x.abs() < 9.0e15 {
        Value::from(x as i64)
    } else {
        Value::from(x)
    }
}

/// The EPSG code a `.prj` WKT names, for the common geographic and Web Mercator cases. Other
/// projections are left without a CRS rather than guessed.
fn epsg_from_wkt(wkt: &str) -> Option<String> {
    if let Some(i) = wkt.rfind("AUTHORITY[\"EPSG\",\"") {
        let rest = &wkt[i + 18..];
        return rest.split('"').next().map(|c| format!("EPSG:{c}"));
    }
    let w = wkt.to_ascii_uppercase();
    if w.starts_with("GEOGCS[\"GCS_WGS_1984\"") || w.starts_with("GEOGCS[\"WGS 84\"") {
        return Some("EPSG:4326".into());
    }
    None
}

/// The GeoPackage layers (feature tables) in a file.
pub fn gpkg_layers(path: &Path) -> Result<Vec<String>> {
    let conn =
        rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|e| bad(format!("GeoPackage: {e}")))?;
    let mut stmt = conn
        .prepare("select table_name from gpkg_geometry_columns order by table_name")
        .map_err(|e| bad(format!("GeoPackage: {e}")))?;
    let names = stmt
        .query_map([], |r| r.get(0))
        .map_err(|e| bad(format!("GeoPackage: {e}")))?
        .collect::<rusqlite::Result<Vec<String>>>()
        .map_err(|e| bad(format!("GeoPackage: {e}")))?;
    Ok(names)
}

/// One GeoPackage layer: its SQLite table, with the geometry blobs turned into WKB.
pub(crate) fn read_gpkg(path: &Path, layer: Option<&str>) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    let layers = gpkg_layers(path)?;
    let layer = match (layer, layers.as_slice()) {
        (Some(l), _) => l.to_string(),
        (None, [one]) => one.clone(),
        (None, []) => return Err(bad(format!("{path:?} has no feature layers"))),
        (None, many) => {
            return Err(bad(format!(
                "{path:?} has {} layers; name one of {}",
                many.len(),
                many.join(", ")
            )));
        }
    };
    let conn =
        rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|e| bad(format!("GeoPackage: {e}")))?;
    let (column, srs): (String, i64) = conn
        .query_row(
            "select column_name, srs_id from gpkg_geometry_columns where table_name = ?1",
            [&layer],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(|e| bad(format!("GeoPackage layer {layer:?}: {e}")))?;
    let crs: Option<String> = conn
        .query_row(
            "select organization, organization_coordsys_id from gpkg_spatial_ref_sys where srs_id = ?1",
            [srs],
            |r| Ok(format!("{}:{}", r.get::<_, String>(0)?.to_ascii_uppercase(), r.get::<_, i64>(1)?)),
        )
        .ok();
    let schema = crate::sqlite::schema(path, Some(&layer))?;
    let batches = crate::sqlite::read(path, Some(&layer), schema.clone())?;
    let gi = schema.index_of(&column)?;
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields[gi] = geometry_field(&column, crs.as_deref());
    let out_schema = Arc::new(Schema::new(fields));
    let mut out = Vec::new();
    for b in batches {
        let mut cols = b.columns().to_vec();
        let mut wkb = BinaryBuilder::new();
        let blobs = arrow::compute::cast(&cols[gi], &DataType::Binary)?;
        let blobs = arrow::array::AsArray::as_binary::<i32>(&blobs);
        for blob in blobs.iter() {
            wkb.append_option(blob.map(gpkg_to_wkb).transpose()?);
        }
        cols[gi] = Arc::new(wkb.finish());
        out.push(RecordBatch::try_new(out_schema.clone(), cols)?);
    }
    Ok((out_schema, out))
}

/// Strips a GeoPackage geometry blob's header, leaving standard WKB.
fn gpkg_to_wkb(blob: &[u8]) -> Result<&[u8]> {
    if blob.len() < 8 || &blob[..2] != b"GP" {
        return Err(bad("not a GeoPackage geometry blob"));
    }
    let envelope = match (blob[3] >> 1) & 0b111 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        other => return Err(bad(format!("bad GeoPackage envelope code {other}"))),
    };
    blob.get(8 + envelope..)
        .ok_or_else(|| bad("truncated GeoPackage geometry"))
}

/// Marks GeoParquet geometry columns (from the file's `geo` metadata) as GeoArrow, with their
/// CRS. Columns without a stated CRS are OGC:CRS84, as the GeoParquet spec says.
pub(crate) fn annotate_geoparquet(schema: &Schema, kv: Option<&String>) -> SchemaRef {
    let Some(geo) = kv.and_then(|s| serde_json::from_str::<Value>(s).ok()) else {
        return Arc::new(schema.clone());
    };
    let columns = geo.get("columns").and_then(Value::as_object);
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            let meta = columns.and_then(|c| c.get(f.name()));
            match meta {
                Some(m) if m.get("encoding").and_then(Value::as_str) == Some("WKB") => {
                    let crs = match m.get("crs") {
                        None => Some("OGC:CRS84".to_string()),
                        Some(Value::Null) => None,
                        Some(c) => c.get("id").and_then(|id| {
                            Some(format!(
                                "{}:{}",
                                id.get("authority")?.as_str()?,
                                id.get("code")?
                            ))
                        }),
                    };
                    geometry_field(f.name(), crs.as_deref())
                }
                _ => f.as_ref().clone(),
            }
        })
        .collect();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geojson_round_trips_through_wkb() {
        let g: Value = serde_json::json!({
            "type": "MultiPolygon",
            "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 0]]], [[[5, 5], [6, 5], [6, 6], [5, 5]]]]
        });
        let wkb = geojson_to_wkb(&g).unwrap();
        let (kind, v) = wkb_vertices(&wkb).unwrap();
        assert_eq!(kind, "MultiPolygon");
        assert_eq!(v.len(), 8);
        assert_eq!(v[4], [5.0, 5.0]);
    }

    #[test]
    fn reads_wkt_authorities() {
        assert_eq!(
            epsg_from_wkt("PROJCS[\"x\",AUTHORITY[\"EPSG\",\"3035\"]]").as_deref(),
            Some("EPSG:3035")
        );
        assert_eq!(
            epsg_from_wkt("GEOGCS[\"GCS_WGS_1984\",DATUM[]]").as_deref(),
            Some("EPSG:4326")
        );
    }
}
