//! Reads vector and raster geospatial files and checks them against GDAL (pyogrio, rasterio).
//! The fixtures and reference come from `scripts/reference/make-geo-fixtures.py`.

use std::path::PathBuf;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};
use dv_data::array;
use dv_data::geo::{field_crs, wkb_vertices};
use dv_data::table::{Format, Table};
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn reference() -> Json {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("reference/geo.json")).unwrap())
        .unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn check_prop(batch: &RecordBatch, row: usize, name: &str, want: &Json, what: &str) {
    let col = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("{what}: no column {name}"));
    match want {
        Json::Null => assert!(col.is_null(row), "{what}: {name} should be null"),
        Json::String(s) => {
            let t = cast(col, &DataType::Utf8).unwrap();
            assert_eq!(t.as_string::<i32>().value(row), s, "{what}: {name}");
        }
        Json::Bool(b) => {
            // Shapefiles may store booleans as 0/1 integers.
            let t = cast(col, &DataType::Boolean).unwrap();
            assert_eq!(t.as_boolean().value(row), *b, "{what}: {name}");
        }
        Json::Number(n) => {
            let t = cast(col, &DataType::Float64).unwrap();
            let v = t.as_primitive::<Float64Type>().value(row);
            assert!(
                close(v, n.as_f64().unwrap()),
                "{what}: {name} {v} vs GDAL {n}"
            );
        }
        other => panic!("{what}: unexpected {other}"),
    }
}

#[test]
fn vector_formats_match_gdal() {
    let r = reference();
    for (key, file, format) in [
        ("mixed.geojson", "mixed.geojson", None),
        ("polys.shp", "polys.shp", None),
        ("polys-3857.fgb", "polys-3857.fgb", None),
        (
            "layers.gpkg#polys",
            "layers.gpkg",
            Some(Format::GeoPackage {
                layer: Some("polys".into()),
            }),
        ),
        (
            "layers.gpkg#points",
            "layers.gpkg",
            Some(Format::GeoPackage {
                layer: Some("points".into()),
            }),
        ),
        ("polys.parquet", "polys.parquet", None),
    ] {
        let path = fixtures().join("geo").join(file);
        let table = match format {
            Some(f) => Table::open_as(&path, f).unwrap(),
            None => Table::open(&path).unwrap(),
        };
        let batch = table.read_all().unwrap_or_else(|e| panic!("{key}: {e}"));
        let want = &r[key];
        let features = want["features"].as_array().unwrap();
        assert_eq!(batch.num_rows(), features.len(), "{key}: rows");

        let gi = batch.schema().fields().iter().position(|f| {
            f.metadata().get("ARROW:extension:name").map(String::as_str) == Some("geoarrow.wkb")
        });
        let gi = gi.unwrap_or_else(|| panic!("{key}: no GeoArrow column"));
        let field = batch.schema().field(gi).clone();
        assert_eq!(
            field_crs(&field),
            Some(format!("EPSG:{}", want["crs_epsg"])),
            "{key}: CRS"
        );
        let geoms = cast(batch.column(gi), &DataType::Binary).unwrap();
        let geoms = geoms.as_binary::<i32>();

        for (i, f) in features.iter().enumerate() {
            let what = format!("{key} feature {i}");
            for (name, v) in f["props"].as_object().unwrap() {
                check_prop(&batch, i, name, v, &what);
            }
            let (kind, mut coords) = wkb_vertices(geoms.value(i)).unwrap();
            assert_eq!(kind, f["type"].as_str().unwrap(), "{what}: type");
            coords.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
            let want_coords = f["coords"].as_array().unwrap();
            assert_eq!(coords.len(), want_coords.len(), "{what}: vertices");
            for (c, w) in coords.iter().zip(want_coords) {
                let (wx, wy) = (w[0].as_f64().unwrap(), w[1].as_f64().unwrap());
                assert!(
                    close(c[0], wx) && close(c[1], wy),
                    "{what}: {c:?} vs GDAL [{wx}, {wy}]"
                );
            }
        }
    }
}

#[test]
fn geopackages_with_several_layers_need_one_named() {
    let err = Table::open(fixtures().join("geo/layers.gpkg")).unwrap_err();
    assert!(err.to_string().contains("points, polys"), "{err}");
}

#[test]
fn rasters_match_rasterio() {
    let r = reference();
    for file in ["dem-utm33.tif", "class-4326.tif"] {
        let want = &r[file];
        let src = array::open(fixtures().join("geo").join(file)).unwrap();
        let crs = format!("EPSG:{}", want["crs_epsg"]);
        assert_eq!(
            src.attrs()["crs"].as_text(),
            Some(crs.as_str()),
            "{file}: CRS"
        );
        let band = src.variable("band_1").unwrap();
        let shape: Vec<u64> = want["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        assert_eq!(band.shape, shape, "{file}: shape");
        for axis in ["x", "y"] {
            let got = array::read_all(src.as_ref(), axis).unwrap();
            let w: Vec<f64> = want[axis]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            assert_eq!(got.len(), w.len());
            assert!(
                got.iter().zip(&w).all(|(a, b)| close(*a, *b)),
                "{file}: {axis} coordinates"
            );
        }
        let got = array::read_all(src.as_ref(), "band_1").unwrap();
        for (i, (g, w)) in got
            .iter()
            .zip(want["values"].as_array().unwrap())
            .enumerate()
        {
            match w.as_f64() {
                None => assert!(g.is_nan(), "{file}[{i}]: {g}, expected nodata"),
                Some(w) => assert!(close(*g, w), "{file}[{i}]: {g} vs GDAL {w}"),
            }
        }
    }
}
