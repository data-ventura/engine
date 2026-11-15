//! Geospatial tables query like any other: GeoParquet scans lazily with its geometry marked, and
//! the in-memory vector formats register too. Totals come from the GDAL reference in
//! `fixtures/reference/geo.json`.

use std::path::PathBuf;

use arrow::array::AsArray;
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};
use dv_data::budget::Budget;
use dv_data::table::Table;
use dv_query::Engine;
use serde_json::Value as Json;

#[tokio::test]
async fn geo_tables_query_like_tables() {
    let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let r: Json =
        serde_json::from_str(&std::fs::read_to_string(fx.join("reference/geo.json")).unwrap())
            .unwrap();
    let e = Engine::new(Budget::new(1 << 30)).unwrap();
    for (name, file, key) in [
        ("gp", "geo/polys.parquet", "polys.parquet"),
        ("fgb", "geo/polys-3857.fgb", "polys-3857.fgb"),
        ("shp", "geo/polys.shp", "polys.shp"),
    ] {
        e.register(name, &Table::open(fx.join(file)).unwrap())
            .await
            .unwrap();
        let want: f64 = r[key]["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["props"]["pop"].as_f64().unwrap())
            .sum();
        let df = e
            .sql(&format!(
                "select count(*), sum(pop), count(geometry) from {name}"
            ))
            .await
            .unwrap();
        let b = &df.collect().await.unwrap()[0];
        let n = |i: usize| {
            let c = cast(b.column(i), &DataType::Float64).unwrap();
            c.as_primitive::<Float64Type>().value(0)
        };
        assert_eq!(n(0), 12.0, "{name}: rows");
        assert_eq!(n(1), want, "{name}: total population");
        assert_eq!(n(2), 12.0, "{name}: geometries");
    }
}
