//! Runs formulas on the trips fixture through DataFusion and checks every value against polars.
//! The reference comes from `scripts/reference/formulas.py`.

use std::path::PathBuf;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::compute::{cast, concat_batches, sort_to_indices, take_record_batch};
use arrow::datatypes::{DataType, Float64Type};
use datafusion::prelude::ident;
use dv_compute::{Context, compile};
use dv_data::budget::Budget;
use dv_data::table::Table;
use dv_model::field::{Dataset, Family};
use dv_query::Engine;
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

#[tokio::test]
async fn formulas_match_polars() {
    let table = Table::open(fixtures().join("tables/trips.csv")).unwrap();
    let dataset = Dataset::from_schema("trips", Family::Table, table.schema());
    let ctx = Context {
        dataset: &dataset,
        order_by: Some("id"),
    };
    let engine = Engine::new(Budget::new(1 << 30)).unwrap();
    engine.register("trips", &table).await.unwrap();
    let reference: serde_json::Map<String, Json> = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("reference/formulas.json")).unwrap(),
    )
    .unwrap();
    assert!(reference.len() >= 15);

    for (src, want) in &reference {
        let formula = compile(src, &ctx).unwrap_or_else(|e| panic!("{src}: {e}"));
        let df = engine
            .table("trips")
            .await
            .unwrap()
            .select(vec![ident("id"), formula.expr.alias("v")])
            .unwrap();
        let batches = df.collect().await.unwrap();
        let b = concat_batches(&batches[0].schema(), &batches).unwrap();
        let order = sort_to_indices(b.column(0), None, None).unwrap();
        let b: RecordBatch = take_record_batch(&b, &order).unwrap();
        let got = b.column(1);
        let want = want.as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{src}: rows");
        for (i, w) in want.iter().enumerate() {
            let at = format!("{src}, id {}", i + 1);
            match w {
                Json::Null => assert!(got.is_null(i), "{at}: expected null"),
                Json::Bool(x) => {
                    assert_eq!(got.as_boolean().value(i), *x, "{at}");
                    assert!(!got.is_null(i), "{at}: unexpected null");
                }
                Json::String(s) => {
                    assert!(!got.is_null(i), "{at}: unexpected null");
                    assert_eq!(got.as_string::<i32>().value(i), s, "{at}");
                }
                Json::Number(n) => {
                    assert!(!got.is_null(i), "{at}: unexpected null");
                    let v = cast(got, &DataType::Float64).unwrap();
                    let v = v.as_primitive::<Float64Type>().value(i);
                    let x = n.as_f64().unwrap();
                    assert!(
                        (v - x).abs() <= 1e-9 * x.abs().max(1.0),
                        "{at}: {v} vs polars {x}"
                    );
                }
                other => panic!("{at}: unexpected {other}"),
            }
        }
    }
}
