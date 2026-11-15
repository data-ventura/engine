//! Checks selection masks against DuckDB row counts. The reference comes from
//! `scripts/reference/selection.py`.

use std::path::PathBuf;

use dv_data::parquet::ParquetTable;
use dv_model::selection::{Predicate, Selection};
use serde_json::Value as Json;

#[test]
fn taxi_selections_match_duckdb() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let table = ParquetTable::open(fixtures.join("taxi-20k.parquet")).unwrap();
    let batch = table.sample(usize::MAX).unwrap().batch;
    let cases: Vec<Json> = serde_json::from_str(
        &std::fs::read_to_string(fixtures.join("reference/taxi-20k.selection.json")).unwrap(),
    )
    .unwrap();
    for c in cases {
        let predicates: Vec<Predicate> = serde_json::from_value(c["predicates"].clone()).unwrap();
        let selection = Selection {
            predicates,
            generation: 0,
        };
        let rows = selection.mask(&batch).unwrap().true_count() as u64;
        assert_eq!(rows, c["rows"].as_u64().unwrap(), "{}", c["sql"]);
    }
}
