//! Reads one small table in every table format and checks it against DuckDB reading the same file
//! (pandas with openpyxl for Excel, pyarrow for Arrow). The fixtures and reference come from
//! `scripts/reference/make-table-fixtures.py`.

use std::collections::HashSet;
use std::path::PathBuf;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type, Int64Type, TimeUnit};
use dv_data::table::{Format, Table};
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn reference() -> Json {
    serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("reference/tables.json")).unwrap(),
    )
    .unwrap()
}

fn f64s(a: &ArrayRef) -> Vec<f64> {
    let a = cast(a, &DataType::Float64).unwrap();
    a.as_primitive::<Float64Type>().iter().flatten().collect()
}

fn micros(a: &ArrayRef) -> Vec<i64> {
    let a = cast(a, &DataType::Timestamp(TimeUnit::Microsecond, None)).unwrap();
    let a = cast(&a, &DataType::Int64).unwrap();
    a.as_primitive::<Int64Type>().iter().flatten().collect()
}

fn check(name: &str, batch: &RecordBatch, expected: &Json) {
    assert_eq!(
        batch.num_rows() as u64,
        expected["rows"].as_u64().unwrap(),
        "{name}: rows"
    );
    for (col, want) in expected["columns"].as_object().unwrap() {
        let a = batch
            .column_by_name(col)
            .unwrap_or_else(|| panic!("{name}: no column {col}"));
        let what = format!("{name}.{col} ({})", a.data_type());
        assert_eq!(
            a.null_count() as u64,
            want["nulls"].as_u64().unwrap(),
            "{what}: nulls"
        );
        if let Some(sum) = want.get("sum") {
            let v = f64s(a);
            let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * y.abs().max(1.0);
            assert!(close(v.iter().sum(), sum.as_f64().unwrap()), "{what}: sum");
            assert!(
                close(
                    v.iter().copied().fold(f64::INFINITY, f64::min),
                    want["min"].as_f64().unwrap()
                ),
                "{what}: min"
            );
            assert!(
                close(
                    v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    want["max"].as_f64().unwrap()
                ),
                "{what}: max"
            );
        } else if col == "pickup" {
            assert!(
                matches!(a.data_type(), DataType::Timestamp(..)),
                "{what}: a timestamp"
            );
            let v = micros(a);
            assert_eq!(v.iter().min().copied(), want["min"].as_i64(), "{what}: min");
            assert_eq!(v.iter().max().copied(), want["max"].as_i64(), "{what}: max");
        } else if let Some(t) = want.get("true") {
            assert_eq!(a.data_type(), &DataType::Boolean, "{what}");
            assert_eq!(
                a.as_boolean().true_count() as u64,
                t.as_u64().unwrap(),
                "{what}: true"
            );
        } else {
            let s = cast(a, &DataType::Utf8).unwrap();
            let distinct: HashSet<&str> = s.as_string::<i32>().iter().flatten().collect();
            assert_eq!(
                distinct.len() as u64,
                want["distinct"].as_u64().unwrap(),
                "{what}: distinct"
            );
        }
    }
}

#[test]
fn every_table_format_matches_its_reference() {
    let reference = reference();
    for (key, file) in [
        ("csv", "trips.csv"),
        ("tsv", "trips.tsv"),
        ("ndjson", "trips.ndjson"),
        ("json", "trips.json"),
        ("parquet", "trips.parquet"),
        ("arrow", "trips.arrow"),
        ("xlsx", "trips.xlsx"),
        ("sqlite", "trips.sqlite"),
    ] {
        let table = Table::open(fixtures().join("tables").join(file)).unwrap();
        let batch = table.read_all().unwrap();
        check(key, &batch, &reference[key]);
        let sample = table.sample(100).unwrap();
        assert_eq!(sample.batch.num_rows(), 100, "{key}: sample");
        assert_eq!(sample.of_rows, 500, "{key}: sample counts the table");
    }
}

#[test]
fn detects_formats_by_content() {
    let dir = std::env::temp_dir().join(format!("dv-detect-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (src, want) in [
        ("trips.parquet", Format::Parquet),
        ("trips.arrow", Format::Arrow),
        ("trips.sqlite", Format::Sqlite { table: None }),
        ("trips.json", Format::Json),
        ("trips.ndjson", Format::Ndjson),
    ] {
        let to = dir.join(src.replace('.', "_"));
        std::fs::copy(fixtures().join("tables").join(src), &to).unwrap();
        assert_eq!(Format::detect(&to).unwrap(), want, "{src}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn folders_and_globs_read_as_one_table() {
    let dir = std::env::temp_dir().join(format!("dv-glob-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for part in ["a", "b"] {
        std::fs::copy(
            fixtures().join("tables/trips.csv"),
            dir.join(format!("{part}.csv")),
        )
        .unwrap();
    }
    std::fs::write(dir.join("README.txt"), "not a table").unwrap();
    let folder = Table::open(&dir).unwrap();
    assert_eq!(folder.files().len(), 2);
    assert_eq!(folder.read_all().unwrap().num_rows(), 1000);
    let glob = Table::open(dir.join("a*.csv")).unwrap();
    assert_eq!(glob.read_all().unwrap().num_rows(), 500);
    std::fs::remove_dir_all(&dir).unwrap();
}
