//! Reads one small table in every table format and checks it against DuckDB reading the same file
//! (pandas with openpyxl for Excel, pyarrow for Arrow). The fixtures and reference come from
//! `scripts/reference/make-table-fixtures.py`, and for the DuckDB database from
//! `scripts/reference/make-duckdb-fixture.py`.

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
        ("trips.duckdb", Format::DuckDb { table: None }),
        ("trips.json", Format::Json),
        ("trips.ndjson", Format::Ndjson),
    ] {
        let to = dir.join(src.replace('.', "_"));
        std::fs::copy(fixtures().join("tables").join(src), &to).unwrap();
        assert_eq!(Format::detect(&to).unwrap(), want, "{src}");
    }
    // `.db` is used by both SQLite and DuckDB, so their contents decide.
    for (src, want) in [
        ("trips.sqlite", Format::Sqlite { table: None }),
        ("trips.duckdb", Format::DuckDb { table: None }),
    ] {
        let to = dir.join(format!("{}.db", src.replace('.', "_")));
        std::fs::copy(fixtures().join("tables").join(src), &to).unwrap();
        assert_eq!(Format::detect(&to).unwrap(), want, "{src} as .db");
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

#[cfg(feature = "duckdb")]
mod duckdb {
    use super::*;

    fn reference() -> Json {
        serde_json::from_str(
            &std::fs::read_to_string(fixtures().join("reference/duckdb.json")).unwrap(),
        )
        .unwrap()
    }

    fn db() -> PathBuf {
        fixtures().join("tables/trips.duckdb")
    }

    fn open(table: &str) -> Table {
        Table::open_as(
            db(),
            Format::DuckDb {
                table: Some(table.into()),
            },
        )
        .unwrap()
    }

    #[test]
    fn lists_tables_and_views() {
        let want: Vec<String> = reference()["tables"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_string())
            .collect();
        assert_eq!(dv_data::duckdb::tables(&db()).unwrap(), want);
        // Several tables, so one must be named.
        let err = Table::open(db()).unwrap_err().to_string();
        assert!(
            err.contains("name one of long_trips, trips, extra.types"),
            "{err}"
        );
        let err = Table::open_as(
            db(),
            Format::DuckDb {
                table: Some("missing".into()),
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no table \"missing\""), "{err}");
    }

    #[test]
    fn tables_and_views_match_duckdb() {
        let reference = reference();
        for name in ["trips", "long_trips"] {
            let table = open(name);
            let batch = table.read_all().unwrap();
            check(name, &batch, &reference[name]);
            for (col, want) in reference[name]["columns"].as_object().unwrap() {
                let f = table.schema().field_with_name(col).unwrap();
                let arrow = match f.data_type() {
                    DataType::Int64 => "int64",
                    DataType::Float64 => "double",
                    DataType::Utf8 => "string",
                    DataType::Boolean => "bool",
                    DataType::Timestamp(TimeUnit::Microsecond, None) => "timestamp[us]",
                    other => panic!("{name}.{col}: {other}"),
                };
                assert_eq!(arrow, want["arrow"], "{name}.{col}: type");
            }
        }
        let sample = open("trips").sample(100).unwrap();
        assert_eq!(sample.batch.num_rows(), 100);
        assert_eq!(sample.of_rows, 500);
    }

    #[test]
    fn duckdb_types_keep_their_arrow_types() {
        use arrow::array::{Decimal128Array, IntervalMonthDayNanoArray};
        use arrow::datatypes::{Int32Type, IntervalUnit};

        let want = &reference()["extra.types"];
        let table = open("extra.types");
        let batch = table.read_all().unwrap();
        assert_eq!(batch.num_rows() as u64, want["rows"].as_u64().unwrap());
        let cols = want["columns"].as_object().unwrap();
        for (col, w) in cols {
            let a = batch.column_by_name(col).unwrap();
            assert_eq!(
                a.null_count() as u64,
                w["nulls"].as_u64().unwrap(),
                "{col}: nulls"
            );
            // pyarrow's name for the type DuckDB exports.
            let arrow = match a.data_type() {
                DataType::Int64 => "int64".to_string(),
                DataType::Decimal128(p, s) => format!("decimal128({p}, {s})"),
                DataType::Date32 => "date32[day]".into(),
                DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) => {
                    format!("timestamp[us, tz={tz}]")
                }
                DataType::Dictionary(k, v) if **k == DataType::UInt8 && **v == DataType::Utf8 => {
                    "dictionary<values=string, indices=uint8, ordered=0>".into()
                }
                DataType::Interval(IntervalUnit::MonthDayNano) => "month_day_nano_interval".into(),
                DataType::Utf8 => "string".into(),
                DataType::Binary => "binary".into(),
                other => panic!("{col}: unexpected {other}"),
            };
            assert_eq!(arrow, w["arrow"], "{col}: type");
        }

        let close = |x: f64, y: &Json| (x - y.as_f64().unwrap()).abs() <= 1e-9 * x.abs().max(1.0);
        for col in ["id", "price", "big"] {
            let w = &cols[col];
            let a = batch.column_by_name(col).unwrap();
            let v: Vec<f64> = match a.data_type() {
                DataType::Decimal128(_, s) => {
                    let d = a.as_any().downcast_ref::<Decimal128Array>().unwrap();
                    d.iter()
                        .flatten()
                        .map(|x| x as f64 / 10f64.powi(i32::from(*s)))
                        .collect()
                }
                _ => f64s(a),
            };
            assert!(close(v.iter().sum(), &w["sum"]), "{col}: sum");
            assert!(
                close(v.iter().copied().fold(f64::INFINITY, f64::min), &w["min"]),
                "{col}: min"
            );
            assert!(
                close(
                    v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    &w["max"]
                ),
                "{col}: max"
            );
        }
        let stamp = micros(batch.column_by_name("stamp").unwrap());
        assert_eq!(stamp.iter().min().copied(), cols["stamp"]["min"].as_i64());
        assert_eq!(stamp.iter().max().copied(), cols["stamp"]["max"].as_i64());
        let day = cast(batch.column_by_name("day").unwrap(), &DataType::Int32).unwrap();
        let day: Vec<i32> = day.as_primitive::<Int32Type>().iter().flatten().collect();
        assert_eq!(
            day.iter().min().map(|d| i64::from(*d)),
            cols["day"]["min"].as_i64()
        );
        assert_eq!(
            day.iter().max().map(|d| i64::from(*d)),
            cols["day"]["max"].as_i64()
        );
        // INTERVAL keeps its parts: `to_hours(i) + to_days(i % 4)` for i = 1..=40.
        let span = batch.column_by_name("span").unwrap();
        let span = span
            .as_any()
            .downcast_ref::<IntervalMonthDayNanoArray>()
            .unwrap();
        let first = span.value(0);
        assert_eq!(
            (first.months, first.days, first.nanoseconds),
            (0, 1, 3_600_000_000_000)
        );
    }

    #[test]
    fn reads_in_large_batches_and_stops_early() {
        let table = open("trips");
        let batches: Vec<_> = table.batches().unwrap().collect::<Result<_, _>>().unwrap();
        assert_eq!(batches.len(), 1, "DuckDB's small chunks are joined");
        assert_eq!(batches[0].num_rows(), 500);
        // Dropping the reader part-way ends its thread rather than blocking it.
        let mut it = table.batches().unwrap();
        assert!(it.next().is_some());
        drop(it);
    }

    #[test]
    fn opens_read_only() {
        // A copy, so a write that slipped through couldn't touch the fixture.
        let dir = std::env::temp_dir().join(format!("dv-duckdb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trips.duckdb");
        std::fs::copy(db(), &path).unwrap();
        let before = std::fs::read(&path).unwrap();
        let t = Table::open_as(
            &path,
            Format::DuckDb {
                table: Some("trips".into()),
            },
        )
        .unwrap();
        t.read_all().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert!(!dir.join("trips.duckdb.wal").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
