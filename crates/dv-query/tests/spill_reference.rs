//! Runs sorts, aggregates and joins under a memory limit far smaller than their data, so they
//! spill, and checks them against the same queries run without a limit and against DuckDB. The
//! reference comes from `scripts/reference/spill.py`, which also defines the tables.

use std::path::{Path, PathBuf};

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch};
use arrow::compute::{cast, concat_batches};
use arrow::datatypes::{DataType, Float64Type, Int64Type};
use datafusion::functions_aggregate::expr_fn::{count, min, sum};
use datafusion::logical_expr::lit;
use datafusion::prelude::ident;
use dv_data::budget::{Budget, Consumer};
use dv_data::request;
use dv_query::join::{JoinReport, KeyJoin, key_join};
use dv_query::spill::Cause;
use dv_query::{Config, Engine, Error, OutOfMemory};
use serde_json::Value as Json;

/// Far less than any of the queries needs: the sort alone holds about 30 MB of rows.
const LIMIT: u64 = 16 << 20;

/// Spilling this much under so small a limit makes hundreds of spill files, and DataFusion keeps
/// each one open until it's merged. Running the tests one at a time keeps the process within the
/// open-file limit (256 by default on macOS).
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn reference() -> Json {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/spill.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The tables, as DataFusion spells `spill.py`'s formulas.
fn tables(n: u64) -> String {
    format!(
        "with facts as (
            select value as id,
                   (value * 7919) % 50021 as key,
                   'g' || cast(value % 97 as varchar) as grp,
                   cast((value * 104729) % 100003 as double) / 100 as value
            from range(0, {n})
        ), labels as (
            select (value * 314159) % {n} as id,
                   'l' || cast(value % 1009 as varchar) as label,
                   value % 13 + 1 as weight
            from range(0, {n})
        ) "
    )
}

fn spill_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("spill-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn small(name: &str) -> Engine {
    Engine::with_config(Config {
        query_memory: Some(LIMIT),
        spill_dir: Some(spill_dir(name)),
        ..Config::new(Budget::new(1 << 30))
    })
    .unwrap()
}

fn unlimited() -> Engine {
    Engine::with_config(Config {
        max_spill: 0,
        ..Config::new(Budget::new(16 << 30))
    })
    .unwrap()
}

async fn run(e: &Engine, sql: &str) -> Result<RecordBatch, Error> {
    let (submitter, _requests) = request::channel::<()>();
    let generation = submitter.submit(());
    let df = e
        .sql(&format!(
            "{}{sql}",
            tables(reference()["n"].as_u64().unwrap())
        ))
        .await?;
    let batches = e.collect(df, &generation).await?;
    Ok(concat_batches(&batches[0].schema(), &batches).unwrap())
}

fn f64_at(a: &ArrayRef, i: usize) -> Option<f64> {
    let a = cast(a, &DataType::Float64).unwrap();
    let a = a.as_primitive::<Float64Type>();
    (!a.is_null(i)).then(|| a.value(i))
}

fn i64_col(b: &RecordBatch, c: usize) -> Vec<i64> {
    let a = cast(b.column(c), &DataType::Int64).unwrap();
    a.as_primitive::<Int64Type>().values().to_vec()
}

fn f64_col(b: &RecordBatch, c: usize) -> Vec<f64> {
    let a = cast(b.column(c), &DataType::Float64).unwrap();
    a.as_primitive::<Float64Type>().values().to_vec()
}

fn close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() <= 1e-9 * b.abs().max(1.0),
        "{what}: {a} vs {b}"
    );
}

/// Compares rows `from..` of a result with DuckDB's, cell by cell: numbers within 1e-9, text
/// exactly.
fn same_rows(got: &RecordBatch, from: usize, want: &Json, what: &str) {
    for (r, row) in want.as_array().unwrap().iter().enumerate() {
        for (c, cell) in row.as_array().unwrap().iter().enumerate() {
            let a = got.column(c);
            let at = format!("{what} row {} col {c}", from + r);
            match cell {
                Json::Number(n) => close(f64_at(a, from + r).unwrap(), n.as_f64().unwrap(), &at),
                Json::String(s) => {
                    let text = cast(a, &DataType::Utf8).unwrap();
                    assert_eq!(text.as_string::<i32>().value(from + r), s, "{at}");
                }
                other => panic!("{at}: unexpected {other}"),
            }
        }
    }
}

/// Compares two engines' results: integers and text exactly, floats within 1e-9, since spilled
/// aggregates add partial sums in a different order.
fn same_batches(got: &RecordBatch, want: &RecordBatch, what: &str) {
    assert_eq!(got.schema(), want.schema(), "{what}: schema");
    assert_eq!(got.num_rows(), want.num_rows(), "{what}: rows");
    for c in 0..got.num_columns() {
        if got.column(c).data_type().is_floating() {
            for (i, (a, b)) in f64_col(got, c).iter().zip(f64_col(want, c)).enumerate() {
                close(*a, b, &format!("{what} row {i} col {c}"));
            }
        } else {
            assert_eq!(got.column(c), want.column(c), "{what} col {c}");
        }
    }
}

fn spilled(e: &Engine, before: u64, what: &str) {
    assert!(e.spilled() > before, "{what} didn't spill");
}

#[tokio::test]
async fn sorts_spill_and_match_duckdb() {
    let _serial = SERIAL.lock().await;
    let want = &reference()["sort"];
    let sql = want["sql"].as_str().unwrap();
    let e = small("sort");
    let got = run(&e, sql).await.unwrap();
    spilled(&e, 0, "the sort");

    assert_eq!(got.num_rows() as u64, want["rows"].as_u64().unwrap());
    let checksum: i64 = i64_col(&got, 0)
        .iter()
        .enumerate()
        .map(|(i, id)| (i as i64 + 1) * id)
        .sum();
    assert_eq!(checksum, want["checksum"].as_i64().unwrap(), "order");
    same_rows(&got, 0, &want["head"], "head");
    same_rows(&got, got.num_rows() - 5, &want["tail"], "tail");

    let full = run(&unlimited(), sql).await.unwrap();
    assert_eq!(got, full, "the same rows as without a limit");
}

#[tokio::test]
async fn aggregates_spill_and_match_duckdb() {
    let _serial = SERIAL.lock().await;
    let want = &reference()["group"];
    let sql = want["sql"].as_str().unwrap();
    let e = small("group");
    let got = run(&e, sql).await.unwrap();
    spilled(&e, 0, "the aggregate");

    let (key, n, grps) = (i64_col(&got, 0), i64_col(&got, 1), i64_col(&got, 5));
    let summary = want["summary"].as_array().unwrap();
    assert_eq!(
        got.num_rows() as u64,
        summary[0].as_u64().unwrap(),
        "groups"
    );
    let key_n: i64 = key.iter().zip(&n).map(|(k, n)| k * n).sum();
    assert_eq!(key_n, summary[1].as_i64().unwrap(), "key * n");
    assert_eq!(
        grps.iter().sum::<i64>(),
        summary[2].as_i64().unwrap(),
        "grps"
    );
    for (c, s) in [(2, 3), (3, 4), (4, 5)] {
        let total: f64 = f64_col(&got, c).iter().sum();
        close(
            total,
            summary[s].as_f64().unwrap(),
            &format!("sum of col {c}"),
        );
    }
    same_rows(&got, 0, &want["head"], "head");

    let full = run(&unlimited(), sql).await.unwrap();
    same_batches(&got, &full, "without a limit");
}

#[tokio::test]
async fn joins_spill_and_match_duckdb() {
    let _serial = SERIAL.lock().await;
    let want = &reference()["join"];
    let sql = want["sql"].as_str().unwrap();
    let e = small("join");
    let got = run(&e, sql).await.unwrap();
    spilled(&e, 0, "the join");
    assert_eq!(got.num_rows(), want["rows"].as_array().unwrap().len());
    same_rows(&got, 0, &want["rows"], "join");

    let full = run(&unlimited(), sql).await.unwrap();
    same_batches(&got, &full, "without a limit");
}

/// A key join, its match report included, under the same limit.
#[tokio::test]
async fn key_joins_spill_and_match_duckdb() {
    let _serial = SERIAL.lock().await;
    let want = &reference()["join"];
    let n = reference()["n"].as_u64().unwrap();
    let e = small("key-join");
    let facts = e
        .sql(&format!("{}select * from facts", tables(n)))
        .await
        .unwrap();
    let labels = e
        .sql(&format!("{}select * from labels", tables(n)))
        .await
        .unwrap();
    let spec = KeyJoin {
        left_keys: vec!["id".into()],
        right_keys: vec!["id".into()],
        duplicates: None,
    };
    let (joined, report) = key_join(facts, labels, &spec).await.unwrap();
    assert_eq!(
        report,
        JoinReport {
            left_rows: n,
            matched: n,
            unmatched: 0,
            ambiguous: 0
        }
    );
    let df = joined
        .aggregate(
            vec![ident("label")],
            vec![
                count(lit(1)),
                sum(ident("value") * ident("weight")),
                min(ident("id")),
            ],
        )
        .unwrap()
        .sort(vec![ident("label").sort(true, false)])
        .unwrap();
    let (submitter, _requests) = request::channel::<()>();
    let batches = e.collect(df, &submitter.submit(())).await.unwrap();
    spilled(&e, 0, "the key join");
    same_rows(
        &concat_batches(&batches[0].schema(), &batches).unwrap(),
        0,
        &want["rows"],
        "key join",
    );
}

fn files_in(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            if path.is_dir() { files_in(&path) } else { 1 }
        })
        .sum()
}

#[tokio::test]
async fn spill_files_are_removed() {
    let _serial = SERIAL.lock().await;
    let dir = spill_dir("cleanup");
    let budget = Budget::new(1 << 30);
    let e = Engine::with_config(Config {
        query_memory: Some(LIMIT),
        spill_dir: Some(dir.clone()),
        ..Config::new(budget.clone())
    })
    .unwrap();
    assert!(dir.is_dir(), "the spill directory is created");
    run(&e, reference()["sort"]["sql"].as_str().unwrap())
        .await
        .unwrap();
    spilled(&e, 0, "the sort");
    let own = e.spill_dir().unwrap();
    assert!(own.starts_with(&dir), "{own:?} is inside {dir:?}");
    assert_eq!(files_in(&dir), 0, "spill files are deleted after the query");
    assert_eq!(e.spilling(), 0);
    assert_eq!(
        budget.used_by(Consumer::Query),
        0,
        "memory returns to the budget"
    );
    drop(e);
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "the engine's directory goes with it"
    );
}

fn out_of_memory(result: Result<RecordBatch, Error>) -> OutOfMemory {
    match result {
        Err(Error::OutOfMemory(oom)) => oom,
        Err(other) => panic!("expected out of memory, got {other}"),
        Ok(_) => panic!("expected out of memory, got a result"),
    }
}

#[tokio::test]
async fn without_spilling_a_large_sort_fails_clearly() {
    let _serial = SERIAL.lock().await;
    let e = Engine::with_config(Config {
        query_memory: Some(LIMIT),
        max_spill: 0,
        ..Config::new(Budget::new(1 << 30))
    })
    .unwrap();
    let oom = out_of_memory(run(&e, reference()["sort"]["sql"].as_str().unwrap()).await);
    assert_eq!(oom.cause, Cause::SpillOff);
    assert_eq!(oom.limit, LIMIT);
    let message = oom.to_string();
    assert!(
        message.starts_with(
            "the query needs more than its 16 MB of memory, and spilling to disk is off"
        ),
        "{message}"
    );
}

#[tokio::test]
async fn the_spill_limit_stops_a_query_clearly() {
    let _serial = SERIAL.lock().await;
    let e = Engine::with_config(Config {
        query_memory: Some(LIMIT),
        spill_dir: Some(spill_dir("limit")),
        max_spill: 1 << 20,
        ..Config::new(Budget::new(1 << 30))
    })
    .unwrap();
    let oom = out_of_memory(run(&e, reference()["sort"]["sql"].as_str().unwrap()).await);
    assert_eq!(oom.cause, Cause::SpillLimit, "{oom}");
    assert_eq!(
        oom.to_string(),
        "the query needs more than its 16 MB of memory, and its spill files reached their 1 MB limit"
    );
    assert_eq!(e.spilling(), 0, "partial spill files are deleted");
}

/// Collecting every row into one list can't spill: the list grows past the limit by itself.
#[tokio::test]
async fn what_cannot_spill_fails_clearly() {
    let _serial = SERIAL.lock().await;
    let e = Engine::with_config(Config {
        query_memory: Some(8 << 20),
        spill_dir: Some(spill_dir("unspillable")),
        ..Config::new(Budget::new(1 << 30))
    })
    .unwrap();
    let oom = out_of_memory(run(&e, "select array_agg(grp) from facts").await);
    assert_eq!(oom.cause, Cause::Unspillable, "{oom}");
    assert!(
        oom.to_string().starts_with(
            "the query needs more than its 8 MB of memory, even after spilling to disk"
        ),
        "{oom}"
    );
}
