//! Checks aggregates, selections and joins against DuckDB. The reference comes from
//! `scripts/reference/query.py`.

use std::path::PathBuf;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch};
use arrow::compute::{cast, concat_batches};
use arrow::datatypes::{DataType, Float64Type, Int64Type};
use datafusion::functions_aggregate::expr_fn::{avg, count, max, sum};
use datafusion::logical_expr::{col, lit};
use datafusion::prelude::ident;
use dv_data::budget::{Budget, Consumer};
use dv_data::request;
use dv_data::table::Table;
use dv_model::selection::{Predicate, Selection};
use dv_query::join::{Aggregate, Duplicates, JoinError, JoinReport, KeyJoin, key_join};
use dv_query::{Config, Engine, Error, selection_expr};
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn reference() -> Json {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("reference/query.json")).unwrap())
        .unwrap()
}

async fn engine() -> Engine {
    let e = Engine::new(Budget::new(1 << 30)).unwrap();
    for (name, file) in [
        ("trips", "tables/trips.csv"),
        ("types", "tables/payment-types.csv"),
        ("fees", "tables/payment-fees.csv"),
        ("taxi", "taxi-20k.parquet"),
    ] {
        e.register(name, &Table::open(fixtures().join(file)).unwrap())
            .await
            .unwrap();
    }
    e
}

fn one(batches: &[RecordBatch]) -> RecordBatch {
    concat_batches(&batches[0].schema(), batches).unwrap()
}

fn f64_at(a: &ArrayRef, i: usize) -> Option<f64> {
    let a = cast(a, &DataType::Float64).unwrap();
    let a = a.as_primitive::<Float64Type>();
    (!a.is_null(i)).then(|| a.value(i))
}

fn close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() <= 1e-9 * b.abs().max(1.0),
        "{what}: {a} vs DuckDB {b}"
    );
}

/// Compares a result's rows with DuckDB's, cell by cell: numbers within 1e-9, text exactly.
fn same_rows(got: &RecordBatch, want: &Json, what: &str) {
    let want = want.as_array().unwrap();
    assert_eq!(got.num_rows(), want.len(), "{what}: rows");
    for (r, row) in want.iter().enumerate() {
        for (c, cell) in row.as_array().unwrap().iter().enumerate() {
            let a = got.column(c);
            let at = format!("{what} row {r} col {c}");
            match cell {
                Json::Null => assert!(a.is_null(r), "{at}: null"),
                Json::Number(n) => close(f64_at(a, r).unwrap(), n.as_f64().unwrap(), &at),
                Json::String(s) => {
                    let text = cast(a, &DataType::Utf8).unwrap();
                    assert_eq!(
                        text.as_string::<i32>().value(r).replace('T', " "),
                        *s,
                        "{at}"
                    );
                }
                other => panic!("{at}: unexpected {other}"),
            }
        }
    }
}

#[tokio::test]
async fn sql_aggregates_match_duckdb() {
    let e = engine().await;
    let df = e
        .sql(
            "select payment, count(*), avg(fare), sum(distance), min(pickup) from trips \
             group by payment order by payment nulls last",
        )
        .await
        .unwrap();
    same_rows(
        &one(&df.collect().await.unwrap()),
        &reference()["group_by_payment"],
        "group by",
    );
}

#[tokio::test]
async fn selections_filter_like_duckdb() {
    let e = engine().await;
    let taxi = e.table("taxi").await.unwrap();
    let selection = Selection {
        predicates: vec![Predicate::Range {
            field: "tripDistance".into(),
            min: Some(1.0),
            max: Some(3.0),
        }],
        generation: 0,
    };
    let filter = selection_expr(&selection, taxi.schema().as_arrow())
        .unwrap()
        .unwrap();
    let df = taxi
        .filter(filter)
        .unwrap()
        .aggregate(
            vec![ident("paymentType")],
            vec![
                count(lit(1)),
                avg(ident("tipAmount")),
                max(ident("totalAmount")),
            ],
        )
        .unwrap()
        .sort(vec![ident("paymentType").sort(true, false)])
        .unwrap();
    same_rows(
        &one(&df.collect().await.unwrap()),
        &reference()["taxi_filtered"],
        "filtered",
    );
}

fn payment_join(duplicates: Option<Duplicates>) -> KeyJoin {
    KeyJoin {
        left_keys: vec!["payment".into()],
        right_keys: vec!["code".into()],
        duplicates,
    }
}

/// Joins fees onto trips and returns the row count, the sum of fees and the report.
async fn fee_totals(
    e: &Engine,
    duplicates: Option<Duplicates>,
) -> Result<(u64, f64, JoinReport), Error> {
    let (df, report) = key_join(
        e.table("trips").await?,
        e.table("fees").await?,
        &payment_join(duplicates),
    )
    .await?;
    let agg = df.aggregate(vec![], vec![count(lit(1)), sum(col("fee"))])?;
    let b = one(&agg.collect().await?);
    let n = cast(b.column(0), &DataType::Int64).unwrap();
    let n = n.as_primitive::<Int64Type>().value(0) as u64;
    Ok((n, f64_at(b.column(1), 0).unwrap(), report))
}

#[tokio::test]
async fn key_joins_match_duckdb() {
    let e = engine().await;
    let r = reference();

    // A lookup with unique keys needs no policy. Unmatched rows (code 3 and nulls) are kept.
    let (df, report) = key_join(
        e.table("trips").await.unwrap(),
        e.table("types").await.unwrap(),
        &payment_join(None),
    )
    .await
    .unwrap();
    assert_eq!(report.left_rows, r["join_report"][0].as_u64().unwrap());
    assert_eq!(report.matched, r["join_report"][1].as_u64().unwrap());
    assert_eq!(report.ambiguous, r["join_report"][2].as_u64().unwrap());
    assert_eq!(report.unmatched, 12);
    assert!(
        !df.schema().has_column_with_unqualified_name("code"),
        "right keys are dropped"
    );
    let by_name = df
        .aggregate(vec![col("name")], vec![count(lit(1))])
        .unwrap()
        .sort(vec![col("name").sort(true, false)])
        .unwrap();
    same_rows(
        &one(&by_name.collect().await.unwrap()),
        &r["join_types"],
        "join types",
    );

    // Several fees per code: refused without a policy.
    match fee_totals(&e, None).await {
        Err(Error::Join(JoinError::Ambiguous {
            ambiguous,
            left_rows,
        })) => {
            assert_eq!(ambiguous, r["fees_report"][2].as_u64().unwrap());
            assert_eq!(left_rows, 500);
        }
        other => panic!("expected an ambiguous join, got {other:?}"),
    }
    for (policy, key) in [
        (Duplicates::KeepAll, "fees_keep_all"),
        (
            Duplicates::First {
                order_by: "since".into(),
            },
            "fees_first",
        ),
        (
            Duplicates::Last {
                order_by: "since".into(),
            },
            "fees_last",
        ),
        (Duplicates::Aggregate(Aggregate::Mean), "fees_mean"),
    ] {
        let (n, s, report) = fee_totals(&e, Some(policy)).await.unwrap();
        assert_eq!(n, r[key][0].as_u64().unwrap(), "{key}: rows");
        close(s, r[key][1].as_f64().unwrap(), key);
        assert_eq!(
            report.matched,
            r["fees_report"][1].as_u64().unwrap(),
            "{key}: matched"
        );
    }
}

#[tokio::test]
async fn sql_is_read_only() {
    let e = engine().await;
    for sql in [
        "create table x as select 1",
        "insert into trips values (1)",
        "drop table trips",
        "set datafusion.execution.batch_size = 1",
    ] {
        assert!(e.sql(sql).await.is_err(), "{sql} should be refused");
    }
}

#[tokio::test]
async fn stale_generations_are_cancelled() {
    let e = engine().await;
    let (submitter, _requests) = request::channel::<()>();
    let old = submitter.submit(());
    let df = e.sql("select * from taxi").await.unwrap();
    let current = submitter.submit(());
    assert!(matches!(
        e.collect(df.clone(), &old).await,
        Err(Error::Cancelled(_))
    ));
    let batches = e.collect(df, &current).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 20_000);
}

#[tokio::test]
async fn query_memory_comes_from_the_budget() {
    let budget = Budget::new(64 * 1024);
    let e = Engine::with_config(Config {
        max_spill: 0,
        ..Config::new(budget.clone())
    })
    .unwrap();
    e.register(
        "taxi",
        &Table::open(fixtures().join("taxi-20k.parquet")).unwrap(),
    )
    .await
    .unwrap();
    let df = e
        .sql("select * from taxi order by \"fareAmount\"")
        .await
        .unwrap();
    let (submitter, _requests) = request::channel::<()>();
    let err = e.collect(df, &submitter.submit(())).await.unwrap_err();
    assert!(matches!(err, Error::OutOfMemory(_)), "{err}");
    assert!(err.to_string().contains("66 KB of memory"), "{err}");
    assert_eq!(
        budget.used_by(Consumer::Query),
        0,
        "memory returns to the budget"
    );
}

/// The same selections `dv-model` checks as masks, run as DataFusion filters.
#[tokio::test]
async fn selection_filters_agree_with_masks() {
    let e = engine().await;
    let cases: Vec<Json> = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("reference/taxi-20k.selection.json")).unwrap(),
    )
    .unwrap();
    for c in cases {
        let selection = Selection {
            predicates: serde_json::from_value(c["predicates"].clone()).unwrap(),
            generation: 0,
        };
        let taxi = e.table("taxi").await.unwrap();
        let filter = selection_expr(&selection, taxi.schema().as_arrow())
            .unwrap()
            .unwrap();
        let rows = taxi.filter(filter).unwrap().count().await.unwrap();
        assert_eq!(rows as u64, c["rows"].as_u64().unwrap(), "{}", c["sql"]);
    }
}
