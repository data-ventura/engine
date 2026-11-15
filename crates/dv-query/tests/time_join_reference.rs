//! Checks nearest-time joins against polars' join_asof. The fixtures and reference come from
//! `scripts/reference/time-join.py`.

use std::path::PathBuf;

use arrow::array::{AsArray, RecordBatch};
use arrow::compute::{cast, concat_batches};
use arrow::datatypes::{DataType, Int64Type};
use dv_data::budget::Budget;
use dv_data::table::Table;
use dv_query::Engine;
use dv_query::time::{Direction, TimeJoin, time_join};
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

async fn engine() -> Engine {
    let e = Engine::new(Budget::new(1 << 30)).unwrap();
    for name in ["trips", "obs", "visits"] {
        let file = fixtures().join("tables").join(format!("{name}.csv"));
        e.register(name, &Table::open(file).unwrap()).await.unwrap();
    }
    e
}

fn ints(b: &RecordBatch, name: &str) -> Vec<Option<i64>> {
    let c = cast(b.column_by_name(name).unwrap(), &DataType::Int64).unwrap();
    c.as_primitive::<Int64Type>().iter().collect()
}

#[tokio::test]
async fn time_joins_match_polars() {
    let e = engine().await;
    let reference: Json = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("reference/time-join.json")).unwrap(),
    )
    .unwrap();
    let cases = [
        (
            "trips_nearest_10m",
            "trips",
            "pickup",
            "id",
            Direction::Nearest,
            10,
            false,
        ),
        (
            "trips_backward_10m",
            "trips",
            "pickup",
            "id",
            Direction::Backward,
            10,
            false,
        ),
        (
            "trips_forward_10m",
            "trips",
            "pickup",
            "id",
            Direction::Forward,
            10,
            false,
        ),
        (
            "visits_by_station_nearest_15m",
            "visits",
            "at",
            "visit",
            Direction::Nearest,
            15,
            true,
        ),
    ];
    for (key, left, time, id, direction, minutes, by_station) in cases {
        let spec = TimeJoin {
            left_time: time.into(),
            right_time: "time".into(),
            tolerance_us: minutes * 60_000_000,
            direction,
            by: if by_station {
                vec![("station".into(), "station".into())]
            } else {
                vec![]
            },
        };
        let (df, report) = time_join(
            &e,
            e.table(left).await.unwrap(),
            e.table("obs").await.unwrap(),
            &spec,
        )
        .await
        .unwrap();
        let want = &reference[key];
        let batches = df.collect().await.unwrap();
        let b = concat_batches(&batches[0].schema(), &batches).unwrap();
        let (ids, obs) = (ints(&b, id), ints(&b, "obs_id"));
        let left_rows = b.num_rows();
        let mut compared = 0;
        for (i, o) in ids.iter().zip(&obs) {
            match want["matches"].get(i.unwrap().to_string()) {
                Some(w) => {
                    assert_eq!(*o, w.as_i64(), "{key}: {id} {}", i.unwrap());
                    compared += 1;
                }
                // Left rows polars couldn't take (null times) must be unmatched.
                None => assert_eq!(*o, None, "{key}: {id} {} has no time", i.unwrap()),
            }
        }
        assert_eq!(
            compared,
            want["matches"].as_object().unwrap().len(),
            "{key}"
        );
        assert_eq!(report.left_rows as usize, left_rows);
        assert_eq!(
            report.matched,
            want["matched"].as_u64().unwrap(),
            "{key}: matched"
        );
        assert_eq!(
            report.max_gap_us,
            want["max_abs_us"].as_i64(),
            "{key}: max gap"
        );
        let mean = want["mean_abs_us"].as_f64().unwrap();
        assert!(
            (report.mean_gap_us.unwrap() - mean).abs() < 1e-6 * mean,
            "{key}: mean gap"
        );
        if by_station {
            assert!(
                b.schema().column_with_name("station_right").is_none(),
                "by keys are dropped"
            );
        }
    }
}

#[tokio::test]
async fn naive_and_zoned_times_dont_join() {
    let e = engine().await;
    let zoned = e
        .sql("select id, arrow_cast(pickup, 'Timestamp(Microsecond, Some(\"UTC\"))') as pickup from trips")
        .await
        .unwrap();
    let spec = TimeJoin {
        left_time: "pickup".into(),
        right_time: "time".into(),
        tolerance_us: 60_000_000,
        direction: Direction::Nearest,
        by: vec![],
    };
    let err = time_join(&e, zoned, e.table("obs").await.unwrap(), &spec)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("time zone"), "{err}");
}
