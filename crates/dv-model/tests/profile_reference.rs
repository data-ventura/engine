//! Checks profiles against DuckDB on a fixture small enough to profile in full, so every sampled
//! statistic is exact. The reference comes from `scripts/reference/profile.py`.

use std::path::PathBuf;

use dv_data::parquet::ParquetTable;
use dv_model::profile::{ExactStats, Value, profile};
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn close(a: f64, b: f64, what: &str) {
    let tol = 1e-9 * a.abs().max(b.abs()).max(1.0);
    assert!((a - b).abs() <= tol, "{what}: {a} vs DuckDB {b}");
}

fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Integer(i) => *i as f64,
        Value::Float(f) => *f,
        other => panic!("not numeric: {other:?}"),
    }
}

fn same_value(ours: &Value, theirs: &Json, what: &str) {
    match ours {
        Value::Text(s) => assert_eq!(Some(s.as_str()), theirs.as_str(), "{what}"),
        Value::Boolean(b) => assert_eq!(Some(*b), theirs.as_bool(), "{what}"),
        v => close(as_f64(v), theirs.as_f64().expect("number"), what),
    }
}

#[test]
fn taxi_profile_matches_duckdb() {
    let table = ParquetTable::open(fixtures().join("taxi-20k.parquet")).unwrap();
    let reference: Json = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("reference/taxi-20k.profile.json")).unwrap(),
    )
    .unwrap();

    let exact = table
        .column_stats()
        .unwrap()
        .into_iter()
        .map(|s| ExactStats {
            nulls: s.nulls,
            min: s.min.as_ref().and_then(Value::from_array),
            max: s.max.as_ref().and_then(Value::from_array),
        })
        .collect();
    let sample = table.sample(usize::MAX).unwrap();
    let p = profile(&sample.batch, sample.of_rows, exact);

    assert_eq!(p.rows, reference["rows"].as_u64().unwrap());
    assert_eq!(
        p.sample_rows as u64, p.rows,
        "a full read samples every row"
    );

    for c in &p.columns {
        let r = &reference["columns"][&c.name];
        let at = |what: &str| format!("{}.{what}", c.name);
        assert_eq!(c.exact.nulls, r["nulls"].as_u64(), "{}", at("nulls"));
        assert_eq!(
            c.sample.nulls as u64,
            r["nulls"].as_u64().unwrap(),
            "{}",
            at("sample nulls")
        );
        assert_eq!(
            c.sample.distinct as u64,
            r["distinct"].as_u64().unwrap(),
            "{}",
            at("distinct")
        );
        for (ours, key) in [(&c.exact.min, "min"), (&c.exact.max, "max")] {
            match ours {
                Some(v) => same_value(v, &r[key], &at(key)),
                None => assert!(r[key].is_null(), "{}: missing", at(key)),
            }
        }
        if let Some(n) = &c.sample.numeric {
            assert_eq!(
                n.count as u64,
                r["count"].as_u64().unwrap(),
                "{}",
                at("count")
            );
            close(n.mean, r["mean"].as_f64().unwrap(), &at("mean"));
            close(n.std, r["std"].as_f64().unwrap(), &at("std"));
            for (i, q) in n.quantiles.iter().enumerate() {
                close(
                    *q,
                    r["quantiles"][i].as_f64().unwrap(),
                    &at(&format!("q{i}")),
                );
            }
            let h = &r["histogram"];
            close(n.histogram.lo, h["lo"].as_f64().unwrap(), &at("hist.lo"));
            close(n.histogram.hi, h["hi"].as_f64().unwrap(), &at("hist.hi"));
            let counts: Vec<u64> = h["counts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(n.histogram.counts, counts, "{}", at("hist.counts"));
            assert_eq!(
                n.histogram.below,
                h["below"].as_u64().unwrap(),
                "{}",
                at("hist.below")
            );
            assert_eq!(
                n.histogram.above,
                h["above"].as_u64().unwrap(),
                "{}",
                at("hist.above")
            );
        } else {
            assert!(
                r.get("mean").is_none(),
                "{}: expected numeric stats",
                c.name
            );
        }
        let top = r
            .get("top")
            .and_then(Json::as_array)
            .cloned()
            .unwrap_or_default();
        assert_eq!(c.sample.top.len(), top.len(), "{}", at("top length"));
        for ((v, n), t) in c.sample.top.iter().zip(&top) {
            same_value(v, &t[0], &at("top value"));
            assert_eq!(*n as u64, t[1].as_u64().unwrap(), "{}", at("top count"));
        }
    }
}
