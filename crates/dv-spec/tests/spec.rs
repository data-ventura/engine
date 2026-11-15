//! Workspaces round-trip and validate against the JSON Schema, problems are caught, and lock
//! fingerprints match hashlib (`scripts/reference/fingerprints.py`).

use std::path::PathBuf;

use dv_spec::lock::{Lock, Replay, fingerprint, replay};
use dv_spec::workspace::{ReadError, Workspace};
use serde_json::Value as Json;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn example() -> String {
    std::fs::read_to_string(root().join("fixtures/spec/example.dvspec")).unwrap()
}

fn schema() -> jsonschema::Validator {
    let s: Json = serde_json::from_str(
        &std::fs::read_to_string(root().join("schema/dvspec.schema.json")).unwrap(),
    )
    .unwrap();
    jsonschema::validator_for(&s).unwrap()
}

fn as_json(toml_text: &str) -> Json {
    let v: toml::Value = toml::from_str(toml_text).unwrap();
    serde_json::to_value(v).unwrap()
}

#[test]
fn example_round_trips_and_matches_the_schema() {
    let w = Workspace::from_toml(&example()).unwrap();
    assert_eq!(w.sources.len(), 4);
    assert_eq!(w.joins.len(), 2);
    let again = Workspace::from_toml(&w.to_toml()).unwrap();
    assert_eq!(w, again, "TOML round trip");

    let v = schema();
    let errors: Vec<String> = v
        .iter_errors(&as_json(&example()))
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let errors: Vec<String> = v
        .iter_errors(&as_json(&w.to_toml()))
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "written workspace: {errors:?}");
}

/// The engine's checks and the schema agree on structural mistakes, and the engine catches the
/// ones a schema can't (references, units, secrets).
#[test]
fn problems_are_reported() {
    let base = example();
    let v = schema();
    let cases: [(&str, &str, bool); 7] = [
        ("dvspec = 0", "dvspec = 1", true),
        ("role = \"category\"", "role = \"colour\"", true),
        ("unit = \"mi\"", "unit = \"furlongs\"", false),
        ("tolerance = \"10m\"", "tolerance = \"soon\"", true),
        ("right = \"types\"", "right = \"nope\"", false),
        (
            "url = \"../tables/obs.csv\"",
            "url = \"https://me:hunter2@example.com/obs.csv\"",
            false,
        ),
        ("columns = 2", "columns = 2\napi_token = \"abc\"", false),
    ];
    for (from, to, schema_catches) in cases {
        assert!(base.contains(from), "{from}");
        let text = base.replacen(from, to, 1);
        match Workspace::from_toml(&text) {
            Err(ReadError::Invalid(p)) => assert!(!p.is_empty(), "{to}"),
            Err(ReadError::Toml(e)) => panic!("{to}: {e}"),
            Ok(_) => panic!("{to} should be refused"),
        }
        assert_eq!(
            !v.is_valid(&as_json(&text)),
            schema_catches,
            "schema on {to}"
        );
    }
    assert!(matches!(
        Workspace::from_toml("dvspec = 0\nsurprise = 1"),
        Err(ReadError::Toml(_))
    ));
}

#[test]
fn fingerprints_match_hashlib() {
    let r: Json = serde_json::from_str(
        &std::fs::read_to_string(root().join("fixtures/reference/fingerprints.json")).unwrap(),
    )
    .unwrap();
    for (rel, want) in r.as_object().unwrap() {
        let f = fingerprint(&root().join("fixtures").join(rel), rel, true).unwrap();
        assert_eq!(f.size, want["size"].as_u64().unwrap(), "{rel}: size");
        assert_eq!(
            f.metadata_sha256,
            want["metadata"].as_str().unwrap(),
            "{rel}: metadata"
        );
        assert_eq!(
            f.content_sha256.as_deref(),
            want["content"].as_str(),
            "{rel}: content"
        );
    }
}

#[test]
fn replay_levels() {
    let fp = |rel: &str| fingerprint(&root().join("fixtures").join(rel), rel, false).unwrap();
    let mut locked = Lock::new();
    locked.sources.insert("taxi".into(), fp("taxi-20k.parquet"));
    locked
        .sources
        .insert("trips".into(), fp("tables/trips.csv"));
    locked.operations.insert("formula".into(), 1);
    let text = locked.to_toml();
    assert_eq!(Lock::from_toml(&text).unwrap(), locked, "lock round trip");

    let mut now = locked.clone();
    now.sources.get_mut("taxi").unwrap().modified += 3600;
    assert_eq!(
        replay(&locked, &now),
        Replay::Exact,
        "a new mtime alone is still exact"
    );

    now.sources.insert("trips".into(), fp("tables/obs.csv"));
    now.operations.insert("formula".into(), 2);
    match replay(&locked, &now) {
        Replay::BestEffort(d) => {
            assert_eq!(d.len(), 2, "{d:?}");
            assert!(d[0].contains("trips") && d[1].contains("formula"), "{d:?}");
        }
        other => panic!("{other:?}"),
    }

    now.sources.remove("taxi");
    assert!(matches!(replay(&locked, &now), Replay::Unavailable(m) if m[0].contains("taxi")));
}
