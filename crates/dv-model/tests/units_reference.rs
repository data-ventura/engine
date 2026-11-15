//! Checks unit conversions against pint. The reference comes from `scripts/reference/units.py`.

use std::path::PathBuf;

use dv_model::units::Unit;
use serde_json::Value as Json;

#[test]
fn conversions_match_pint() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/units.json");
    let cases: Vec<Json> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert!(cases.len() >= 15);
    for c in cases {
        let (from, to) = (c["from"].as_str().unwrap(), c["to"].as_str().unwrap());
        let v = c["value"].as_f64().unwrap();
        let expected = c["result"].as_f64().unwrap();
        let a = Unit::parse(from).unwrap();
        let b = Unit::parse(to).unwrap();
        let got = a
            .converter(&b)
            .unwrap_or_else(|| panic!("{from} -> {to}"))
            .apply(v);
        let tol = 1e-12 * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tol,
            "{v} {from} -> {to}: {got} vs pint {expected}"
        );
    }
}
