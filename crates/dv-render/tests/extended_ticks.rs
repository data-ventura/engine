//! Checks Wilkinson's extended tick values against the authors' R implementation
//! (`labeling::extended` 0.4.3) on 4,716 cases. The reference comes from
//! `scripts/reference/extended-ticks.R`.

use std::path::PathBuf;

use dv_render::plot::ticks::extended;

#[test]
fn ticks_match_r_labeling() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/reference/extended-ticks.json");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = reference["cases"].as_array().unwrap();
    assert!(cases.len() > 4000);
    let mut failures = Vec::new();
    for c in cases {
        let (dmin, dmax) = (c["dmin"].as_f64().unwrap(), c["dmax"].as_f64().unwrap());
        let m = c["m"].as_u64().unwrap() as usize;
        let loose = c["only_loose"].as_bool().unwrap();
        let want: Vec<f64> = c["ticks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let got = extended(dmin, dmax, m, loose);
        let scale = (dmax - dmin).abs().max(f64::MIN_POSITIVE);
        let same = got.len() == want.len()
            && got
                .iter()
                .zip(&want)
                .all(|(g, w)| (g - w).abs() <= 1e-9 * scale);
        if !same {
            failures.push(format!(
                "extended({dmin}, {dmax}, {m}, only.loose={loose}): got {got:?}, R {want:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} differ, first: {:#?}",
        failures.len(),
        cases.len(),
        &failures[..failures.len().min(5)]
    );
}
