//! Checks colour-vision-deficiency simulation against colorspacious, CIELAB and CIEDE2000 against
//! colour-science, and the palette check against the same pipeline in Python. The reference comes
//! from `scripts/reference/cvd.py`.

use std::path::PathBuf;

use dv_render::color::{delta_e2000, lab};
use dv_render::cvd::{self, Deficiency};
use dv_render::plot::chart::{DARK, LIGHT};
use dv_render::vector::Rgba;
use serde_json::Value;

fn reference() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/cvd.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn rgba(v: &Value) -> Rgba {
    let c = floats(v);
    Rgba::rgb(c[0] as u8, c[1] as u8, c[2] as u8)
}

fn deficiency(v: &Value) -> Option<Deficiency> {
    match v.as_str() {
        None => None,
        Some("protan") => Some(Deficiency::Protan),
        Some("deutan") => Some(Deficiency::Deutan),
        Some("tritan") => Some(Deficiency::Tritan),
        Some(other) => panic!("unknown deficiency {other}"),
    }
}

#[test]
fn simulation_matches_colorspacious() {
    let r = reference();
    let cases = r["simulations"].as_array().unwrap();
    assert!(cases.len() > 500);
    for c in cases {
        let d = deficiency(&c["kind"]).unwrap();
        let severity = c["severity"].as_f64().unwrap();
        let got = d.simulate(rgba(&c["rgb"]), severity);
        let want = floats(&c["srgb"]);
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() < 1e-9,
                "{c}: got {got:?}, want {want:?}"
            );
        }
    }
}

#[test]
fn lab_matches_colour_science() {
    let r = reference();
    for c in r["labs"].as_array().unwrap() {
        let got = lab(rgba(&c["rgb"]));
        let want = floats(&c["lab"]);
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() < 1e-9,
                "{c}: got {got:?}, want {want:?}"
            );
        }
    }
}

#[test]
fn ciede2000_matches_colour_science() {
    let r = reference();
    for p in r["pairs"].as_array().unwrap() {
        let (a, b) = (floats(&p["a"]), floats(&p["b"]));
        let got = delta_e2000([a[0], a[1], a[2]], [b[0], b[1], b[2]]);
        let want = p["delta_e"].as_f64().unwrap();
        assert!((got - want).abs() < 1e-9, "{p}: got {got}");
    }
}

#[test]
fn palette_check_matches_reference() {
    let r = reference();
    let min = r["min_delta_e"].as_f64().unwrap();
    assert_eq!(min, cvd::MIN_DELTA_E);
    for (name, case) in r["checks"].as_object().unwrap() {
        let palette: Vec<Rgba> = case["palette"]
            .as_array()
            .unwrap()
            .iter()
            .map(rgba)
            .collect();
        let got = cvd::check(&palette, 1.0, min);
        let want = case["flagged"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{name}: {got:?}");
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.a, w["a"].as_u64().unwrap() as usize, "{name}");
            assert_eq!(g.b, w["b"].as_u64().unwrap() as usize, "{name}");
            assert_eq!(g.deficiency, deficiency(&w["kind"]), "{name}");
            assert!(
                (g.delta_e - w["delta_e"].as_f64().unwrap()).abs() < 1e-9,
                "{name}: {g:?} vs {w}"
            );
        }
    }
}

/// Both themes' categorical palettes pass the check.
#[test]
fn theme_palettes_pass() {
    for theme in [LIGHT, DARK] {
        assert_eq!(cvd::check(&theme.palette, 1.0, cvd::MIN_DELTA_E), []);
    }
}
