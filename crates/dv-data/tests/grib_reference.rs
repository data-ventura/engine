//! Reads GRIB 1 and 2 fixtures and checks every variable, coordinate and value against cfgrib.
//! The fixtures and reference come from `scripts/reference/make-grib-fixtures.py`.
#![cfg(feature = "grib")]

use std::path::PathBuf;

use dv_data::array::{self, ArraySource, AttrValue, Variable};
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn reference() -> Json {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("reference/grib.json")).unwrap())
        .unwrap()
}

fn open(file: &str) -> Box<dyn ArraySource> {
    array::open(fixtures().join("arrays").join(file)).unwrap()
}

/// A name without the `_2`, `_3`… that tells copies of a coordinate apart.
fn base(name: &str) -> &str {
    match name.rsplit_once('_') {
        Some((b, n)) if n.parse::<u32>().is_ok_and(|n| n >= 2) => b,
        _ => name,
    }
}

fn same(what: &str, got: &[f64], want: &Json) {
    let want = want.as_array().unwrap();
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        match w.as_f64() {
            None => assert!(g.is_nan(), "{what}[{i}]: {g}, expected missing"),
            Some(w) => assert!(
                (g - w).abs() <= 1e-9 * w.abs().max(1.0),
                "{what}[{i}]: {g} vs cfgrib {w}"
            ),
        }
    }
}

fn coordinates(v: &Variable) -> Vec<String> {
    v.attr("coordinates")
        .and_then(AttrValue::as_text)
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn check_file(file: &str, want: &Json) {
    let src = open(file);
    for (k, w) in want["__attrs__"].as_object().unwrap() {
        let got = src
            .attrs()
            .get(k)
            .unwrap_or_else(|| panic!("{file}: no {k}"));
        match w {
            Json::String(s) => assert_eq!(got.as_text(), Some(s.as_str()), "{file} {k}"),
            n => assert_eq!(got.as_f64(), n.as_f64(), "{file} {k}"),
        }
    }
    for (name, w) in want.as_object().unwrap() {
        if name == "__attrs__" {
            continue;
        }
        let what = format!("{file} {name}");
        let v = src
            .variable(name)
            .unwrap_or_else(|| panic!("{what}: missing"));
        let dims: Vec<&str> = v.dims.iter().map(|d| base(d)).collect();
        assert_eq!(
            dims,
            w["dims"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_str().unwrap())
                .collect::<Vec<_>>(),
            "{what}: dims"
        );
        let shape: Vec<u64> = w["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        assert_eq!(v.shape, shape, "{what}: shape");
        assert_eq!(v.dtype, "f4", "{what}: cfgrib reads float32");
        same(
            &what,
            &array::read_all(src.as_ref(), name).unwrap(),
            &w["values"],
        );
        for (k, a) in w["attrs"].as_object().unwrap() {
            let got = v
                .attr(k)
                .unwrap_or_else(|| panic!("{what}: no attribute {k}"));
            match a {
                Json::String(s) => assert_eq!(got.as_text(), Some(s.as_str()), "{what} {k}"),
                n => assert_eq!(got.as_f64(), n.as_f64(), "{what} {k}"),
            }
        }

        // Each of cfgrib's coordinates: a dimension of the variable, or one its `coordinates`
        // attribute names.
        let aux = coordinates(v);
        for (c, cw) in w["coords"].as_object().unwrap() {
            let mine = v
                .dims
                .iter()
                .chain(&aux)
                .find(|n| base(n) == c)
                .unwrap_or_else(|| panic!("{what}: no coordinate {c} in {:?} or {aux:?}", v.dims));
            let cv = src
                .variable(mine)
                .unwrap_or_else(|| panic!("{what}: no variable {mine}"));
            let cdims: Vec<&str> = cv.dims.iter().map(|d| base(d)).collect();
            let want_dims: Vec<&str> = cw["dims"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_str().unwrap())
                .collect();
            assert_eq!(cdims, want_dims, "{what} {c} ({mine}): dims");
            // A coordinate's dimensions are the variable's own.
            assert!(
                cv.dims.iter().all(|d| v.dims.contains(d)),
                "{what} {mine}: dims {:?}",
                cv.dims
            );
            same(
                &format!("{what} {c} ({mine})"),
                &array::read_all(src.as_ref(), mine).unwrap(),
                &cw["values"],
            );
            if let Some(u) = cw["units"].as_str() {
                assert_eq!(
                    cv.attr("units").and_then(AttrValue::as_text),
                    Some(u),
                    "{what} {c}: units"
                );
            }
        }
        assert_eq!(
            aux.len() + v.dims.iter().filter(|d| src.variable(d).is_some()).count(),
            w["coords"].as_object().unwrap().len(),
            "{what}: coordinates {aux:?}"
        );
    }
}

#[test]
fn every_variable_matches_cfgrib() {
    let reference = reference();
    for file in [
        "forecast.grib2",
        "analysis.grib1",
        "grids.grib2",
        "ensemble.grib2",
    ] {
        check_file(file, &reference[file]);
    }
}

#[test]
fn hyperslabs_match_whole_reads() {
    let src = open("forecast.grib2");
    // t is (step 2, isobaricInhPa 3, latitude 5, longitude 6).
    let all = array::read_all(src.as_ref(), "t").unwrap();
    let (start, count) = ([1, 1, 1, 2], [1, 2, 3, 3]);
    let got = array::read(src.as_ref(), "t", &start, &count).unwrap();
    let mut want = Vec::new();
    for l in 1..3 {
        for y in 1..4 {
            for x in 2..5 {
                want.push(all[((3 + l) * 5 + y) * 6 + x]);
            }
        }
    }
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
    );
    assert!(array::read(src.as_ref(), "t", &[2, 0, 0, 0], &[1, 1, 1, 1]).is_err());
}

#[test]
fn clashing_coordinates_get_numbered_copies() {
    let src = open("forecast.grib2");
    let aux = |v: &str| coordinates(src.variable(v).unwrap());
    // 2 m and 10 m: two heightAboveGround coordinates.
    assert!(
        aux("t2m").contains(&"heightAboveGround".to_string()),
        "{:?}",
        aux("t2m")
    );
    assert!(
        aux("u10").contains(&"heightAboveGround_2".to_string()),
        "{:?}",
        aux("u10")
    );
    // tp's single step (6 h) isn't the step dimension the others share.
    assert_eq!(src.variable("t").unwrap().dims[0], "step");
    assert!(aux("tp").contains(&"step_2".to_string()), "{:?}", aux("tp"));
    // One reference time for all: one shared `time`.
    let times = src
        .variables()
        .iter()
        .filter(|v| base(&v.name) == "time")
        .count();
    assert_eq!(times, 1);
}

#[test]
fn lambert_grid_gets_its_own_y_and_x() {
    let src = open("grids.grib2");
    // The polar stereographic `z` came first, so the Lambert `r` gets numbered copies.
    let z = src.variable("z").unwrap();
    let r = src.variable("r").unwrap();
    assert_eq!(z.dims, ["y", "x"]);
    assert_eq!(r.dims, ["y_2", "x_2"]);
    assert_eq!(r.shape, [5, 7]);
    assert_eq!(
        r.attr("GRIB_gridType").and_then(AttrValue::as_text),
        Some("lambert")
    );
    // Latitudes: `latitude` for the reduced Gaussian grid, `_2` polar stereographic, `_3` Lambert.
    let aux = coordinates(r);
    assert!(aux.contains(&"latitude_3".to_string()), "{aux:?}");
    let lat = src.variable("latitude_3").unwrap();
    assert_eq!(lat.dims, ["y_2", "x_2"]);
    // A hyperslab of the Lambert field and of its latitudes, against cfgrib.
    let reference = reference();
    let want = &reference["grids.grib2"]["r"];
    let pick = |w: &Json, y: usize, x: usize| w.as_array().unwrap()[y * 7 + x].as_f64().unwrap();
    let got = array::read(src.as_ref(), "r", &[1, 2], &[2, 3]).unwrap();
    let lats = array::read(src.as_ref(), "latitude_3", &[1, 2], &[2, 3]).unwrap();
    for (i, (y, x)) in [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(got[i], pick(&want["values"], y, x), "r[{y}, {x}]");
        let lat = pick(&want["coords"]["latitude"]["values"], y, x);
        assert!(
            (lats[i] - lat).abs() <= 1e-9 * lat.abs(),
            "latitude[{y}, {x}]"
        );
    }
}

#[test]
fn ensemble_members_are_a_dimension() {
    let src = open("ensemble.grib2");
    let v = src.variable("t2m").unwrap();
    assert_eq!(v.dims, ["number", "step", "latitude", "longitude"]);
    assert_eq!(
        array::read_all(src.as_ref(), "number").unwrap(),
        [0.0, 1.0, 2.0, 3.0]
    );
    // Member 2 has no step 12: that block reads as missing, the rest as data.
    let block = array::read(src.as_ref(), "t2m", &[2, 0, 0, 0], &[1, 2, 5, 6]).unwrap();
    assert!(block[..30].iter().all(|x| x.is_finite()));
    assert!(block[30..].iter().all(|x| x.is_nan()));
}
