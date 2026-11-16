//! Checks conservative regridding against xarray-regrid and S2 (spherely), curvilinear cell
//! lookup against S2, and that regridding preserves the area-weighted integral. The reference
//! comes from `scripts/reference/regrid.py`.

use std::path::PathBuf;

use dv_grid::conservative::{Conservative, Rectilinear};
use dv_grid::curvilinear::Curvilinear;
use dv_grid::mesh::EARTH_RADIUS;
use serde_json::Value as Json;

fn reference() -> Json {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/regrid.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn values(v: &Json) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn same(got: f64, want: f64, tol: f64, what: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: {got}, expected missing");
    } else {
        assert!((got - want).abs() <= tol, "{what}: {got} vs {want}");
    }
}

fn all_same(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        same(*g, *w, tol, &format!("{what} cell {i}"));
    }
}

/// Relative difference, for sums of areas and integrals.
fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// Sum of value × valid area over the target, and over the source (whose cells with no geometry
/// have no area).
fn integrals(w: &Conservative, src: &[f64], dst: &[f64]) -> (f64, f64) {
    let frac = w.valid_fraction(src);
    let target = dst
        .iter()
        .zip(&frac)
        .zip(w.target_areas())
        .filter(|((v, _), _)| v.is_finite())
        .map(|((v, f), a)| v * f * a)
        .sum();
    let source = src
        .iter()
        .zip(w.source_areas())
        .filter(|(v, a)| v.is_finite() && a.is_finite())
        .map(|(v, a)| v * a)
        .sum();
    (target, source)
}

#[test]
fn nested_lat_lon_matches_xarray_regrid() {
    let r = reference();
    for case in ["nested_global", "nested_regional"] {
        let c = &r[case];
        let src = Rectilinear::from_centers(&values(&c["src_lon"]), &values(&c["src_lat"]), true)
            .unwrap();
        let dst = Rectilinear::from_centers(&values(&c["dst_lon"]), &values(&c["dst_lat"]), true)
            .unwrap();
        let w = Conservative::rectilinear(&src, &dst).unwrap();
        let v = values(&c["values"]);
        for (key, min_valid) in [("min_valid_0", 0.0), ("min_valid_0.6", 0.6)] {
            let want = values(&c[key]);
            assert!(
                want.iter().any(|x| x.is_nan()),
                "{case} {key} has missing cells"
            );
            all_same(
                &w.apply(&v, min_valid),
                &want,
                1e-9,
                &format!("{case} {key}"),
            );
        }
    }
}

#[test]
fn planar_matches_xarray_regrid() {
    let c = &reference()["planar"];
    let src = Rectilinear::from_centers(&values(&c["src_x"]), &values(&c["src_y"]), false).unwrap();
    let dst = Rectilinear::from_centers(&values(&c["dst_x"]), &values(&c["dst_y"]), false).unwrap();
    let w = Conservative::rectilinear(&src, &dst).unwrap();
    all_same(
        &w.apply(&values(&c["values"]), 0.0),
        &values(&c["min_valid_0"]),
        1e-9,
        "planar",
    );
}

/// Areas in steradians, so the two spheres' radii don't matter.
fn steradians(areas: &[f64], radius: f64) -> Vec<f64> {
    areas.iter().map(|a| a / (radius * radius)).collect()
}

fn check_areas(w: &Conservative, c: &Json, radius: f64, tol: f64, what: &str) {
    for (got, key) in [
        (w.source_areas(), "src_area"),
        (w.target_areas(), "dst_area"),
    ] {
        let (got, want) = (
            steradians(got, EARTH_RADIUS),
            steradians(&values(&c[key]), radius),
        );
        for (i, (g, s)) in got.iter().zip(&want).enumerate() {
            if s.is_nan() {
                assert!(g.is_nan(), "{what} {key} {i}: {g}, expected no geometry");
            } else {
                assert!(rel(*g, *s) < tol, "{what} {key} {i}: {g} vs {s}");
            }
        }
    }
}

#[test]
fn unnested_lat_lon_matches_s2() {
    let r = reference();
    let c = &r["unnested"];
    let src = Rectilinear::from_edges(values(&c["src_x_edges"]), values(&c["src_y_edges"]), true)
        .unwrap();
    let dst = Rectilinear::from_edges(values(&c["dst_x_edges"]), values(&c["dst_y_edges"]), true)
        .unwrap();
    let w = Conservative::rectilinear(&src, &dst).unwrap();
    // S2 follows the parallels with 0.02° great-circle segments, off by up to 2e-8 of a cell's
    // area in the polar caps; values then agree well within 1e-6.
    check_areas(&w, c, r["radius"].as_f64().unwrap(), 5e-8, "unnested");
    let v = values(&c["values"]);
    let ones = vec![1.0; v.len()];
    all_same(
        &w.valid_fraction(&ones),
        &values(&c["fraction"]),
        1e-8,
        "fraction",
    );
    for (key, min_valid) in [("min_valid_0", 0.0), ("min_valid_0.5", 0.5)] {
        all_same(&w.apply(&v, min_valid), &values(&c[key]), 1e-6, key);
    }
}

fn curvilinear(c: &Json) -> Curvilinear {
    let n = |k: &str| c[k].as_u64().unwrap() as usize;
    Curvilinear::from_corners(
        n("nx"),
        n("ny"),
        &values(&c["corner_lon"]),
        &values(&c["corner_lat"]),
    )
    .unwrap()
}

fn lat_lon(c: &Json) -> Rectilinear {
    Rectilinear::from_edges(values(&c["dst_x_edges"]), values(&c["dst_y_edges"]), true).unwrap()
}

#[test]
fn curvilinear_matches_s2() {
    let r = reference();
    let radius = r["radius"].as_f64().unwrap();
    for (case, thresholds) in [("rotated", &[0.0, 0.5, 0.9][..]), ("polar", &[0.0][..])] {
        let c = &r[case];
        let src = curvilinear(c);
        let dst = lat_lon(c).to_curvilinear().unwrap();
        let w = Conservative::curvilinear(&src, &dst);
        // The same great-circle cells as S2: agreement to rounding.
        check_areas(&w, c, radius, 1e-13, case);
        let v = values(&c["values"]);
        let ones = vec![1.0; v.len()];
        let frac = w.valid_fraction(&ones);
        all_same(
            &frac,
            &values(&c["fraction"]),
            1e-12,
            &format!("{case} fraction"),
        );
        assert!(
            frac.iter().any(|f| *f > 0.0 && *f < 0.99),
            "{case}: partly covered cells"
        );
        for t in thresholds {
            let key = format!("min_valid_{t}");
            all_same(
                &w.apply(&v, *t),
                &values(&c[&key]),
                1e-10,
                &format!("{case} {key}"),
            );
        }
    }
}

#[test]
fn curvilinear_lookup_matches_s2() {
    let r = reference();
    for case in ["rotated", "polar"] {
        let c = &r[case];
        let grid = curvilinear(c);
        let l = &c["lookup"];
        let (lon, lat) = (values(&l["lon"]), values(&l["lat"]));
        let cells = l["cell"].as_array().unwrap();
        let mut found = 0;
        for (i, want) in cells.iter().enumerate() {
            let want = usize::try_from(want.as_i64().unwrap()).ok();
            found += want.is_some() as usize;
            assert_eq!(
                grid.locate(lon[i], lat[i]),
                want,
                "{case} ({}, {})",
                lon[i],
                lat[i]
            );
        }
        assert!(
            found > 100 && found < cells.len(),
            "{case}: points inside and out"
        );
    }
}

/// Regridding between global grids keeps the area-weighted integral, with missing values too.
#[test]
fn global_rectilinear_conserves() {
    let r = reference();
    let c = &r["unnested"];
    let src = Rectilinear::from_edges(values(&c["src_x_edges"]), values(&c["src_y_edges"]), true)
        .unwrap();
    let mut lat = values(&c["dst_y_edges"]);
    lat.insert(0, -90.0);
    lat.push(90.0);
    let dst = Rectilinear::from_edges(values(&c["dst_x_edges"]), lat, true).unwrap();
    let w = Conservative::rectilinear(&src, &dst).unwrap();
    let sphere = 4.0 * std::f64::consts::PI * EARTH_RADIUS * EARTH_RADIUS;
    assert!(rel(w.source_areas().iter().sum(), sphere) < 1e-12);
    assert!(rel(w.target_areas().iter().sum(), sphere) < 1e-12);

    let v = values(&c["values"]);
    assert!(v.iter().any(|x| x.is_nan()));
    let full: Vec<f64> = v
        .iter()
        .map(|x| if x.is_nan() { 270.0 } else { *x })
        .collect();
    for data in [&full, &v] {
        let (target, source) = integrals(&w, data, &w.apply(data, 0.0));
        assert!(rel(target, source) < 1e-12, "{target} vs {source}");
    }
    // Without missing values the plain integrals agree too.
    let out = w.apply(&full, 0.0);
    let plain: f64 = out.iter().zip(w.target_areas()).map(|(v, a)| v * a).sum();
    let source: f64 = full.iter().zip(w.source_areas()).map(|(v, a)| v * a).sum();
    assert!(rel(plain, source) < 1e-12);
}

/// Curvilinear grids regridded onto a target that covers them keep the area-weighted integral:
/// the rotated-pole grid onto a global grid, the polar grid onto its reference target.
#[test]
fn curvilinear_conserves() {
    let r = reference();
    let global = Rectilinear::from_edges(
        (0..=180).map(|i| -180.0 + 2.0 * i as f64).collect(),
        (0..=90).map(|i| -90.0 + 2.0 * i as f64).collect(),
        true,
    )
    .unwrap()
    .to_curvilinear()
    .unwrap();
    let sphere = 4.0 * std::f64::consts::PI * EARTH_RADIUS * EARTH_RADIUS;
    assert!(rel(global.areas().iter().sum(), sphere) < 1e-12);

    for (case, dst) in [
        ("rotated", global),
        ("polar", lat_lon(&r["polar"]).to_curvilinear().unwrap()),
    ] {
        let c = &r[case];
        let src = curvilinear(c);
        let w = Conservative::curvilinear(&src, &dst);
        // Every source cell with geometry is wholly covered by target cells.
        let mut covered = vec![0.0; w.sources()];
        for d in 0..w.targets() {
            for (s, a) in w.overlaps(d) {
                covered[s] += a;
            }
        }
        for (s, (got, area)) in covered.iter().zip(w.source_areas()).enumerate() {
            if area.is_nan() {
                assert_eq!(*got, 0.0, "{case} cell {s} has no geometry");
            } else {
                assert!(rel(*got, *area) < 1e-10, "{case} cell {s}: {got} of {area}");
            }
        }
        let v = values(&c["values"]);
        let (target, source) = integrals(&w, &v, &w.apply(&v, 0.0));
        assert!(rel(target, source) < 1e-10, "{case}: {target} vs {source}");
    }
}
