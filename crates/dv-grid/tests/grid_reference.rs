//! Checks transforms against PROJ (via pyproj), interpolation against xarray and nearest nodes
//! against scipy. The reference comes from `scripts/reference/grid.py`.

use std::path::PathBuf;

use dv_grid::crs::{Crs, ProjData, Transformer};
use dv_grid::mesh::Nodes;
use dv_grid::regular::{Axis, Grid};
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn reference() -> Json {
    let path = fixtures().join("reference/grid.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// PROJ's database with the grids pyproj saw, whatever the machine has installed.
fn no_grids() -> ProjData {
    ProjData::grids_in(Vec::<PathBuf>::new())
}

fn conus_grid() -> ProjData {
    ProjData::grids_in([fixtures().join("proj")])
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

#[test]
fn transforms_match_proj() {
    let wgs84 = Crs::wgs84();
    for case in reference()["transforms"].as_array().unwrap() {
        let code = case["epsg"].as_u64().unwrap() as u16;
        let crs = Crs::epsg(code).unwrap();
        assert!(!crs.is_geographic());
        let forward = Transformer::with_data(&wgs84, &crs, &no_grids()).unwrap();
        let inverse = Transformer::with_data(&crs, &wgs84, &no_grids()).unwrap();
        let (lon, lat) = (values(&case["lon"]), values(&case["lat"]));
        let (x, y) = (values(&case["x"]), values(&case["y"]));
        let (mut bx, mut by) = (lon.clone(), lat.clone());
        forward.transform(&mut bx, &mut by);
        for i in 0..lon.len() {
            let what = format!("EPSG:{code} ({}, {})", lon[i], lat[i]);
            let (px, py) = forward.point(lon[i], lat[i]).expect(&what);
            // The same library and database as pyproj: agreement to well under a millimetre.
            same(px, x[i], 1e-6, &what);
            same(py, y[i], 1e-6, &what);
            same(bx[i], x[i], 1e-6, &format!("{what} batch"));
            same(by[i], y[i], 1e-6, &format!("{what} batch"));
            // The inverse of a Helmert datum shift isn't exact: about a millimetre, in PROJ too.
            let (blon, blat) = inverse.point(px, py).expect(&what);
            let tol = if code == 27700 { 1e-7 } else { 1e-8 };
            same(blon, lon[i], tol, &format!("{what} back"));
            same(blat, lat[i], tol, &format!("{what} back"));
        }
    }
}

/// NAD27 to NAD83 without the NADCON grid falls back to a 14 m operation; with it, PROJ uses the
/// 0.15 m grid shift. The engine matches pyproj either way and says which it used.
#[test]
fn datum_shifts_match_proj_and_report_their_accuracy() {
    let r = reference();
    let d = &r["datum"];
    let (lon, lat) = (values(&d["lon"]), values(&d["lat"]));
    let (nad27, nad83) = (Crs::epsg(4267).unwrap(), Crs::epsg(4269).unwrap());
    assert!(nad27.is_geographic());
    for (case, data) in [("no_grids", no_grids()), ("conus_grid", conus_grid())] {
        let want = &d[case];
        let t = Transformer::with_data(&nad27, &nad83, &data).unwrap();
        let (x, y) = (values(&want["x"]), values(&want["y"]));
        for i in 0..lon.len() {
            let what = format!("{case} ({}, {})", lon[i], lat[i]);
            let (px, py) = t.point(lon[i], lat[i]).expect(&what);
            same(px, x[i], 1e-9, &what);
            same(py, y[i], 1e-9, &what);
        }
        let op = t.operation_at(lon[0], lat[0]).unwrap();
        let name = want["operation"].as_str().unwrap();
        assert!(name.starts_with(&op.name), "{case}: {} vs {name}", op.name);
        assert_eq!(op.accuracy_m, want["accuracy_m"].as_f64(), "{case}");
        assert!(!op.ballpark, "{case}");
    }

    // Without grids, the engine names the grids that would make NAD27 and British National Grid
    // accurate; every one is a grid pyproj also reports missing.
    for (src, dst, key, best) in [
        (4267, 4269, "unavailable_nad27", "us_noaa_conus.tif"),
        (
            4326,
            27700,
            "unavailable_bng",
            "uk_os_OSTN15_NTv2_OSGBtoETRS.tif",
        ),
    ] {
        let t = Transformer::with_data(
            &Crs::epsg(src).unwrap(),
            &Crs::epsg(dst).unwrap(),
            &no_grids(),
        )
        .unwrap();
        let missing = t.missing_grids().unwrap();
        let pyproj: Vec<&str> = d[key]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|o| o["grids"].as_array().unwrap())
            .map(|g| g.as_str().unwrap())
            .collect();
        assert!(!missing.is_empty(), "{key}");
        for m in &missing {
            for g in &m.grids {
                assert!(pyproj.contains(&g.as_str()), "{key}: {g} not in {pyproj:?}");
            }
            assert!(
                m.urls.iter().flatten().all(|u| u.starts_with("https://")),
                "{m:?}"
            );
        }
        assert!(
            missing.iter().any(|m| m.grids.iter().any(|g| g == best)),
            "{key}: {missing:?}"
        );
    }
    // With the grid installed, NADCON is no longer missing.
    let t = Transformer::with_data(&nad27, &nad83, &conus_grid()).unwrap();
    let missing = t.missing_grids().unwrap();
    assert!(
        missing
            .iter()
            .all(|m| m.grids.iter().all(|g| g != "us_noaa_conus.tif")),
        "{missing:?}"
    );
}

#[test]
fn crs_parsing() {
    assert!(Crs::parse("WGS84").unwrap().is_geographic());
    assert_eq!(
        Crs::epsg(27700).unwrap().name(),
        "OSGB36 / British National Grid"
    );
    let utm = Crs::parse("+proj=utm +zone=33 +datum=WGS84").unwrap();
    assert!(!utm.is_geographic());
    assert!(
        Crs::parse("+proj=longlat +ellps=airy +towgs84=446,-125,542,0,0,0,0")
            .unwrap()
            .is_geographic()
    );
    assert!(Crs::parse("EPSG:999999").is_err());
    assert!(Crs::parse("not a crs").is_err());
    // A transform can be made and used on any thread.
    let t = Transformer::new(&Crs::wgs84(), &utm).unwrap();
    let here = t.point(15.0, 52.0).unwrap();
    let there = std::thread::spawn(move || t.point(15.0, 52.0).unwrap())
        .join()
        .unwrap();
    assert_eq!(here, there);
}

#[test]
fn interpolation_matches_xarray() {
    let r = reference();
    let g = &r["grid"];
    let (lon, lat, data) = (values(&g["lon"]), values(&g["lat"]), values(&g["values"]));
    let grid = Grid {
        x: Axis::from_coords(&lon).unwrap().longitude(),
        y: Axis::from_coords(&lat).unwrap(),
        values: &data,
    };
    assert!(grid.x.periodic);
    let p = &r["points"];
    let (qlon, qlat) = (values(&p["lon"]), values(&p["lat"]));
    let (bil, near) = (values(&p["bilinear"]), values(&p["nearest"]));
    for i in 0..qlon.len() {
        let what = format!("({}, {})", qlon[i], qlat[i]);
        same(
            grid.bilinear(qlon[i], qlat[i]),
            bil[i],
            1e-9,
            &format!("bilinear {what}"),
        );
        same(
            grid.nearest(qlon[i], qlat[i]),
            near[i],
            0.0,
            &format!("nearest {what}"),
        );
    }

    let t = &r["regrid"];
    let x = Axis::from_coords(&values(&t["lon"])).unwrap();
    let y = Axis::from_coords(&values(&t["lat"])).unwrap();
    let out = grid.regrid_bilinear(&x, &y);
    for (i, (a, b)) in out.iter().zip(values(&t["values"])).enumerate() {
        same(*a, b, 1e-9, &format!("regrid cell {i}"));
    }
}

#[test]
fn regional_grids_never_clamp() {
    let r = &reference()["regional"];
    let data = values(&r["values"]);
    let grid = Grid {
        x: Axis::from_coords(&values(&r["lon"])).unwrap().longitude(),
        y: Axis::from_coords(&values(&r["lat"])).unwrap(),
        values: &data,
    };
    assert!(!grid.x.periodic);
    let (qlon, qlat, want) = (
        values(&r["qlon"]),
        values(&r["qlat"]),
        values(&r["bilinear"]),
    );
    let outside = want.iter().filter(|v| v.is_nan()).count();
    assert!(outside > 20, "the reference has points outside the grid");
    for i in 0..qlon.len() {
        same(
            grid.bilinear(qlon[i], qlat[i]),
            want[i],
            1e-9,
            &format!("({}, {})", qlon[i], qlat[i]),
        );
    }
}

#[test]
fn nearest_nodes_match_scipy() {
    let m = &reference()["mesh"];
    let nodes = Nodes::new(&values(&m["lon"]), &values(&m["lat"]));
    let (qlon, qlat) = (values(&m["qlon"]), values(&m["qlat"]));
    let meters = values(&m["meters"]);
    for (i, want) in m["nearest"].as_array().unwrap().iter().enumerate() {
        let (node, d) = nodes.nearest(qlon[i], qlat[i], f64::INFINITY).unwrap();
        assert_eq!(node as u64, want.as_u64().unwrap(), "query {i}");
        same(d, meters[i], 1e-3, &format!("query {i} distance"));
        // Just inside and just outside the maximum distance.
        assert!(nodes.nearest(qlon[i], qlat[i], d + 1.0).is_some());
        assert!(nodes.nearest(qlon[i], qlat[i], d - 1.0).is_none());
    }
}
