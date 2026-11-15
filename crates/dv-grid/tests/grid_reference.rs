//! Checks transforms against PROJ (via pyproj), interpolation against xarray and nearest nodes
//! against scipy. The reference comes from `scripts/reference/grid.py`.

use std::path::PathBuf;

use dv_grid::crs::{Crs, Transformer};
use dv_grid::mesh::Nodes;
use dv_grid::regular::{Axis, Grid};
use serde_json::Value as Json;

fn reference() -> Json {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/grid.json");
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

#[test]
fn transforms_match_proj() {
    let wgs84 = Crs::wgs84();
    for case in reference()["transforms"].as_array().unwrap() {
        let code = case["epsg"].as_u64().unwrap() as u16;
        let crs = Crs::epsg(code).unwrap();
        let forward = Transformer::new(&wgs84, &crs);
        let inverse = Transformer::new(&crs, &wgs84);
        let (lon, lat) = (values(&case["lon"]), values(&case["lat"]));
        let (x, y) = (values(&case["x"]), values(&case["y"]));
        for i in 0..lon.len() {
            let what = format!("EPSG:{code} ({}, {})", lon[i], lat[i]);
            let (px, py) = forward.point(lon[i], lat[i]).expect(&what);
            // Within a millimeter, or a centimeter where a datum shift is involved: proj4rs and
            // PROJ apply the same Helmert parameters with slightly different arithmetic.
            let tol = if code == 27700 { 1e-2 } else { 1e-3 };
            same(px, x[i], tol, &what);
            same(py, y[i], tol, &what);
            let (blon, blat) = inverse.point(px, py).expect(&what);
            let tol = if code == 27700 { 1e-7 } else { 1e-8 };
            same(blon, lon[i], tol, &format!("{what} back"));
            same(blat, lat[i], tol, &format!("{what} back"));
        }
    }
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
