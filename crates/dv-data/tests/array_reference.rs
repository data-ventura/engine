//! Reads array fixtures and checks every decoded value against xarray. The fixtures and
//! reference come from `scripts/reference/make-array-fixtures.py`.

use std::path::PathBuf;

use dv_data::array::{self, ArraySource};
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn reference() -> Json {
    serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("reference/arrays.json")).unwrap(),
    )
    .unwrap()
}

fn check(file: &str, src: &dyn ArraySource, want: &Json) {
    for (var, w) in want.as_object().unwrap() {
        if var == "time" {
            continue;
        }
        let v = src
            .variable(var)
            .unwrap_or_else(|| panic!("{file}: no {var}"));
        let shape: Vec<u64> = w["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        assert_eq!(v.shape, shape, "{file} {var}: shape");
        if let Some(dims) = w.get("dims") {
            let dims: Vec<&str> = dims
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_str().unwrap())
                .collect();
            assert_eq!(v.dims, dims, "{file} {var}: dims");
        }
        let got = array::read_all(src, var).unwrap();
        let want = w["values"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{file} {var}: length");
        for (i, (g, x)) in got.iter().zip(want).enumerate() {
            match x.as_f64() {
                None => assert!(g.is_nan(), "{file} {var}[{i}]: {g}, expected missing"),
                Some(x) => assert!(
                    (g - x).abs() <= 1e-9 * x.abs().max(1.0),
                    "{file} {var}[{i}]: {g} vs xarray {x}"
                ),
            }
        }
    }
}

#[test]
fn numpy_files_match() {
    let r = reference();
    for file in ["grid.npy", "fortran.npy", "bundle.npz"] {
        let src = array::open(fixtures().join("arrays").join(file)).unwrap();
        check(file, src.as_ref(), &r[file]);
    }
}

#[test]
fn hyperslabs_are_row_major() {
    let src = array::open(fixtures().join("arrays/grid.npy")).unwrap();
    let all = array::read_all(src.as_ref(), "grid").unwrap();
    // grid is (3, 5, 8): take time 1, rows 2..4, columns 3..6.
    let part = array::read(src.as_ref(), "grid", &[1, 2, 3], &[1, 2, 3]).unwrap();
    let expect: Vec<f64> = (2..4)
        .flat_map(|y| (3..6).map(move |x| (y, x)))
        .map(|(y, x)| all[40 + y * 8 + x])
        .collect();
    assert_eq!(part, expect);
    assert!(
        array::read(src.as_ref(), "grid", &[2, 0, 0], &[2, 1, 1]).is_err(),
        "out of range"
    );
}

#[cfg(feature = "netcdf")]
#[test]
fn netcdf_and_hdf5_match_xarray() {
    let r = reference();
    for file in ["packed.nc", "classic.nc", "plain.h5"] {
        let src = array::open(fixtures().join("arrays").join(file)).unwrap();
        check(file, src.as_ref(), &r[file]);
    }
    let src = array::open(fixtures().join("arrays/packed.nc")).unwrap();
    assert_eq!(src.attrs()["title"].as_text(), Some("dv fixture"));
    let t2m = src.variable("t2m").unwrap();
    assert_eq!(t2m.dtype, "i2");
    // Time stays numeric, with the attributes dv_model::time decodes it from (checked against
    // cftime in dv-model's tests).
    let time = src.variable("time").unwrap();
    assert_eq!(
        time.attr("calendar").and_then(|a| a.as_text()),
        Some("noleap")
    );
    assert_eq!(
        time.attr("units").and_then(|a| a.as_text()),
        Some("days since 2000-02-27")
    );
    assert_eq!(
        array::read_all(src.as_ref(), "time").unwrap(),
        [0.0, 1.0, 2.0]
    );
}

#[cfg(not(feature = "netcdf"))]
#[test]
fn netcdf_needs_the_feature() {
    let err = array::open(fixtures().join("arrays/packed.nc")).unwrap_err();
    assert!(err.to_string().contains("netcdf"), "{err}");
}
