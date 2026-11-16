//! Checks Zarr decoding against xarray.
//!
//! `chunks_match_xarray` reads single chunks of a fixture in ARCO-ERA5's layout (Zarr format 2,
//! Blosc LZ4 with shuffle, NaN fill, one step per chunk, one chunk missing), from
//! `scripts/reference/make-zarr-fixture.py`. The rest open Zarr format 2 and 3 stores and an
//! Icechunk repository as array sources and check every decoded value; those fixtures and their
//! reference come from `scripts/reference/make-zarr-stores.py`.

use std::path::PathBuf;

use dv_data::array::{self, ArraySource};
use dv_data::zarr::{IcechunkVersion, ZarrArray, ZarrSource};
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn reference() -> Json {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("reference/zarr.json")).unwrap())
        .unwrap()
}

fn assert_matches(what: &str, got: &[f64], want: &[Json]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, x)) in got.iter().zip(want).enumerate() {
        match x.as_f64() {
            None => assert!(g.is_nan(), "{what}[{i}]: {g}, expected missing"),
            Some(x) => assert!(
                (g - x).abs() <= 1e-9 * x.abs().max(1.0),
                "{what}[{i}]: {g} vs xarray {x}"
            ),
        }
    }
}

/// Every variable xarray decoded, apart from time (checked as numbers below).
fn check(what: &str, src: &dyn ArraySource, want: &Json) {
    for (var, w) in want.as_object().unwrap() {
        if var == "time" {
            continue;
        }
        let v = src
            .variable(var)
            .unwrap_or_else(|| panic!("{what}: no {var}"));
        let shape: Vec<u64> = serde_json::from_value(w["shape"].clone()).unwrap();
        let dims: Vec<String> = serde_json::from_value(w["dims"].clone()).unwrap();
        assert_eq!(v.shape, shape, "{what} {var}: shape");
        assert_eq!(v.dims, dims, "{what} {var}: dims");
        let got = array::read_all(src, var).unwrap();
        assert_matches(
            &format!("{what} {var}"),
            &got,
            w["values"].as_array().unwrap(),
        );
    }
    assert_eq!(
        src.variables().len(),
        want.as_object().unwrap().len(),
        "{what}: variables"
    );
    // Time stays numeric, with the attributes dv_model::time decodes it from.
    let time = src.variable("time").unwrap();
    assert_eq!(
        time.attr("calendar").and_then(|a| a.as_text()),
        Some("noleap")
    );
    assert_eq!(
        time.attr("units").and_then(|a| a.as_text()),
        Some("days since 2000-02-27")
    );
    assert_eq!(array::read_all(src, "time").unwrap(), [0.0, 1.0, 2.0]);
}

#[tokio::test]
async fn chunks_match_xarray() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures.join("reference/arco-like.json")).unwrap(),
    )
    .unwrap();
    let array = ZarrArray::open_dir(fixtures.join("arco-like.zarr"), "/t2m")
        .await
        .unwrap();
    assert_eq!(array.shape(), [4, 9, 16]);
    assert_eq!(array.chunk_grid(), [4, 1, 1]);

    let want = reference["values"].as_array().unwrap();
    let per_step = 9 * 16;
    for t in 0..4u64 {
        let got = array.chunk_f32(&[t, 0, 0]).await.unwrap();
        let expected = &want[t as usize * per_step..(t as usize + 1) * per_step];
        for (i, (g, w)) in got.iter().zip(expected).enumerate() {
            match w.as_f64() {
                // xarray reads float32 values; JSON stores them as the same f64.
                Some(w) => assert_eq!(f64::from(*g), w, "step {t} value {i}"),
                None => assert!(g.is_nan(), "step {t} value {i} should be missing"),
            }
        }
        // A subset of the chunk (rows 2..5, columns 3..7) matches the same values.
        let sub = array
            .chunk_subset_f32(&[t, 0, 0], &[0..1, 2..5, 3..7])
            .await
            .unwrap();
        let mut k = 0;
        for r in 2..5 {
            for c in 3..7 {
                let (g, w) = (sub[k], &expected[r * 16 + c]);
                match w.as_f64() {
                    Some(w) => assert_eq!(f64::from(g), w, "step {t} subset ({r}, {c})"),
                    None => assert!(g.is_nan(), "step {t} subset ({r}, {c}) should be missing"),
                }
                k += 1;
            }
        }
        assert_eq!(sub.len(), k);
    }
}

#[test]
fn zarr_stores_match_xarray() {
    let r = reference();
    for store in ["v2.zarr", "v3.zarr"] {
        let src = array::open(fixtures().join("zarr").join(store)).unwrap();
        check(store, src.as_ref(), &r[store]);
        assert_eq!(src.attrs()["title"].as_text(), Some("dv zarr fixture"));
        let t2m = src.variable("t2m").unwrap();
        assert_eq!(t2m.dtype, "i2", "{store}");
        assert_eq!(
            t2m.attr("_FillValue").and_then(|a| a.as_f64()),
            Some(-32767.0)
        );
        assert!(t2m.attr("_ARRAY_DIMENSIONS").is_none(), "{store}");
        assert_eq!(src.variable("cloud").unwrap().dtype, "i1", "{store}");
    }
}

#[test]
fn groups_and_arrays_open_by_path() {
    let src = ZarrSource::open_dir(fixtures().join("zarr/v3.zarr"), "/precip").unwrap();
    assert_eq!(src.variables().len(), 1);
    assert_eq!(src.variables()[0].name, "precip");
    assert_eq!(src.variables()[0].dims, ["time", "lat", "lon"]);
    let url = format!(
        "file://{}",
        fixtures()
            .join("zarr/v2.zarr")
            .canonicalize()
            .unwrap()
            .display()
    );
    let src = ZarrSource::open(&url, "/").unwrap();
    assert_eq!(src.chunk_shape("t2m"), Some(vec![1, 3, 4]));
    assert_eq!(src.chunk_shape("fortran"), Some(vec![2, 3]));
    assert!(ZarrSource::open_dir(fixtures().join("zarr/v2.zarr"), "/nothing").is_err());
}

/// Hyperslabs that cross chunk (and shard) boundaries read only what they cover, row-major, and
/// work from inside an async runtime, both blocking and awaited.
#[tokio::test]
async fn hyperslabs_cross_chunks() {
    let r = reference();
    for store in ["v2.zarr", "v3.zarr"] {
        let src = ZarrSource::open_dir(fixtures().join("zarr").join(store), "/").unwrap();
        for var in ["t2m", "precip"] {
            let all = r[store][var]["values"].as_array().unwrap();
            // Steps 1..3, rows 2..5, columns 3..7 of (3, 5, 8), over chunks of (1, 3, 4).
            let want: Vec<Json> = (1..3)
                .flat_map(|t| (2..5).flat_map(move |y| (3..7).map(move |x| (t, y, x))))
                .map(|(t, y, x)| all[t * 40 + y * 8 + x].clone())
                .collect();
            let got = array::read(&src, var, &[1, 2, 3], &[2, 3, 4]).unwrap();
            assert_matches(&format!("{store} {var} slab"), &got, &want);
            let raw = src
                .read_raw_async(var, &[1, 2, 3], &[2, 3, 4])
                .await
                .unwrap();
            assert_eq!(
                raw.len(),
                want.len(),
                "{store} {var}: async read of the same slab"
            );
        }
        let cloud = r[store]["cloud"]["values"].as_array().unwrap();
        let want: Vec<Json> = (1..4).map(|y| cloud[y * 8 + 6].clone()).collect();
        let got = array::read(&src, "cloud", &[1, 6], &[3, 1]).unwrap();
        assert_matches(&format!("{store} cloud column"), &got, &want);
        assert!(
            src.read_raw("t2m", &[2, 0, 0], &[2, 1, 1]).is_err(),
            "out of range"
        );
    }
}

#[cfg(feature = "icechunk")]
#[test]
fn icechunk_versions_match_xarray() {
    let r = reference();
    let repo = &r["repo.icechunk"];
    let location = fixtures().join("zarr/repo.icechunk");
    let location = location.to_str().unwrap();
    let first = repo["snapshots"]["first"].as_str().unwrap().to_string();
    for (label, version) in [
        ("main", IcechunkVersion::Branch("main".into())),
        ("dev", IcechunkVersion::Branch("dev".into())),
        ("v1", IcechunkVersion::Tag("v1".into())),
        ("first", IcechunkVersion::Snapshot(first)),
    ] {
        let src = ZarrSource::open_icechunk(location, &version, "/").unwrap();
        check(&format!("icechunk {label}"), &src, &repo[label]);
    }
    // A directory with Icechunk's layout opens at main.
    let src = array::open(fixtures().join("zarr/repo.icechunk")).unwrap();
    check("icechunk (open)", src.as_ref(), &repo["main"]);
    let (branches, tags) = ZarrSource::icechunk_refs(location).unwrap();
    assert_eq!(branches, ["dev", "main"]);
    assert_eq!(tags, ["v1"]);
    let missing = IcechunkVersion::Branch("nope".into());
    assert!(ZarrSource::open_icechunk(location, &missing, "/").is_err());
}

#[cfg(not(feature = "icechunk"))]
#[test]
fn icechunk_needs_the_feature() {
    let location = fixtures().join("zarr/repo.icechunk");
    let main = IcechunkVersion::Branch("main".into());
    let err = ZarrSource::open_icechunk(location.to_str().unwrap(), &main, "/").unwrap_err();
    assert!(err.to_string().contains("icechunk"), "{err}");
}
