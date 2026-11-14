//! Checks Zarr chunk decoding against xarray on a fixture in ARCO-ERA5's layout (Zarr format 2,
//! Blosc LZ4 with shuffle, NaN fill, one step per chunk, one chunk missing). The fixture and
//! reference come from `scripts/reference/make-zarr-fixture.py`.

use std::path::PathBuf;

use dv_data::zarr::ZarrArray;

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
