//! Links the system ecCodes library for the `grib` feature.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "grib")]
    if let Err(e) = pkg_config::Config::new()
        .atleast_version("2.20")
        .probe("eccodes")
    {
        panic!(
            "the `grib` feature needs ecCodes 2.20 or later (`brew install eccodes`, \
             `apt install libeccodes-dev`): {e}"
        );
    }
}
