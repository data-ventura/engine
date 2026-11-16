# Data Ventura engine

The open-source engine behind [Data Ventura](https://dataventura.com), a fast, native app for exploring any dataset: tables, N-dimensional arrays, geospatial data and time series.

This repository holds everything that reads, computes and draws data, so results stay reproducible and citable. It's Apache-2.0. The desktop apps live in a separate, private repository and depend on this one.

> **Status:** pre-alpha. The M0 spike is done and M1 (the engine crates) is in progress; nothing here is stable yet.

## Crates

| Crate | Purpose |
|---|---|
| `dv-data` | Readers for tables (Parquet, CSV, Arrow, JSON, Excel, SQLite/DuckDB), arrays (NetCDF, HDF5, Zarr/Icechunk, GRIB, .npy) and geospatial data (GeoParquet, GeoJSON, Shapefile, GeoPackage, FlatGeobuf, GeoTIFF/COG); object storage and SFTP |
| `dv-query` | Lazy table queries on Apache DataFusion, spilling sorts, aggregates and joins to disk when they outgrow their memory |
| `dv-model` | Datasets, fields, roles, units, profiles, selections |
| `dv-grid` | Projections (PROJ), regridding, curvilinear and unstructured meshes |
| `dv-compute` | Expression language, window functions, reductions, verification metrics |
| `dv-render` | wgpu renderer, including `dv-plot` (charts drawn with vello) and SVG/PDF/PNG export |
| `dv-spec` | The `.dvspec` workspace format, JSON Schema and lock files |
| `dv-ext` | The `dv:ext` WebAssembly extension host and Rust SDK |
| `dv-py` | The connected-runtime host (Arrow shared memory) for Python, later R and Julia |
| `dv-serve` | Headless remote helper, run over SSH |

Crate directories use short names. Package names use the `dataventura-` prefix (for example `dataventura-ext`), and library names use `dv_` (for example `dv_ext`).

## Other directories

- `python/dataventura`: the Python package for connected-runtime extensions (published to PyPI at M3).
- `wit/`: the `dv:ext` WIT contract for WebAssembly extensions (draft).
- `schema/`: the `.dvspec` JSON Schema (draft).
- `fixtures/`: small test datasets; `scripts/fetch-spike-data.sh` downloads the larger M0 spike data.

## Building

Requires the Rust toolchain in `rust-toolchain.toml` (installed automatically by rustup).

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```

Projections use PROJ 9.9 or later (`brew install proj`). Without it, the build compiles PROJ from source, which needs CMake, SQLite and libtiff (`apt install cmake sqlite3 libtiff-dev`). The engine never downloads PROJ's datum grids; it uses the ones installed and reports when a more accurate transform needs a grid that isn't.

NetCDF and HDF5 support is behind `dv-data`'s `netcdf` feature, which links the system netCDF-C library (`brew install netcdf` or `apt install libnetcdf-dev`). GRIB 1 and 2 support is behind its `grib` feature, which links the system ecCodes library through pkg-config (`brew install eccodes` or `apt install libeccodes-dev`). DuckDB database files are behind its `duckdb` feature, which compiles DuckDB from source (no system library, but several minutes of C++ on a clean build). Icechunk repositories are behind its `icechunk` feature (pure Rust, no system libraries, but a large dependency tree); plain Zarr stores need neither. CI builds and tests with `--all-features`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions use the Developer Certificate of Origin (`git commit -s`); there is no CLA.

## License

Apache-2.0. See [LICENSE](LICENSE).
