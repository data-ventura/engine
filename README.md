# Data Ventura engine

The open-source engine behind [Data Ventura](https://dataventura.com), a fast, native app for exploring any dataset: tables, N-dimensional arrays, geospatial data and time series.

This repository holds everything that reads, computes and draws data, so results stay reproducible and citable. It's Apache-2.0. The desktop apps live in a separate, private repository and depend on this one.

> **Status:** pre-alpha. The M0 spike is in progress; nothing here is usable yet.

## Crates

| Crate | Purpose |
|---|---|
| `dv-data` | Readers for tables (Parquet, CSV, Arrow, JSON, Excel, SQLite/DuckDB), arrays (NetCDF, HDF5, Zarr/Icechunk, GRIB, .npy) and geospatial data (GeoParquet, GeoJSON, Shapefile, GeoPackage, FlatGeobuf, GeoTIFF/COG); object storage and SFTP |
| `dv-query` | Lazy table queries on Apache DataFusion |
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions use the Developer Certificate of Origin (`git commit -s`); there is no CLA.

## License

Apache-2.0. See [LICENSE](LICENSE).
