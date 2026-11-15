# Fixtures

Small datasets for tests live here (each under a few MB, with its source noted). Larger data for the M0 spike is downloaded by `scripts/fetch-spike-data.sh` into `fixtures/spike/data/`, which is git-ignored.

| Fixture | What it is | Source | Made by |
|---|---|---|---|
| `taxi-20k.parquet` | 20,000 evenly spaced rows of NYC yellow taxi trips, Jan–Mar 2015 | NYC Taxi & Limousine Commission trip records (public), via Azure Open Datasets | `scripts/reference/make-taxi-fixture.py` |
| `tables/trips.*` | 500 taxi trips with text, boolean and null edge cases, in CSV, TSV, NDJSON, JSON, Parquet, Arrow, Excel and SQLite | Derived from `taxi-20k.parquet` | `scripts/reference/make-table-fixtures.py` |
| `tables/payment-types.csv`, `tables/payment-fees.csv` | Lookup tables for join tests: unique keys, and several rows per key | Hand-written | — |
| `tables/obs.csv`, `tables/visits.csv` | Synthetic observations at two stations (odd-second spacing, so nearest is never a tie) and station visits, for time joins | Generated | `scripts/reference/time-join.py` |
| `arrays/*` | Tiny arrays: packed NetCDF 4 (int16 + scale/offset, `_FillValue`, `missing_value`, `_Unsigned`, noleap time), NetCDF 3, plain HDF5, `.npy` (C and Fortran order), `.npz` | Generated | `scripts/reference/make-array-fixtures.py` |
| `geo/*` | Small vector layers (GeoJSON with mixed types, Shapefile, FlatGeobuf in EPSG:3857, a two-layer GeoPackage, GeoParquet) with non-ASCII names, a hole and a multipolygon; and two GeoTIFFs (tiled deflate float32 in UTM 33N with nodata, striped LZW uint8 in WGS 84) | Generated | `scripts/reference/make-geo-fixtures.py` |
| `arco-like.zarr` | A tiny synthetic array in ARCO-ERA5's layout: Zarr format 2, Blosc LZ4 + shuffle, NaN fill, one chunk missing | Generated (no external data) | `scripts/reference/make-zarr-fixture.py` |

Reference results in `reference/` are computed by other tools, so tests compare against an independent implementation:

| Reference | Computed with |
|---|---|
| `reference/taxi-20k.profile.json` | DuckDB (`scripts/reference/profile.py`) |
| `reference/arco-like.json` | xarray (`scripts/reference/make-zarr-fixture.py`) |
| `reference/taxi-20k.selection.json` | DuckDB (`scripts/reference/selection.py`) |
| `reference/units.json` | pint (`scripts/reference/units.py`) |
| `reference/tables.json` | DuckDB reading each file; pandas + openpyxl for Excel, pyarrow for Arrow (`scripts/reference/make-table-fixtures.py`) |
| `reference/query.json` | DuckDB (`scripts/reference/query.py`) |
| `reference/formulas.json` | polars (`scripts/reference/formulas.py`) |
| `reference/verification.json` | Synthetic data and xskillscore scores (`scripts/reference/verification.py`) |
| `reference/cftime.json` | cftime and xarray `convert_calendar` (`scripts/reference/cftime-calendars.py`) |
| `reference/time-join.json` | polars `join_asof` (`scripts/reference/time-join.py`) |
| `reference/grid.json` | pyproj (PROJ) transforms, xarray pointwise interpolation, scipy nearest nodes, on synthetic grids (`scripts/reference/grid.py`) |
| `reference/arrays.json` | xarray with cftime (`scripts/reference/make-array-fixtures.py`) |
| `reference/geo.json` | GDAL through pyogrio and rasterio; geopandas for GeoParquet (`scripts/reference/make-geo-fixtures.py`) |
