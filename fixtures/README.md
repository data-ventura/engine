# Fixtures

Small datasets for tests live here (each under a few MB, with its source noted). Larger data for the M0 spike is downloaded by `scripts/fetch-spike-data.sh` into `fixtures/spike/data/`, which is git-ignored.

| Fixture | What it is | Source | Made by |
|---|---|---|---|
| `taxi-20k.parquet` | 20,000 evenly spaced rows of NYC yellow taxi trips, Jan–Mar 2015 | NYC Taxi & Limousine Commission trip records (public), via Azure Open Datasets | `scripts/reference/make-taxi-fixture.py` |
| `tables/trips.*` | 500 taxi trips with text, boolean and null edge cases, in CSV, TSV, NDJSON, JSON, Parquet, Arrow, Excel and SQLite | Derived from `taxi-20k.parquet` | `scripts/reference/make-table-fixtures.py` |
| `tables/payment-types.csv`, `tables/payment-fees.csv` | Lookup tables for join tests: unique keys, and several rows per key | Hand-written | — |
| `tables/obs.csv`, `tables/visits.csv` | Synthetic observations at two stations (odd-second spacing, so nearest is never a tie) and station visits, for time joins | Generated | `scripts/reference/time-join.py` |
| `arrays/*` | Tiny arrays: packed NetCDF 4 (int16 + scale/offset, `_FillValue`, `missing_value`, `_Unsigned`, noleap time), NetCDF 3, plain HDF5, `.npy` (C and Fortran order), `.npz` | Generated | `scripts/reference/make-array-fixtures.py` |
| `arrays/*.grib1`, `arrays/*.grib2` | Tiny GRIB files written by ecCodes: GRIB 2 forecasts on a regular grid (pressure levels out of order, two steps, CCSDS and complex packing, a bitmap of missing points, clashing height levels), GRIB 1 analyses at two times on a south-to-north grid, reduced Gaussian, polar stereographic and Lambert conformal grids, and a four-member GRIB 2 ensemble with one member missing at one step | Generated | `scripts/reference/make-grib-fixtures.py` |
| `tables/trips.duckdb` | The `trips` table in a DuckDB database, with a view and a second schema holding DECIMAL, DATE, TIMESTAMPTZ, HUGEINT, ENUM, INTERVAL, UUID and BLOB columns | Derived from `tables/trips.parquet` | `scripts/reference/make-duckdb-fixture.py` |
| `geo/*` | Small vector layers (GeoJSON with mixed types, Shapefile, FlatGeobuf in EPSG:3857, a two-layer GeoPackage, GeoParquet) with non-ASCII names, a hole and a multipolygon; and two GeoTIFFs (tiled deflate float32 in UTM 33N with nodata, striped LZW uint8 in WGS 84) | Generated | `scripts/reference/make-geo-fixtures.py` |
| `geo/cog-*.tif` | Cloud Optimized GeoTIFFs written by GDAL's COG driver: a float32 DEM (deflate + floating-point predictor, nodata, partial edge tiles, three average overviews) and a 3-band uint8 image (zstd, pixel-interleaved, three nearest overviews) | Generated | `scripts/reference/make-cog-fixtures.py` |
| `proj/us_noaa_conus.tif` | NADCON grid for NAD27 to NAD83 in the conterminous US (173 KB) | NOAA, public domain, from PROJ-data (`cdn.proj.org`) | Copied unchanged |
| `spec/example.dvspec` | A workspace using every recipe kind | Hand-written | — |
| `zarr/*` | The `packed.nc` variables as a Zarr format 2 store (consolidated; Blosc zstd, gzip, zlib, `/` chunk keys, a Fortran-order array, a missing chunk), a Zarr format 3 store (unconsolidated; zstd, gzip + crc32c, sharding with Blosc LZ4, a missing chunk) and an Icechunk repository with two branches, a tag and three commits | Generated | `scripts/reference/make-zarr-stores.py` |
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
| `reference/spill.json` | DuckDB, on 1,000,000-row tables both sides make from the same integer formulas, too large to check in (`scripts/reference/spill.py`) |
| `reference/formulas.json` | polars (`scripts/reference/formulas.py`) |
| `reference/verification.json` | Synthetic data and xskillscore scores (`scripts/reference/verification.py`) |
| `reference/cftime.json` | cftime and xarray `convert_calendar` (`scripts/reference/cftime-calendars.py`) |
| `reference/time-join.json` | polars `join_asof` (`scripts/reference/time-join.py`) |
| `reference/regrid.json` | xarray-regrid conservative regridding (nested lat-lon and planar grids), and S2 through spherely for spherical overlaps (lat-lon cells with densified parallels, rotated-pole and polar stereographic curvilinear grids) and point-in-cell lookup, on synthetic grids (`scripts/reference/regrid.py`) |
| `reference/grid.json` | pyproj (PROJ) transforms, xarray pointwise interpolation, scipy nearest nodes, on synthetic grids (`scripts/reference/grid.py`) |
| `reference/zarr.json` | xarray on zarr-python 3, and Icechunk read-only sessions (`scripts/reference/make-zarr-stores.py`) |
| `reference/arrays.json` | xarray with cftime (`scripts/reference/make-array-fixtures.py`) |
| `reference/grib.json` | cfgrib (`open_datasets`) on ecCodes (`scripts/reference/make-grib-fixtures.py`) |
| `reference/duckdb.json` | DuckDB reading the database, and the Arrow types it exports (`scripts/reference/make-duckdb-fixture.py`) |
| `reference/geo.json` | GDAL through pyogrio and rasterio; geopandas for GeoParquet (`scripts/reference/make-geo-fixtures.py`) |
| `reference/cog.json` | GDAL through rasterio: levels, windows, tile sizes (`BLOCK_SIZE`) and the overviews reads into smaller buffers chose (`scripts/reference/make-cog-fixtures.py`) |
| `reference/fingerprints.json` | Python hashlib (`scripts/reference/fingerprints.py`) |
| `reference/cvd.json` | colorspacious (Machado et al. 2009 colour-vision-deficiency simulation) and colour-science (CIELAB, CIEDE2000), and palette checks combining them (`scripts/reference/cvd.py`) |
| `reference/pdf-text.json` | pdfium (through pypdfium2): the characters and boxes it extracts from dv-render's golden PDFs (`scripts/reference/pdf-text.py`) |
