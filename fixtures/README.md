# Fixtures

Small datasets for tests live here (each under a few MB, with its source noted). Larger data for the M0 spike is downloaded by `scripts/fetch-spike-data.sh` into `fixtures/spike/data/`, which is git-ignored.

| Fixture | What it is | Source | Made by |
|---|---|---|---|
| `taxi-20k.parquet` | 20,000 evenly spaced rows of NYC yellow taxi trips, Jan–Mar 2015 | NYC Taxi & Limousine Commission trip records (public), via Azure Open Datasets | `scripts/reference/make-taxi-fixture.py` |
| `tables/trips.*` | 500 taxi trips with text, boolean and null edge cases, in CSV, TSV, NDJSON, JSON, Parquet, Arrow, Excel and SQLite | Derived from `taxi-20k.parquet` | `scripts/reference/make-table-fixtures.py` |
| `tables/payment-types.csv`, `tables/payment-fees.csv` | Lookup tables for join tests: unique keys, and several rows per key | Hand-written | — |
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
