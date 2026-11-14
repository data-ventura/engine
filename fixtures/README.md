# Fixtures

Small datasets for tests live here (each under a few MB, with its source noted). Larger data for the M0 spike is downloaded by `scripts/fetch-spike-data.sh` into `fixtures/spike/data/`, which is git-ignored.

| Fixture | What it is | Source | Made by |
|---|---|---|---|
| `taxi-20k.parquet` | 20,000 evenly spaced rows of NYC yellow taxi trips, Jan–Mar 2015 | NYC Taxi & Limousine Commission trip records (public), via Azure Open Datasets | `scripts/reference/make-taxi-fixture.py` |
| `arco-like.zarr` | A tiny synthetic array in ARCO-ERA5's layout: Zarr format 2, Blosc LZ4 + shuffle, NaN fill, one chunk missing | Generated (no external data) | `scripts/reference/make-zarr-fixture.py` |

Reference results in `reference/` are computed by other tools, so tests compare against an independent implementation:

| Reference | Computed with |
|---|---|
| `reference/taxi-20k.profile.json` | DuckDB (`scripts/reference/profile.py`) |
| `reference/arco-like.json` | xarray (`scripts/reference/make-zarr-fixture.py`) |
