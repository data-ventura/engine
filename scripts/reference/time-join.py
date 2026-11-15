"""Makes observation tables for time-join tests, and joins them with polars' join_asof.

Usage: time-join.py <fixtures dir> <out.json>

Run with: uv run --with polars python scripts/reference/time-join.py fixtures fixtures/reference/time-join.json

Observations are spaced by odd numbers of seconds, so no left time (whole seconds) is ever exactly
halfway between two observations and "nearest" is never a tie. Left rows with a null time can't
be passed to join_asof; the engine must leave them unmatched.
"""

import json
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl

fx, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(3)


def series(start, n, station):
    t = start
    rows = []
    for i in range(n):
        t += timedelta(seconds=int(rng.integers(300, 1500)) * 2 + 1)
        rows.append((t, station, round(float(rng.normal(5, 3)), 2)))
    return rows


start = datetime(2014, 12, 31, 23, 0, 0)
rows = series(start, 2600, "A") + series(start, 900, "B")
obs = pl.DataFrame(rows, schema=["time", "station", "temp"], orient="row")
obs = obs.with_row_index("obs_id", offset=1).with_columns(pl.col("obs_id").cast(pl.Int64))
obs.write_csv(f"{fx}/tables/obs.csv", datetime_format="%Y-%m-%d %H:%M:%S")

visits = pl.DataFrame({
    "visit": list(range(1, 301)),
    "station": [["A", "B", "C"][i % 3] for i in range(300)],
    "at": [start + timedelta(seconds=int(s)) for s in rng.integers(3600, 3600 * 24 * 20, 300)],
})
visits.write_csv(f"{fx}/tables/visits.csv", datetime_format="%Y-%m-%d %H:%M:%S")

trips = pl.read_csv(f"{fx}/tables/trips.csv", try_parse_dates=True).select("id", "pickup")
obs = pl.read_csv(f"{fx}/tables/obs.csv", try_parse_dates=True)
visits = pl.read_csv(f"{fx}/tables/visits.csv", try_parse_dates=True)


def join(left, left_on, by, strategy, tolerance):
    l = left.drop_nulls(left_on).sort(left_on)
    r = obs.sort("time")
    j = l.join_asof(
        r.with_columns(pl.col("time").alias("obs_time")),
        left_on=left_on,
        right_on="time",
        by=by,
        strategy=strategy,
        tolerance=tolerance,
    )
    key = "id" if "id" in left.columns else "visit"
    j = j.with_columns(((pl.col("obs_time") - pl.col(left_on)).dt.total_microseconds().abs()).alias("d"))
    matched = j.filter(pl.col("obs_id").is_not_null())
    return {
        "matches": {str(k): v for k, v in zip(j[key].to_list(), j["obs_id"].to_list())},
        "matched": matched.height,
        "mean_abs_us": matched["d"].mean(),
        "max_abs_us": matched["d"].max(),
    }


ref = {
    "trips_nearest_10m": join(trips, "pickup", None, "nearest", "10m"),
    "trips_backward_10m": join(trips, "pickup", None, "backward", "10m"),
    "trips_forward_10m": join(trips, "pickup", None, "forward", "10m"),
    "visits_by_station_nearest_15m": join(visits, "at", "station", "nearest", "15m"),
}
with open(out, "w") as f:
    json.dump(ref, f)
print({k: v["matched"] for k, v in ref.items()})
