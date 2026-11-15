"""Evaluates formulas with polars, for dv-compute's tests.

Usage: formulas.py <trips.csv> <out.json>

Run with: uv run --with polars python scripts/reference/formulas.py fixtures/tables/trips.csv fixtures/reference/formulas.json

Each case is a dv-compute formula and the polars expression that means the same, evaluated on
the trips fixture ordered by id. Windows follow polars' defaults: rolling windows need a full
window of non-null values, cum_sum leaves nulls in place, and rank(method="min") is SQL's RANK.
"""

import json
import math
import sys

import polars as pl

path, out = sys.argv[1], sys.argv[2]
df = pl.read_csv(path, try_parse_dates=True).sort("id")
c = pl.col

CASES = {
    "fare / (distance + 1)": c("fare") / (c("distance") + 1),
    'if(distance > 2, "long", "short")': pl.when(c("distance") > 2).then(pl.lit("long")).otherwise(pl.lit("short")),
    "sqrt(distance) * 10": c("distance").sqrt() * 10,
    "fare - mean(fare)": c("fare") - c("fare").mean(),
    "std(fare)": c("fare").std().repeat_by(pl.len()).explode(),
    "lag(fare, 2)": c("fare").shift(2),
    "lead(distance)": c("distance").shift(-1),
    "diff(fare)": c("fare").diff(),
    "cumsum(distance)": c("distance").cum_sum(),
    "rolling_mean(fare, 3)": c("fare").rolling_mean(window_size=3),
    "rolling_max(fare, 4)": c("fare").rolling_max(window_size=4),
    "rank(fare)": c("fare").rank(method="min"),
    "coalesce(distance, 0) * 2": c("distance").fill_null(0) * 2,
    "year(pickup) * 100 + month(pickup)": c("pickup").dt.year().cast(pl.Int64) * 100 + c("pickup").dt.month().cast(pl.Int64),
    "payment == 1 and tipped": (c("payment") == 1) & c("tipped"),
    "5 / 2 + id % 7": 5 / 2 + c("id") % 7,
    "max(fare, distance * 3)": pl.max_horizontal(c("fare"), c("distance") * 3),
    "-fare ^ 2 + abs(-distance)": -(c("fare") ** 2) + c("distance").abs(),
    'upper(note) == "PLAIN" or contains(note, ",")': (c("note").str.to_uppercase() == "PLAIN") | c("note").str.contains(",", literal=True),
}


def clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return v


result = {}
for formula, expr in CASES.items():
    values = df.select(expr.alias("v"))["v"].to_list()
    result[formula] = [clean(v) for v in values]
with open(out, "w") as f:
    json.dump(result, f)
print(len(result), "formulas,", len(df), "rows")
