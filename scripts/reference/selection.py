"""Counts the rows DuckDB selects for a few selections on a Parquet file.

Usage: selection.py <file.parquet> <out.json>

Run with: uv run --with duckdb python scripts/reference/selection.py fixtures/taxi-20k.parquet fixtures/reference/taxi-20k.selection.json

Each case is a list of predicates, ANDed, in dv_model::selection's JSON form, and the SQL that
means the same. Ranges are inclusive; timestamps compare as microseconds since the epoch.
"""

import json
import sys

import duckdb

path, out = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute(f"create view t as select * from read_parquet('{path}')")

CASES = [
    (
        [{"kind": "range", "field": "tripDistance", "min": 1.0, "max": 3.0}],
        '"tripDistance" between 1.0 and 3.0',
    ),
    (
        [{"kind": "range", "field": "fareAmount", "min": None, "max": 10.0}],
        '"fareAmount" <= 10.0',
    ),
    (
        [
            {"kind": "range", "field": "tpepPickupDateTime", "min": 1422748800000000, "max": None},
            {"kind": "one_of", "field": "paymentType", "values": ["1", "2"]},
        ],
        "epoch_us(\"tpepPickupDateTime\") >= 1422748800000000 and \"paymentType\" in ('1', '2')",
    ),
    (
        [{"kind": "one_of", "field": "passengerCount", "values": [1, 5]}],
        '"passengerCount" in (1, 5)',
    ),
    (
        [{"kind": "range", "field": "startLat", "min": 40.7, "max": 40.8}, {"kind": "not_null", "field": "endLat"}],
        '"startLat" between 40.7 and 40.8 and "endLat" is not null',
    ),
]

result = []
for predicates, sql in CASES:
    n = con.execute(f"select count(*) from t where {sql}").fetchone()[0]
    result.append({"predicates": predicates, "sql": sql, "rows": n})

with open(out, "w") as f:
    json.dump(result, f, indent=1)
print([r["rows"] for r in result])
