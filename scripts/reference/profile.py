"""Computes reference profile statistics for a Parquet file with DuckDB.

Usage: profile.py <file.parquet> <out.json>

Matches dv_model::profile's definitions: linear-interpolation quantiles (quantile_cont), sample
standard deviation, NaN excluded from numeric statistics, 40 equal-width bins between the 0.5% and
99.5% quantiles with the last bin including the upper edge, and the 10 most frequent values with ties
broken by value. Timestamps are microseconds since the epoch.
"""

import json
import sys

import duckdb

QUANTILES = [0.01, 0.25, 0.5, 0.75, 0.99]
BINS = 40

path, out = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute(f"create view t as select * from read_parquet('{path}')")
cols = con.execute("select column_name, column_type from (describe t)").fetchall()
rows = con.execute("select count(*) from t").fetchone()[0]


def numeric_expr(name, typ):
    q = f'"{name}"'
    if typ.startswith("TIMESTAMP"):
        return f"epoch_us({q})::DOUBLE"
    return f"{q}::DOUBLE"


result = {"rows": rows, "columns": {}}
for name, typ in cols:
    q = f'"{name}"'
    nulls = con.execute(f"select count(*) - count({q}) from t").fetchone()[0]
    col = {"type": typ, "nulls": nulls}
    numeric = typ in ("BIGINT", "INTEGER", "DOUBLE", "FLOAT") or typ.startswith("TIMESTAMP")
    if numeric:
        x = numeric_expr(name, typ)
        con.execute(f"create or replace temp view v as select {x} as x from t where {q} is not null and not isnan({x})")
        n, mean, std, distinct = con.execute(
            "select count(x), avg(x), coalesce(stddev_samp(x), 0), count(distinct x) from v"
        ).fetchone()
        qs = con.execute(
            f"select quantile_cont(x, {QUANTILES}), quantile_cont(x, 0.005), quantile_cont(x, 0.995) from v"
        ).fetchone()
        lo, hi = qs[1], qs[2]
        bins = BINS if hi > lo else 1
        width = (hi - lo) / bins
        counts = [0] * bins
        hist = con.execute(
            f"""select least(case when {width} > 0 then floor((x - {lo}) / {width})::BIGINT else 0 end, {bins - 1}) as b,
                       count(*) from v where x >= {lo} and x <= {hi} group by b"""
        ).fetchall()
        for b, c in hist:
            counts[b] = c
        below = con.execute(f"select count(*) from v where x < {lo}").fetchone()[0]
        above = con.execute(f"select count(*) from v where x > {hi}").fetchone()[0]
        col.update(
            count=n, mean=mean, std=std, distinct=distinct, quantiles=list(qs[0]),
            histogram={"lo": lo, "hi": hi, "counts": counts, "below": below, "above": above},
        )
        if distinct <= 20:
            top = con.execute(f"select x, count(*) c from v group by x order by c desc, x limit 10").fetchall()
            col["top"] = [[v, c] for v, c in top]
    else:
        distinct = con.execute(f"select count(distinct {q}) from t").fetchone()[0]
        top = con.execute(
            f"select {q}::VARCHAR v, count(*) c from t where {q} is not null group by v order by c desc, v limit 10"
        ).fetchall()
        col.update(distinct=distinct, top=[[v, c] for v, c in top])
    if numeric or typ == "VARCHAR":
        mn, mx = con.execute(f"select min({q}), max({q}) from t").fetchone()
        if typ.startswith("TIMESTAMP"):
            mn, mx = con.execute(f"select epoch_us(min({q})), epoch_us(max({q})) from t").fetchone()
        col.update(min=mn, max=mx)
    result["columns"][name] = col

with open(out, "w") as f:
    json.dump(result, f, indent=1)
print(f"{len(cols)} columns")
