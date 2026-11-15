"""Computes reference aggregates and joins with DuckDB for dv-query's tests.

Usage: query.py <fixtures dir> <out.json>

Run with: uv run --with duckdb python scripts/reference/query.py fixtures fixtures/reference/query.json

Joins follow dv_query::join's rules: a left join that keeps unmatched rows with nulls, null keys
never match, and rows with several matches resolved by the stated policy.
"""

import json
import sys

import duckdb

fx, out = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute(f"create view trips as select * from read_csv('{fx}/tables/trips.csv')")
con.execute(f"create view types as select * from read_csv('{fx}/tables/payment-types.csv')")
con.execute(f"create view fees as select * from read_csv('{fx}/tables/payment-fees.csv')")
con.execute(f"create view taxi as select * from read_parquet('{fx}/taxi-20k.parquet')")


def rows(sql):
    return [list(r) for r in con.execute(sql).fetchall()]


ref = {
    "group_by_payment": rows(
        "select payment, count(*), avg(fare), sum(distance), min(pickup)::varchar from trips"
        " group by payment order by payment nulls last"
    ),
    "taxi_filtered": rows(
        'select "paymentType", count(*), avg("tipAmount"), max("totalAmount") from taxi'
        ' where "tripDistance" between 1.0 and 3.0 and not isnan("tripDistance")'
        ' group by "paymentType" order by "paymentType"'
    ),
    "join_report": rows(
        "select count(*), count(t.code), count(*) filter (where t.n > 1) from trips"
        " left join (select code, count(*) n from types group by code) t on payment = t.code"
    )[0],
    "fees_report": rows(
        "select count(*), count(f.code), count(*) filter (where f.n > 1) from trips"
        " left join (select code, count(*) n from fees group by code) f on payment = f.code"
    )[0],
    "join_types": rows(
        "select name, count(*) from trips left join types on payment = code group by name order by name nulls last"
    ),
    "fees_keep_all": rows("select count(*), sum(fee) from trips left join fees on payment = code")[0],
    "fees_first": rows(
        "select count(*), sum(fee) from trips left join"
        " (select * from fees qualify row_number() over (partition by code order by since) = 1) f"
        " on payment = f.code"
    )[0],
    "fees_last": rows(
        "select count(*), sum(fee) from trips left join"
        " (select * from fees qualify row_number() over (partition by code order by since desc) = 1) f"
        " on payment = f.code"
    )[0],
    "fees_mean": rows(
        "select count(*), sum(fee) from trips left join"
        " (select code, avg(fee) fee from fees group by code) f on payment = f.code"
    )[0],
}
with open(out, "w") as f:
    json.dump(ref, f, indent=1)
for k, v in ref.items():
    print(k, v)
