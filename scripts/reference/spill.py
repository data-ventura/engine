"""Computes reference sorts, aggregates and joins with DuckDB for dv-query's spill tests.

Usage: spill.py <out.json>

Run with: uv run --with duckdb python scripts/reference/spill.py fixtures/reference/spill.json

The tables are too large to check in, so both sides make them from the same integer formulas:
`facts` has one row per id with a value and two keys, and `labels` gives every id a label and a
weight, in a shuffled order (314159 is coprime to N, so `j * 314159 % N` visits every id once).
The queries are written once here, stored with their results and run by the Rust tests as they
are; only the two table definitions differ between DuckDB and DataFusion. Results too large to
store are reduced to checksums the tests recompute from the engine's rows.
"""

import json
import sys

import duckdb

out = sys.argv[1]
N = 1_000_000

con = duckdb.connect()
con.execute(f"""
create view facts as
select range as id,
       (range * 7919) % 50021 as key,
       'g' || (range % 97)::varchar as grp,
       ((range * 104729) % 100003)::double / 100 as value
from range(0, {N})
""")
con.execute(f"""
create view labels as
select (range * 314159) % {N} as id,
       'l' || (range % 1009)::varchar as label,
       range % 13 + 1 as weight
from range(0, {N})
""")


def rows(sql):
    return [list(r) for r in con.execute(sql).fetchall()]


SORT = "select id, value, grp from facts order by value desc, id"
GROUP = (
    "select key, count(*) as n, sum(value) as total, min(value) as lo, max(value) as hi,"
    " count(distinct grp) as grps from facts group by key order by key"
)
JOIN = (
    "select l.label, count(*) as n, sum(f.value * l.weight) as total, min(f.id) as first"
    " from facts f join labels l on f.id = l.id group by l.label order by l.label"
)

ref = {
    "n": N,
    "sort": {
        "sql": SORT,
        "rows": N,
        # Sum of position (from 1) times id, in the sorted order.
        "checksum": con.execute(
            f"select sum(rn * id)::bigint from (select row_number() over () as rn, id from ({SORT}))"
        ).fetchone()[0],
        "head": rows(f"{SORT} limit 5"),
        "tail": rows(f"select * from ({SORT}) offset {N - 5}"),
    },
    "group": {
        "sql": GROUP,
        # Groups, sum of key * n, sum of grps, sum of totals, sum of lo, sum of hi.
        "summary": rows(
            f"select count(*), sum(key * n)::bigint, sum(grps)::bigint, sum(total), sum(lo), sum(hi)"
            f" from ({GROUP})"
        )[0],
        "head": rows(f"{GROUP} limit 5"),
    },
    "join": {"sql": JOIN, "rows": rows(JOIN)},
}

with open(out, "w") as f:
    json.dump(ref, f, indent=1)
    f.write("\n")
