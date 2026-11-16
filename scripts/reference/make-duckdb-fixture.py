"""Makes fixtures/tables/trips.duckdb and what DuckDB reads from it.

Usage: make-duckdb-fixture.py <tables dir> <reference.json>

Run with: uv run --with duckdb==2.0.0 --with pyarrow python scripts/reference/make-duckdb-fixture.py fixtures/tables fixtures/reference/duckdb.json

The database holds the trips table from trips.parquet (made by make-table-fixtures.py), a view
over it, and, in a second schema, a table of types the other formats lack: DECIMAL, DATE,
TIMESTAMPTZ, HUGEINT, ENUM, INTERVAL, UUID and BLOB, with nulls. The reference is DuckDB's own
read of the file: row counts, nulls and sums or ranges per column, and the Arrow types DuckDB
exports.
"""

import json
import os
import sys

import duckdb

tables, ref = sys.argv[1], sys.argv[2]
db = f"{tables}/trips.duckdb"
for f in (db, db + ".wal"):
    if os.path.exists(f):
        os.remove(f)

con = duckdb.connect(db)
con.execute(f"create table trips as select * from read_parquet('{tables}/trips.parquet') order by id")
con.execute("create view long_trips as select id, pickup, distance, fare from trips where distance > 5")
con.execute("create schema extra")
con.execute("create type mood as enum ('sad', 'ok', 'happy')")
con.execute(
    """
    create table extra.types as
    select i as id,
           case when i % 7 = 3 then null else (i * 12.345)::decimal(10, 3) end as price,
           case when i % 7 = 4 then null else date '2024-02-27' + i::integer end as day,
           case when i % 7 = 5 then null
                else timestamptz '2024-03-30 22:00:00+00' + to_minutes(i * 37) end as stamp,
           case when i % 7 = 6 then null else (i::hugeint * 1000000000000)::hugeint end as big,
           (['sad', 'ok', 'happy'])[i % 3 + 1]::mood as feeling,
           to_hours(i) + to_days(i % 4) as span,
           ('00000000-0000-4000-8000-' || lpad(i::varchar, 12, '0'))::uuid as tag,
           ('\\x' || lpad(hex(i), 4, '0'))::blob as bytes
    from range(1, 41) t(i)
    """
)
con.close()

# The reference reads the finished file back, read-only.
c = duckdb.connect(db, read_only=True)
c.execute("set TimeZone = 'UTC'")


def stats(table, numeric=(), times=(), dates=()):
    n = c.execute(f"select count(*) from {table}").fetchone()[0]
    cols = {}
    names = [r[0] for r in c.execute(f"describe {table}").fetchall()]
    for name in names:
        q = f'"{name}"'
        s = {"nulls": c.execute(f"select count(*) - count({q}) from {table}").fetchone()[0]}
        if name in numeric:
            s["sum"], s["min"], s["max"] = c.execute(
                f"select sum({q}::double), min({q}::double), max({q}::double) from {table}"
            ).fetchone()
        elif name in times:
            s["min"], s["max"] = c.execute(f"select epoch_us(min({q})), epoch_us(max({q})) from {table}").fetchone()
        elif name in dates:
            s["min"], s["max"] = c.execute(
                f"select min({q}) - date '1970-01-01', max({q}) - date '1970-01-01' from {table}"
            ).fetchone()
        cols[name] = s
    # A cursor of its own, since fetchone leaves the connection's last result open. TIMESTAMPTZ
    # exports in the session's time zone: UTC, as in a DuckDB built without ICU.
    cur = c.cursor()
    cur.execute("set TimeZone = 'UTC'")
    arrow = cur.sql(f"select * from {table}").to_arrow_table().schema
    for field in arrow:
        cols[field.name]["arrow"] = str(field.type)
    return {"rows": n, "columns": cols}


reference = {
    "tables": [r[0] for r in c.execute(
        "select case when schema_name = 'main' then name else schema_name || '.' || name end from ("
        " select schema_name, table_name as name from duckdb_tables() union all"
        " select schema_name, view_name from duckdb_views() where not internal)"
        " order by schema_name = 'main' desc, schema_name, name"
    ).fetchall()],
    "trips": stats("trips", numeric=("id", "distance", "fare"), times=("pickup",)),
    "long_trips": stats("long_trips", numeric=("id", "distance", "fare"), times=("pickup",)),
    "extra.types": stats("extra.types", numeric=("id", "price", "big"), times=("stamp",), dates=("day",)),
}
# Count the trips table's true booleans and distinct text values, as tables.json does.
reference["trips"]["columns"]["tipped"]["true"] = c.execute("select count(*) from trips where tipped").fetchone()[0]
for col in ("payment", "note"):
    reference["trips"]["columns"][col]["distinct"] = c.execute(f"select count(distinct {col}) from trips").fetchone()[0]
c.close()

with open(ref, "w") as f:
    json.dump(reference, f, indent=1)
print(reference["tables"], {k: v["rows"] for k, v in reference.items() if k != "tables"})
