"""Makes fixtures/tables/trips.* (one small table in every table format) and its references.

Usage: make-table-fixtures.py <taxi-20k.parquet> <out dir> <reference.json>

Run with: uv run --with duckdb --with pandas --with openpyxl --with pyarrow python scripts/reference/make-table-fixtures.py fixtures/taxi-20k.parquet fixtures/tables fixtures/reference/tables.json

The table is 500 taxi trips with a few columns, plus a text column with commas, quotes and
non-ASCII text, a boolean, and nulls in every column. Statistics for each file are computed by
DuckDB reading that file (pandas with openpyxl for Excel), so each reader is checked against an
independent reader of the same bytes.
"""

import json
import os
import sqlite3
import sys

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

src, out, ref = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
con = duckdb.connect()
df = con.execute(
    f"""
    select row_number() over () as id,
           tpepPickupDateTime as pickup,
           tripDistance as distance,
           fareAmount as fare,
           paymentType as payment
    from read_parquet('{src}') limit 500
    """
).df()
notes = ["plain", 'say "hi", twice', "comma, here", "naïve café", "日本語"]
df["note"] = [notes[i % len(notes)] for i in range(len(df))]
df["tipped"] = [i % 3 == 0 for i in range(len(df))]
# Nulls in every column but id, on different rows.
for k, col in enumerate(["pickup", "distance", "fare", "payment", "note", "tipped"]):
    df[col] = df[col].astype(object)
    df.loc[df.index % 50 == 7 + k, col] = None
df["pickup"] = pd.to_datetime(df["pickup"])
df["distance"] = df["distance"].astype("float64")
df["fare"] = df["fare"].astype("float64")
df["tipped"] = df["tipped"].astype("boolean")

con.register("df", df)
con.execute(f"copy df to '{out}/trips.csv' (header, timestampformat '%Y-%m-%d %H:%M:%S')")
con.execute(f"copy df to '{out}/trips.tsv' (header, delimiter '\t', timestampformat '%Y-%m-%d %H:%M:%S')")
con.execute(f"copy df to '{out}/trips.ndjson' (format json, timestampformat '%Y-%m-%dT%H:%M:%S')")
con.execute(f"copy df to '{out}/trips.json' (format json, array true, timestampformat '%Y-%m-%dT%H:%M:%S')")
con.execute(f"copy df to '{out}/trips.parquet' (format parquet)")
feather.write_feather(pa.Table.from_pandas(df, preserve_index=False), f"{out}/trips.arrow", compression="uncompressed")
df.to_excel(f"{out}/trips.xlsx", index=False, sheet_name="trips")

db = f"{out}/trips.sqlite"
if os.path.exists(db):
    os.remove(db)
lite = sqlite3.connect(db)
lite.execute(
    "create table trips (id integer primary key, pickup timestamp, distance real, fare real,"
    " payment text, note text, tipped boolean)"
)
rows = [
    (
        int(r.id),
        None if pd.isna(r.pickup) else r.pickup.strftime("%Y-%m-%d %H:%M:%S"),
        None if pd.isna(r.distance) else float(r.distance),
        None if pd.isna(r.fare) else float(r.fare),
        r.payment,
        r.note,
        None if pd.isna(r.tipped) else int(bool(r.tipped)),
    )
    for r in df.itertuples()
]
lite.executemany("insert into trips values (?, ?, ?, ?, ?, ?, ?)", rows)
lite.commit()
lite.close()


def stats(rel_sql, **frames):
    """Row count, and per column: nulls, then sum/min/max or distinct count."""
    c = duckdb.connect()
    for name, frame in frames.items():
        c.register(name, frame)
    c.execute(f"create view t as {rel_sql}")
    n = c.execute("select count(*) from t").fetchone()[0]
    cols = {}
    for name in ["id", "pickup", "distance", "fare", "payment", "note", "tipped"]:
        q = f'"{name}"'
        s = {"nulls": c.execute(f"select count(*) - count({q}) from t").fetchone()[0]}
        if name in ("id", "distance", "fare"):
            s["sum"], s["min"], s["max"] = c.execute(f"select sum({q}::double), min({q}::double), max({q}::double) from t").fetchone()
        elif name == "pickup":
            s["min"], s["max"] = c.execute(f"select epoch_us(min({q}::timestamp)), epoch_us(max({q}::timestamp)) from t").fetchone()
        elif name == "tipped":
            s["true"] = c.execute(f"select count(*) from t where {q}::boolean").fetchone()[0]
        else:
            s["distinct"] = c.execute(f"select count(distinct {q}) from t").fetchone()[0]
        cols[name] = s
    return {"rows": n, "columns": cols}


reference = {
    "csv": stats(f"select * from read_csv('{out}/trips.csv')"),
    "tsv": stats(f"select * from read_csv('{out}/trips.tsv', delim='\t')"),
    "ndjson": stats(f"select * from read_json('{out}/trips.ndjson')"),
    "json": stats(f"select * from read_json('{out}/trips.json')"),
    "parquet": stats(f"select * from read_parquet('{out}/trips.parquet')"),
    "sqlite": stats(f"select * from sqlite_scan('{db}', 'trips')"),
}
xl = pd.read_excel(f"{out}/trips.xlsx", sheet_name="trips", engine="openpyxl")
reference["xlsx"] = stats("select * from xl", xl=xl)
# Arrow is checked against pyarrow's own read of the file.
reference["arrow"] = stats("select * from arrow_tbl", arrow_tbl=feather.read_table(f"{out}/trips.arrow"))

with open(ref, "w") as f:
    json.dump(reference, f, indent=1)
print({k: v["rows"] for k, v in reference.items()})
