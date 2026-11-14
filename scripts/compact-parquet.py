"""Compacts a directory of Parquet part files into one file with fixed-size row groups.

Usage: compact-parquet.py <parts-dir> <out.parquet>

Files are read in name order, so rows keep their month order. Timestamps are stored as microseconds,
every column is written with Zstandard, and a page index lets readers skip pages. Used by fetch-spike-data.sh.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROW_GROUP = 1_000_000


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(src.glob("*.parquet"))
    if not files:
        sys.exit(f"no Parquet files in {src}")

    schema = pq.read_schema(files[0])
    schema = pa.schema(
        [
            f.with_type(pa.timestamp("us")) if pa.types.is_timestamp(f.type) else f
            for f in schema
        ]
    )

    rows = 0
    with pq.ParquetWriter(dst, schema, compression="zstd", write_statistics=True, write_page_index=True) as w:
        pending: list[pa.RecordBatch] = []
        pending_rows = 0
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=65_536):
                batch = pa.Table.from_batches([batch]).cast(schema).to_batches()[0]
                pending.append(batch)
                pending_rows += batch.num_rows
                if pending_rows >= ROW_GROUP:
                    t = pa.Table.from_batches(pending, schema)
                    w.write_table(t.slice(0, ROW_GROUP), row_group_size=ROW_GROUP)
                    rest = t.slice(ROW_GROUP)
                    pending, pending_rows = rest.to_batches(), rest.num_rows
                    rows += ROW_GROUP
        if pending_rows:
            w.write_table(pa.Table.from_batches(pending, schema), row_group_size=ROW_GROUP)
            rows += pending_rows
    print(f"{rows} rows, {dst.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
