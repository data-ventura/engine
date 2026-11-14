"""Makes fixtures/taxi-20k.parquet: 20,000 evenly spaced rows of the spike's taxi file.

Usage: make-taxi-fixture.py <nyc-taxi-2015q1.parquet> <out.parquet>
"""

import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROWS = 20_000

src = pq.ParquetFile(sys.argv[1])
total = src.metadata.num_rows
step = total // ROWS
parts, offset = [], 0
for g in range(src.num_row_groups):
    t = src.read_row_group(g)
    first = (-offset) % step
    parts.append(t.take(pa.array(range(first, t.num_rows, step))))
    offset += t.num_rows
table = pa.concat_tables(parts).slice(0, ROWS)
pq.write_table(table, sys.argv[2], compression="zstd", row_group_size=5_000)
print(table.num_rows, "rows")
