#!/usr/bin/env bash
# Downloads the M0 spike's local test data into fixtures/spike/data/ (git-ignored).
#
# S1-S3: NYC yellow taxi trips, January-March 2015 (38.5M rows), with the original pickup and dropoff
#        coordinates. The official TLC Parquet files replaced coordinates with zone IDs, so this uses the
#        Azure Open Datasets copy, which kept them. The 60 Spark part files are compacted into one Parquet
#        file with 1M-row groups: fixtures/spike/data/nyc-taxi-2015q1.parquet.
#        Source: NYC Taxi & Limousine Commission trip records, via Azure Open Datasets (public).
# S4/S9: ERA5 is read directly from Google's public ARCO-ERA5 Zarr store; nothing is downloaded.
#
# Needs curl and uv (for the one-off compaction with pyarrow).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
data="$root/fixtures/spike/data"
out="$data/nyc-taxi-2015q1.parquet"
parts="$data/nyc-taxi-2015q1.parts"
base="https://azureopendatastorage.blob.core.windows.net/nyctlc"

if [[ -s "$out" ]]; then
  echo "have $out"
else
  mkdir -p "$parts"
  for m in 1 2 3; do
    curl -fsS "$base?restype=container&comp=list&prefix=yellow/puYear=2015/puMonth=$m/part" \
      | grep -o '<Name>[^<]*\.parquet</Name>' | sed -E 's#</?Name>##g' \
      | while read -r name; do
          f="$parts/2015-$m-$(basename "$name")"
          if [[ -s "$f" ]]; then continue; fi
          echo "fetching $name"
          curl -fL --retry 3 -sS -o "$f.part" "$base/$name"
          mv "$f.part" "$f"
        done
  done
  echo "compacting into $out"
  uv run -q --with 'pyarrow>=17' python -I "$root/scripts/compact-parquet.py" "$parts" "$out.tmp"
  mv "$out.tmp" "$out"
  rm -rf "$parts"
fi

cat <<'NOTE'

Done. ERA5 for S4 is read in place from the public ARCO store (Zarr format 2, one chunk per hourly step):
  gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
NOTE
