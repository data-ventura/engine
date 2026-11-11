#!/usr/bin/env bash
# Downloads the M0 spike's local test data into fixtures/spike/data/ (git-ignored).
#
# S1-S3: NYC TLC yellow taxi trips for 2025, 12 monthly Parquet files (about 40M rows, roughly 0.7 GB).
#        Source: NYC Taxi & Limousine Commission trip record data (public).
# S4:    ERA5 is read directly from Google's public ARCO-ERA5 Zarr store; nothing is downloaded.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/fixtures/spike/data/nyc-taxi-2025"
mkdir -p "$out"

for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  f="yellow_tripdata_2025-$m.parquet"
  if [[ -s "$out/$f" ]]; then
    echo "have $f"
    continue
  fi
  echo "fetching $f"
  curl -fL --retry 3 -o "$out/$f.part" "https://d37ci6vzurychx.cloudfront.net/trip-data/$f"
  mv "$out/$f.part" "$out/$f"
done

cat <<'NOTE'

Done. ERA5 for S4 is read in place from the public ARCO store:
  gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
NOTE
