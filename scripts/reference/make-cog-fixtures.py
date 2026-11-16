"""Makes fixtures/geo/cog-*.tif (Cloud Optimized GeoTIFFs with overviews) and what GDAL reads from them.

Usage: make-cog-fixtures.py <out dir> <reference.json>

Run with: uv run --with rasterio python scripts/reference/make-cog-fixtures.py fixtures/geo fixtures/reference/cog.json

Both files are written by GDAL's COG driver: a float32 DEM in UTM 33N (deflate with the
floating-point predictor, nodata, average overviews, a size that leaves partial edge tiles) and a
3-band uint8 image (zstd, pixel-interleaved, nearest overviews). For each, the reference keeps:

- every level (the full image, then each overview) with its size and geotransform;
- raw values (nodata not masked) of windows at each level, and the bytes of the tiles each window
  touches (GDAL's BLOCK_SIZE), so a range read can be checked to fetch no more;
- for a set of reads of a window into a smaller buffer, the levels GDAL could have read: those
  whose values contain every value the read returned. Nearest-neighbour reads copy values, and
  each level's values are distinct, so this finds the overview GDAL chose (DEM only).
"""

import json
import os
import sys

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from rasterio.windows import Window

out, ref = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(12)


def write_cog(path, data, crs, transform, nodata, **opts):
    count, height, width = data.shape
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff", width=width, height=height, count=count, dtype=data.dtype,
            crs=crs, transform=transform, nodata=nodata,
        ) as tmp:
            tmp.write(data)
        with mem.open() as tmp:
            rio_copy(tmp, path, driver="COG", **opts)


# A smooth surface with noise, so overview averages differ from every full-resolution value.
h, w = 151, 203
yy, xx = np.mgrid[0:h, 0:w]
dem = (500 + 80 * np.sin(xx / 17.0) * np.cos(yy / 23.0) + rng.normal(0, 3, (h, w))).astype("float32")
dem[40:47, 60:75] = -9999
write_cog(
    f"{out}/cog-dem.tif", dem[None], "EPSG:32633", from_origin(500000, 5300000, 30, 30), -9999,
    BLOCKSIZE=32, COMPRESS="DEFLATE", PREDICTOR="YES", OVERVIEW_RESAMPLING="AVERAGE",
)

rgb = rng.integers(0, 256, (3, 70, 90)).astype("uint8")
write_cog(
    f"{out}/cog-rgb.tif", rgb, "EPSG:4326", from_origin(-5.0, 52.0, 0.01, 0.01), None,
    BLOCKSIZE=16, COMPRESS="ZSTD", OVERVIEW_RESAMPLING="NEAREST",
)


def open_level(path, level):
    return rasterio.open(path) if level == 0 else rasterio.open(path, overview_level=level - 1)


def tile_bytes(src, window):
    """Bytes of the distinct tiles (of band 1; pixel-interleaved tiles hold every band) that
    `window` touches."""
    bh, bw = src.block_shapes[0]
    rows = range(window.row_off // bh, (window.row_off + window.height - 1) // bh + 1)
    cols = range(window.col_off // bw, (window.col_off + window.width - 1) // bw + 1)
    return sum(src.block_size(1, i, j) for i in rows for j in cols)


def window_ref(path, level, bands, window):
    with open_level(path, level) as src:
        return {
            "level": level,
            "bands": bands,
            "start": [window.row_off, window.col_off],
            "count": [window.height, window.width],
            "values": [src.read(b, window=window).astype("float64").ravel().tolist() for b in bands],
            "tile_bytes": tile_bytes(src, window),
        }


def summarize(path, windows, choices):
    with rasterio.open(path) as src:
        n = len(src.overviews(1)) + 1
        nodata = src.nodata
    levels = []
    for k in range(n):
        with open_level(path, k) as src:
            levels.append({"width": src.width, "height": src.height, "transform": list(src.transform)[:6]})
    level_values = []
    for k in range(n):
        with open_level(path, k) as src:
            v = src.read(1).ravel()
            level_values.append(set(v[v != nodata].tolist()) if nodata is not None else set(v.tolist()))
    picked = []
    with rasterio.open(path) as src:
        for (row, col, wh, ww), (oh, ow) in choices:
            got = src.read(1, window=Window(col, row, ww, wh), out_shape=(oh, ow)).ravel()
            got = set((got[got != nodata] if nodata is not None else got).tolist())
            picked.append({
                "start": [row, col],
                "count": [wh, ww],
                "out": [oh, ow],
                "levels": [k for k in range(n) if got <= level_values[k]],
            })
    return {
        "size": os.path.getsize(path),
        "nodata": nodata,
        "levels": levels,
        "windows": [window_ref(path, k, b, win) for k, b, win in windows],
        "choices": picked,
    }


full = (0, 0, h, w)
dem_choices = [
    (full, o)
    for o in [(151, 203), (140, 190), (127, 170), (100, 140), (80, 110), (76, 102), (60, 80),
              (40, 52), (38, 51), (30, 40), (19, 26), (16, 21), (10, 13), (5, 7), (1, 50), (50, 1)]
] + [
    ((10, 20, 64, 96), o)
    for o in [(64, 96), (50, 80), (32, 48), (27, 40), (20, 30), (16, 24), (8, 12), (4, 6), (50, 40)]
]


def whole(path, k):
    with open_level(path, k) as src:
        return Window(0, 0, src.width, src.height)


reference = {
    "cog-dem.tif": summarize(
        f"{out}/cog-dem.tif",
        [
            (0, [1], Window(25, 20, 70, 50)),
            (0, [1], Window(190, 140, 13, 11)),
            (0, [1], Window(5, 38, 80, 12)),
            *[(k, [1], whole(f"{out}/cog-dem.tif", k)) for k in (1, 2, 3)],
            (1, [1], Window(30, 10, 20, 40)),
        ],
        dem_choices,
    ),
    "cog-rgb.tif": summarize(
        f"{out}/cog-rgb.tif",
        [
            (0, [1, 2, 3], Window(5, 10, 30, 20)),
            (1, [1, 2, 3], whole(f"{out}/cog-rgb.tif", 1)),
            (3, [1, 2, 3], whole(f"{out}/cog-rgb.tif", 3)),
        ],
        # uint8 values repeat across levels, so reads into a buffer can't tell levels apart.
        [],
    ),
}

with open(ref, "w") as f:
    json.dump(reference, f)
print({k: [(l["width"], l["height"]) for l in v["levels"]] for k, v in reference.items()})
print({k: [(c["out"], c["levels"]) for c in v["choices"]] for k, v in reference.items()})
