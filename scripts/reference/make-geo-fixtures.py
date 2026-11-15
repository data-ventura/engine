"""Makes fixtures/geo/* (vector and raster files) and what GDAL reads from them.

Usage: make-geo-fixtures.py <out dir> <reference.json>

Run with: uv run --with geopandas --with pyogrio --with pyarrow --with rasterio --with shapely python scripts/reference/make-geo-fixtures.py fixtures/geo fixtures/reference/geo.json

Vector files are written and read back with pyogrio (GDAL); GeoParquet with geopandas. For each
feature the reference keeps the properties, the geometry's type, its bounds and its vertices
sorted, so the check doesn't depend on ring order or orientation, which formats may change.
Rasters are written and read back with rasterio (GDAL).
"""

import json
import os
import sys
from datetime import date

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

out, ref = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(9)


def square(x, y, s):
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y)]


polys = []
for i in range(12):
    x, y = rng.uniform(-10, 30), rng.uniform(35, 65)
    if i == 3:
        g = Polygon(square(x, y, 2.0), [square(x + 0.5, y + 0.5, 0.5)[::-1]])
    elif i == 7:
        g = MultiPolygon([Polygon(square(x, y, 1.0)), Polygon(square(x + 3, y, 1.0))])
    else:
        g = Polygon(square(x, y, float(rng.uniform(0.2, 1.5))))
    polys.append(g)
names = ["Zürich", "Århus", "Łódź", "plain", "with, comma", "東京"] * 2
props = {
    "name": names,
    "pop": rng.integers(100, 100000, 12).astype("int64"),
    "density": np.round(rng.random(12) * 1000, 3),
    "capital": [i % 4 == 0 for i in range(12)],
}
gdf = gpd.GeoDataFrame(props, geometry=polys, crs="EPSG:4326")
points = gpd.GeoDataFrame(
    {"name": [f"p{i}" for i in range(8)], "value": rng.random(8)},
    geometry=[Point(rng.uniform(-10, 30), rng.uniform(35, 65)) for _ in range(8)],
    crs="EPSG:4326",
)
lines = gpd.GeoDataFrame(
    {"name": ["a", "b"]},
    geometry=[LineString([(0, 50), (1, 51), (2, 50.5)]), LineString([(5, 45), (6, 46)])],
    crs="EPSG:4326",
)
mixed = gpd.GeoDataFrame(
    {"kind": ["poly"] * 3 + ["point"] * 3 + ["line"] * 2},
    geometry=list(gdf.geometry[:3]) + list(points.geometry[:3]) + list(lines.geometry),
    crs="EPSG:4326",
)

pyogrio.write_dataframe(mixed, f"{out}/mixed.geojson", driver="GeoJSON")
pyogrio.write_dataframe(gdf, f"{out}/polys.shp", driver="ESRI Shapefile", encoding="UTF-8")
pyogrio.write_dataframe(gdf.to_crs("EPSG:3857"), f"{out}/polys-3857.fgb", driver="FlatGeobuf", spatial_index=True)
gpkg = f"{out}/layers.gpkg"
if os.path.exists(gpkg):
    os.remove(gpkg)
pyogrio.write_dataframe(gdf, gpkg, layer="polys", driver="GPKG")
pyogrio.write_dataframe(points, gpkg, layer="points", driver="GPKG", append=True)
gdf.to_parquet(f"{out}/polys.parquet")


def summarize(frame):
    feats = []
    for _, row in frame.iterrows():
        g = row.geometry
        coords = []
        geoms = getattr(g, "geoms", [g])
        for part in geoms:
            rings = [part.exterior, *part.interiors] if part.geom_type == "Polygon" else [part]
            for r in rings:
                coords.extend([list(c[:2]) for c in r.coords])
        p = {k: (v if not hasattr(v, "item") else v.item()) for k, v in row.drop(labels="geometry").items()}
        feats.append({
            "props": p,
            "type": g.geom_type,
            "bounds": list(g.bounds),
            "coords": sorted(coords),
        })
    crs = frame.crs.to_epsg() if frame.crs else None
    return {"crs_epsg": crs, "features": feats}


reference = {
    "mixed.geojson": summarize(pyogrio.read_dataframe(f"{out}/mixed.geojson")),
    "polys.shp": summarize(pyogrio.read_dataframe(f"{out}/polys.shp")),
    "polys-3857.fgb": summarize(pyogrio.read_dataframe(f"{out}/polys-3857.fgb")),
    "layers.gpkg#polys": summarize(pyogrio.read_dataframe(gpkg, layer="polys")),
    "layers.gpkg#points": summarize(pyogrio.read_dataframe(gpkg, layer="points")),
    "polys.parquet": summarize(gpd.read_parquet(f"{out}/polys.parquet")),
}

# Rasters: a tiled, deflate-compressed float32 in UTM 33N with nodata (COG-like), and a striped
# LZW uint8 in WGS 84.
data = (rng.random((40, 48)) * 100).astype("float32")
data[5:8, 10:14] = -9999
with rasterio.open(
    f"{out}/dem-utm33.tif", "w", driver="GTiff", width=48, height=40, count=1, dtype="float32",
    crs="EPSG:32633", transform=from_origin(500000, 5300000, 30, 30), nodata=-9999,
    tiled=True, blockxsize=16, blockysize=16, compress="deflate",
) as dst:
    dst.write(data, 1)
small = rng.integers(0, 255, (20, 30)).astype("uint8")
with rasterio.open(
    f"{out}/class-4326.tif", "w", driver="GTiff", width=30, height=20, count=1, dtype="uint8",
    crs="EPSG:4326", transform=from_origin(-5.0, 52.0, 0.25, 0.25), compress="lzw",
) as dst:
    dst.write(small, 1)

for name in ["dem-utm33.tif", "class-4326.tif"]:
    with rasterio.open(f"{out}/{name}") as src:
        band = src.read(1, masked=True).astype("float64").filled(np.nan)
        t = src.transform
        reference[name] = {
            "crs_epsg": src.crs.to_epsg(),
            "shape": [src.height, src.width],
            "nodata": src.nodata,
            # Pixel centers.
            "x": [t.c + t.a * (i + 0.5) for i in range(src.width)],
            "y": [t.f + t.e * (j + 0.5) for j in range(src.height)],
            "values": [None if np.isnan(v) else float(v) for v in band.ravel()],
        }

with open(ref, "w") as f:
    json.dump(reference, f, ensure_ascii=False)
print({k: len(v.get("features", v.get("values", []))) for k, v in reference.items()})
