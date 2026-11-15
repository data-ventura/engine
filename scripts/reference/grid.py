"""Reference transforms (pyproj/PROJ), interpolation (xarray) and nearest nodes (scipy).

Usage: grid.py <out.json>

Run with: uv run --with pyproj --with xarray --with scipy python scripts/reference/grid.py fixtures/reference/grid.json

Transforms go straight from EPSG codes. pyproj wheels ship no grid files, and the engine's tests
point PROJ at fixtures/proj only, so both see the same grids. The datum cases transform NAD27 to
NAD83 without and then with the NADCON grid in fixtures/proj (us_noaa_conus.tif), and record the
more accurate operations that need grids that aren't installed.
Bilinear interpolation on a global grid
pads one longitude column at 360° so xarray sees the wrap the engine does.
"""

import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
from scipy.spatial import cKDTree

rng = np.random.default_rng(11)
# pyproj warns when the best transformation needs a missing grid; the datum cases record that.
warnings.filterwarnings("ignore", message="Best transformation is not available")


def clean(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=float).ravel()]


# Transforms from WGS 84 to projected CRSs, with points inside each CRS's area of use.
transforms = []
for code, lon_rng, lat_rng in [
    (3857, (-179, 179), (-80, 80)),
    (32633, (9, 21), (0, 80)),
    (3035, (-10, 40), (35, 70)),
    (2154, (-5, 9), (41, 51)),
    (3413, (-180, 180), (60, 89)),
    (27700, (-7, 2), (50, 60)),
    (5070, (-125, -67), (25, 49)),
    (3031, (-180, 180), (-89, -60)),
]:
    lon = rng.uniform(*lon_rng, 40)
    lat = rng.uniform(*lat_rng, 40)
    t = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{code}", always_xy=True)
    x, y = t.transform(lon, lat)
    transforms.append({
        "epsg": code, "lon": clean(lon), "lat": clean(lat), "x": clean(x), "y": clean(y),
        "pipeline": t.description, "accuracy_m": t.accuracy,
    })

# A global 2.5° grid, latitude descending as in ERA5, with gaps.
lon = np.arange(0, 360, 2.5)
lat = np.linspace(90, -90, 73)
data = 250 + 40 * rng.random((lat.size, lon.size))
data[rng.random(data.shape) < 0.03] = np.nan
da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
padded = xr.concat([da, da.isel(lon=[0]).assign_coords(lon=[360.0])], dim="lon")
qlon = np.concatenate([rng.uniform(0, 360, 300), rng.uniform(357.5, 360, 50), [0.0, 357.5, 180.0]])
qlat = np.concatenate([rng.uniform(-90, 90, 350), [90.0, -90.0, 0.0]])
points = padded.interp(lon=xr.DataArray(qlon, dims="p"), lat=xr.DataArray(qlat, dims="p"), method="linear")
nearest = padded.interp(lon=xr.DataArray(qlon, dims="p"), lat=xr.DataArray(qlat, dims="p"), method="nearest")
target_lon = np.arange(1.0, 360, 5.0)
target_lat = np.arange(-88.0, 89, 4.0)
# Regridding is checked through pointwise interp (scipy's RegularGridInterpolator), the engine's
# rule. xarray's orthogonal interp works one dimension at a time and differs on grid lines next to
# a missing cell, so it isn't used here.
tlon, tlat = np.meshgrid(target_lon, target_lat)
regrid = padded.interp(
    lon=xr.DataArray(tlon.ravel(), dims="p"), lat=xr.DataArray(tlat.ravel(), dims="p"), method="linear"
)

# A regional grid: points outside it must be NaN, never the edge value.
regional = da.sel(lat=slice(60, 30), lon=slice(10, 40))
rq_lon = rng.uniform(0, 50, 100)
rq_lat = rng.uniform(20, 70, 100)
regional_points = regional.interp(lon=xr.DataArray(rq_lon, dims="p"), lat=xr.DataArray(rq_lat, dims="p"))

# Unstructured nodes; nearest by great-circle distance (cKDTree on unit vectors).
def unit(lo, la):
    lo, la = np.radians(lo), np.radians(la)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

n_lon = rng.uniform(-180, 180, 5000)
n_lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 5000)))
m_lon = rng.uniform(-180, 180, 400)
m_lat = rng.uniform(-90, 90, 400)
chord, idx = cKDTree(unit(n_lon, n_lat)).query(unit(m_lon, m_lat))
meters = 2 * np.arcsin(np.minimum(chord / 2, 1)) * 6371008.8

# Datum shifts, with their own random stream so the cases above don't change.
drng = np.random.default_rng(27)
d_lon = drng.uniform(-120, -75, 30)
d_lat = drng.uniform(28, 47, 30)


def unavailable(src, dst):
    group = pyproj.transformer.TransformerGroup(src, dst, always_xy=True)
    return [
        {"operation": op.name, "accuracy_m": op.accuracy,
         "grids": [g.short_name for g in op.grids if not g.available]}
        for op in group.unavailable_operations
    ]


def datum(src, dst):
    t = pyproj.Transformer.from_crs(src, dst, always_xy=True)
    x, y = t.transform(d_lon, d_lat)
    # The operation PROJ picked for the first point.
    t.transform(d_lon[0], d_lat[0])
    used = t.get_last_used_operation()
    return {"x": clean(x), "y": clean(y), "operation": used.description, "accuracy_m": used.accuracy}


datum_cases = {"lon": clean(d_lon), "lat": clean(d_lat)}
datum_cases["no_grids"] = datum("EPSG:4267", "EPSG:4269")
datum_cases["unavailable_nad27"] = unavailable("EPSG:4267", "EPSG:4269")
datum_cases["unavailable_bng"] = unavailable("EPSG:4326", "EPSG:27700")
pyproj.datadir.append_data_dir(str(Path(__file__).resolve().parents[2] / "fixtures" / "proj"))
datum_cases["conus_grid"] = datum("EPSG:4267", "EPSG:4269")

ref = {
    "transforms": transforms,
    "datum": datum_cases,
    "grid": {"lon": clean(lon), "lat": clean(lat), "values": clean(data)},
    "points": {"lon": clean(qlon), "lat": clean(qlat), "bilinear": clean(points.values), "nearest": clean(nearest.values)},
    "regrid": {"lon": clean(target_lon), "lat": clean(target_lat), "values": clean(regrid.values)},
    "regional": {
        "lon": clean(regional.lon.values), "lat": clean(regional.lat.values), "values": clean(regional.values),
        "qlon": clean(rq_lon), "qlat": clean(rq_lat), "bilinear": clean(regional_points.values),
    },
    "mesh": {"lon": clean(n_lon), "lat": clean(n_lat), "qlon": clean(m_lon), "qlat": clean(m_lat),
             "nearest": [int(i) for i in idx], "meters": clean(meters)},
}
with open(sys.argv[1], "w") as f:
    json.dump(ref, f)
print("ok; NaN bilinear:", int(np.isnan(points.values).sum()), "of", points.size)
