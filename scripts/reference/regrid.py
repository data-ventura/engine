"""Reference conservative regridding (xarray-regrid, S2 via spherely) and curvilinear cell lookup.

Usage: regrid.py <out.json>

Run with: uv run --with xarray-regrid --with spherely --with pyproj python scripts/reference/regrid.py fixtures/reference/regrid.json

Rectilinear grids are checked against xarray-regrid's conservative method where it is exact: on
the sphere when source cells nest inside target cells (its latitude correction weights whole
source cells), and on planar grids always. Cells that aren't nested are checked against S2
(spherely), with each latitude edge densified so S2's great-circle edges follow the parallel to
well under 1e-9 of a cell's area.

Curvilinear cells have great-circle edges between their corners, as in ESMF and S2. Their overlaps
with target cells come from S2 polygon intersections; the missing-value rule (ESMF's and
xESMF's `skipna`: renormalise by the valid overlap, missing below a valid fraction of the target
cell) is then applied here. Cell lookup uses S2's point-in-polygon test.
"""

import json
import sys

import numpy as np
import pyproj
import spherely
import xarray as xr
import xarray_regrid  # noqa: F401  (registers the .regrid accessor)

rng = np.random.default_rng(14)
R = spherely.EARTH_RADIUS_METERS


def clean(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=float).ravel()]


def field(lon, lat, missing=0.05):
    """A smooth field with noise and scattered missing values."""
    v = 280 + 20 * np.cos(np.radians(lat)) * np.sin(np.radians(2 * lon)) + rng.normal(0, 2, np.shape(lon))
    v[rng.random(v.shape) < missing] = np.nan
    return v


# --- Rectilinear, nested, on the sphere: xarray-regrid. ---------------------------------------
# A 1° global source with latitude descending (as in ERA5) and a land block of missing values, to
# 3° targets: one inside the source and one hanging over the edge of a regional source.
def nested(src_lon, src_lat, dst_lon, dst_lat):
    lon2, lat2 = np.meshgrid(src_lon, src_lat)
    v = field(lon2, lat2)
    v[(lon2 > 40) & (lon2 < 55) & (lat2 > 10) & (lat2 < 22)] = np.nan
    da = xr.DataArray(v, coords={"lat": src_lat, "lon": src_lon}, dims=("lat", "lon"))
    target = xr.Dataset(coords={"lat": dst_lat, "lon": dst_lon})
    out = {"src_lon": clean(src_lon), "src_lat": clean(src_lat), "dst_lon": clean(dst_lon),
           "dst_lat": clean(dst_lat), "values": clean(v)}
    for thr in (1.0, 0.4):
        r = da.regrid.conservative(target, latitude_coord="lat", skipna=True, nan_threshold=thr)
        # min_valid in the engine is xarray-regrid's 1 - nan_threshold.
        out[f"min_valid_{1 - thr:g}"] = clean(r.transpose("lat", "lon").values)
    return out


nested_global = nested(
    np.arange(0.5, 360, 1.0), np.arange(89.5, -90, -1.0),
    np.arange(31.5, 120, 3.0), np.arange(-28.5, 60, 3.0),
)
# The regional source covers lon 0–60, lat 0–50; target cells outside it must be missing.
nested_regional = nested(
    np.arange(0.5, 60, 1.0), np.arange(0.5, 50, 1.0),
    np.arange(-9.0, 70, 2.0), np.arange(-5.0, 60, 2.0),
)

# --- Rectilinear, planar: xarray-regrid (no latitude correction), cells not nested. -------------
px = np.arange(250.0, 20000, 500.0)
py = np.arange(15000.0, 0, -500.0)
pv = field(px[None, :] / 100, py[:, None] / 300)
pda = xr.DataArray(pv, coords={"y": py, "x": px}, dims=("y", "x"))
tx = np.arange(1000.0, 19000, 730.0)
ty = np.arange(1200.0, 14000, 610.0)
pr = pda.regrid.conservative(xr.Dataset(coords={"y": ty, "x": tx}), latitude_coord=None, skipna=True)
planar = {"src_x": clean(px), "src_y": clean(py), "dst_x": clean(tx), "dst_y": clean(ty),
          "values": clean(pv), "min_valid_0": clean(pr.transpose("y", "x").values)}


# --- Spherical polygons with S2. ----------------------------------------------------------------
def polygon(lon, lat):
    """An S2 polygon from corner lists, dropping repeated corners (at a pole)."""
    pts = []
    for x, y in zip(lon, lat):
        p = (float(x), float(y))
        if pts and (p == pts[-1] or (abs(y) == 90 and abs(pts[-1][1]) == 90)):
            continue
        pts.append(p)
    if len(pts) > 1 and (pts[0] == pts[-1] or (abs(pts[0][1]) == 90 and abs(pts[-1][1]) == 90)):
        pts.pop()
    return spherely.create_polygon(pts)


def latlon_cell(x0, x1, y0, y1, step=None):
    """A lat-lon cell; with `step` (degrees) its parallels are densified to follow the parallel."""
    if step is None:
        return polygon([x0, x1, x1, x0], [y0, y0, y1, y1])
    n = max(2, int(np.ceil(abs(x1 - x0) / step)) + 1)
    xs = np.linspace(x0, x1, n)
    return polygon(np.concatenate([xs, xs[::-1]]), np.concatenate([np.full(n, y0), np.full(n, y1)]))


def overlaps(dst, src, dst_box, src_box):
    """Overlap areas (m²) between every target and source polygon whose boxes meet."""
    rows = []
    for d, (poly, box) in enumerate(zip(dst, dst_box)):
        cand = [s for s, b in enumerate(src_box) if src[s] is not None and boxes_meet(box, b)]
        if poly is None or not cand:
            rows.append(([], []))
            continue
        a = spherely.area(spherely.intersection(poly, np.array([src[s] for s in cand])), R)
        keep = a > 0
        rows.append(([cand[i] for i in np.nonzero(keep)[0]], list(a[keep])))
    return rows


def boxes_meet(a, b):
    # Boxes are (unit-vector min, unit-vector max), padded generously.
    return np.all(a[0] <= b[1]) and np.all(b[0] <= a[1])


def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def box(lon, lat):
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        return (np.full(3, np.inf), np.full(3, -np.inf))
    u = unit(np.asarray(lon), np.asarray(lat))
    return (u.min(axis=0) - 0.05, u.max(axis=0) + 0.05)


def apply(rows, dst_area, v, min_valid):
    out = np.full(len(rows), np.nan)
    for d, (idx, a) in enumerate(rows):
        idx, a = np.array(idx, dtype=int), np.array(a)
        ok = np.isfinite(v[idx]) if len(idx) else np.array([], dtype=bool)
        valid = a[ok].sum()
        if valid > 0 and valid >= min_valid * dst_area[d]:
            out[d] = (a[ok] * v[idx][ok]).sum() / valid
    return out


def cells_from_corners(clon, clat):
    ny, nx = clon.shape[0] - 1, clon.shape[1] - 1
    polys, boxes = [], []
    for j in range(ny):
        for i in range(nx):
            lo = [clon[j, i], clon[j, i + 1], clon[j + 1, i + 1], clon[j + 1, i]]
            la = [clat[j, i], clat[j, i + 1], clat[j + 1, i + 1], clat[j + 1, i]]
            ok = np.all(np.isfinite(lo)) and np.all(np.isfinite(la))
            polys.append(polygon(lo, la) if ok else None)
            boxes.append(box(lo, la))
    return polys, boxes


def latlon_cells(xe, ye, step=None):
    polys, boxes = [], []
    for j in range(len(ye) - 1):
        for i in range(len(xe) - 1):
            polys.append(latlon_cell(xe[i], xe[i + 1], ye[j], ye[j + 1], step))
            boxes.append(box([xe[i], xe[i + 1], xe[i + 1], xe[i], (xe[i] + xe[i + 1]) / 2],
                             [ye[j], ye[j], ye[j + 1], ye[j + 1], (ye[j] + ye[j + 1]) / 2]))
    return polys, boxes


def spherical_case(src, src_box, dst, dst_box, v, thresholds):
    rows = overlaps(dst, src, dst_box, src_box)
    dst_area = np.array([spherely.area(p, R) if p is not None else np.nan for p in dst])
    src_area = np.array([spherely.area(p, R) if p is not None else np.nan for p in src])
    out = {"values": clean(v), "dst_area": clean(dst_area), "src_area": clean(src_area),
           "fraction": clean([sum(a) / dst_area[d] for d, (_, a) in enumerate(rows)])}
    for t in thresholds:
        out[f"min_valid_{t:g}"] = clean(apply(rows, dst_area, v, t))
    return out


# Rectilinear, not nested, across the antimeridian: a 4° × 3° source on 0–360 to 7.2° cells on
# −180–180 with uneven latitude edges. Parallels densified every 0.02°.
s_xe = np.arange(0.0, 360.1, 4.0)
s_ye = np.arange(-90.0, 90.1, 3.0)
d_xe = np.arange(-180.0, 180.1, 7.2)
d_ye = np.array([-80.0, -71.3, -60.0, -44.4, -30.5, -12.0, -3.3, 0.0, 8.8, 25.0, 41.7, 55.5, 70.2, 85.0])
src, src_box = latlon_cells(s_xe, s_ye, step=0.02)
dst, dst_box = latlon_cells(d_xe, d_ye, step=0.02)
sx, sy = np.meshgrid((s_xe[:-1] + s_xe[1:]) / 2, (s_ye[:-1] + s_ye[1:]) / 2)
unnested = spherical_case(src, src_box, dst, dst_box, field(sx, sy).ravel(), (0, 0.5))
unnested.update({"src_x_edges": clean(s_xe), "src_y_edges": clean(s_ye),
                 "dst_x_edges": clean(d_xe), "dst_y_edges": clean(d_ye)})

# Curvilinear: a rotated-pole grid over Europe (EURO-CORDEX's pole) of 1.6° × 1.5° cells, with two
# missing corners, to 1.5° lat-lon cells that reach past it on every side.
def rotated_pole(rlon, rlat, pole_lon=-162.0, pole_lat=39.25):
    """Geographic coordinates of rotated-pole ones (CF `rotated_latitude_longitude`)."""
    u = unit(rlon, rlat)
    t = np.radians(pole_lat - 90)
    x = np.cos(t) * u[..., 0] + np.sin(t) * u[..., 2]
    z = -np.sin(t) * u[..., 0] + np.cos(t) * u[..., 2]
    a = np.radians(pole_lon + 180)
    x, y = np.cos(a) * x - np.sin(a) * u[..., 1], np.sin(a) * x + np.cos(a) * u[..., 1]
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(np.clip(z, -1, 1)))


rl, rp = np.meshgrid(np.arange(-20.0, 20.1, 1.6), np.arange(-18.0, 18.1, 1.5))
clon, clat = rotated_pole(rl, rp)
clon[3, 5] = clat[3, 5] = np.nan
clon[10, 0] = clat[10, 0] = np.nan
src, src_box = cells_from_corners(clon, clat)
d_xe = np.arange(-30.0, 45.1, 1.5)
d_ye = np.arange(28.5, 75.1, 1.5)
dst, dst_box = latlon_cells(d_xe, d_ye)
centers = rotated_pole((rl[:-1, :-1] + rl[1:, 1:]) / 2, (rp[:-1, :-1] + rp[1:, 1:]) / 2)
rotated = spherical_case(src, src_box, dst, dst_box, field(*centers).ravel(), (0, 0.5, 0.9))
rotated.update({"corner_lon": clean(clon), "corner_lat": clean(clat), "nx": clon.shape[1] - 1,
                "ny": clon.shape[0] - 1, "dst_x_edges": clean(d_xe), "dst_y_edges": clean(d_ye)})

# Curvilinear around the North Pole, across the antimeridian: polar stereographic (as in sea-ice
# products) to 6° × 2.5° cells from 40°N, the top row meeting at the pole.
stere = pyproj.Transformer.from_crs("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +R=6371007.181",
                                    "+proj=longlat +R=6371007.181", always_xy=True)
x, y = np.meshgrid(np.arange(-3_000_000.0, 3_000_001, 250_000), np.arange(3_000_000.0, -3_000_001, -250_000))
plon, plat = stere.transform(x, y)
src, src_box = cells_from_corners(plon, plat)
d_xe = np.arange(-180.0, 180.1, 6.0)
d_ye = np.arange(40.0, 90.1, 2.5)
dst, dst_box = latlon_cells(d_xe, d_ye)
centers = stere.transform((x[:-1, :-1] + x[1:, 1:]) / 2, (y[:-1, :-1] + y[1:, 1:]) / 2)
polar = spherical_case(src, src_box, dst, dst_box, field(*centers, missing=0.1).ravel(), (0,))
polar.update({"corner_lon": clean(plon), "corner_lat": clean(plat), "nx": plon.shape[1] - 1,
              "ny": plon.shape[0] - 1, "dst_x_edges": clean(d_xe), "dst_y_edges": clean(d_ye)})


# --- Cell lookup on the curvilinear grids. --------------------------------------------------------
def lookup(clon, clat, qlon, qlat):
    polys, _ = cells_from_corners(clon, clat)
    found = []
    for x, y in zip(qlon, qlat):
        p = spherely.create_point(float(x), float(y))
        hit = [i for i, c in enumerate(polys) if c is not None and spherely.contains(c, p)]
        assert len(hit) <= 1, (x, y, hit)
        found.append(hit[0] if hit else -1)
    return found


q_rot = (rng.uniform(-35, 50, 600), rng.uniform(20, 80, 600))
q_pol = (rng.uniform(-180, 180, 600), np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(35)), 1, 600))))
rotated["lookup"] = {"lon": clean(q_rot[0]), "lat": clean(q_rot[1]), "cell": lookup(clon, clat, *q_rot)}
polar["lookup"] = {"lon": clean(q_pol[0]), "lat": clean(q_pol[1]), "cell": lookup(plon, plat, *q_pol)}

ref = {
    "radius": R,
    "nested_global": nested_global,
    "nested_regional": nested_regional,
    "planar": planar,
    "unnested": unnested,
    "rotated": rotated,
    "polar": polar,
}
with open(sys.argv[1], "w") as f:
    json.dump(ref, f)
print("ok; lookups found:", sum(c >= 0 for c in rotated["lookup"]["cell"]), sum(c >= 0 for c in polar["lookup"]["cell"]))
