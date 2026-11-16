"""Makes fixtures/arrays/*.grib1 and *.grib2 and what cfgrib reads from them.

Usage: make-grib-fixtures.py <out dir> <reference.json>

Run with: uv run --with eccodes==2.50.0 --with cfgrib==0.9.15.1 --with xarray python scripts/reference/make-grib-fixtures.py fixtures/arrays fixtures/reference/grib.json

forecast.grib2 is a 6 x 5 regular lat/lon grid (north to south) with temperature on three
pressure levels (written out of order, CCSDS packing) at steps 0 and 6, 2 m temperature with a
bitmap of missing points, 10 m wind (so two heightAboveGround levels clash) and a 0-6 h
accumulation (second-order complex packing). analysis.grib1 is GRIB 1 on a south-to-north grid,
with mean sea level pressure at two analysis times and temperature at one. grids.grib2 has a
reduced Gaussian field (one `values` dimension) and a polar stereographic one (y and x, with
two-dimensional latitude and longitude).

The reference is what `cfgrib.open_datasets` reads, with times left as numbers (seconds since
1970 for times, hours for steps), one entry per data variable.
"""

import json
import os
import sys

import cfgrib
import eccodes as ec
import numpy as np

out, ref = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def message(sample, **keys):
    h = ec.codes_grib_new_from_samples(sample)
    for k, v in keys.items():
        ec.codes_set(h, k, v)
    return h


def write(path, handles):
    with open(path, "wb") as f:
        for h in handles:
            ec.codes_write(h, f)
            ec.codes_release(h)


def regrid(h, grid):
    for k, v in grid.items():
        ec.codes_set(h, k, v)
    # Placeholder values, so the point count matches the new grid.
    ec.codes_set_values(h, np.zeros(grid["Ni"] * grid["Nj"]))


def field(h, fn):
    """Values from fn(lat, lon) at each point, in the message's own point order."""
    lats = ec.codes_get_double_array(h, "latitudes")
    lons = ec.codes_get_double_array(h, "longitudes")
    return fn(lats, lons)


# forecast.grib2: a regular lat/lon grid from 50N to 40N and 5W to 7.5E.
grid2 = dict(
    Ni=6, Nj=5,
    latitudeOfFirstGridPointInDegrees=50.0, latitudeOfLastGridPointInDegrees=40.0,
    longitudeOfFirstGridPointInDegrees=355.0, longitudeOfLastGridPointInDegrees=7.5,
    iDirectionIncrementInDegrees=2.5, jDirectionIncrementInDegrees=2.5,
    jScansPositively=0,
)
base2 = dict(dataDate=20260115, dataTime=600)
msgs = []
for step in (0, 6):
    for level in (850, 1000, 500):
        h = message("regular_ll_pl_grib2", **base2)
        regrid(h, grid2)
        ec.codes_set(h, "shortName", "t")
        ec.codes_set(h, "level", level)
        ec.codes_set(h, "step", step)
        ec.codes_set_values(h, field(h, lambda la, lo: 300 - level / 20 - 0.5 * (la - 40) + 0.2 * lo + step / 3))
        ec.codes_set(h, "packingType", "grid_ccsds")
        msgs.append(h)
for step in (0, 6):
    h = message("regular_ll_sfc_grib2", **base2)
    regrid(h, grid2)
    ec.codes_set(h, "shortName", "2t")
    ec.codes_set(h, "typeOfLevel", "heightAboveGround")
    ec.codes_set(h, "level", 2)
    ec.codes_set(h, "step", step)
    ec.codes_set(h, "bitmapPresent", 1)
    v = field(h, lambda la, lo: 280 + np.sin(la / 3) + np.cos(lo / 2) + step / 2)
    v[[3, 7, 20 + step]] = 9999.0  # the default missingValue: no data at these points
    ec.codes_set_values(h, v)
    msgs.append(h)
for step in (0, 6):
    h = message("regular_ll_sfc_grib2", **base2)
    regrid(h, grid2)
    ec.codes_set(h, "shortName", "10u")
    ec.codes_set(h, "typeOfLevel", "heightAboveGround")
    ec.codes_set(h, "level", 10)
    ec.codes_set(h, "step", step)
    ec.codes_set_values(h, field(h, lambda la, lo: 3 * np.sin(lo / 4 + step) - 0.1 * la))
    msgs.append(h)
h = message("regular_ll_sfc_grib2", **base2)
regrid(h, grid2)
ec.codes_set(h, "productDefinitionTemplateNumber", 8)
ec.codes_set(h, "shortName", "tp")
ec.codes_set(h, "stepRange", "0-6")
ec.codes_set_values(h, field(h, lambda la, lo: np.maximum(0, 0.004 * np.sin(la) + 0.002 * np.cos(lo))))
ec.codes_set(h, "packingType", "grid_complex_spatial_differencing")
msgs.append(h)
write(f"{out}/forecast.grib2", msgs)

# analysis.grib1: GRIB 1, a grid from 30N to 40N (south to north) and 0E to 12.5E.
grid1 = dict(
    Ni=6, Nj=5,
    latitudeOfFirstGridPointInDegrees=30.0, latitudeOfLastGridPointInDegrees=40.0,
    longitudeOfFirstGridPointInDegrees=0.0, longitudeOfLastGridPointInDegrees=12.5,
    iDirectionIncrementInDegrees=2.5, jDirectionIncrementInDegrees=2.5,
    jScansPositively=1,
)
msgs = []
for time in (0, 1200):
    h = message("regular_ll_sfc_grib1", dataDate=20251231, dataTime=time)
    regrid(h, grid1)
    ec.codes_set(h, "shortName", "msl")
    ec.codes_set(h, "typeOfLevel", "meanSea")
    ec.codes_set_values(h, field(h, lambda la, lo: 101325 + 40 * (la - 35) - 15 * lo + time / 10))
    msgs.append(h)
for level in (500, 850):
    h = message("regular_ll_pl_grib1", dataDate=20251231, dataTime=0)
    regrid(h, grid1)
    ec.codes_set(h, "shortName", "t")
    ec.codes_set(h, "level", level)
    ec.codes_set_values(h, field(h, lambda la, lo: 290 - level / 25 - 0.4 * la + 0.1 * lo))
    msgs.append(h)
write(f"{out}/analysis.grib1", msgs)

# grids.grib2: a reduced Gaussian and a polar stereographic grid.
h = message("reduced_gg_pl_32_grib2", dataDate=20260301, dataTime=0, bitsPerValue=12)
ec.codes_set(h, "shortName", "t")
ec.codes_set(h, "level", 1000)
ec.codes_set_values(h, field(h, lambda la, lo: 260 + 40 * np.cos(np.radians(la)) + np.sin(np.radians(lo))))
p = message("polar_stereographic_pl_grib2", dataDate=20260301, dataTime=0)
ec.codes_set(p, "shortName", "z")
ec.codes_set(p, "level", 500)
ec.codes_set_values(p, field(p, lambda la, lo: 55000 + 100 * la + np.cos(np.radians(lo))))
write(f"{out}/grids.grib2", [h, p])


def values(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype="float64").ravel()]


ATTRS = ["units", "long_name", "standard_name", "GRIB_shortName", "GRIB_paramId", "GRIB_typeOfLevel",
         "GRIB_stepType", "GRIB_gridType", "GRIB_cfVarName", "GRIB_missingValue"]

refs = {}
for name in ("forecast.grib2", "analysis.grib1", "grids.grib2"):
    file_ref = {}
    for ds in cfgrib.open_datasets(f"{out}/{name}", backend_kwargs={"indexpath": ""},
                                   decode_times=False, decode_timedelta=False):
        for var, da in ds.data_vars.items():
            file_ref[var] = {
                "dims": list(da.dims),
                "shape": list(da.shape),
                "values": values(da.values),
                "attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in da.attrs.items() if k in ATTRS},
                "coords": {
                    c: {"dims": list(ds[c].dims), "values": values(ds[c].values),
                        "units": ds[c].attrs.get("units")}
                    for c in da.coords
                },
            }
        file_ref["__attrs__"] = {k: v for k, v in ds.attrs.items() if k.startswith("GRIB_")}
    refs[name] = file_ref

with open(ref, "w") as f:
    json.dump(refs, f)
for name, r in refs.items():
    print(name, {v: (d["dims"], d["shape"]) for v, d in r.items() if v != "__attrs__"})
