"""Makes fixtures/arrays/* (NetCDF 3 and 4, HDF5, .npy, .npz) and what xarray decodes from them.

Usage: make-array-fixtures.py <out dir> <reference.json>

Run with: uv run --with xarray --with netCDF4 --with h5py --with numpy --with cftime python scripts/reference/make-array-fixtures.py fixtures/arrays fixtures/reference/arrays.json

packed.nc (NetCDF 4) stores temperature as int16 with scale_factor/add_offset and a _FillValue,
precipitation as float32 with a missing_value (not a _FillValue), a uint8 cloud cover stored
as signed bytes with _Unsigned, and time in the noleap calendar. classic.nc is NetCDF 3. plain.h5
is HDF5 without NetCDF's dimension scales. The reference is what xarray (with cftime) reads back.
"""

import json
import os
import sys

import h5py
import numpy as np
import xarray as xr

out, ref = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(5)
t, ny, nx = 3, 5, 8
temp = 250 + 40 * rng.random((t, ny, nx))
temp[0, 1, 2] = np.nan
precip = (5 * rng.random((t, ny, nx))).astype("float32")
precip[2, 4, 7] = -999.0
cloud = rng.integers(0, 256, (ny, nx)).astype("uint8")

ds = xr.Dataset(
    {
        "t2m": (("time", "lat", "lon"), temp, {"units": "K", "standard_name": "air_temperature"}),
        "precip": (("time", "lat", "lon"), precip, {"units": "mm/day", "missing_value": np.float32(-999.0)}),
        "cloud": (("lat", "lon"), cloud.view("int8"), {"_Unsigned": "true", "units": "%"}),
    },
    coords={
        "time": ("time", [0.0, 1.0, 2.0], {"units": "days since 2000-02-27", "calendar": "noleap"}),
        "lat": ("lat", np.linspace(60, 20, ny), {"units": "degrees_north"}),
        "lon": ("lon", np.arange(nx) * 5.0 - 10, {"units": "degrees_east"}),
    },
    attrs={"title": "dv fixture"},
)
enc = {
    "t2m": {"dtype": "int16", "scale_factor": 0.01, "add_offset": 273.15, "_FillValue": np.int16(-32767)},
    "precip": {"_FillValue": None},
    "time": {"_FillValue": None},
    "lat": {"_FillValue": None},
    "lon": {"_FillValue": None},
}
ds.to_netcdf(f"{out}/packed.nc", format="NETCDF4", encoding=enc)
plain = xr.Dataset(
    {"t2m": (("time", "lat", "lon"), temp.astype("float32"))},
    coords={"lat": ("lat", np.linspace(60, 20, ny), {"units": "degrees_north"}), "lon": ("lon", np.arange(nx) * 5.0 - 10)},
)
plain.to_netcdf(f"{out}/classic.nc", format="NETCDF3_CLASSIC", encoding={"t2m": {"_FillValue": np.float32(9.96921e36)}})
with h5py.File(f"{out}/plain.h5", "w") as h:
    h.create_dataset("grid", data=temp[1].astype("float64"))
    h["grid"].attrs["units"] = "K"
np.save(f"{out}/grid.npy", temp.astype("float32"))
np.save(f"{out}/fortran.npy", np.asfortranarray(temp[:, :, :4]))
np.savez(f"{out}/bundle.npz", counts=rng.integers(-5, 5, (4, 3)).astype("int16"), weights=rng.random(6))


def clean(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype="float64").ravel()]


refs = {}
for name in ["packed.nc", "classic.nc"]:
    d = xr.open_dataset(f"{out}/{name}", use_cftime=True)
    refs[name] = {
        v: {"dims": list(d[v].dims), "shape": list(d[v].shape), "values": clean(d[v].values)}
        for v in d.data_vars
    }
    if "time" in d:
        refs[name]["time"] = {"dates": [x.isoformat() for x in d.time.values]}
refs["plain.h5"] = {"grid": {"shape": list(temp[1].shape), "values": clean(temp[1])}}
refs["grid.npy"] = {"grid": {"shape": [t, ny, nx], "values": clean(np.load(f"{out}/grid.npy"))}}
refs["fortran.npy"] = {"fortran": {"shape": [t, ny, 4], "values": clean(np.load(f"{out}/fortran.npy"))}}
z = np.load(f"{out}/bundle.npz")
refs["bundle.npz"] = {k: {"shape": list(z[k].shape), "values": clean(z[k])} for k in z.files}
with open(ref, "w") as f:
    json.dump(refs, f)
print({k: list(v) for k, v in refs.items()})
