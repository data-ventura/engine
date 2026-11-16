"""Makes fixtures/zarr/* (Zarr format 2 and 3 stores and an Icechunk repository) and what xarray
decodes from them.

Usage: make-zarr-stores.py <out dir> <reference.json>

Run with: uv run --python 3.12 --with "zarr>=3.1,<3.2" --with "xarray>=2025.10" --with "icechunk>=2.3,<2.4" --with numcodecs --with cftime python -I scripts/reference/make-zarr-stores.py fixtures/zarr fixtures/reference/zarr.json

The variables match fixtures/arrays/packed.nc, so Zarr decodes as NetCDF does: temperature as
int16 with scale_factor/add_offset and a _FillValue, precipitation as float32 with a
missing_value, a uint8 cloud cover stored as signed bytes with _Unsigned, and time in the noleap
calendar. Each store uses different codecs:

- v2.zarr: Zarr format 2 with consolidated metadata; Blosc zstd + bit shuffle, gzip, zlib,
  "/" chunk keys, a Fortran-order array and one chunk missing (read as the fill value).
- v3.zarr: Zarr format 3 without consolidated metadata, so readers list the store; zstd,
  gzip + crc32c, Blosc LZ4 and a sharded array, with one chunk missing.
- repo.icechunk: an Icechunk repository. The first commit is tagged v1, the second (new
  temperatures) is on main, and branch dev adds a third (new precipitation).

The reference is what xarray reads back: open_zarr for the stores, and open_zarr on a read-only
session at each branch, tag and snapshot for the repository.
"""

import json
import os
import shutil
import sys

import icechunk
import numpy as np
import xarray as xr
import zarr
from numcodecs import GZip, Zlib
from numcodecs import Blosc as BloscV2
from zarr.codecs import BloscCodec, Crc32cCodec, GzipCodec, ZstdCodec

out, ref = sys.argv[1], sys.argv[2]
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
rng = np.random.default_rng(11)
t, ny, nx = 3, 5, 8
temp = 250 + 40 * rng.random((t, ny, nx))
temp[0, 1, 2] = np.nan
precip = (5 * rng.random((t, ny, nx))).astype("float32")
precip[2, 4, 7] = -999.0
cloud = rng.integers(0, 256, (ny, nx)).astype("uint8")


def dataset(temp, precip):
    return xr.Dataset(
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
        attrs={"title": "dv zarr fixture"},
    )


ds = dataset(temp, precip)
packed = {"dtype": "int16", "scale_factor": 0.01, "add_offset": 273.15, "_FillValue": np.int16(-32767)}
chunks = (1, 3, 4)

# Zarr format 2.
enc2 = {
    "t2m": {**packed, "chunks": chunks, "compressors": BloscV2(cname="zstd", clevel=3, shuffle=BloscV2.BITSHUFFLE)},
    "precip": {"chunks": chunks, "compressors": GZip(level=5), "_FillValue": None},
    "cloud": {"chunks": (2, 8), "compressors": Zlib(level=1)},
}
ds.to_zarr(f"{out}/v2.zarr", zarr_format=2, encoding=enc2, consolidated=False)
root = zarr.open_group(f"{out}/v2.zarr", mode="a", zarr_format=2)
fortran = np.arange(ny * nx, dtype="float64").reshape(ny, nx) / 4
f = root.create_array(
    "fortran", shape=(ny, nx), chunks=(2, 3), dtype="float64", order="F", fill_value=None,
    compressors=GZip(level=1), chunk_key_encoding={"name": "v2", "separator": "/"},
    attributes={"_ARRAY_DIMENSIONS": ["lat", "lon"]},
)
f[:] = fortran
zarr.consolidate_metadata(f"{out}/v2.zarr", zarr_format=2)
os.remove(f"{out}/v2.zarr/precip/1.1.0")

# Zarr format 3.
enc3 = {
    "t2m": {**packed, "chunks": chunks, "compressors": ZstdCodec(level=3)},
    "precip": {"chunks": chunks, "compressors": [GzipCodec(level=5), Crc32cCodec()], "_FillValue": None},
    "cloud": {"chunks": (1, 4), "shards": (5, 8), "compressors": BloscCodec(cname="lz4", shuffle="shuffle")},
}
ds.to_zarr(f"{out}/v3.zarr", zarr_format=3, encoding=enc3, consolidated=False)
os.remove(f"{out}/v3.zarr/t2m/c/2/1/0")

# Icechunk.
repo = icechunk.Repository.create(icechunk.local_filesystem_storage(f"{out}/repo.icechunk"))
session = repo.writable_session("main")
enc_ic = {"t2m": {**packed, "chunks": chunks}, "precip": {"chunks": chunks, "_FillValue": None}}
ds.to_zarr(session.store, zarr_format=3, encoding=enc_ic, consolidated=False)
first = session.commit("first")
repo.create_tag("v1", first)
repo.create_branch("dev", first)
# Later commits write the stored (packed) values straight through zarr.
temp2 = np.round((temp + 1.5 - 273.15) / 0.01)
session = repo.writable_session("main")
zarr.open_group(session.store)["t2m"][:] = np.where(np.isnan(temp2), -32767, temp2).astype("int16")
second = session.commit("second")
precip2 = (precip * 2).astype("float32")
precip2[2, 4, 7] = -999.0
session = repo.writable_session("dev")
zarr.open_group(session.store)["precip"][:] = precip2
third = session.commit("third")


def clean(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype="float64").ravel()]


def describe(d):
    r = {v: {"dims": list(d[v].dims), "shape": list(d[v].shape), "values": clean(d[v].values)} for v in d.variables if v != "time"}
    r["time"] = {"dates": [x.isoformat() for x in d.time.values]}
    return r


refs = {}
for name in ["v2.zarr", "v3.zarr"]:
    refs[name] = describe(xr.open_zarr(f"{out}/{name}", consolidated=name == "v2.zarr", decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)))
refs["repo.icechunk"] = {"snapshots": {"first": first, "second": second, "third": third}}
for label, kw in [("main", {"branch": "main"}), ("dev", {"branch": "dev"}), ("v1", {"tag": "v1"}), ("first", {"snapshot_id": first})]:
    s = repo.readonly_session(**kw)
    refs["repo.icechunk"][label] = describe(xr.open_zarr(s.store, consolidated=False, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)))
with open(ref, "w") as fh:
    json.dump(refs, fh)
print({k: sorted(v) for k, v in refs.items()})
