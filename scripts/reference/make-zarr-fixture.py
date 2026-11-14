"""Makes fixtures/arco-like.zarr and its xarray reference values.

Usage: make-zarr-fixture.py <out.zarr> <reference.json>

Run with zarr-python 2, as ARCO-ERA5 was written: uv run --python 3.12 --with "zarr<3" --with "xarray<2025.9" --with "numcodecs<0.16" .

The array copies ARCO-ERA5's layout at a tiny size: Zarr format 2, float32, one time step per chunk,
Blosc LZ4 with byte shuffle, NaN fill, consolidated metadata. Step 2's chunk is deleted after
writing, so readers must return the fill value for it. The reference is what xarray reads back.
"""

import json
import os
import shutil
import sys

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

out, ref = sys.argv[1], sys.argv[2]
shutil.rmtree(out, ignore_errors=True)
t, ny, nx = 4, 9, 16
rng = np.random.default_rng(7)
data = (250 + 40 * rng.random((t, ny, nx))).astype("float32")
data[1, 3, 5] = np.nan
ds = xr.Dataset(
    {"t2m": (("time", "latitude", "longitude"), data, {"units": "K"})},
    coords={"latitude": np.linspace(90, -90, ny), "longitude": np.arange(nx) * 22.5},
)
enc = {"t2m": {"chunks": (1, ny, nx), "compressor": Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)}}
ds.to_zarr(out, encoding=enc, consolidated=True)
os.remove(os.path.join(out, "t2m", "2.0.0"))

back = xr.open_zarr(out)["t2m"].values
with open(ref, "w") as f:
    json.dump({"shape": list(back.shape), "values": [None if np.isnan(v) else float(v) for v in back.ravel()]}, f)
print(back.shape, int(np.isnan(back).sum()), "NaN")
