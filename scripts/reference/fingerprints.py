"""Computes source fingerprints with hashlib, for dv-spec's lock files.

Usage: fingerprints.py <fixtures dir> <out.json>

Run with: python3 scripts/reference/fingerprints.py fixtures fixtures/reference/fingerprints.json

Metadata hashes: a Parquet file's footer (the bytes from the footer through the trailing length
and magic), a Zarr store's metadata files (relative path, a zero byte, then the file, in sorted
order), and otherwise the first 64 KiB. Content hashes cover every byte (for a directory: each
file's relative path, a zero byte, then its bytes, in sorted order).
"""

import hashlib
import json
import os
import struct
import sys

fx, out = sys.argv[1], sys.argv[2]
META = {".zmetadata", ".zarray", ".zattrs", ".zgroup", "zarr.json"}


def file_fp(path):
    data = open(path, "rb").read()
    if data[:4] == b"PAR1" and data[-4:] == b"PAR1":
        n = struct.unpack("<I", data[-8:-4])[0] + 8
        meta = data[-n:]
    else:
        meta = data[: 64 * 1024]
    return {"size": len(data), "metadata": hashlib.sha256(meta).hexdigest(), "content": hashlib.sha256(data).hexdigest()}


def dir_fp(root):
    files = []
    for d, _, names in os.walk(root):
        for n in names:
            p = os.path.join(d, n)
            files.append(os.path.relpath(p, root).replace(os.sep, "/"))
    files.sort()
    meta, content, size = hashlib.sha256(), hashlib.sha256(), 0
    for rel in files:
        data = open(os.path.join(root, rel), "rb").read()
        size += len(data)
        if rel.rsplit("/", 1)[-1] in META:
            meta.update(rel.encode() + b"\0" + data)
        content.update(rel.encode() + b"\0" + data)
    return {"size": size, "metadata": meta.hexdigest(), "content": content.hexdigest()}


ref = {
    "taxi-20k.parquet": file_fp(f"{fx}/taxi-20k.parquet"),
    "tables/trips.csv": file_fp(f"{fx}/tables/trips.csv"),
    "tables/obs.csv": file_fp(f"{fx}/tables/obs.csv"),
    "arrays/packed.nc": file_fp(f"{fx}/arrays/packed.nc"),
    "arco-like.zarr": dir_fp(f"{fx}/arco-like.zarr"),
}
with open(out, "w") as f:
    json.dump(ref, f, indent=1)
print({k: v["size"] for k, v in ref.items()})
