"""Transforms the dv-py tests and the S6 spike run through a worker."""

import time

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

import dataventura as dv

_kept = []


@dv.transform
def identity(t: pa.Table) -> pa.Table:
    return t


@dv.transform(columns=["x"])
def doubled(df: pl.DataFrame) -> pl.DataFrame:
    return df.select((pl.col("x") * 2).alias("x2"))


@dv.transform
def via_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(x_plus_one=df["x"] + 1)


@dv.transform(columns=["x"])
def total(x: np.ndarray) -> np.ndarray:
    return np.array([x.sum()])


@dv.transform
def chatty(t: pa.Table) -> pa.Table:
    # Prints must not corrupt the protocol channel.
    print("hello from user code")
    return t.slice(0, 1)


@dv.transform
def boom(t: pa.Table) -> pa.Table:
    raise ValueError("bad input")


@dv.transform
def keeps_input(t: pa.Table) -> pa.Table:
    _kept.append(t)
    return t.slice(0, 1)


@dv.transform
def hog(t: pa.Table) -> pa.Table:
    # Touches 2 GB, page by page, so the footprint really grows.
    blocks = []
    for _ in range(64):
        b = bytearray(32 << 20)
        b[::4096] = b"\x01" * len(b[::4096])
        blocks.append(b)
    return t


@dv.transform
def slow(t: pa.Table) -> pa.Table:
    for _ in range(1000):
        time.sleep(0.01)
    return t


@dv.transform
def stubborn(t: pa.Table) -> pa.Table:
    # Ignores cancellation, like code stuck in a long C call.
    end = time.monotonic() + 30
    while time.monotonic() < end:
        try:
            time.sleep(0.01)
        except KeyboardInterrupt:
            pass
    return t


@dv.transform(columns=["startLon", "startLat", "endLon", "endLat", "tpepPickupDateTime", "tpepDropoffDateTime"])
def trip_speed(t: pa.Table) -> pa.Table:
    """Great-circle speed of each taxi trip, km/h: a typical derived-field transform."""
    r = np.pi / 180
    lat1, lat2 = t["startLat"].to_numpy() * r, t["endLat"].to_numpy() * r
    # Difference first, then radians: for trips of a few meters, converting first loses digits.
    dlon = (t["endLon"].to_numpy() - t["startLon"].to_numpy()) * r
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    km = 2 * 6371.0088 * np.arcsin(np.sqrt(a))
    hours = pc.divide(
        pc.cast(pc.subtract(t["tpepDropoffDateTime"], t["tpepPickupDateTime"]), pa.int64()), 3.6e9
    ).to_numpy(zero_copy_only=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(hours > 0, km / hours, np.nan)
    return pa.table({"km": km, "kmh": speed})


@dv.transform
def trip_speed_wide(t: pa.Table) -> pa.Table:
    """The same, given every column: what it costs to send a whole table."""
    return trip_speed(t.select(["startLon", "startLat", "endLon", "endLat", "tpepPickupDateTime", "tpepDropoffDateTime"]))


@dv.transform(columns=["startLon", "startLat", "endLon", "endLat", "tpepPickupDateTime", "tpepDropoffDateTime"])
def trip_speed_polars(df: pl.DataFrame) -> pl.DataFrame:
    r = np.pi / 180
    lat1, lat2 = pl.col("startLat") * r, pl.col("endLat") * r
    dlat, dlon = lat2 - lat1, (pl.col("endLon") - pl.col("startLon")) * r
    a = (dlat / 2).sin() ** 2 + lat1.cos() * lat2.cos() * (dlon / 2).sin() ** 2
    km = 2 * 6371.0088 * a.sqrt().arcsin()
    hours = (pl.col("tpepDropoffDateTime") - pl.col("tpepPickupDateTime")).dt.total_microseconds() / 3.6e9
    return df.select(km.alias("km"), pl.when(hours > 0).then(km / hours).alias("kmh"))
