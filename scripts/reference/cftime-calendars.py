"""Decodes CF time values in every calendar with cftime, and converts calendars with xarray.

Usage: cftime-calendars.py <out.json>

Run with: uv run --with cftime --with "xarray[parallel]" --with numpy python scripts/reference/cftime-calendars.py fixtures/reference/cftime.json
"""

import json
import sys
import warnings

import cftime
import numpy as np
import xarray as xr

# Years before 1 in real-world calendars trigger a CF warning about year zero; cftime has no
# year zero there (1 BC is followed by AD 1), which is what the engine follows.
warnings.simplefilter("ignore")

CALENDARS = ["standard", "proleptic_gregorian", "julian", "noleap", "all_leap", "360_day"]
UNITS = [
    "days since 1850-01-01",
    "hours since 1900-01-01 06:00:00",
    "seconds since 1582-10-01 12:00:00",
    "minutes since 0001-01-01",
    "days since 2000-02-28T18:30:00",
]
VALUES = [
    0, 1, -1, 0.5, 1.25, 59, 60, 365, 366, 1460.75, 36524.25, -100000.5, 1e6, 2.5e6, 123456.789,
    86400 * 10, 86400 * 400.5,
]


def iso(d):
    s = f"{d.year:05d}" if d.year < 0 else f"{d.year:04d}"
    s += f"-{d.month:02d}-{d.day:02d}T{d.hour:02d}:{d.minute:02d}:{d.second:02d}"
    if d.microsecond:
        s += f".{d.microsecond:06d}"
    return s


decode = []
for cal in CALENDARS:
    for units in UNITS:
        try:
            dates = cftime.num2date(VALUES, units, calendar=cal, only_use_cftime_datetimes=True)
        except ValueError as e:
            decode.append({"calendar": cal, "units": units, "error": str(e)})
            continue
        decode.append({
            "calendar": cal,
            "units": units,
            "values": VALUES,
            "dates": [iso(d) for d in dates],
            # Encoding the decoded dates must give the values back.
            "encoded": [float(x) for x in cftime.date2num(dates, units, calendar=cal)],
        })

convert = []
for src, dst, align in [
    ("noleap", "360_day", "year"),
    ("noleap", "360_day", "date"),
    ("360_day", "standard", "year"),
    ("360_day", "standard", "date"),
    ("all_leap", "noleap", None),
    ("standard", "noleap", None),
    ("proleptic_gregorian", "all_leap", None),
    ("julian", "proleptic_gregorian", None),
]:
    times = xr.date_range("2000-01-01T06:00:00", "2001-12-30T06:00:00", freq="D", calendar=src, use_cftime=True)
    da = xr.DataArray(np.arange(len(times)), coords={"time": times}, dims="time")
    out = da.convert_calendar(dst, align_on=align, use_cftime=True)
    mapped = {int(v): iso(t) for v, t in zip(out.values, out.time.values)}
    convert.append({
        "from": src,
        "to": dst,
        "align_on": align,
        "dates": [iso(t) for t in times],
        "converted": [mapped.get(i) for i in range(len(times))],
    })

with open(sys.argv[1], "w") as f:
    json.dump({"decode": decode, "convert": convert}, f)
print(len(decode), "decode cases,", sum("error" in d for d in decode), "errors;", len(convert), "conversions")
