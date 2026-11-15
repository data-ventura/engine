"""Computes reference unit conversions with pint.

Usage: units.py <out.json>

Run with: uv run --with pint python scripts/reference/units.py fixtures/reference/units.json

Each case converts a value between two UDUNITS spellings. pint spells a few units differently, so
each case also gives pint's spelling. pint has no UDUNITS dependency, so it's an independent check.
"""

import json
import sys

import pint

ureg = pint.UnitRegistry()
Q = ureg.Quantity

# (ours from, ours to, pint from, pint to, value)
CASES = [
    ("km", "m", "km", "m", 3.5),
    ("hPa", "Pa", "hPa", "Pa", 1013.25),
    ("mbar", "kPa", "mbar", "kPa", 850.0),
    ("degC", "K", "degC", "K", -40.0),
    ("degF", "degC", "degF", "degC", 98.6),
    ("K", "degF", "K", "degF", 300.0),
    ("knots", "m/s", "knot", "m/s", 25.0),
    ("mph", "km/h", "mph", "km/h", 60.0),
    ("ft", "m", "ft", "m", 1000.0),
    ("inch", "mm", "inch", "mm", 2.0),
    ("W m-2", "J m-2 h-1", "W/m**2", "J/m**2/h", 340.0),
    ("days", "hours", "day", "hour", 1.5),
    ("%", "1", "percent", "dimensionless", 42.0),
    ("ppm", "1", "ppm", "dimensionless", 410.0),
    ("degrees_north", "rad", "degree", "radian", 45.0),
    ("m3 s-1", "L/min", "m**3/s", "L/min", 0.25),
    ("g/kg", "1", "g/kg", "dimensionless", 12.0),
    ("t", "lb", "metric_ton", "lb", 2.0),
    ("ha", "km2", "hectare", "km**2", 250.0),
]

out = []
for a, b, pa, pb, v in CASES:
    out.append({"from": a, "to": b, "value": v, "result": Q(v, pa).to(pb).magnitude})

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1)
print(len(out), "cases")
