"""Computes reference colour-vision-deficiency simulations, CIELAB values, CIEDE2000 differences
and palette checks for dv-render's `cvd` and `color` modules.

Usage: cvd.py <out.json>

Run with: uv run --with colorspacious --with colour-science --with numpy python scripts/reference/cvd.py fixtures/reference/cvd.json

Simulation is colorspacious's "sRGB1+CVD" space (Machado et al. 2009). CIELAB (D65) and CIEDE2000
are colour-science's. The palette check simulates each color, clips it to [0, 1] in linear sRGB,
and flags pairs closer than 10 CIEDE2000 with typical vision or any deficiency at severity 1.
"""

import itertools
import json
import sys

import colorspacious
import colour
import numpy as np

D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
SRGB = colour.RGB_COLOURSPACES["sRGB"]
KINDS = {"protan": "protanomaly", "deutan": "deuteranomaly", "tritan": "tritanomaly"}
MIN_DELTA_E = 10.0

rng = np.random.default_rng(20091)
okabe_ito = [
    [230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66],
    [0, 114, 178], [213, 94, 0], [204, 121, 167],
]
colors = okabe_ito + [
    [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128], [10, 10, 10],
] + rng.integers(0, 256, size=(24, 3)).tolist()


def lab_of_linear(lin):
    return colour.XYZ_to_Lab(colour.RGB_to_XYZ(lin, SRGB, illuminant=D65), D65)


def linear(rgb8):
    return colour.cctf_decoding(np.asarray(rgb8) / 255.0, function="sRGB")


def seen(rgb8, kind, severity):
    """Linear sRGB as seen, clipped to the gamut."""
    if kind is None:
        lin = linear(rgb8)
    else:
        space = {"name": "sRGB1+CVD", "cvd_type": KINDS[kind], "severity": severity * 100}
        sim = colorspacious.cspace_convert(np.asarray(rgb8) / 255.0, space, "sRGB1")
        lin = colorspacious.cspace_convert(sim, "sRGB1", "sRGB1-linear")
    return np.clip(lin, 0.0, 1.0)


simulations = []
for c in colors:
    for kind in KINDS:
        for severity in [0.0, 0.35, 0.5, 0.73, 1.0]:
            space = {"name": "sRGB1+CVD", "cvd_type": KINDS[kind], "severity": severity * 100}
            out = colorspacious.cspace_convert(np.asarray(c) / 255.0, space, "sRGB1")
            simulations.append({"rgb": c, "kind": kind, "severity": severity, "srgb": out.tolist()})

labs = [{"rgb": c, "lab": lab_of_linear(linear(c)).tolist()} for c in colors]

pairs = []
for _ in range(200):
    a = [rng.uniform(0, 100), rng.uniform(-100, 100), rng.uniform(-100, 100)]
    b = [rng.uniform(0, 100), rng.uniform(-100, 100), rng.uniform(-100, 100)]
    pairs.append({"a": a, "b": b})
# Near-identical, achromatic and hue-wrapping pairs.
for a, b in [
    ([50, 0, 0], [50, 0, 0]), ([50, 0, 0], [60, 0, 0]), ([50, 0, 0], [50, -1, 2]),
    ([60, 20, -1], [60, 20, 1]), ([40, -30, -0.5], [40, -30, 0.5]), ([70, 0.2, 0.1], [70, -0.2, -0.1]),
    ([50, 2.5, 0], [73, 25, -18]), ([50, 2.6772, -79.7751], [50, 0, -82.7485]),
]:
    pairs.append({"a": a, "b": b})
for p in pairs:
    p["delta_e"] = float(colour.delta_E(np.asarray(p["a"]), np.asarray(p["b"]), method="CIE 2000"))

palettes = {
    "okabe-ito": okabe_ito,
    # Red and green of similar lightness, and two blues: flagged for deutans, protans and everyone.
    "traffic": [[220, 50, 47], [110, 140, 30], [38, 139, 210], [42, 130, 220], [181, 137, 0]],
    "random": rng.integers(0, 256, size=(10, 3)).tolist(),
}
checks = {}
for name, palette in palettes.items():
    flagged = []
    for i, j in itertools.combinations(range(len(palette)), 2):
        for kind in [None, *KINDS]:
            d = float(colour.delta_E(
                lab_of_linear(seen(palette[i], kind, 1.0)),
                lab_of_linear(seen(palette[j], kind, 1.0)),
                method="CIE 2000",
            ))
            if d < MIN_DELTA_E:
                flagged.append({"a": i, "b": j, "kind": kind, "delta_e": d})
    checks[name] = {"palette": palette, "flagged": flagged}

json.dump(
    {
        "min_delta_e": MIN_DELTA_E,
        "simulations": simulations,
        "labs": labs,
        "pairs": pairs,
        "checks": checks,
        "versions": {"colorspacious": colorspacious.__version__, "colour": colour.__version__},
    },
    open(sys.argv[1], "w"),
    indent=1,
)
