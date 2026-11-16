"""Writes crates/dv-render/src/cvd/machado.rs: Machado et al. (2009) colour-vision-deficiency
simulation matrices, from colorspacious's copy of the authors' supplementary data.

Usage: gen-cvd-matrices.py <out.rs>

Run with: uv run --with colorspacious python scripts/gen-cvd-matrices.py crates/dv-render/src/cvd/machado.rs

Machado, Oliveira & Fernandes (2009), "A Physiologically-based Model for Simulation of Color Vision
Deficiency", IEEE TVCG 15(6), doi:10.1109/TVCG.2009.113. colorspacious is MIT-licensed.
"""

import sys

from colorspacious.cvd import MACHADO_ET_AL_MATRICES

lines = [
    "//! Machado et al. (2009) simulation matrices for linear sRGB, at severities 0, 0.1, …, 1, generated",
    "//! by scripts/gen-cvd-matrices.py from colorspacious's copy of the authors' supplementary data.",
    "",
]
for kind in ["protanomaly", "deuteranomaly", "tritanomaly"]:
    name = {"protanomaly": "PROTAN", "deuteranomaly": "DEUTAN", "tritanomaly": "TRITAN"}[kind]
    lines.append(f"pub(super) const {name}: [[[f64; 3]; 3]; 11] = [")
    for severity in range(0, 101, 10):
        lines.append("    [")
        for row in MACHADO_ET_AL_MATRICES[kind][severity]:
            # Positive zeros: the source writes some as -0.000000.
            lines.append("        [" + ", ".join(f"{v + 0.0:.6f}" for v in row) + "],")
        lines.append("    ],")
    lines.append("];")
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
