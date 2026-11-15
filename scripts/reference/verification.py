"""Makes synthetic forecast/observation data and its verification scores, computed by xskillscore.

Usage: verification.py <out.json>

Run with: uv run --with xskillscore --with properscoring python scripts/reference/verification.py fixtures/reference/verification.json

Scores skip pairs where either side is NaN. Events for the contingency table and the Brier score
are values at or above the threshold.
"""

import json
import sys

import numpy as np
import xarray as xr
import xskillscore as xs

rng = np.random.default_rng(42)
n, members, threshold = 200, 10, 1.0
obs = rng.normal(0.0, 1.5, n)
fc = obs * 0.8 + rng.normal(0.3, 0.7, n)
ens = obs[:, None] * 0.7 + rng.normal(0.2, 1.0, (n, members))
obs[[5, 77]] = np.nan
fc[[12, 77, 150]] = np.nan
ens[33, :] = np.nan

o = xr.DataArray(obs, dims="t")
f = xr.DataArray(fc, dims="t")
e = xr.DataArray(ens, dims=("t", "member"))

prob = (ens >= threshold).mean(axis=1)
prob[33] = np.nan
p = xr.DataArray(prob, dims="t")

edges = np.array([-np.inf, threshold, np.inf])
table = xs.Contingency(o, f, edges, edges, dim="t")
ok = ~(np.isnan(obs) | np.isnan(fc))
hits = int(((fc >= threshold) & (obs >= threshold) & ok).sum())
assert int(table.hits().values) == hits, (int(table.hits().values), hits)

valid_bs = ~(np.isnan(obs) | np.isnan(prob))
ref = {
    "threshold": threshold,
    "obs": [None if np.isnan(v) else float(v) for v in obs],
    "fc": [None if np.isnan(v) else float(v) for v in fc],
    "ens": [[None if np.isnan(v) else float(v) for v in row] for row in ens],
    "prob": [None if np.isnan(v) else float(v) for v in prob],
    "scores": {
        "bias": float(xs.me(f, o, dim="t", skipna=True)),
        "mae": float(xs.mae(f, o, dim="t", skipna=True)),
        "mse": float(xs.mse(f, o, dim="t", skipna=True)),
        "rmse": float(xs.rmse(f, o, dim="t", skipna=True)),
        "pearson_r": float(xs.pearson_r(f, o, dim="t", skipna=True)),
        # xskillscore's spearman_r returns NaN with skipna, so it gets the valid pairs only.
        "spearman_r": float(xs.spearman_r(f[ok], o[ok], dim="t")),
        "brier_score": float(xs.brier_score((o >= threshold).where(valid_bs), p.where(valid_bs), dim="t").mean()),
        "hits": int(table.hits().values),
        "misses": int(table.misses().values),
        "false_alarms": int(table.false_alarms().values),
        "correct_negatives": int(table.correct_negatives().values),
        "hit_rate": float(table.hit_rate().values),
        "false_alarm_ratio": float(table.false_alarm_ratio().values),
        "threat_score": float(table.threat_score().values),
        "peirce_score": float(table.peirce_score().values),
        "heidke_score": float(table.heidke_score().values),
        "equit_threat_score": float(table.equit_threat_score().values),
        "bias_score": float(table.bias_score().values),
    },
}
# CRPS over the observations where the observation and every member are present.
keep = ~(np.isnan(obs) | np.isnan(ens).any(axis=1))
ref["scores"]["crps_ensemble"] = float(xs.crps_ensemble(o[keep], e[keep], dim="t"))
with open(sys.argv[1], "w") as fh:
    json.dump(ref, fh)
print(json.dumps(ref["scores"], indent=1))
