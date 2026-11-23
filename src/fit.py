"""Fit the identifiable calibration parameters (Powell, bounded) on training weeks.
Fixed at documented values: envelope RSI (EnerGuide), furnace low-fire output (plate), exhaust x0.6,
fireplace damper open-minus-closed ELA = 150 cm2 (ASHRAE Fundamentals component table: open 350 / closed 43 per
fireplace, scaled to a ~0.5 m2 opening). Damper vs general leakage can't be separated before 2026-11-22.
    python3 src/fit.py   -> runs/calib/fit/log.jsonl, best.json"""
import sys, json, copy
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "model"))
from screen import apply
from calib import evaluate_subprocess

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="fit2"); ap.add_argument("--maxfev", type=int, default=120)
A = ap.parse_args()
# fit2: envelope and basement insulation free; SHGC limited to what low-E argon units can plausibly have
FREE = {"ela_mult": (0.3, 1.4), "win_shgc_new": (0.35, 0.55), "gains_fraction_lost": (0.0, 0.4),
        "ac_cap_mult": (0.8, 1.2), "soil_k": (0.8, 2.6), "env_r_mult": (0.9, 1.2),
        "bsmt_ins_depth_below_grade": (0.5, 1.57), "t_neighbour": (18.5, 22.0)}
X0 = {"ela_mult": 0.55, "win_shgc_new": 0.40, "gains_fraction_lost": 0.2, "ac_cap_mult": 1.0, "soil_k": 1.2,
      "env_r_mult": 1.1, "bsmt_ins_depth_below_grade": 1.2, "t_neighbour": 20.0}
names = list(FREE); lo = np.array([FREE[n][0] for n in names]); hi = np.array([FREE[n][1] for n in names])
OD = ROOT / "runs/calib" / A.tag; OD.mkdir(parents=True, exist_ok=True)
log = open(OD / "log.jsonl", "a"); n_eval = [0]; best = [np.inf, None]

def f(u):
    x = dict(zip(names, lo + np.clip(u, 0, 1) * (hi - lo)))
    n_eval[0] += 1
    J, sc = evaluate_subprocess(apply(x), f"{A.tag}/e{n_eval[0]:03d}")
    log.write(json.dumps(dict(i=n_eval[0], x=x, J=J)) + "\n"); log.flush()
    if J < best[0]:
        best[:] = [J, x]
        (OD / "best.json").write_text(json.dumps(dict(J=J, x=x, i=n_eval[0], scores=sc), indent=1, default=float))
    print(n_eval[0], round(J, 4), {k: round(v, 3) for k, v in x.items()}, flush=True)
    return J

u0 = (np.array([X0[n] for n in names]) - lo) / (hi - lo)
r = minimize(f, u0, method="Powell", bounds=[(0, 1)] * len(names), options=dict(maxfev=A.maxfev, xtol=0.02, ftol=1e-3))
print("done", r.fun, best[1])
