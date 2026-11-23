"""v0.5 refit: thermal mass free (zone air capacitance multiplier; InternalMass area scaling makes results jump
because it changes the radiant enclosure), plus targets from the night-time RC regression and the daily change-point slope.

Objective = calib.objective (training weeks)
          + W_C     * ((C_model - C_MEAS) / C_MEAS)^2        C from rc_fit on the model's cal-year output (zone air temp,
                                                              same quantity the temperature terms compare with the ecobee)
          + W_SLOPE * ((slope_model - slope_meas) / slope_meas)^2   3P CPM slope of daily furnace runtime, cal year
Measured targets (benchmark-cpm-rc.md, 2026-11-23): C = 22.8 MJ/K, daily 3P slope = 0.2588 h/day/K (117 W/K).
Fixed at v0.4.1 values: soil k and basement insulation depth (at bounds), furnace output (plate), leakage split.

    python3 src/fit_mass.py --tag fit5 --maxfev 150   -> runs/calib/<tag>/log.jsonl, best.json
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "model"))
from screen import apply
from calib import evaluate_subprocess
import rc_fit, cpm

C_MEAS, SLOPE_MEAS = 22.8, 0.2588
W_C, W_SLOPE = 0.1, 2.0

FREE = {"air_cap_mult": (1.0, 100.0), "ela_mult": (0.6, 1.4), "env_r_mult": (0.9, 1.25),
        "gains_fraction_lost": (0.0, 0.4), "win_shgc_new": (0.35, 0.55), "ac_cap_mult": (0.8, 1.2),
        "t_neighbour": (18.5, 22.0)}
# x0 as used for fit6 (screen.BASE was v0.4.1 then; it is now v0.5); air capacitance started at x40 (scan: C 16 MJ/K)
X0 = {"air_cap_mult": 40.0, "ela_mult": 1.0, "env_r_mult": 1.0, "gains_fraction_lost": 0.123,
      "win_shgc_new": 0.353, "ac_cap_mult": 1.0, "t_neighbour": 19.48}


def dyn_terms(run_dir):
    """C (MJ/K, night RC regression) and daily 3P slope (h/day/K) from a calib case folder."""
    fr = rc_fit.model_frame("cal", run_dir)
    b, _ = rc_fit.ols(fr)
    C = float(b[2] * 3.6)
    from results import hourly
    import pandas as pd
    h = hourly(Path(run_dir) / "eplusout.sql", ["Heating Coil Runtime Fraction"])
    rt = h.filter(like="Heating Coil Runtime").iloc[:, 0].resample("D").sum().rename("heat_h_mod")
    d = cpm.dataset(); d = d[d.case == "cal"].join(rt, how="inner")
    slope = cpm.fit(d, ["dT"], "heat", "heat_h_mod")["beta"]["dT"]
    return C, float(slope)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="fit5"); ap.add_argument("--maxfev", type=int, default=150)
    A = ap.parse_args()
    names = list(FREE); lo = np.array([FREE[n][0] for n in names]); hi = np.array([FREE[n][1] for n in names])
    od = ROOT / "runs/calib" / A.tag; od.mkdir(parents=True, exist_ok=True)
    log = open(od / "log.jsonl", "a"); n = [0]; best = [np.inf]

    def f(u):
        x = dict(zip(names, lo + np.clip(u, 0, 1) * (hi - lo)))
        n[0] += 1; tag = f"{A.tag}/e{n[0]:03d}"
        J0, sc = evaluate_subprocess(apply(x), tag)
        C, s = dyn_terms(ROOT / "runs/calib" / tag / "cal")
        J = J0 + W_C * ((C - C_MEAS) / C_MEAS) ** 2 + W_SLOPE * ((s - SLOPE_MEAS) / SLOPE_MEAS) ** 2
        rec = dict(i=n[0], x=x, J=J, J0=J0, C=C, slope=s)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if J < best[0]:
            best[0] = J
            (od / "best.json").write_text(json.dumps(dict(rec, scores=sc), indent=1, default=float))
        print(n[0], round(J, 4), round(J0, 4), round(C, 1), round(s * 10842 / 24, 1),
              {k: round(v, 3) for k, v in x.items()}, flush=True)
        return J

    u0 = (np.array([X0[k] for k in names]) - lo) / (hi - lo)
    r = minimize(f, u0, method="Powell", bounds=[(0, 1)] * len(names), options=dict(maxfev=A.maxfev, xtol=0.02, ftol=1e-3))
    print("done", r.fun)


if __name__ == "__main__":
    main()
