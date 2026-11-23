"""Morris elementary-effects screening of calibration parameters (2 worker processes).
    python3 src/screen.py [r]      -> runs/calib/screen/morris.json"""
import sys, json, copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "model"))
import build_model
from calib import evaluate_subprocess as evaluate

BASE = copy.deepcopy(build_model.PARAMS)
# name: (low, high) in the units of apply(); base values shown in comments
SPACE = {
    "ela_mult": (0.6, 1.4),            # x current ELA (0.5 x EnerGuide)
    "damper_ela_cm2": (0.0, 300.0),
    "win_shgc_new": (0.30, 0.60),
    "gains_fraction_lost": (0.0, 0.40),
    "soil_k": (1.0, 2.5),
    "mass_mult": (0.5, 3.0),
    "exhaust_mult": (0.3, 1.0),
    "ac_cap_mult": (0.8, 1.1),
    "env_r_mult": (0.8, 1.3),          # walls + ceiling effective RSI
    "furnace_out_mult": (0.85, 1.05),  # delivered output at low fire vs 37 kBtu/h plate (eff <= 1)
    "bsmt_ins_depth_below_grade": (0.5, 1.5),
}


def apply(x):
    p = copy.deepcopy(BASE)
    for k, v in x.items():
        if k == "ela_mult":
            p["infil_mult"] = BASE["infil_mult"] * v
        elif k == "ac_cap_mult":
            p["ac_cap_W"] = BASE["ac_cap_W"] * v
        elif k == "env_r_mult":
            for n in ["rsi_wall_f1", "rsi_wall_f2", "rsi_ceiling"]:
                p[n] = BASE[n] * v
        elif k == "furnace_out_mult":
            p["furnace_output_W"] = BASE["furnace_output_W"] * v
            p["furnace_eff"] = BASE["furnace_eff"] * v     # same gas input (39 kBtu/h), more/less heat
        else:
            p[k] = v
    return p


def outputs(J, sc):
    o = {"J": J}
    for r in sc:
        c = r["case"]
        o[f"{c}_heat_nmbe"] = r["heat_h_train"]["nmbe"]; o[f"{c}_heat_cv"] = r["heat_h_train"]["cvrmse"]
        o[f"{c}_cool_nmbe"] = r["cool_h_train"]["nmbe"]
        o[f"{c}_T1_rmse"] = r["T_FLOOR1"]["rmse"]; o[f"{c}_T2_bias"] = r["T_FLOOR2"]["bias"]
    return o


def job(arg):
    i, u = arg
    x = {k: lo + ui * (hi - lo) for (k, (lo, hi)), ui in zip(SPACE.items(), u)}
    J, sc = evaluate(apply(x), f"screen/{i:03d}")
    return i, x, outputs(J, sc)


if __name__ == "__main__":
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    k, levels = len(SPACE), 4
    delta = levels / (2 * (levels - 1))
    rng = np.random.default_rng(1)
    pts, meta = [], []
    for t in range(r):                       # Morris trajectories on a 4-level grid
        u = rng.integers(0, levels // 2, k) / (levels - 1)
        traj = [u.copy()]
        for j in rng.permutation(k):
            u = u.copy(); u[j] = u[j] + delta if u[j] + delta <= 1 else u[j] - delta
            traj.append(u); meta.append((t, j))
        pts += traj
    with ThreadPoolExecutor(2) as pool:
        res = sorted(pool.map(job, list(enumerate(pts))))
    out = [o for _, _, o in res]
    names = list(SPACE); keys = list(out[0])
    ee = {n: {kk: [] for kk in keys} for n in names}
    for t in range(r):
        base_i = t * (k + 1)
        for s in range(k):
            a, b = res[base_i + s], res[base_i + s + 1]
            j = [n for n in names if abs(a[1][n] - b[1][n]) > 1e-9][0]
            sgn = np.sign(b[1][j] - a[1][j])
            for kk in keys:
                ee[j][kk].append(float(sgn * (b[2][kk] - a[2][kk]) / delta))
    summ = {n: {kk: dict(mu_star=float(np.mean(np.abs(v))), mu=float(np.mean(v)), sigma=float(np.std(v)))
                for kk, v in d.items()} for n, d in ee.items()}
    od = ROOT / "runs/calib/screen"; od.mkdir(parents=True, exist_ok=True)
    (od / "morris.json").write_text(json.dumps(dict(space=SPACE, runs=[dict(x=x, o=o) for _, x, o in res], summary=summ), indent=1))
    for kk in ["J", "cal_heat_nmbe", "fall_heat_nmbe", "cal_cool_nmbe", "cal_T1_rmse"]:
        print(f"\n{kk}: " + ", ".join(f"{n} {summ[n][kk]['mu']:+.2f}/{summ[n][kk]['mu_star']:.2f}"
                                       for n in sorted(names, key=lambda n: -summ[n][kk]["mu_star"])))
