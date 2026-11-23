"""Calibration harness: run the two calibration cases and score them against measured data.

Cases
  cal   2025-08-22 .. 2026-07-31, Kanata_actual_2025-08_2026-07.epw, sched_8760.csv
  fall  2026-08-01 .. 2026-11-20, Kanata_2026_obs.epw,              sched_fall2026.csv
Targets (all LST)
  daily furnace runtime h (ecobee heat_s)     heating days (>= 0.25 h measured)
  daily AC runtime h (ecobee cool_s)          cooling days (>= 0.25 h measured)
  hourly Floor1/Floor2 temperature (ecobee t_main / t_up)
  gas between actual Enbridge reads (meter) - checks the runtime->gas link and DHW
Hold-out: every 4th ISO week (week % 4 == 3) is excluded from the objective and reported separately.

    python3 src/calib.py            # run both cases with PARAMS and print the score card
"""
import sys, json, copy, shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "model"))
import build_model
from run_ep import run
from results import hourly

C, W = ROOT / "data/clean", ROOT / "weather"
HV_J = 37.8e6
CASES = {
    "cal": dict(weather="actual", period=None, sched="sched_8760.csv", epw="Kanata_actual_2025-08_2026-07.epw",
                span=("2025-08-29", "2026-07-31")),          # first week = warm-up margin
    "fall": dict(weather="custom", period=("2026-08-01", "2026-11-20"), sched="sched_fall2026.csv",
                 epw="Kanata_2026_obs.epw", span=("2026-08-08", "2026-11-20")),
}
VARS = ["Heating Coil Runtime Fraction", "Cooling Coil Runtime Fraction", "Zone Mean Air Temperature",
        "NaturalGas:Facility", "Electricity:Facility", "Site Outdoor Air Drybulb Temperature"]

_meas = None


def measured():
    global _meas
    if _meas is None:
        e = pd.read_csv(C / "ecobee_5min.csv", index_col=0, parse_dates=True)
        x = e[["heat_s", "cool_s"]].fillna(0)
        daily = (x.resample("D").sum() / 3600).rename(columns={"heat_s": "heat_h", "cool_s": "cool_h"})
        ok = e.t_main.resample("D").count() >= 250          # >= ~21 h of 5-min records
        daily = daily[ok]
        temps = e[["t_main", "t_up"]].resample("h").mean()
        g = pd.read_csv(C / "enbridge_reads.csv", parse_dates=["billed_to"])
        a = g[g["Read Type"] == "A"].sort_values("billed_to").reset_index(drop=True)
        reads = [(a.billed_to[i - 1] + pd.Timedelta(days=1), a.billed_to[i] + pd.Timedelta(days=1),
                  float(a["Meter Read"][i] - a["Meter Read"][i - 1])) for i in range(1, len(a))]
        _meas = dict(daily=daily, temps=temps, reads=reads,
                     kwh=pd.read_csv(C / "hydro_hourly.csv", index_col=0, parse_dates=True).kwh)
    return _meas


def holdout(idx):
    return pd.Index(idx).isocalendar().week.values % 4 == 3 if len(idx) else np.array([], bool)


def run_case(case, p, outdir):
    c = CASES[case]
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    m = build_model.build(c["weather"], p=p, period=c["period"], sched=c["sched"])
    (outdir / "house.epJSON").write_text(json.dumps(m))
    rc = run(outdir / "house.epJSON", W / c["epw"], outdir)
    if rc != 0:
        raise RuntimeError(f"E+ failed for {case} in {outdir}")
    d = hourly(outdir / "eplusout.sql", VARS)
    for f in outdir.iterdir():           # keep the folder light: sql + inputs only
        if f.name not in ("eplusout.sql", "house.epJSON", "eplusout.err"):
            f.unlink()
    return d


def stats(m, s):
    d = pd.concat([m, s], axis=1, sort=True).dropna(); m, s = d.iloc[:, 0], d.iloc[:, 1]
    n = len(d)
    if n < 3 or m.mean() == 0:
        return dict(n=n, nmbe=np.nan, cvrmse=np.nan)
    return dict(n=n, nmbe=float(100 * (m - s).sum() / ((n - 1) * m.mean())),
                cvrmse=float(100 * np.sqrt(((m - s) ** 2).sum() / (n - 1)) / m.mean()))


def score_case(case, d):
    M = measured(); s0, s1 = CASES[case]["span"]
    mod_daily = pd.DataFrame({"heat_h": d["Heating Coil Runtime Fraction[FURNACE AC HEATING COIL]"].resample("D").sum()
                              if "Heating Coil Runtime Fraction[FURNACE AC HEATING COIL]" in d else
                              d.filter(like="Heating Coil Runtime").iloc[:, 0].resample("D").sum(),
                              "cool_h": d.filter(like="Cooling Coil Runtime").iloc[:, 0].resample("D").sum()}).loc[s0:s1]
    meas = M["daily"].loc[s0:s1]
    j = meas.join(mod_daily, rsuffix="_mod", how="inner")
    ho = holdout(j.index)
    r = {"case": case}
    for k in ["heat_h", "cool_h"]:
        act = j[k] >= 0.25
        for lab, sel in [("train", act & ~ho), ("hold", act & ho)]:
            r[f"{k}_{lab}"] = stats(j.loc[sel, k], j.loc[sel, k + "_mod"])
        r[f"{k}_tot"] = (float(j.loc[act, k].sum()), float(j.loc[act, k + "_mod"].sum()))
    th = M["temps"].loc[s0:s1]
    tm = d.filter(like="Zone Mean Air Temperature").loc[s0:s1]
    hot = holdout(th.index)
    for meas_c, z in [("t_main", "FLOOR1"), ("t_up", "FLOOR2")]:
        diff = (tm[f"Zone Mean Air Temperature[{z}]"] - th[meas_c]).dropna()
        tr = diff[~holdout(diff.index)]
        r[f"T_{z}"] = dict(bias=float(tr.mean()), rmse=float(np.sqrt((tr ** 2).mean())))
    tb = tm.get("Zone Mean Air Temperature[BASEMENT]")
    win = tb[tb.index.month.isin([12, 1, 2])] if tb is not None else pd.Series(dtype=float)
    r["T_BSMT_DJF"] = float(win.mean()) if len(win) else None
    gm = d["NaturalGas:Facility"] / HV_J
    r["gas_reads"] = [dict(start=str(a.date()), end=str(b.date()), meas=v, model=round(float(gm.loc[a:b - pd.Timedelta(hours=1)].sum()), 1))
                      for a, b, v in M["reads"] if a >= d.index.min() and b <= d.index.max() + pd.Timedelta(hours=1)]
    em = d["Electricity:Facility"].loc[s0:s1] / 3.6e6
    r["elec_hourly"] = stats(M["kwh"].loc[s0:s1], em)
    return r


def objective(scores):
    """Scalar to minimise (training data only). Weights: furnace runtime dominates (gas forecast)."""
    J = 0.0
    for r in scores:
        h = r["heat_h_train"]; c = r["cool_h_train"]
        J += 1.0 * (h["cvrmse"] / 100) ** 2 + 2.0 * (h["nmbe"] / 100) ** 2
        if c["n"] >= 10:
            J += 0.3 * (c["cvrmse"] / 100) ** 2 + 0.6 * (c["nmbe"] / 100) ** 2
        for z in ["T_FLOOR1", "T_FLOOR2"]:
            J += 0.05 * r[z]["rmse"] ** 2
        tb = r.get("T_BSMT_DJF")
        if tb is not None:     # J-P: basement is 16 C at worst in winter (no sensor) -> soft band 16-18 C
            J += 0.5 * (max(0.0, 16.0 - tb) + max(0.0, tb - 18.5)) ** 2
    return J


def evaluate(p, tag, root=ROOT / "runs/calib"):
    scores = [score_case(c, run_case(c, p, root / tag / c)) for c in CASES]
    return objective(scores), scores


def card(J, scores):
    lines = [f"J = {J:.4f}"]
    for r in scores:
        h, ho, c = r["heat_h_train"], r["heat_h_hold"], r["cool_h_train"]
        lines.append(f"[{r['case']}] furnace h meas/model {r['heat_h_tot'][0]:.0f}/{r['heat_h_tot'][1]:.0f}  "
                     f"daily train NMBE {h['nmbe']:+.1f}% CV {h['cvrmse']:.1f}% | hold NMBE {ho['nmbe']:+.1f}% CV {ho['cvrmse']:.1f}%")
        lines.append(f"       AC h {r['cool_h_tot'][0]:.0f}/{r['cool_h_tot'][1]:.0f}  train NMBE {c['nmbe']:+.1f}% CV {c['cvrmse']:.1f}%  "
                     f"T1 bias {r['T_FLOOR1']['bias']:+.2f} rmse {r['T_FLOOR1']['rmse']:.2f}  T2 bias {r['T_FLOOR2']['bias']:+.2f} rmse {r['T_FLOOR2']['rmse']:.2f}  "
                     f"elec NMBE {r['elec_hourly']['nmbe']:+.1f}%" + (f"  Tbsmt DJF {r['T_BSMT_DJF']:.1f}" if r.get("T_BSMT_DJF") else ""))
        for g in r["gas_reads"]:
            lines.append(f"       gas {g['start']}..{g['end']}: meter {g['meas']:.0f} model {g['model']:.0f} m3 ({100 * (g['model'] / g['meas'] - 1):+.0f}%)")
    return "\n".join(lines)


def evaluate_subprocess(p, tag, root=ROOT / "runs/calib"):
    """Same as evaluate() but in a fresh Python process (the E+ API is not fork-safe)."""
    import subprocess
    od = root / tag; od.mkdir(parents=True, exist_ok=True)
    (od / "params.json").write_text(json.dumps(p))
    r = subprocess.run([sys.executable, str(Path(__file__)), tag, "--params", str(od / "params.json"), "--root", str(root)],
                       capture_output=True, text=True, timeout=600, cwd=str(od))
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    res = json.loads((od / "scores.json").read_text())
    return res["J"], res["scores"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("tag", nargs="?", default="base")
    ap.add_argument("--params"); ap.add_argument("--root", default=str(ROOT / "runs/calib"))
    A = ap.parse_args()
    p = json.loads(Path(A.params).read_text()) if A.params else copy.deepcopy(build_model.PARAMS)
    J, sc = evaluate(p, A.tag, Path(A.root))
    (Path(A.root) / A.tag / "scores.json").write_text(json.dumps(dict(J=J, scores=sc), indent=1, default=float))
    if not A.params:
        print(card(J, sc))
