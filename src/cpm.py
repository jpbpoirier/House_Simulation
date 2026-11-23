"""Change-point model (CPM) benchmark for daily furnace and AC runtime (BLDG 5301 lecture 2; Kissock et al. 1998).

Same targets, days, hold-out and NMBE/CV(RMSE) definitions as src/calib.py, so results compare directly with E+.
Weather = the EPW files the calibration cases use (daily mean dry-bulb, daily global horizontal irradiation, mean wind).

    python3 src/cpm.py [cal_run_dir fall_run_dir]   # optional: E+ run folders (eplusout.sql) from calib.run_case, for comparison
Results -> runs/cpm_results.json. Findings (2026-11-23): claude.ai project doc benchmark-cpm-rc.md.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calib import CASES, measured, holdout, stats

W = ROOT / "weather"
EPW_COLS = {0: "year", 1: "month", 2: "day", 3: "hour", 6: "t", 13: "ghi", 21: "wind"}


def epw_daily(name):
    e = pd.read_csv(W / name, skiprows=8, header=None, usecols=list(EPW_COLS)).rename(columns=EPW_COLS)
    ts = pd.to_datetime(e[["year", "month", "day"]]) + pd.to_timedelta(e.hour - 1, unit="h")  # hour-ending -> start
    e.index = ts
    return pd.DataFrame({"t": e.t.resample("D").mean(), "ghi": e.ghi.resample("D").sum() / 1000,   # kWh/m2/day
                         "wind": e.wind.resample("D").mean(), "t_lag": e.t.resample("D").mean().shift(1)})


def dataset():
    M = measured()
    eco = pd.read_csv(ROOT / "data/clean/ecobee_5min.csv", index_col=0, parse_dates=True)
    sp = eco.sp_heat.resample("D").mean()
    rows = []
    for case, c in CASES.items():
        w = epw_daily(c["epw"]).loc[c["span"][0]:c["span"][1]]
        d = w.join(M["daily"], how="inner").join(sp.rename("sp_heat"))
        d["case"] = case
        rows.append(d)
    d = pd.concat(rows)
    d["hold"] = holdout(d.index)
    return d.dropna(subset=["t", "heat_h", "cool_h"])


# ---- models -------------------------------------------------------------------------------------------------
# Heating: y = max(0, X(Tb) @ beta), X = [Tb-T, ghi, wind*(Tb-T), sp_heat-20, Tb-T_lag], used only where T < Tb.
HEAT_SPECS = {"3P": ["dT"], "3P+solar": ["dT", "ghi"], "3P+solar+wind": ["dT", "ghi", "wdT"],
              "3P+solar+wind+setpoint": ["dT", "ghi", "wdT", "dsp"],
              "3P+solar+wind+setpoint+lag": ["dT", "ghi", "wdT", "dsp", "dTlag"]}
COOL_SPECS = {"3P": ["dT"], "3P+solar": ["dT", "ghi"]}


def features(d, tb, mode):
    s = 1 if mode == "heat" else -1
    dT = (s * (tb - d.t)).clip(lower=0)
    on = dT > 0
    f = pd.DataFrame({"dT": dT, "ghi": d.ghi * on, "wdT": d.wind * dT, "dsp": (d.sp_heat - 20) * on,
                      "dTlag": (s * (tb - d.t_lag)).clip(lower=0)}, index=d.index)
    return f, on


def fit(d, cols, mode, y):
    best = None
    for tb in np.arange(8, 24.01, 0.25):
        f, on = features(d, tb, mode)
        X = f.loc[on, cols].values
        if on.sum() < len(cols) + 5:
            continue
        beta, *_ = np.linalg.lstsq(X, d.loc[on, y].values, rcond=None)
        pred = np.where(on, f[cols].values @ beta, 0).clip(min=0)
        sse = ((d[y].values - pred) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, tb, beta)
    return dict(tb=float(best[1]), beta=dict(zip(cols, map(float, best[2]))))


def predict(m, d, cols, mode):
    f, on = features(d, m["tb"], mode)
    return pd.Series(np.where(on, f[cols].values @ np.array([m["beta"][c] for c in cols]), 0).clip(min=0), d.index)


def score(d, y, pred):
    act = d[y] >= 0.25
    out = {}
    for case in CASES:
        for lab, sel in [("train", ~d.hold), ("hold", d.hold)]:
            s = (d.case == case) & act & sel
            out[f"{case}_{lab}"] = stats(d.loc[s, y], pred[s])
    return out


def run(eplus_dirs=None):
    d = dataset()
    res = {}
    for y, mode, specs in [("heat_h", "heat", HEAT_SPECS), ("cool_h", "cool", COOL_SPECS)]:
        for name, cols in specs.items():
            # A: trained on training weeks of both cases (same data E+ v0.4.1 was calibrated on)
            m = fit(d[~d.hold], cols, mode, y)
            # B: trained on the calibration year only -> fall 2026 is fully out of sample
            mb = fit(d[(~d.hold) & (d.case == "cal")], cols, mode, y)
            sb = score(d, y, predict(mb, d, cols, mode))
            res[f"{y} {name}"] = dict(model=m, score=score(d, y, predict(m, d, cols, mode)), model_calonly=mb,
                                      fall_oos=dict(all=stats(d.loc[(d.case == "fall") & (d[y] >= 0.25), y],
                                                              predict(mb, d, cols, mode)[(d.case == "fall") & (d[y] >= 0.25)])))
    if eplus_dirs:
        from results import hourly
        parts = []
        for c, rd in eplus_dirs.items():
            h = hourly(Path(rd) / "eplusout.sql", ["Heating Coil Runtime Fraction", "Cooling Coil Runtime Fraction"])
            parts.append(pd.DataFrame({"heat_h_mod": h.filter(like="Heating Coil Runtime").iloc[:, 0].resample("D").sum(),
                                       "cool_h_mod": h.filter(like="Cooling Coil Runtime").iloc[:, 0].resample("D").sum()})
                         .loc[CASES[c]["span"][0]:CASES[c]["span"][1]])
        ep = pd.concat(parts)
        dd = d.join(ep, how="inner")
        for y in ["heat_h", "cool_h"]:
            res[f"{y} E+"] = dict(score=score(dd, y, dd[y + "_mod"]),
                                         fall_oos=dict(all=stats(dd.loc[(dd.case == "fall") & (dd[y] >= 0.25), y],
                                                                 dd.loc[(dd.case == "fall") & (dd[y] >= 0.25), y + "_mod"])))
    return d, res


if __name__ == "__main__":
    d, res = run(dict(zip(CASES, sys.argv[1:3])) if len(sys.argv) > 2 else None)
    for k, v in res.items():
        s = v["score"]
        cells = "  ".join(f"{c}: {s[c]['nmbe']:+5.1f}/{s[c]['cvrmse']:4.1f} (n={s[c]['n']})" for c in s)
        extra = f"  Tb={v['model']['tb']}" if "model" in v else ""
        print(f"{k:42s} {cells}  fall OOS {v['fall_oos']['all']['nmbe']:+5.1f}/{v['fall_oos']['all']['cvrmse']:4.1f}{extra}")
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/cpm_results.json").write_text(json.dumps(res, indent=1))
