"""Grey-box UA / C identification from night-time data (BLDG 5301 lecture 4: Q_mass = Q_env + Q_hvac + Q_misc).

Whole-house, single-node energy balance on hourly data, night hours only (no solar):
    Q_furnace + E_elec = UA*(T_in - T_oa) + K_w*wind*(T_in - T_oa) + C*dT_in/dt - G_u
  Q_furnace = runtime fraction x 10.84 kW (low-fire output, 37 kBtu/h)
  E_elec    = whole-house electricity (all of it treated as internal heat; includes the blower)
  T_in      = mean of main-floor and upstairs sensors (basement unmeasured -> its losses land in the intercept)
  dT_in/dt  = centred difference of hourly means (K/h)
  G_u       = unknown constant gains/losses (people, DHW tank, ground, party wall offsets)
The identical regression is run on EnergyPlus output (run folders given as arguments) so measured and model UA/C compare like for like.
UA here is the effective heat-loss coefficient seen by the above-grade sensors (conduction + infiltration at mean wind).

    python3 src/rc_fit.py [cal_run_dir fall_run_dir]   # optional: E+ run folders (eplusout.sql) from calib.run_case
Results -> runs/rc_fit_results.json. Findings (2026-11-23): claude.ai project doc benchmark-cpm-rc.md.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "model"))
from calib import CASES
from results import hourly
import build_model

C_DIR, W = ROOT / "data/clean", ROOT / "weather"
Q_FURN_KW = build_model.PARAMS["furnace_output_W"] / 1000
HOURS = [21, 22, 23, 0, 1, 2, 3, 4, 5]          # start hours; sun down Nov-Mar, after cooking/dryer peak
MONTHS = [11, 12, 1, 2, 3]
COLS = ["dT", "wdT", "dTdt", "one"]


def epw_hourly(name):
    e = pd.read_csv(W / name, skiprows=8, header=None, usecols=[0, 1, 2, 3, 6, 21])
    e.columns = ["year", "month", "day", "hour", "t_oa", "wind"]
    e.index = pd.to_datetime(e[["year", "month", "day"]]) + pd.to_timedelta(e.hour - 1, unit="h")
    return e[["t_oa", "wind"]]


def frame(q_kw, e_kw, t_in, wx):
    d = pd.DataFrame({"q": q_kw, "e": e_kw, "t_in": t_in}).join(wx, how="inner")
    d["dTdt"] = (d.t_in.shift(-1) - d.t_in.shift(1)) / 2
    d["dT"] = d.t_in - d.t_oa
    d["wdT"] = d.wind * d.dT
    d["one"] = 1.0
    d["y"] = d.q + d.e
    d = d[d.index.hour.isin(HOURS) & d.index.month.isin(MONTHS)]
    return d.dropna()


def measured_frame(case):
    s0, s1 = CASES[case]["span"]
    e = pd.read_csv(C_DIR / "ecobee_5min.csv", index_col=0, parse_dates=True).loc[s0:s1]
    h = e[["heat_s", "t_main", "t_up"]].resample("h").agg({"heat_s": "sum", "t_main": "mean", "t_up": "mean"})
    n = e.t_main.resample("h").count()
    h = h[n >= 11]
    kwh = pd.read_csv(C_DIR / "hydro_hourly.csv", index_col=0, parse_dates=True).kwh
    q = (h.heat_s / 3600).clip(upper=1) * Q_FURN_KW
    return frame(q, kwh, (h.t_main + h.t_up) / 2, epw_hourly(CASES[case]["epw"]))


def model_frame(case, run_dir):
    s0, s1 = CASES[case]["span"]
    d = hourly(Path(run_dir) / "eplusout.sql", ["Heating Coil Runtime Fraction", "Zone Mean Air Temperature",
                                                 "Electricity:Facility"]).loc[s0:s1]
    q = d.filter(like="Heating Coil Runtime").iloc[:, 0] * Q_FURN_KW
    t_in = (d["Zone Mean Air Temperature[FLOOR1]"] + d["Zone Mean Air Temperature[FLOOR2]"]) / 2
    return frame(q, d["Electricity:Facility"] / 3.6e6, t_in, epw_hourly(CASES[case]["epw"]))


def ols(d):
    X, y = d[COLS].values, d.y.values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, float(1 - r.var() / y.var())


def fit(d, n_boot=500, seed=0):
    """OLS + bootstrap over nights (resampling whole nights keeps the within-night correlation)."""
    b, r2 = ols(d)
    night = (d.index - pd.Timedelta(hours=12)).date
    nights = pd.unique(night)
    groups = {k: g for k, g in d.groupby(night)}
    rng = np.random.default_rng(seed)
    bs = np.array([ols(pd.concat([groups[k] for k in rng.choice(nights, len(nights))]))[0] for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [5, 95], axis=0)
    wbar = d.wind.mean()
    ua_eff = b[0] + b[1] * wbar
    ua_bs = bs[:, 0] + bs[:, 1] * wbar
    return dict(n_hours=len(d), n_nights=len(nights), r2=round(r2, 3), mean_wind=round(float(wbar), 2),
                mean_dT=round(float(d.dT.mean()), 1),
                UA_W_K=round(1000 * ua_eff, 1), UA_90=[round(1000 * x, 1) for x in np.percentile(ua_bs, [5, 95])],
                UA0_W_K=round(1000 * b[0], 1), Kw_W_K_per_ms=round(1000 * b[1], 2),
                C_MJ_K=round(b[2] * 3.6, 1), C_90=[round(3.6 * lo[2], 1), round(3.6 * hi[2], 1)],
                tau_h=round(b[2] / ua_eff, 1), Gu_W=round(-1000 * b[3], 0), Gu_90=[round(-1000 * hi[3]), round(-1000 * lo[3])])


if __name__ == "__main__":
    runs = dict(zip(["cal", "fall"], sys.argv[1:3]))
    out = {}
    for case in CASES:
        out[f"measured {case}"] = fit(measured_frame(case))
        if case in runs:
            out[f"E+ {Path(runs[case]).name} {case}"] = fit(model_frame(case, runs[case]))
    for k, v in out.items():
        print(f"{k:18s}", {kk: v[kk] for kk in ["n_nights", "r2", "UA_W_K", "UA_90", "Kw_W_K_per_ms", "C_MJ_K", "C_90", "tau_h", "Gu_W"]})
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/rc_fit_results.json").write_text(json.dumps(out, indent=1))
