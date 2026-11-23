"""Compare a model run (actual-year weather) with measured data. Writes runs/<run>/compare.json.
    python3 src/compare.py runs/v0_actual"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from results import hourly

ROOT = Path(__file__).resolve().parents[1]
HV_J = 37.8e6            # J per m3 natural gas (HHV, assumed; confirm Enbridge heat value)

def metrics(meas, mod):
    d = pd.concat([meas, mod], axis=1, sort=True).dropna(); m, s = d.iloc[:, 0], d.iloc[:, 1]
    n = len(d)
    return dict(n=n, nmbe=100 * (m - s).sum() / ((n - 1) * m.mean()),
                cvrmse=100 * np.sqrt(((m - s) ** 2).sum() / (n - 1)) / m.mean())

def compare(run):
    run = Path(run)
    d = hourly(run / "eplusout.sql", ["NaturalGas:Facility", "Heating:NaturalGas", "WaterSystems:NaturalGas",
                                       "Electricity:Facility", "Cooling:Electricity", "Fans:Electricity",
                                       "Zone Mean Air Temperature", "Site Outdoor Air Drybulb Temperature"])
    C = ROOT / "data/clean"
    hh = pd.read_csv(C / "hydro_hourly.csv", index_col=0, parse_dates=True).kwh
    e = pd.read_csv(C / "ecobee_5min.csv", index_col=0, parse_dates=True)
    g = pd.read_csv(C / "enbridge_reads.csv", parse_dates=["billed_to"])
    out = {}
    # --- electricity
    em = d["Electricity:Facility"] / 3.6e6
    out["elec_hourly"] = metrics(hh, em)
    out["elec_monthly"] = metrics(hh.resample("MS").sum().loc["2025-09":"2026-07"], em.resample("MS").sum().loc["2025-09":"2026-07"])
    # --- gas between actual meter reads within the run
    a = g[g["Read Type"] == "A"].reset_index(drop=True)
    periods = []
    gm = d["NaturalGas:Facility"] / HV_J
    for i in range(1, len(a)):
        t0, t1 = a.billed_to[i - 1] + pd.Timedelta(days=1), a.billed_to[i] + pd.Timedelta(days=1)
        if t0 < d.index.min() or t1 > d.index.max() + pd.Timedelta(hours=1):
            continue
        periods.append(dict(start=str(t0.date()), end=str(t1.date()), measured_m3=float(a["Meter Read"][i] - a["Meter Read"][i - 1]),
                            model_m3=round(float(gm.loc[t0:t1 - pd.Timedelta(hours=1)].sum()), 1)))
    out["gas_periods"] = periods
    # --- indoor temperature (ecobee main = Floor1, upstairs = Floor2)
    eh = e[["t_main", "t_up", "t_avg"]].resample("h").mean()
    for meas, zone in [("t_main", "FLOOR1"), ("t_up", "FLOOR2")]:
        mz = d[f"Zone Mean Air Temperature[{zone}]"]
        diff = (mz - eh[meas]).dropna()
        out[f"temp_{zone}"] = dict(bias=float(diff.mean()), rmse=float(np.sqrt((diff ** 2).mean())))
    # --- monthly table for the dashboard
    mon = pd.DataFrame({
        "elec_meas_kwh": hh.resample("MS").sum(), "elec_model_kwh": em.resample("MS").sum(),
        "gas_model_m3": gm.resample("MS").sum(), "heat_gas_model_m3": (d["Heating:NaturalGas"] / HV_J).resample("MS").sum(),
        "dhw_gas_model_m3": (d["WaterSystems:NaturalGas"] / HV_J).resample("MS").sum(),
        "t_out_mean": d.filter(like="Site Outdoor Air Drybulb").iloc[:, 0].resample("MS").mean(),
    }).loc["2025-08":"2026-07"]
    out["monthly"] = mon.round(1).reset_index(names="month").astype({"month": str}).to_dict("records")
    (run / "compare.json").write_text(json.dumps(out, indent=1, default=float))
    return out, d

if __name__ == "__main__":
    out, _ = compare(sys.argv[1])
    print(json.dumps({k: v for k, v in out.items() if k != "monthly"}, indent=1, default=lambda x: round(x, 2)))
