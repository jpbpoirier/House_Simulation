"""Export model-vs-measured results for the dashboard as JSON {rows, meta}.
    python3 src/export_dashboard.py runs/v01_actual   -> dashboard/{daily,gas_periods,monthly}.json"""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1]); out = ROOT / "dashboard"; out.mkdir(exist_ok=True)
code = Path(__file__).read_text()
cmp = json.loads((run / "compare.json").read_text())
meta = [{"label": "export_dashboard.py", "value": code, "type": "code", "lang": "python"},
        {"label": "Model run", "value": run.name},
        {"label": "Weather", "value": "ERA5 (Open-Meteo archive) at 45.31,-75.90, LST"},
        {"label": "Gas heating value", "value": "37.8 MJ/m3 (assumed)"}]

from compare import compare
_, d = compare(run)
C = ROOT / "data/clean"; HV = 37.8e6
hh = pd.read_csv(C / "hydro_hourly.csv", index_col=0, parse_dates=True).kwh
e = pd.read_csv(C / "ecobee_5min.csv", index_col=0, parse_dates=True)
daily = pd.DataFrame({
    "elec_meas_kwh": hh.resample("D").sum(min_count=20),
    "elec_model_kwh": (d["Electricity:Facility"] / 3.6e6).resample("D").sum(),
    "gas_model_m3": (d["NaturalGas:Facility"] / HV).resample("D").sum(),
    "heat_gas_model_m3": (d["Heating:NaturalGas"] / HV).resample("D").sum(),
    "furnace_runtime_h": e.heat_s.resample("D").sum() / 3600,
    "t_out_c": d.filter(like="Site Outdoor").iloc[:, 0].resample("D").mean(),
    "t_main_meas_c": e.t_main.resample("D").mean(), "t_main_model_c": d["Zone Mean Air Temperature[FLOOR1]"].resample("D").mean(),
    "t_up_meas_c": e.t_up.resample("D").mean(), "t_up_model_c": d["Zone Mean Air Temperature[FLOOR2]"].resample("D").mean(),
}).loc["2025-08-22":"2026-07-31"]
daily["gas_est_meas_m3"] = daily.furnace_runtime_h * 39000 * 1055.06 / HV + 0.75
daily = daily.round(2); daily.index.name = "date"; daily = daily.reset_index()
daily["date"] = daily.date.dt.strftime("%Y-%m-%d"); daily.to_csv(run / "daily.csv", index=False)
(out / "daily.json").write_text(json.dumps({"rows": json.loads(daily.to_json(orient="records")), "meta": meta + [
    {"label": "Measured-gas estimate", "value": "furnace runtime (ecobee) x 39,000 Btu/h low-fire input + 0.75 m3/day water-heater baseline; fitted to Enbridge actual meter reads"}]}))
gp = pd.DataFrame(cmp["gas_periods"])
gp["error_pct"] = (100 * (gp.model_m3 - gp.measured_m3) / gp.measured_m3).round(1)
gp["label"] = pd.to_datetime(gp.start).dt.strftime("%b %d, %Y") + " – " + pd.to_datetime(gp.end).dt.strftime("%b %d, %Y")
(out / "gas_periods.json").write_text(json.dumps({"rows": json.loads(gp.to_json(orient="records")), "meta": meta + [
    {"label": "Measured", "value": "Enbridge meter-read differences between actual (A) reads"}]}))
mon = pd.DataFrame(cmp["monthly"])
(out / "monthly.json").write_text(json.dumps({"rows": json.loads(mon.to_json(orient="records")), "meta": meta}))
print("wrote", [p.name for p in out.glob("*.json")])
