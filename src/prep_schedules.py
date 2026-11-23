"""Hourly schedules (local standard time) for EnergyPlus Schedule:File, from ecobee + Hydro Ottawa.
Output data/clean/sched_8760.csv: 8760 rows indexed by hour-of-year (Jan 1 00:00 ... Dec 31 23:00, non-leap).
The calibration period (2025-08-22 .. 2026-08-01) never repeats a calendar day, so Aug-Dec come from 2025
and Jan-Jul from 2026. Days with no data (Aug 1-21, Aug 2-31 2026 side) are filled from the same
hour of the nearest available week.
Columns (Schedule:File 2-10): sp_heat, sp_cool (degC; mode-adjusted), gains_W (non-HVAC, non-dryer electricity),
occ (0-1), w_up (ecobee upstairs weight), dhw (fraction of peak DHW flow), dryer_W (detected dryer electricity),
dryer_on (0-1, dryer exhaust fan), damper (1 = fireplace damper assumed open: Oct-Apr, closed from 2026-11-22),
fan_on (column 11: 1 = furnace blower in continuous 'fan on' mode, i.e. ecobee fan-only > 50% of the hour).
"""
from pathlib import Path
import argparse, sys
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--aug-dec-year", type=int, default=2025, help="year supplying Aug-Dec (Jan-Jul come from 2026)")
ap.add_argument("--out", default="sched_8760.csv")
A = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "data/clean"
sys.path.insert(0, str(ROOT / "src"))
from loads import split_loads
DAMPER_CLOSED_FROM = pd.Timestamp("2026-11-22")   # J-P closed the fireplace damper
HEAT_OFF, COOL_OFF = 10.0, 35.0

e = pd.read_csv(C / "ecobee_5min.csv", index_col=0, parse_dates=True)
hh = pd.read_csv(C / "hydro_hourly.csv", index_col=0, parse_dates=True)

sp_h, sp_c = e.sp_heat.copy(), e.sp_cool.copy()
sp_h[e.setting.isin(["cool", "off"])] = HEAT_OFF
sp_c[e.setting.isin(["heat", "off"])] = COOL_OFF
L = split_loads(e, hh.kwh)
H = pd.DataFrame({"sp_heat": sp_h.resample("h").mean(), "sp_cool": sp_c.resample("h").mean()})
H = H.join(L[["gains_W", "dryer_W", "dryer_on"]], how="left")
H["fan_on"] = (L.fanonly_h > 0.5).astype(float).reindex(H.index)
vac = e.event.eq("Vacation").resample("h").mean() > 0.5
H["occ"] = np.where(vac.reindex(H.index, fill_value=False), 0.0, np.nan)

# canonical non-leap calendar year
yr = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
def src_ts(t):  # Aug-Dec from A.aug_dec_year, Jan-Jul from 2026
    return t.replace(year=A.aug_dec_year) if t.month >= 8 else t.replace(year=2026)
COLS = ["sp_heat", "sp_cool", "gains_W", "dryer_W", "dryer_on", "fan_on"]
out = pd.DataFrame(index=yr, columns=COLS, dtype=float)
src = [src_ts(t) for t in yr]
out[:] = H.reindex(src)[COLS].values
# fill gaps with same hour-of-week from nearest data (forward then backward by 7 days)
for _ in range(6):
    m = out.isna().any(axis=1)
    if not m.any(): break
    out = out.fillna(out.shift(168)).fillna(out.shift(-168))
out = out.ffill().bfill()
# occupancy: 2 people, simple profile (home nights/evenings, ~half daytime), zero on vacation
hr = out.index.hour
occ = np.where((hr >= 22) | (hr < 7), 1.0, np.where((hr >= 17), 0.9, 0.5))
vac_c = pd.Series([bool(vac.get(s, False)) for s in src], index=yr)
out["occ"] = np.where(vac_c, 0.0, occ)
# ecobee control temperature = (1-w)*main + w*upstairs. Fit w by hour of day (heating season,
# intervals where the sensors differ by >0.4 K): Sleep comfort setting uses upstairs only.
ee = e.dropna(subset=["t_avg", "t_main", "t_up"])
ww = ((ee.t_avg - ee.t_main) / (ee.t_up - ee.t_main)).where((ee.t_up - ee.t_main).abs() > 0.4).clip(0, 1)
ww = ww[ww.index.month.isin([10, 11, 12, 1, 2, 3, 4])]
w_hod = ww.groupby(ww.index.hour).median().reindex(range(24)).fillna(0.5)
out["w_up"] = w_hod.reindex(out.index.hour).values
print("upstairs weight by hour:", w_hod.round(2).to_dict())
# hot-water draw: fixed hourly profile x (daily household water from the City of Ottawa meter / mean).
# Column 'dhw' is the fraction of the model's peak DHW flow; the model sets the peak flow so that
# mean hot water = dhw_hot_fraction x mean household water.
DHW_PROFILE = {7: .06, 8: .03, 12: .015, 18: .04, 19: .04, 21: .017}
wr = pd.read_csv(C / "water_reads.csv", parse_dates=["start", "read_date"])
w_mean = 1000 * wr.usage_m3.sum() / wr.days.sum()
def water_scale(t):
    m = wr[(wr.start <= t) & (t < wr.read_date)]
    return float(m.L_per_day.iloc[-1] / w_mean) if len(m) else 1.0
day_scale = {d: water_scale(pd.Timestamp(d)) for d in sorted(set(s_.normalize() for s_ in src))}
out["dhw"] = [DHW_PROFILE.get(t.hour, 0.0) * day_scale[s_.normalize()] for t, s_ in zip(out.index, src)]
print(f"water: mean {w_mean:.0f} L/day; daily scale range {min(day_scale.values()):.2f}-{max(day_scale.values()):.2f}")
out["damper"] = [1.0 if (s_.month in (10, 11, 12, 1, 2, 3, 4) and s_ < DAMPER_CLOSED_FROM) else 0.0 for s_ in src]
out = out[["sp_heat", "sp_cool", "gains_W", "occ", "w_up", "dhw", "dryer_W", "dryer_on", "damper", "fan_on"]]
print(f"fan on (continuous): {int(out.fan_on.sum())} h/yr")
print(f"dryer: {(out.dryer_W > 0).sum()} h/yr, {out.dryer_W.sum() / 1000:.0f} kWh/yr")
out.round(4).to_csv(C / A.out, index_label="hour_lst")
print(out.describe().round(2).to_string())
print("calibration-period gains mean W:", round(H.gains_W.loc["2025-08-22":"2026-08-01"].mean(), 0))
