"""Fireplace-damper check: model vs measured before and after the damper was closed (2026-11-22 ~21:00).

Weather: Open-Meteo GEM past-31-day series (data/raw/weather/openmeteo_gem_past31_<date>.json, fetched on the
Mac) for Oct 29 - Nov 27 2026, stitched over the observed 2026 EPW. Schedules: actual ecobee setpoints and
Hydro Ottawa plug loads (sched_fall2026.csv). Runs Oct 15 - Nov 27 2026 twice: damper closed from Nov 22 21:00
(as J-P reported) and damper left open; compares daily furnace runtime and gas with ecobee and the meter.
    python3 src/damper_check.py data/raw/weather/openmeteo_gem_past31_2026-11-28.json
"""
import sys, json, copy
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "model"))
import build_model
from make_epw import load_openmeteo, write_epw
from stitch_epw import stitch
from run_ep import run
from results import hourly

C, W, OUT = ROOT / "data/clean", ROOT / "weather", ROOT / "runs/damper_check"
HV_J = 37.8e6
M3_PER_H = 39000 * 1055.06 / HV_J            # low-fire input, m3/h
CLOSE = pd.Timestamp("2026-11-22 21:00")      # damper closed (local time ~ EST in November)
RUN = ("2026-10-15", "2026-11-27")


def weather(js):
    d = json.loads(Path(js).read_text())
    om = pd.DataFrame(d["hourly"]); om["time"] = pd.to_datetime(om.time)
    csv = W / "openmeteo_gem_recent_2026-10-28_2026-11-28.csv"
    om.to_csv(csv, index=False)
    epw = W / "Kanata_gem_2026-10-29_2026-11-27.epw"
    write_epw(load_openmeteo(csv), "2026-10-29", "2026-11-27", epw, "Open-Meteo GEM past days (analysis/short-lead)")
    out = OUT / "run_weather.epw"
    origin = stitch("2025-11-29", out, [str(epw), "Kanata_2026_obs.epw", "Kanata_actual_2025-08_2026-07.epw"])
    return out, origin


def sched(open_after):
    s = pd.read_csv(C / "sched_fall2026.csv", index_col=0, parse_dates=True)
    lst = s.index.map(lambda t: t.replace(year=2026) if t.month >= 8 else t)   # rows are hours of 2026 Aug-Dec
    s.loc[(lst >= "2026-10-01") & (lst < CLOSE), "damper"] = 1.0
    s.loc[lst >= CLOSE, "damper"] = 1.0 if open_after else 0.0
    fn = f"sched_damper_{'open' if open_after else 'closed'}.csv"
    s.to_csv(C / fn)
    return fn


def simulate(tag, open_after, epw):
    fn = sched(open_after)
    od = OUT / tag; od.mkdir(parents=True, exist_ok=True)
    m = build_model.build("custom", p=build_model.PARAMS, period=RUN, sched=fn)
    (od / "house.epJSON").write_text(json.dumps(m))
    assert run(od / "house.epJSON", epw, od) == 0, tag
    (C / fn).unlink()
    return hourly(od / "eplusout.sql", ["Heating Coil Runtime Fraction", "NaturalGas:Facility", "Heating:NaturalGas",
                                         "Zone Mean Air Temperature"])


def interval_sum(series_h, a, b):
    """Sum an hourly series (hour-start index) over [a, b) with partial first/last hours."""
    s = series_h
    t0, t1 = pd.Timestamp(a), pd.Timestamp(b)
    full = s[(s.index >= t0.ceil("h")) & (s.index < t1.floor("h"))].sum()
    first = s.get(t0.floor("h"), 0) * ((t0.ceil("h") - t0) / pd.Timedelta("1h")) if t0 != t0.ceil("h") else 0
    last = s.get(t1.floor("h"), 0) * ((t1 - t1.floor("h")) / pd.Timedelta("1h"))
    return float(full + first + last)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    epw, origin = weather(sys.argv[1])
    print("weather hours by source:", origin)
    e = pd.read_csv(C / "ecobee_5min.csv", index_col=0, parse_dates=True)
    meas_h = (e.heat_s.fillna(0).resample("h").sum() / 3600).loc["2026-10-15":"2026-11-27"]
    ok = e.t_main.resample("D").count() >= 250
    res = {}
    for tag, op in [("closed", False), ("open", True)]:
        d = simulate(tag, op, epw)
        res[tag] = dict(rt=d.filter(like="Heating Coil Runtime").iloc[:, 0],
                        gas=d["NaturalGas:Facility"] / HV_J, heat=d["Heating:NaturalGas"] / HV_J)
    rows = []
    for name, a, b in [("before: Nov 1-21 (damper open)", "2026-11-01", "2026-11-22"),
                       ("after: Nov 23-27 (closed)", "2026-11-23", "2026-11-28")]:
        days = [dd for dd in pd.date_range(a, b, freq="D", inclusive="left") if ok.get(dd, False)]
        mask = meas_h.index.normalize().isin(days)
        r = dict(period=name, days=len(days), meas_h=meas_h[mask].sum())
        for tag in res:
            r[f"model_{tag}_h"] = res[tag]["rt"][res[tag]["rt"].index.normalize().isin(days)].sum()
        rows.append(r)
    df = pd.DataFrame(rows)
    for tag in res:
        df[f"err_{tag}_%"] = 100 * (df[f"model_{tag}_h"] / df.meas_h - 1)
    print(df.round(1).to_string(index=False))
    # meter: first manual read to the last read inside the run
    g = pd.read_csv(ROOT / "data/manual/gas_meter_manual.csv", parse_dates=["datetime_local"])
    g = g[g.datetime_local <= "2026-11-28 00:00"]
    a, b = g.iloc[0], g.iloc[-1]
    meter = b.reading_m3 - a.reading_m3
    run_h = interval_sum(meas_h, a.datetime_local, b.datetime_local)
    print(f"\nmeter {a.datetime_local} -> {b.datetime_local}: {meter:.0f} m3 (whole-m3 reads: {meter - 1:.0f}-{meter + 1:.0f})")
    print(f"ecobee runtime same span: {run_h:.1f} h -> {run_h * M3_PER_H:.1f} m3 furnace at low fire")
    for tag in res:
        print(f"model damper {tag}: total gas {interval_sum(res[tag]['gas'], a.datetime_local, b.datetime_local):.1f} m3, "
              f"furnace {interval_sum(res[tag]['heat'], a.datetime_local, b.datetime_local):.1f} m3, "
              f"runtime {interval_sum(res[tag]['rt'], a.datetime_local, b.datetime_local):.1f} h")
    df.to_csv(OUT / "summary.csv", index=False)
