"""Daily energy and cost forecast for 3 DeCarlo Gardens (runs on the Mac, ~10 s).

    python3 src/forecast.py                     # live: Open-Meteo GEM forecast
    python3 src/forecast.py --offline-test 2026-11-10   # test: pretend it's that day, use archived weather

Steps
 1. Weather: Open-Meteo forecast API (Canadian GEM 'seamless': HRDPS 2.5 km -> RDPS -> GDPS), 31 past days
    + 11 forecast days, via the system curl. Converted to an EPW in local standard time.
 2. 365-day EPW ending on the last forecast day: forecast > observed archive EPWs > Ottawa CWEC.
 3. Schedules: current ecobee program (see thermostat-schedule.md), Away heating raised by the eco+
    Smart Home override seen last winter; non-HVAC electricity = mean by weekday/weekend and hour of
    the latest 8 weeks of Hydro Ottawa data (mean); hot water at the mean metered household use.
 4. EnergyPlus run: 14 spin-up days + HORIZON forecast days.
 5. dashboard/forecast.json: one row per forecast day with weather, gas, electricity and all-in cost
    (src/tariffs.py); a dated copy goes to dashboard/forecast_archive/ for forecast-vs-actual later.
"""
import argparse, datetime as dt, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "model"))
import build_model, tariffs
from loads import split_loads
from make_epw import epw_rows, write_epw, load_openmeteo
from stitch_epw import stitch
from run_ep import run
from results import hourly

LAT, LON = 45.31, -75.90
HORIZON, SPINUP, PAST = 10, 14, 31
HV_J = 37.8e6
VARS = ("temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,wind_speed_10m,wind_direction_10m,"
        "cloud_cover,shortwave_radiation,direct_normal_irradiance,diffuse_radiation,snow_depth")
W, C, OUT = ROOT / "weather", ROOT / "data/clean", ROOT / "runs/forecast"
OBSERVED = ["Kanata_2026_obs.epw", "Kanata_actual_2025-08_2026-07.epw"]

# current ecobee program (local clock time). (start hh:mm, comfort) per day type; comfort -> (heat, cool) degC
COMFORT = {"Away": (20.0, 23.0), "Home": (21.0, 22.0), "Sleep": (18.0, 21.0), "Weekend Home": (21.0, 21.0)}
PROGRAM = {"weekday": [("00:00", "Sleep"), ("06:30", "Home"), ("08:00", "Away"), ("15:30", "Home"), ("19:00", "Sleep")],
           "weekend": [("00:00", "Sleep"), ("08:00", "Weekend Home"), ("19:00", "Sleep")]}
AWAY_HEAT_ECO = 0.45       # eco+ Smart Home kept Away at 21 instead of 20 ~45% of the time (calibration year)


def curl_json(url):
    r = subprocess.run(["curl", "-sSfL", "--max-time", "120", "-A", "house-sim/0.1", url], capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode().strip())
    return json.loads(r.stdout)


def fetch_forecast(today, offline=None):
    if offline:   # archived ERA5 stands in for the forecast
        om = pd.concat([pd.read_csv(W / f, parse_dates=["time"]) for f in sorted(W.glob("openmeteo_archive_*.csv"))])
        om = om.drop_duplicates("time").set_index("time").sort_index()
        om = om.loc[str(today - dt.timedelta(PAST + 1)):str(today + dt.timedelta(HORIZON + 1))].reset_index()
        source = "OFFLINE TEST: ERA5 archive used as forecast"
    else:
        last_err = None
        import time as _t
        for attempt, model in enumerate(["gem_seamless", "gem_seamless", "best_match", "gem_seamless"]):
            if attempt:
                _t.sleep(20 * attempt)      # Open-Meteo returns HTTP 429 when busy; back off and retry
            try:
                d = curl_json(f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={VARS}"
                              f"&models={model}&past_days={PAST}&forecast_days={HORIZON + 1}&timezone=GMT&wind_speed_unit=ms")
                om = pd.DataFrame(d["hourly"]); om["time"] = pd.to_datetime(om.time)
                source = f"Open-Meteo forecast API, model {model}"
                break
            except Exception as e:
                last_err = e
        else:
            raise RuntimeError(f"forecast fetch failed: {last_err}")
    om.to_csv(OUT / "forecast_weather_raw.csv", index=False)
    om = om.set_index("time")
    num = om.columns
    om[num] = om[num].apply(pd.to_numeric, errors="coerce").interpolate(limit=6).ffill().bfill()
    om["snow_depth"] = om.get("snow_depth", 0).fillna(0)
    om = om.reset_index()
    om["lst"] = om.time - pd.Timedelta(hours=5)
    return om.set_index("lst"), source


def program_setpoints(ts_lst):
    """Heating/cooling setpoint for the hour starting at ts_lst (EST), averaged over the hour (half-hour edges)."""
    loc = pd.Timestamp(ts_lst).tz_localize("Etc/GMT+5").tz_convert("America/Toronto").tz_localize(None)
    prog = PROGRAM["weekend" if loc.weekday() >= 5 else "weekday"]
    hs, cs = [], []
    for minute in (0, 30):
        m = loc.hour * 60 + minute
        name = [c for start, c in prog if int(start[:2]) * 60 + int(start[3:]) <= m][-1]
        h, c = COMFORT[name]
        if name == "Away":
            h += AWAY_HEAT_ECO
        hs.append(h); cs.append(c)
    heat, cool = np.mean(hs), np.mean(cs)
    mo = loc.month
    if mo in (10, 11, 12, 1, 2, 3, 4):
        cool = 35.0              # system in heat mode
    elif mo in (6, 7, 8):
        heat = 10.0              # system in cool mode
    return heat, cool


def typical_gains():
    """Mean plug gains, dryer W and dryer-on fraction by (weekend?, hour LST) over the latest 8 weeks (mean keeps energy)."""
    e = pd.read_csv(C / "ecobee_5min.csv", index_col=0, parse_dates=True)
    hh = pd.read_csv(C / "hydro_hourly.csv", index_col=0, parse_dates=True).kwh
    g = split_loads(e, hh)[["gains_W", "dryer_W", "dryer_on"]].dropna()
    g = g.loc[g.index.max() - pd.Timedelta(days=56):]
    return g.groupby([g.index.dayofweek >= 5, g.index.hour]).mean(), (g.index.min(), g.index.max())


def forecast_schedules(window_start, window_end):
    """8760 rows by hour-of-year for the run year(s); each day-of-year is mapped to its date in the window."""
    base = pd.read_csv(C / "sched_8760.csv", index_col=0, parse_dates=True)
    w_up = base.w_up.groupby(base.index.hour).median()
    gains, span = typical_gains()
    prof = {7: .06, 8: .03, 12: .015, 18: .04, 19: .04, 21: .017}
    days = pd.date_range(window_start, window_end, freq="D")
    by_md = {(d.month, d.day): d for d in days}
    rows = []
    for t in pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h"):
        d = by_md.get((t.month, t.day), pd.Timestamp(year=window_end.year, month=t.month, day=t.day))
        ts = d + pd.Timedelta(hours=t.hour)
        heat, cool = program_setpoints(ts)
        hr = ts.hour
        occ = 1.0 if (hr >= 22 or hr < 7) else (0.9 if hr >= 17 else 0.5)
        k = (ts.dayofweek >= 5, hr)
        gw, dw, don = gains.loc[k].values if k in gains.index else (400.0, 0.0, 0.0)
        # fireplace damper closed from 2026-11-22 (J-P) -> 0
        rows.append((t, heat, cool, gw, occ, w_up.get(hr, 0.5), prof.get(hr, 0.0), dw, don, 0.0))
    s = pd.DataFrame(rows, columns=["hour_lst", "sp_heat", "sp_cool", "gains_W", "occ", "w_up", "dhw",
                                    "dryer_W", "dryer_on", "damper"]).set_index("hour_lst")
    s.round(4).to_csv(C / "sched_forecast.csv")
    return span


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--offline-test", default=None)
    A = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    now = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=-5)))  # EST clock
    today = dt.date.fromisoformat(A.offline_test) if A.offline_test else now.date()
    first_fc, last_fc = today, today + dt.timedelta(HORIZON - 1)
    spin0 = today - dt.timedelta(SPINUP)

    # 1-2. weather
    om, source = fetch_forecast(today, A.offline_test)
    have_first = (om.index.min() + pd.Timedelta(hours=23)).normalize().date()
    fc_first = have_first          # use all fetched past days (more recent than the archive EPWs)
    fc_epw = OUT / "forecast_weather.epw"
    write_epw(om, fc_first, last_fc, fc_epw, source)
    win_start = last_fc - dt.timedelta(364)
    origin = stitch(win_start, OUT / "run_weather.epw", [str(fc_epw)] + OBSERVED)

    # 3. schedules
    span = forecast_schedules(pd.Timestamp(win_start), pd.Timestamp(last_fc))

    # 4. EnergyPlus
    model = build_model.build("custom", period=(spin0.isoformat(), last_fc.isoformat()), sched="sched_forecast.csv")
    (OUT / "house.epJSON").write_text(json.dumps(model))
    rc = run(OUT / "house.epJSON", OUT / "run_weather.epw", OUT)
    if rc != 0:
        raise SystemExit(f"EnergyPlus failed (exit {rc}); see {OUT / 'eplusout.err'}")

    # 5. results and costs
    d = hourly(OUT / "eplusout.sql", ["Electricity:Facility", "NaturalGas:Facility", "Heating:NaturalGas",
                                      "WaterSystems:NaturalGas", "Cooling:Electricity", "Fans:Electricity",
                                      "Site Outdoor Air Drybulb Temperature", "Zone Mean Air Temperature"])
    d = d.loc[str(first_fc):str(last_fc) + " 23:00"]
    kwh = d["Electricity:Facility"] / 3.6e6
    ec = tariffs.elec_cost(kwh)
    tout = d.filter(like="Site Outdoor").iloc[:, 0]
    gas = d["NaturalGas:Facility"] / HV_J
    daily = pd.DataFrame({
        "t_out_min_c": tout.resample("D").min(), "t_out_max_c": tout.resample("D").max(), "t_out_mean_c": tout.resample("D").mean(),
        "gas_m3": gas.resample("D").sum(), "heat_gas_m3": (d["Heating:NaturalGas"] / HV_J).resample("D").sum(),
        "dhw_gas_m3": (d["WaterSystems:NaturalGas"] / HV_J).resample("D").sum(),
        "elec_kwh": kwh.resample("D").sum(),
        "hvac_kwh": ((d["Cooling:Electricity"] + d["Fans:Electricity"]) / 3.6e6).resample("D").sum(),
        "cost_elec": ec.allin.resample("D").sum(),
        "t_main_c": d["Zone Mean Air Temperature[FLOOR1]"].resample("D").mean(),
    })
    month_m3 = daily.gas_m3.mean() * 30.4
    gas_rate = tariffs.gas_marginal_cents(month_m3) / 100
    daily["cost_gas"] = daily.gas_m3 * gas_rate + tariffs.GAS["customer_month"] * (1 + tariffs.HST) / 30.4
    daily["cost_total"] = daily.cost_gas + daily.cost_elec
    daily = daily.round(2)
    daily.index = daily.index.strftime("%Y-%m-%d"); daily.index.name = "date"
    issued = now.strftime("%Y-%m-%d %H:%M") + " EST" if not A.offline_test else f"{today} (offline test)"
    daily.insert(0, "issued", issued)
    rows = json.loads(daily.reset_index().to_json(orient="records"))
    # hourly detail (for the dashboard's "Today" block): hour-start in local standard time (EST)
    hgas = d["Heating:NaturalGas"] / HV_J
    hr = pd.DataFrame({
        "date": d.index.strftime("%Y-%m-%d"), "hour": d.index.hour, "t_out_c": tout.round(1),
        "gas_m3": gas.round(3), "heat_gas_m3": hgas.round(3),
        "furnace_min": (60 * hgas / (39000 * 1055.06 / HV_J)).clip(upper=60).round(0),   # low-fire 39 kBtu/h input
        "elec_kwh": kwh.round(3), "tier": ec.tier.values,
        "cost_elec": ec.allin.round(3), "cost_gas": (gas * gas_rate).round(3),
        "t_main_c": d["Zone Mean Air Temperature[FLOOR1]"].round(2)})
    hr.insert(0, "issued", issued)
    hrows = json.loads(hr.to_json(orient="records"))
    meta = [{"label": "forecast.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"},
            {"label": "Issued", "value": issued},
            {"label": "Weather", "value": source},
            {"label": "Weather file hours", "value": json.dumps(origin)},
            {"label": "Thermostat", "value": "current ecobee program; Away heat +%.2f K for eco+ Smart Home" % AWAY_HEAT_ECO},
            {"label": "Plug loads", "value": f"mean by weekday/weekend and hour, {span[0]:%Y-%m-%d} to {span[1]:%Y-%m-%d}"},
            {"label": "Prices", "value": "all-in: src/tariffs.py (Hydro Ottawa TOU from Nov 1 2026, Enbridge Rate 1 from Oct 1 2026); "
                                          f"gas marginal {100 * gas_rate:.1f} c/m3 incl. HST"},
            {"label": "Model", "value": "EnergyPlus 26.2, model v0.4.1 (calibrated 2026-11-23)"}]
    dash_dir = ROOT / "dashboard"; (dash_dir / "forecast_archive").mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"rows": rows, "meta": meta})
    (dash_dir / "forecast.json").write_text(payload)
    (dash_dir / "forecast_archive" / f"forecast_{today}.json").write_text(payload)
    (dash_dir / "forecast_hourly.json").write_text(json.dumps({"rows": hrows, "meta": [
        {"label": "Issued", "value": issued},
        {"label": "Hours", "value": "hour-start, local standard time (EST); gas cost is marginal (fixed charges in the daily totals only)"},
        {"label": "Furnace minutes", "value": "heating gas / low-fire input 39 kBtu/h"}]}))
    msg = (f"{issued}: next {HORIZON} days gas {daily.gas_m3.sum():.0f} m3, elec {daily.elec_kwh.sum():.0f} kWh, "
           f"cost ${daily.cost_total.sum():.2f} ({time.time() - t0:.0f} s)")
    with open(dash_dir / "forecast_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        (ROOT / "dashboard").mkdir(parents=True, exist_ok=True)
        with open(ROOT / "dashboard" / "forecast_log.txt", "a") as f:
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M} FAILED: {e}\n")
        raise
