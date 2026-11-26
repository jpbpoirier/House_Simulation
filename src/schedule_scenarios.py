"""Compare thermostat programs over a typical heating season (CWEC 2020, Oct-Apr), model = current PARAMS.

Comfort columns: home_hours_short = hours of the current program's Home periods with the main floor more
than 0.5 K below that scenario's setpoint (mostly recovery after setbacks; the model has no ecobee Smart
Recovery, so real recoveries start earlier and save slightly less). Cost = heating gas x marginal all-in price.

Each scenario changes forecast.COMFORT / PROGRAM / AWAY_HOME_FRAC, rebuilds a full-year schedule with the
forecast's schedule logic (plug loads = latest 8 weeks, damper closed) and runs the CWEC year.
    python3 src/schedule_scenarios.py      -> runs/scenarios/summary.csv (+ printed table)
"""
import sys, json, copy, shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "model"))
import forecast, build_model, tariffs
from run_ep import run
from results import hourly

C, W, OUT = ROOT / "data/clean", ROOT / "weather", ROOT / "runs/scenarios"
HV_J = 37.8e6
BASE_COMFORT = copy.deepcopy(forecast.COMFORT)
BASE_PROGRAM = copy.deepcopy(forecast.PROGRAM)
WD, WE = BASE_PROGRAM["weekday"], BASE_PROGRAM["weekend"]

SCEN = {
    "current": {},
    "previous (to 2026-11-25)": {"comfort": {"Away": (20.0, 23.0)}, "eco": 0.45,
                                 "program": {k: v for k, v in BASE_PROGRAM.items() if k != "wednesday"}},
    "previous, eco 0.03": {"comfort": {"Away": (20.0, 23.0)}, "eco": 0.03,
                          "program": {k: v for k, v in BASE_PROGRAM.items() if k != "wednesday"}},
    "away 18": {"comfort": {"Away": (18.0, 23.0)}},
    "sleep 17": {"comfort": {"Sleep": (17.0, 21.0)}},
    "home 20.5": {"comfort": {"Home": (20.5, 22.0), "Weekend Home": (20.5, 21.0)}},
    "away 18 + sleep 17": {"comfort": {"Away": (18.0, 23.0), "Sleep": (17.0, 21.0)}},
    "sleep from 21:00": {"program": {"weekday": [("00:00", "Sleep"), ("06:30", "Home"), ("08:00", "Away"),
                                                 ("15:30", "Home"), ("21:00", "Sleep")],
                                     "weekend": [("00:00", "Sleep"), ("08:00", "Weekend Home"), ("21:00", "Sleep")]}},
    "no setback (21 all day)": {"comfort": {"Away": (21.0, 23.0), "Sleep": (21.0, 22.0)}, "eco": 0.0},
}


def sched_for(sc, name):
    forecast.COMFORT = {**BASE_COMFORT, **sc.get("comfort", {})}
    forecast.PROGRAM = sc.get("program", BASE_PROGRAM)
    forecast.AWAY_HOME_FRAC = sc.get("eco", 0.45)
    forecast.forecast_schedules(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))
    import re
    fn = "sched_scen_" + re.sub(r"[^A-Za-z0-9.]+", "_", name).strip("_") + ".csv"   # E+ Schedule:File-safe name
    shutil.move(C / "sched_forecast.csv", C / fn)
    return fn


def heating_mask(idx):
    return idx.month.isin([10, 11, 12, 1, 2, 3, 4])


def evaluate(name, sc):
    fn = sched_for(sc, name)
    od = OUT / fn[:-4]; od.mkdir(parents=True, exist_ok=True)
    m = build_model.build("cwec", p=build_model.PARAMS, sched=fn)
    (od / "house.epJSON").write_text(json.dumps(m))
    assert run(od / "house.epJSON", W / "Ottawa_CWEC_2020.epw", od) == 0, name
    d = hourly(od / "eplusout.sql", ["NaturalGas:Facility", "Heating:NaturalGas", "Electricity:Facility",
                                     "Zone Mean Air Temperature"])
    s = pd.read_csv(C / fn, index_col=0)
    s.index = d.index[:len(s)] if len(d) == len(s) else d.index
    hm = heating_mask(d.index)
    heat_m3 = d["Heating:NaturalGas"][hm].sum() / HV_J
    kwh = d["Electricity:Facility"][hm].sum() / 3.6e6
    t1, t2 = d["Zone Mean Air Temperature[FLOOR1]"], d["Zone Mean Air Temperature[FLOOR2]"]
    sp = s.sp_heat.values[:len(d)]
    home = hm & HOME_MASK                         # the current program's Home / Weekend Home hours
    occ_hours = home.sum()
    short = (t1[home] < sp[home] - 0.5).sum()     # main floor > 0.5 K below this scenario's setpoint
    night = hm & (d.index.hour >= 0) & (d.index.hour < 6)
    out = dict(scenario=name, heat_gas_m3=heat_m3, heat_cost=heat_m3 * tariffs.gas_marginal_cents(150) / 100,
               elec_kwh=kwh, home_hours_short=int(short), home_hours=int(occ_hours),
               t_main_home=float(t1[home].mean()), t_up_night_min=float(t2[night].min()),
               t_up_night_mean=float(t2[night].mean()))
    for f in od.iterdir():
        if f.suffix != ".sql":
            f.unlink()
    (C / fn).unlink()
    return out


HOME_MASK = None

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    keep = (C / "sched_forecast.csv").read_bytes()      # forecast_schedules overwrites it; restore at the end
    fn0 = sched_for({}, "mask")
    HOME_MASK = pd.read_csv(C / fn0, index_col=0).sp_heat.values >= 20.9
    (C / fn0).unlink()
    only = sys.argv[1:]
    res = [evaluate(n, sc) for n, sc in SCEN.items() if not only or n in only]
    df = pd.DataFrame(res).set_index("scenario")
    base = df.loc["current"]
    df["d_gas_m3"] = df.heat_gas_m3 - base.heat_gas_m3
    df["d_gas_pct"] = 100 * df.d_gas_m3 / base.heat_gas_m3
    df["d_cost"] = df.heat_cost - base.heat_cost
    (C / "sched_forecast.csv").write_bytes(keep)
    df.round(2).to_csv(OUT / "summary.csv")
    print(df.round(2).to_string())
