"""Phase 1 ingest: ecobee (5-min), Hydro Ottawa (daily), Enbridge (billing reads) -> data/clean.
Run from project root:  python3 src/ingest.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN = ROOT / "data/raw", ROOT / "data/clean"
CLEAN.mkdir(parents=True, exist_ok=True)

def read_ecobee(f):
    lines = open(f).readlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("Date,Time"))
    return pd.read_csv(f, skiprows=k, index_col=False)

def ecobee():
    e = pd.concat([read_ecobee(f) for f in sorted((RAW / "ecobee").rglob("*.csv"))])
    e = e.drop_duplicates(["Date", "Time"])
    # thermostat local time (DST applied per ecobee docs)
    e["ts"] = pd.to_datetime(e.Date + " " + e.Time, format="%Y-%m-%d %H:%M:%S")
    cols = {"System Setting": "setting", "Heat Set Temp (C)": "sp_heat", "Cool Set Temp (C)": "sp_cool",
            "Current Temp (C)": "t_avg", "Thermostat Temperature (C)": "t_main",
            "Upstairs (C)": "t_up", "Current Humidity (%RH)": "rh",
            "Outdoor Temp (C)": "t_out", "Heat Stage 1 (sec)": "heat_s",
            "Cool Stage 1 (sec)": "cool_s", "Fan (sec)": "fan_s",
            "System Mode": "mode", "Program Mode": "prog", "Calendar Event": "event"}
    e = e.set_index("ts").sort_index()[list(cols)].rename(columns=cols)
    # ecobee reports local clock time with DST; convert to local standard time (EST, UTC-5)
    # to match Hydro Ottawa hourly data and EnergyPlus. Ambiguous fall-back hour dropped.
    idx = e.index.tz_localize("America/Toronto", ambiguous="NaT", nonexistent="NaT")
    e = e[~idx.isna()]
    e.index = idx[~idx.isna()].tz_convert("Etc/GMT+5").tz_localize(None)
    e.index.name = "ts_lst"
    return e

def hydro():
    frames = []
    for f in (RAW / "hydro").glob("*.xlsx"):
        d = pd.read_excel(f, header=2)
        frames.append(d[pd.to_datetime(d.Date, errors="coerce").notna()])
    h = pd.concat(frames)
    h["Date"] = pd.to_datetime(h.Date)
    return h.drop_duplicates("Date").set_index("Date").sort_index()

def hydro_hourly():
    frames = []
    for f in (RAW / "hydro_hourly").glob("*.xlsx"):
        d = pd.read_excel(f, header=2)
        d = d[pd.to_datetime(d["Start time"], errors="coerce").notna()]
        frames.append(d)
    h = pd.concat(frames)
    h["start"] = pd.to_datetime(h["Start time"])
    h["kwh"] = pd.to_numeric(h["Total (kWh)"], errors="coerce")
    h["t_out_ho"] = pd.to_numeric(h["Temperature °C"], errors="coerce")
    # Hydro Ottawa hourly timestamps behave as local standard time: ecobee-vs-meter
    # correlation peaks at zero lag in winter and only after ecobee DST->EST in summer.
    # The export skips 02:00 on the spring-forward date; that hour is left missing.
    h = h.drop_duplicates("start").set_index("start").sort_index()
    h.index.name = "start_lst"
    return h[["kwh", "t_out_ho"]]

def water():
    """City of Ottawa water meter reads (m3). Each row covers 'Days' days ending on the read date."""
    w = pd.read_csv(RAW / "water/Ottawa_Water_Usage_History.csv")
    w = w.rename(columns={"Meter Read Date": "read_date", "Read Status": "status", "Current Read": "read_m3"})
    w["read_date"] = pd.to_datetime(w.read_date)
    w["days"] = pd.to_numeric(w.Days)
    w["usage_m3"] = pd.to_numeric(w.Usage.astype(str).str.strip())
    w["start"] = w.read_date - pd.to_timedelta(w.days, "D")
    w["L_per_day"] = (1000 * w.usage_m3 / w.days).round(0)
    return w.sort_values("read_date")[["start", "read_date", "days", "usage_m3", "L_per_day", "status"]]

def enbridge():
    g = pd.concat([pd.read_csv(f) for f in sorted((RAW / "enbridge").glob("*.csv"))])
    g = g.drop_duplicates(["Billed From", "Billed To", "Meter Read"])
    g["billed_to"] = pd.to_datetime(g["Billed To"])
    return g.sort_values("billed_to")[["billed_to", "Meter Read", "Read Type", "Consumption M3"]]

if __name__ == "__main__":
    e, h, g, hh = ecobee(), hydro(), enbridge(), hydro_hourly()
    hh.to_csv(CLEAN / "hydro_hourly.csv")
    wt = water(); wt.to_csv(CLEAN / "water_reads.csv", index=False)
    print(f"water  {wt.start.min().date()} -> {wt.read_date.max().date()} ({len(wt)} reads, mean {1000 * wt.usage_m3.sum() / wt.days.sum():.0f} L/day)")
    print(f"hydro hourly {hh.index.min()} -> {hh.index.max()} ({len(hh)} h, {hh.kwh.isna().sum()} missing)")
    e.to_csv(CLEAN / "ecobee_5min.csv"); h.to_csv(CLEAN / "hydro_daily.csv"); g.to_csv(CLEAN / "enbridge_reads.csv", index=False)
    print(f"ecobee {e.index.min()} -> {e.index.max()} ({len(e)} rows)")
    print(f"hydro  {h.index.min().date()} -> {h.index.max().date()} ({len(h)} days)")
    print(f"gas    {len(g)} bills, {(g['Read Type']=='A').sum()} actual reads")
