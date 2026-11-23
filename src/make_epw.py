"""Open-Meteo hourly weather -> EPW (local standard time, UTC-5), using the Ottawa CWEC header.
    python3 src/make_epw.py --src openmeteo_archive_2025-08_2026-08.csv --first-day 2025-08-01 --last-day 2026-07-31 \
        --out Kanata_actual_2025-08_2026-07.epw
Horizontal IR is left missing (9999) so EnergyPlus computes it from opaque sky cover."""
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "weather"

def load_openmeteo(path):
    om = pd.read_csv(path, parse_dates=["time"])
    om["lst"] = om.time - pd.Timedelta(hours=5)            # GMT -> EST
    return om.set_index("lst")

def epw_rows(om):
    """om: Open-Meteo hourly frame indexed by LST timestamp (value for the hour ending at that time)."""
    rows = []
    for t, r in om.iterrows():
        he = t if t.hour != 0 else t - pd.Timedelta(hours=1)          # hour-ending 24 belongs to previous day
        hour = t.hour if t.hour != 0 else 24
        cc = int(round((r.cloud_cover if r.cloud_cover == r.cloud_cover else 50) / 10))
        sd = r.get("snow_depth", 0.0)
        snow_cm = 0 if sd != sd else round(sd * 100)
        rows.append(",".join(map(str, [
            he.year, he.month, he.day, hour, 0, "?9?9?9?9E0?9?9?9?9?9?9?9*9?9?9?9?9?9*9*_?9*9*9*9*9",
            round(r.temperature_2m, 1), round(min(r.dew_point_2m, r.temperature_2m), 1), int(r.relative_humidity_2m),
            int(round(r.surface_pressure * 100)), 9999, 9999, 9999,
            round(r.shortwave_radiation), round(r.direct_normal_irradiance), round(r.diffuse_radiation),
            999999, 999999, 999999, 9999, int(r.wind_direction_10m), round(r.wind_speed_10m, 1),
            cc, cc, 9999, 99999, 9, 999999999, 999, 0.999, snow_cm, 99, 999, 0, 0])))
    return rows

def write_epw(om, first_day, last_day, out, comment):
    start = pd.Timestamp(first_day) + pd.Timedelta(hours=1)
    end = pd.Timestamp(last_day) + pd.Timedelta(days=1)
    om = om.loc[start:end]
    ndays = (pd.Timestamp(last_day) - pd.Timestamp(first_day)).days + 1
    assert len(om) == ndays * 24, (len(om), ndays * 24)
    hdr = (W / "Ottawa_CWEC_2020.epw").read_text().splitlines()[:8]
    hdr[0] = "LOCATION,KANATA (Open-Meteo),ON,CAN,ActualYear,716280,45.31,-75.90,-5.0,100.0"
    first, last = pd.Timestamp(first_day), pd.Timestamp(last_day)
    hdr[5] = f"COMMENTS 1,{comment} {first_day}..{last_day} LST (45.31,-75.90); header from Ottawa CWEC"
    hdr[7] = f"DATA PERIODS,1,1,Data,{first.day_name()}, {first.month}/{first.day}/{first.year}, {last.month}/{last.day}/{last.year}"
    rows = epw_rows(om)
    Path(out).write_text("\n".join(hdr + rows) + "\n")
    return len(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="openmeteo_archive_2025-08_2026-08.csv")
    ap.add_argument("--first-day", default="2025-08-01")
    ap.add_argument("--last-day", default="2026-07-31")
    ap.add_argument("--out", default="Kanata_actual_2025-08_2026-07.epw")
    A = ap.parse_args()
    n = write_epw(load_openmeteo(W / A.src), A.first_day, A.last_day, W / A.out, "Actual weather (ERA5 archive)")
    print("wrote", W / A.out, n, "hours")
