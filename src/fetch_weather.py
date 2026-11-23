"""Fetch observed hourly weather for the calibration year. Run on the Mac (stdlib + system curl):
    cd ~/Documents/"House Simulation" && python3 src/fetch_weather.py
Writes:
  weather/openmeteo_archive_2025-08_2026-08.csv  (ERA5 reanalysis at the house: T, RH, dew point,
      pressure, wind, cloud, GHI/DNI/DHI; times in UTC)
  weather/eccc_ottawa_intl_YYYY-MM.csv            (ECCC OTTAWA INTL A hourly observations, LST)
"""
import json, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "weather"; OUT.mkdir(exist_ok=True)
LAT, LON = 45.31, -75.90          # Kanata, approx.
# optional: python3 src/fetch_weather.py 2026-08-01 2026-11-21
START, END = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("2025-08-01", "2026-08-31")

def get(url):
    # use the system curl (macOS keychain certificates) instead of Python's SSL stack
    r = subprocess.run(["curl", "-sSfL", "--max-time", "180", "-A", "house-sim/0.1", url],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode().strip() or f"curl exit {r.returncode}")
    return r.stdout

# 1) Open-Meteo historical archive (ERA5 / ERA5-Land), free, no key
vars_ = ("temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,"
         "wind_speed_10m,wind_direction_10m,cloud_cover,"
         "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,snow_depth")
url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}"
       f"&start_date={START}&end_date={END}&hourly={vars_}&timezone=GMT&wind_speed_unit=ms")
try:
    d = json.loads(get(url))["hourly"]
    keys = list(d)
    with open(OUT / f"openmeteo_archive_{START[:7]}_{END[:7]}.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for row in zip(*(d[k] for k in keys)):
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")
    print(f"Open-Meteo: {len(d['time'])} hours written")
except Exception as e:
    print("Open-Meteo failed:", e)

# 2) ECCC hourly observations, OTTAWA INTL A (station ID 49568), one CSV per month
ok = 0
import datetime as _dt
_s, _e = _dt.date.fromisoformat(START), _dt.date.fromisoformat(END)
months = sorted({(d.year, d.month) for d in (_s + _dt.timedelta(n) for n in range((_e - _s).days + 1))})
for y, m in months:
    url = ("https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv"
           f"&stationID=49568&Year={y}&Month={m}&Day=1&timeframe=1&submit=Download+Data")
    try:
        (OUT / f"eccc_ottawa_intl_{y}-{m:02d}.csv").write_bytes(get(url)); ok += 1
        time.sleep(1)
    except Exception as e:
        print(f"ECCC {y}-{m:02d} failed:", e)
print(f"ECCC: {ok}/{len(months)} months written")
