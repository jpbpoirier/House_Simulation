# House Simulation — 3 DeCarlo Gardens

EnergyPlus 26.2 model of the house (semi-detached, Kanata, Ottawa), calibrated to ecobee, Hydro Ottawa and Enbridge data, used to forecast daily energy use and all-in cost from weather forecasts.

Dashboard: https://claude.ai/artifact/7qiLdwotc4fJCvyTDXCc14 · Model inputs, results and decisions: the claude.ai project "House Simulation" (house-inputs.md, model-v0-results.md, forecast-pipeline.md, tariffs.md, ccashp-sizing.md).

## Layout
| Folder | Contents |
|---|---|
| `model/build_model.py` | epJSON generator; all calibrated knobs in `PARAMS` (model v0.4.1) |
| `src/` | ingest, schedules, weather (EPW), run, results, calibration (`calib.py`, `screen.py`, `fit.py`), forecast, tariffs, heat-pump sizing |
| `data/clean/` | cleaned ecobee (5-min), Hydro Ottawa (hourly/daily), Enbridge reads, water reads, Schedule:File inputs. Local standard time (EST) throughout |
| `data/manual/` | manual gas meter readings |
| `weather/` | ERA5/ECCC observations, actual-weather EPWs, CWEC 2020 |
| `dashboard/` | JSON exports for the dashboard, forecast archive and log |
| `scripts/` | cloud setup and daily forecast |

Not in the repo: `data/raw/` (bills, EnerGuide reports, raw exports — kept on the Mac) and `runs/` (simulation outputs).

## Run
Mac (EnergyPlus at `/Applications/EnergyPlus-26-2-0`):
```
python3 src/forecast.py              # 10-day forecast -> dashboard/forecast*.json
python3 src/calib.py base            # calibration score card for current PARAMS
python3 src/ingest.py                # data/raw -> data/clean (after new exports)
python3 src/prep_schedules.py        # data/clean -> sched_8760.csv (add --aug-dec-year 2026 --out sched_fall2026.csv)
```
Cloud / Linux: `bash scripts/setup_cloud.sh` once (downloads EnergyPlus 26.2 to `/home/claude/ep`, or `$EPLUS_DIR`), then the same commands. `bash scripts/daily_forecast.sh` runs setup + forecast and commits the outputs.

Network: the forecast needs `api.open-meteo.com`; setup needs GitHub release downloads.
