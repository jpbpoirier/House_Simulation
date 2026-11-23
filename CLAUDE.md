# Notes for Claude sessions in this repo

Owner: J-P (P.Eng., heat pump researcher). Concise, number-backed answers; flag uncertainty.

- Fresh Linux workspace: run `bash scripts/setup_cloud.sh` before anything that calls EnergyPlus (~1-2 min).
- Times are local standard time (EST, UTC-5) in all data and model outputs. ecobee raw data is in local time with DST and is converted in `src/ingest.py`.
- Always simulate with a full-year (365-day) EPW; part-year files break Foundation:Kiva. `src/stitch_epw.py` builds them.
- E+ API runs are not fork-safe: run parallel cases as separate processes (`calib.evaluate_subprocess`), each with its own working directory (E+ writes `in.idf` to the cwd).
- `data/raw/` is not in git (private documents). New exports come from J-P's Mac; ingest them and commit `data/clean/`.
- Manual gas meter readings go in `data/manual/gas_meter_manual.csv` (`datetime_local,reading_m3,note`; whole m³ only).
- Dashboard artifact slug `3764023b-585c-42c6-b431-5166f53ebc87`. Datasets: daily, gas_periods, monthly (calibration), forecast, forecast_hourly. To publish a file: `upload_asset`, then `db_get` the dataset document for its version and `db_write` an `update` of `source.url` and `updated`.
- Calibration status, parameter meanings and open issues: claude.ai project doc `model-v0-results.md`. Update it when PARAMS change.
- Commit messages end with the session's attribution lines.
