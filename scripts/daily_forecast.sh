#!/usr/bin/env bash
# Daily cloud forecast: setup -> forecast.py -> commit outputs. Publishing to the dashboard is done by
# the Claude scheduled task (artifact tools), not by this script.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_cloud.sh
python3 src/forecast.py
git add dashboard/forecast.json dashboard/forecast_hourly.json dashboard/forecast_archive dashboard/forecast_log.txt
git -c user.name="Claude (house-sim forecast)" -c user.email="noreply@anthropic.com" \
  commit -q -m "Forecast $(date -u +%Y-%m-%dT%H:%MZ)" || echo "nothing to commit"
