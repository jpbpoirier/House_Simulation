#!/bin/bash
# Installs a macOS launchd job that runs src/forecast.py every day at 05:40 (and once now).
# Run once from the project folder:   bash src/install_forecast_job.sh
# Remove:  launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.housesim.forecast.plist
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(command -v python3)"
"$PY" -c "import pandas, numpy" 2>/dev/null || { echo "The python3 at $PY has no pandas/numpy. Activate the Python you use in VS Code and re-run."; exit 1; }
PLIST="$HOME/Library/LaunchAgents/com.housesim.forecast.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$DIR/runs/forecast"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.housesim.forecast</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$DIR/src/forecast.py</string></array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>5</integer><key>Minute</key><integer>40</integer></dict>
  <key>StandardOutPath</key><string>$DIR/runs/forecast/launchd.out</string>
  <key>StandardErrorPath</key><string>$DIR/runs/forecast/launchd.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart -k "gui/$(id -u)/com.housesim.forecast"
echo "Installed: daily 05:40 using $PY. First run started now; see runs/forecast/forecast_log.txt in ~15 s."
