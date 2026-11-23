#!/usr/bin/env bash
# One-time setup for a fresh Linux (Ubuntu 24.04, x86_64) workspace: EnergyPlus 26.2 + Python packages.
# Installs EnergyPlus into $EPLUS_DIR (default /home/claude/ep, which src/run_ep.py uses on Linux). ~1-2 min.
set -euo pipefail
EPLUS_DIR="${EPLUS_DIR:-/home/claude/ep}"
URL="https://github.com/NREL/EnergyPlus/releases/download/v26.2.0/EnergyPlus-26.2.0-4bd7a1f26f-Linux-Ubuntu24.04-x86_64.tar.gz"
if [ ! -x "$EPLUS_DIR/energyplus" ]; then
  echo "Downloading EnergyPlus 26.2.0 ..."
  tmp="$(mktemp -d)"
  curl -fsSL "$URL" -o "$tmp/ep.tgz"
  mkdir -p "$EPLUS_DIR"
  tar -xzf "$tmp/ep.tgz" -C "$EPLUS_DIR" --strip-components=1
  rm -rf "$tmp"
fi
"$EPLUS_DIR/energyplus" --version
python3 -c "import numpy, pandas, scipy" 2>/dev/null || \
  pip install -q -r "$(dirname "$0")/../requirements.txt" --break-system-packages
echo "setup ok"
