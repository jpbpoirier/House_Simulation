"""Run EnergyPlus through the Python API.  python3 src/run_ep.py model.epJSON weather.epw outdir
Set EPLUS_DIR (default: /Applications/EnergyPlus-26-2-0 on macOS)."""
import os, sys, platform
from pathlib import Path
EPLUS_DIR = os.environ.get("EPLUS_DIR", "/Applications/EnergyPlus-26-2-0" if platform.system() == "Darwin"
                           else "/home/claude/ep")
sys.path.insert(0, EPLUS_DIR)
from pyenergyplus.api import EnergyPlusAPI

import subprocess, shutil

def to_idf(model):
    """HVACTemplate expansion (-x) only works on IDF input: convert epJSON -> IDF first."""
    model = Path(model)
    if model.suffix.lower() != ".epjson":
        return model
    exe = Path(EPLUS_DIR) / "ConvertInputFormat"
    subprocess.run([str(exe), "-f", "IDF", "-o", str(model.parent), str(model)], check=True, capture_output=True)
    return model.with_suffix(".idf")

def run(model, epw, outdir, expand=True, quiet=True):
    if expand:
        model = to_idf(model)
    api = EnergyPlusAPI()
    s = api.state_manager.new_state()
    api.runtime.set_console_output_status(s, not quiet)
    args = ["-w", str(epw), "-d", str(outdir)] + (["-x"] if expand else []) + [str(model)]
    rc = api.runtime.run_energyplus(s, args)
    api.state_manager.delete_state(s)
    return rc

if __name__ == "__main__":
    print("exit code", run(*sys.argv[1:4]))
