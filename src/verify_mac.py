"""One-shot native check: rebuild schedules + model, run EnergyPlus 26.2 from /Applications, compare to data.
Run from the project folder (Terminal or VS Code):
    python3 src/verify_mac.py
Needs: pandas, numpy (same Python you used for the class notebook).
Writes runs/mac_check/ (EnergyPlus outputs + compare.json). Claude reads compare.json and eplusout.err from there.
"""
import subprocess, sys, time, json, platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "model"))
out = ROOT / "runs/mac_check"; out.mkdir(parents=True, exist_ok=True)

print("Python", sys.version.split()[0], "on", platform.platform())
print("1/4 schedules ..."); subprocess.run([sys.executable, str(ROOT / "src/prep_schedules.py")], check=True, capture_output=True)
print("2/4 model ...")
import build_model
(out / "house.epJSON").write_text(json.dumps(build_model.build("actual")))
print("3/4 EnergyPlus ...")
from run_ep import run, EPLUS_DIR
t = time.time()
rc = run(out / "house.epJSON", ROOT / "weather/Kanata_actual_2025-08_2026-07.epw", out)
print(f"    {EPLUS_DIR}: exit {rc}, {time.time() - t:.1f} s")
print("    ", (out / "eplusout.err").read_text().strip().splitlines()[-1])
if rc != 0:
    sys.exit("EnergyPlus failed: see runs/mac_check/eplusout.err")
print("4/4 compare ...")
from compare import compare
res, _ = compare(out)
for g in res["gas_periods"]:
    print(f"    gas {g['start']} -> {g['end']}: model {g['model_m3']:7.1f} m3 vs metered {g['measured_m3']:6.0f}")
print(f"    main floor bias {res['temp_FLOOR1']['bias']:+.2f} K, upstairs {res['temp_FLOOR2']['bias']:+.2f} K")
print("Done. Results in runs/mac_check/compare.json")
