"""One-at-a-time sensitivity of winter gas to calibration knobs (actual-year weather)."""
import sys, json, copy
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import build_model as bm
from run_ep import run
from compare import compare

cases = {
    "base": {},
    "ela_x0.5": {"ela_4pa_cm2": bm.PARAMS["ela_4pa_cm2"] * 0.5},
    "bsmt_ins_full": {"bsmt_ins_depth_below_grade": 1.68},
    "walls_ceiling_+30%R": {k: bm.PARAMS[k] * 1.3 for k in ["rsi_wall_f1", "rsi_wall_f2", "rsi_ceiling"]},
    "gains_lost_0": {"gains_fraction_lost": 0.0},
}
res = {}
for name, over in cases.items():
    p = copy.deepcopy(bm.PARAMS); p.update(over)
    m = Path(f"runs/sweep_{name}/house.epJSON"); m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(bm.build("actual", p)))
    run(m, "weather/Kanata_actual_2025-08_2026-07.epw", m.parent)
    out, _ = compare(m.parent)
    g = out["gas_periods"]
    res[name] = [p_["model_m3"] for p_ in g]
    print(f"{name:22s} gas model {res[name]} vs measured {[p_['measured_m3'] for p_ in g]}  T1 bias {out['temp_FLOOR1']['bias']:+.2f}")
