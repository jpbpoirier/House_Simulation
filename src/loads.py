"""Hourly non-HVAC electricity split into plug/lighting gains and clothes-dryer use (local standard time).

The dryer (electric, vented outdoors) is detected as hours whose non-HVAC electricity exceeds the
centred 25-h rolling median by more than DRYER_EXCESS_KWH. In those hours the excess is the dryer;
~230 h/yr at 1.5 kWh, mostly 09:00-21:00 (some cooking-oven hours will be misclassified).
"""
import pandas as pd

AC_KW, BLOWER_HEAT_KW, BLOWER_FAN_KW = 1.34, 0.26, 0.19   # from hourly regressions (phase1 findings)
DRYER_EXCESS_KWH, DRYER_MAX_KW, DRYER_FULL_KW = 1.5, 5.0, 3.0   # detection threshold, cap, kW counted as 'fully on'


def hvac_hours(e):
    x = e[["heat_s", "cool_s", "fan_s"]].fillna(0)
    fanonly = (x.fan_s - x[["heat_s", "cool_s"]].max(axis=1)).clip(lower=0)
    return pd.DataFrame({"heat_h": x.heat_s, "cool_h": x.cool_s, "fanonly_h": fanonly}).resample("h").sum() / 3600


def split_loads(e, kwh):
    """-> DataFrame (hourly): nonhvac_W, gains_W (non-HVAC minus dryer, >=100), dryer_W, dryer_on (0-1)."""
    H = hvac_hours(e).join(kwh.rename("kwh"), how="inner")
    g = H.kwh - AC_KW * H.cool_h - BLOWER_HEAT_KW * H.heat_h - BLOWER_FAN_KW * H.fanonly_h
    base = g.rolling(25, center=True, min_periods=5).median()
    ex = (g - base).where(g - base > DRYER_EXCESS_KWH, 0.0).clip(upper=DRYER_MAX_KW).fillna(0)
    out = pd.DataFrame({"nonhvac_W": g * 1000, "dryer_W": ex * 1000})
    out["gains_W"] = (out.nonhvac_W - out.dryer_W).clip(lower=100)
    out["dryer_on"] = (ex / DRYER_FULL_KW).clip(upper=1.0)
    return out.join(H[["heat_h", "cool_h", "fanonly_h"]])
