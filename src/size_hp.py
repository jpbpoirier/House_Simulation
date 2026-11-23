"""Design heating load with the calibrated v0.4 model: ideal loads, damper closed, design day(s)."""
import os, sys, json, copy, subprocess, sqlite3, platform
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import build_model as bm

EP = os.environ.get("EPLUS_DIR", "/Applications/EnergyPlus-26-2-0" if platform.system() == "Darwin" else "/home/claude/ep")


def make(tdesign, t_up=21.0, t_bsmt=17.0, gains=False, wind=3.6, annual=None, weather_name="cwec"):
    p = copy.deepcopy(bm.PARAMS)
    p["damper_ela_cm2"] = 0.0          # damper now kept closed
    p["thermostat"] = "floor1"         # no EMS blending; ideal loads per zone
    if not gains:
        p["exhaust_mult"] = 0.0 if False else p["exhaust_mult"]
    m = bm.build(weather=weather_name, p=p)
    # strip unitary HVAC template, add ideal loads per zone with constant setpoints
    for k in ["HVACTemplate:Zone:Unitary", "HVACTemplate:System:Unitary", "HVACTemplate:Thermostat"]:
        m.pop(k, None)
    m["Schedule:Constant"]["SP Up"] = dict(schedule_type_limits_name="Temperature", hourly_value=t_up)
    m["Schedule:Constant"]["SP Bsmt"] = dict(schedule_type_limits_name="Temperature", hourly_value=t_bsmt)
    m["Schedule:Constant"]["SP Clg"] = dict(schedule_type_limits_name="Temperature", hourly_value=50)
    m["HVACTemplate:Thermostat"] = {
        "TUp": dict(heating_setpoint_schedule_name="SP Up", cooling_setpoint_schedule_name="SP Clg"),
        "TBsmt": dict(heating_setpoint_schedule_name="SP Bsmt", cooling_setpoint_schedule_name="SP Clg")}
    m["HVACTemplate:Zone:IdealLoadsAirSystem"] = {
        f"{z} Ideal": dict(zone_name=z, template_thermostat_name=("TBsmt" if z == "Basement" else "TUp"),
                           heating_limit="NoLimit", cooling_limit="NoLimit", outdoor_air_method="None")
        for z in ["Basement", "Floor1", "Floor2"]}
    if annual is None:
        # design-day only: no solar, constant temperature, design wind
        dd = m["SizingPeriod:DesignDay"]["Ottawa Htg 99.6"]
        dd.update(maximum_dry_bulb_temperature=tdesign, wetbulb_or_dewpoint_at_maximum_dry_bulb=tdesign,
                  wind_speed=wind)
        m["SizingPeriod:DesignDay"].pop("Ottawa Clg 1")
        m["SimulationControl"]["SimulationControl 1"].update(
            run_simulation_for_sizing_periods="Yes", run_simulation_for_weather_file_run_periods="No",
            do_zone_sizing_calculation="No", do_system_sizing_calculation="No")
        if not gains:
            for cls in ["ElectricEquipment", "People"]:
                m.pop(cls, None)
            m.pop("ZoneVentilation:DesignFlowRate", None)
    else:
        m["SimulationControl"]["SimulationControl 1"].update(
            do_zone_sizing_calculation="No", do_system_sizing_calculation="No")
    m.pop("EnergyManagementSystem:Sensor", None); m.pop("EnergyManagementSystem:Actuator", None)
    m.pop("EnergyManagementSystem:Program", None); m.pop("EnergyManagementSystem:ProgramCallingManager", None)
    m["Output:Variable"] = {f"OV{i}": dict(key_value="*", variable_name=v, reporting_frequency="Hourly")
                            for i, v in enumerate(["Zone Ideal Loads Supply Air Total Heating Rate",
                                                   "Site Outdoor Air Drybulb Temperature",
                                                   "Zone Mean Air Temperature",
                                                   "Zone Infiltration Air Change Rate"])}
    m.pop("Output:Meter", None)
    return m


def run(m, tag, epw):
    d = ROOT / "runs" / "hp_sizing" / tag
    d.mkdir(parents=True, exist_ok=True)
    f = d / "in.epJSON"
    f.write_text(json.dumps(m))
    subprocess.run([f"{EP}/ConvertInputFormat", "-f", "IDF", "-o", str(d), str(f)], check=True, capture_output=True)
    r = subprocess.run([f"{EP}/energyplus", "-x", "-w", epw, "-d", str(d), str(d / "in.idf")],
                       capture_output=True, text=True)
    err = (d / "eplusout.err").read_text()
    sev = [l for l in err.splitlines() if "** Severe" in l or "**  Fatal" in l]
    if sev:
        print(tag, "SEVERE:", sev[:5])
    return d / "eplusout.sql"


def design_load(**kw):
    """Peak total ideal-loads heating (W) on the design day."""
    import collections
    sql = run(make(**kw), f"dd{kw['tdesign']}", str(ROOT / "weather/Ottawa_CWEC_2020.epw"))
    tot = collections.defaultdict(float)
    for k, mo, d, h, v in hourly(sql, "Zone Ideal Loads Supply Air Total Heating Rate"):
        tot[h] += v
    return max(tot.values())


def hourly(sql, var):
    c = sqlite3.connect(sql)
    q = """select d.KeyValue, t.Month, t.Day, t.Hour, r.Value from ReportData r
           join ReportDataDictionary d on r.ReportDataDictionaryIndex=d.ReportDataDictionaryIndex
           join Time t on r.TimeIndex=t.TimeIndex where d.Name=?"""
    return c.execute(q, (var,)).fetchall()


if __name__ == "__main__":
    # v0.4, damper closed, 21 C above grade / 17 C basement, no internal or solar gains
    for t in (-25, -15, 0):
        print(t, "C:", round(design_load(tdesign=t)), "W")
