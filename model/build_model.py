"""Generate the EnergyPlus 26.2 epJSON model of 3 DeCarlo Gardens.

    python3 model/build_model.py [--weather actual|cwec] [--out model/house.epJSON]

All inputs are in PARAMS (calibration knobs) and the geometry/construction sections below.
Sources: house-inputs.md (EnerGuide 2024, plans, rating plates). Coordinates in metres:
x = SW->NE (party wall at x=0), y = front(SE)->rear(NW), z = up, grade at z=0.
North axis 320 deg puts the front facade normal at 140 deg (SE).
"""
import argparse, json, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PARAMS = dict(
    # envelope (effective RSI incl. films, EnerGuide 2024)
    # v0.4 calibration (2026-11-22, src/fit.py fit3): walls + ceiling x1.144 of EnerGuide effective RSI
    # v0.5 (2026-11-23, src/fit_mass.py fit6, with air capacitance free): x1.059
    rsi_wall_f1=2.65 * 1.059, rsi_wall_f2=2.44 * 1.059, rsi_wall_garage=2.29, rsi_ceiling=5.33 * 1.059,
    rsi_exposed_floor=3.93, rsi_header=2.80, rsi_bsmt_int_ins=2.11,
    bsmt_ins_depth_below_grade=1.567,        # fit at its bound (full height); J-P says ~3 ft (0.9 m) -> stands in for low basement losses
    win_u_new=2.05, win_shgc_new=0.353, win_u_side=3.0, win_shgc_side=0.60,
    win_u_bsmt=2.4, win_shgc_bsmt=0.50,
    # infiltration: ELA 668.5 cm2 @10 Pa (CGSB) -> 4 Pa with n=0.65
    # provisional calibration (2026-11-22): x0.5 brings winter gas from +52% to +16%
    # v0.4.1 leakage: EnerGuide blower door (damper closed, J-P) = 668.5 cm2 EqLA @10 Pa, Cd 0.611 (CGSB)
    # -> LBL ELA @4 Pa, Cd 1: 668.5 * 0.611 * 0.4**0.15 = 356 cm2 (v0.4 omitted the Cd factor).
    # One multiplier on all leakage (Sherman-Grimsrud with typical shelter overpredicts for this semi).
    ela_house_cm2=668.5 * 0.611 * (4 / 10) ** (0.65 - 0.5),
    infil_mult=0.396,            # v0.5 fit (v0.4.1: 0.472 = 239 cm2 winter total with damper open / (356 + 150))
    stack_coef=0.000290, wind_coef=0.000231,  # ASHRAE Fundamentals, 2-storey, typical shelter (check)
    # HVAC
    furnace_output_W=37000 / 3.412, furnace_eff=37000 / 39000,   # low fire, rating plate
    ac_cap_W=0.854 * 1.5 * 3517, ac_cop_gross=3.6, ac_shr=0.75,
    supply_flow_m3s=0.29, fan_dp_Pa=400, fan_eff=0.55,
    flow_split=dict(Floor1=0.36, Floor2=0.47, Basement=0.17),   # tuned to ecobee upstairs-main gradient (Dec-Feb +1.1 K)
    # DHW
    dhw_setpoint_C=55, dhw_cap_W=44000 / 3.412, dhw_eff=0.78,
    dhw_pilot_W=130, dhw_ua_W_K=3.0,
    dhw_hot_fraction=0.20,      # hot share of household water (City of Ottawa meter, mean 426 L/day -> ~85 L/day hot)
    # internal gains
    gains_fraction_lost=0.237, gains_split=dict(Floor1=0.45, Floor2=0.30, Basement=0.25),
    people=2,
    thermostat="weighted",   # "weighted": ecobee-style main/upstairs blend via EMS; "floor1": Floor1 only
    attic_ach=2.0, garage_ach=1.0,
    # v0.4: exhaust devices (exhaust-only ZoneVentilation + infiltration by quadrature), dryer, fireplace damper
    exhaust_mult=0.6,            # delivered / rated flow (typical for residential fans with real ducts; calibrate)
    range_cfm=200, bath_cfm=80, dryer_cfm=120,
    dryer_heat_lost=0.85,        # share of dryer electricity leaving with the exhaust
    damper_ela_cm2=150.0,        # fireplace damper open-minus-closed ELA before infil_mult (ASHRAE component table)
    soil_k=0.824, mass_mult=1.0,     # soil k at the fit's lower bound
    # v0.5: extra thermal mass via internal-mass density (mass_mult scales area, which changes the radiant
    # enclosure and made results jump non-monotonically; keep mass_mult = 1)
    mass_dens_mult=1.0,
    # v0.5: zone air capacitance multiplier on Basement/Floor1/Floor2 (fit6: 51.5). Night RC regression C:
    # measured 22.8 MJ/K, v0.4.1 4.9, v0.5 19.1 (benchmark-cpm-rc.md)
    air_cap_mult=51.5,
    # v0.4b: party wall to the neighbour's heated unit (J-P: neighbour keeps ~20 C)
    t_neighbour=19.33, t_neighbour_bsmt=17.0,
)

# ---------------------------------------------------------------- geometry
W, D = 6.60, 8.48              # footprint
NX, NY = 3.86, 1.88            # front-NE notch starts at x=NX, depth NY (garage occupies it)
GX1, GY0 = 7.06, -4.02         # garage extent
Z_AG = 0.61                    # top of foundation above grade (J-P: ~2 ft)
Z_1 = Z_AG + 0.25              # F1 floor: sill plate + 235 mm joists + subfloor on the foundation
Z_B = Z_1 - 2.18 - 0.25        # basement slab: 7'-2" clear to joist underside (plan)
Z_2 = Z_1 + 2.40 + 0.25        # 2.4 m ceilings + floor structure
Z_A = Z_2 + 2.40 + 0.25
RIDGE = 1.77                   # attic ridge height (5/12 pitch over half depth), ridge along x

FOOT_NOTCH = [(0, 0), (NX, 0), (NX, NY), (W, NY), (W, D), (0, D)]   # CCW from above
FOOT_RECT = [(0, 0), (W, 0), (W, D), (0, D)]


def wall_pts(a, b, z0, z1):
    """Wall from a to b (CCW footprint order), seen from outside: UL, LL, LR, UR."""
    return [(a[0], a[1], z1), (a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1)]


def up_pts(poly, z):       # outward normal up (ceiling/roof): CCW from above
    return [(x, y, z) for x, y in poly]


def down_pts(poly, z):     # outward normal down (floor): reverse
    return [(x, y, z) for x, y in reversed(poly)]


def verts(pts):
    return [{"vertex_x_coordinate": round(x, 4), "vertex_y_coordinate": round(y, 4),
             "vertex_z_coordinate": round(z, 4)} for x, y, z in pts]


def build(weather="cwec", p=PARAMS, period=None, sched="sched_8760.csv"):
    """weather: 'cwec' (typical year), 'actual' (calibration year) or 'custom' with period=(start, end) dates."""
    m = {}

    def add(cls, name, **f):
        m.setdefault(cls, {})[name] = f

    add("Version", "Version 1", version_identifier="26.2")
    add("Building", "3 DeCarlo Gardens", north_axis=320, terrain="Suburbs",
        loads_convergence_tolerance_value=0.04, temperature_convergence_tolerance_value=0.4,
        solar_distribution="FullExterior", maximum_number_of_warmup_days=25,
        minimum_number_of_warmup_days=6)
    add("GlobalGeometryRules", "GlobalGeometryRules 1", starting_vertex_position="UpperLeftCorner",
        vertex_entry_direction="Counterclockwise", coordinate_system="Relative")
    add("Timestep", "Timestep 1", number_of_timesteps_per_hour=6)
    # sizing is on only because HVACTemplate leaves terminal-unit fields autosized; all
    # equipment capacities and flows below are hard-set from rating plates / measurements.
    add("SimulationControl", "SimulationControl 1", do_zone_sizing_calculation="Yes",
        do_system_sizing_calculation="Yes", do_plant_sizing_calculation="No",
        run_simulation_for_sizing_periods="No", run_simulation_for_weather_file_run_periods="Yes")
    if weather == "custom":
        s0, s1 = [__import__("datetime").date.fromisoformat(str(x)) for x in period]
        add("RunPeriod", "Custom", begin_month=s0.month, begin_day_of_month=s0.day, begin_year=s0.year,
            end_month=s1.month, end_day_of_month=s1.day, end_year=s1.year, use_weather_file_holidays_and_special_days="No",
            use_weather_file_daylight_saving_period="No", apply_weekend_holiday_rule="No",
            use_weather_file_rain_indicators="Yes", use_weather_file_snow_indicators="Yes", treat_weather_as_actual="Yes")
    elif weather == "actual":
        add("RunPeriod", "Calibration", begin_month=8, begin_day_of_month=22, begin_year=2025,
            end_month=7, end_day_of_month=31, end_year=2026, use_weather_file_holidays_and_special_days="No",
            use_weather_file_daylight_saving_period="No", apply_weekend_holiday_rule="No",
            use_weather_file_rain_indicators="Yes", use_weather_file_snow_indicators="Yes",
            treat_weather_as_actual="Yes")
    else:
        add("RunPeriod", "TypicalYear", begin_month=1, begin_day_of_month=1, end_month=12,
            end_day_of_month=31, use_weather_file_holidays_and_special_days="No",
            use_weather_file_daylight_saving_period="No", apply_weekend_holiday_rule="No",
            use_weather_file_rain_indicators="Yes", use_weather_file_snow_indicators="Yes")
    # ASHRAE 2013 design conditions for Ottawa Intl (from CWEC header): 99.6% heating, 1% cooling
    add("SizingPeriod:DesignDay", "Ottawa Htg 99.6", month=1, day_of_month=21, day_type="WinterDesignDay",
        maximum_dry_bulb_temperature=-24.2, daily_dry_bulb_temperature_range=0,
        humidity_condition_type="WetBulb", wetbulb_or_dewpoint_at_maximum_dry_bulb=-24.2,
        barometric_pressure=99960, wind_speed=3.6, wind_direction=280,
        solar_model_indicator="ASHRAEClearSky", sky_clearness=0)
    add("SizingPeriod:DesignDay", "Ottawa Clg 1", month=7, day_of_month=21, day_type="SummerDesignDay",
        maximum_dry_bulb_temperature=28.9, daily_dry_bulb_temperature_range=10.4,
        humidity_condition_type="WetBulb", wetbulb_or_dewpoint_at_maximum_dry_bulb=20.8,
        barometric_pressure=99960, wind_speed=4.4, wind_direction=220,
        solar_model_indicator="ASHRAEClearSky", sky_clearness=1.0)
    add("Site:WaterMainsTemperature", "Site:WaterMainsTemperature 1",
        calculation_method="CorrelationFromWeatherFile")
    add("ZoneAirHeatBalanceAlgorithm", "ZoneAirHeatBalanceAlgorithm 1", algorithm="AnalyticalSolution")
    if p.get("air_cap_mult", 1.0) != 1.0:
        # v0.5: lumped contents / fast mass at the air node of the conditioned zones (attic and garage stay at 1).
        # Smooth stand-in for furniture, contents and surface layers; also absorbs ecobee sensor lag.
        for z in ["Basement", "Floor1", "Floor2"]:
            add("ZoneCapacitanceMultiplier:ResearchSpecial", f"Contents {z}", zone_or_zonelist_name=z,
                temperature_capacity_multiplier=p["air_cap_mult"], humidity_capacity_multiplier=1.0,
                carbon_dioxide_capacity_multiplier=1.0)

    # ------------------------------------------------------------ schedules
    for n, lo, hi in [("Fraction", 0, 1), ("Temperature", -60, 200), ("AnyNumber", None, None),
                      ("OnOff", 0, 1)]:
        f = {}
        if lo is not None:
            f.update(lower_limit_value=lo, upper_limit_value=hi, numeric_type="Continuous")
        add("ScheduleTypeLimits", n, **f)
    add("Schedule:Constant", "AlwaysOn", schedule_type_limits_name="Fraction", hourly_value=1)
    add("Schedule:Constant", "FanCycling", schedule_type_limits_name="Fraction", hourly_value=0)
    add("Schedule:Constant", "DHW Setpoint", schedule_type_limits_name="Temperature",
        hourly_value=p["dhw_setpoint_C"])
    add("Schedule:Constant", "Activity", schedule_type_limits_name="AnyNumber", hourly_value=110)
    # range hood: evening cooking ~18:00-18:45 daily; bath fan: morning shower ~06:30-07:30
    add("Schedule:Compact", "Range Hood", schedule_type_limits_name="Fraction", data=[{"field": f} for f in [
        "Through: 12/31", "For: AllDays", "Until: 18:00", 0, "Until: 18:50", 1, "Until: 24:00", 0]])
    add("Schedule:Compact", "Bath Fan", schedule_type_limits_name="Fraction", data=[{"field": f} for f in [
        "Through: 12/31", "For: AllDays", "Until: 06:30", 0, "Until: 07:30", 1, "Until: 24:00", 0]])
    sched_csv = str(ROOT / "data/clean" / sched)
    for col, name, lim in [(2, "Heating Setpoint", "Temperature"), (3, "Cooling Setpoint", "Temperature"),
                           (4, "Plug Gains W", "AnyNumber"), (5, "Occupancy", "Fraction"),
                           (6, "Upstairs Weight", "Fraction"), (7, "DHW Draw", "AnyNumber"),
                           (8, "Dryer W", "AnyNumber"), (9, "Dryer On", "Fraction"), (10, "Damper Open", "Fraction")]:
        add("Schedule:File", name, schedule_type_limits_name=lim, file_name=sched_csv, column_number=col,
            rows_to_skip_at_top=1, number_of_hours_of_data=8760, column_separator="Comma",
            interpolate_to_timestep="No", adjust_schedule_for_daylight_savings="No")
    # DHW draw (Schedule:File column 7, from prep_schedules): hourly profile x daily water-meter scale.
    # Peak flow sets the mean daily hot-water volume = dhw_hot_fraction x mean metered household water.
    import csv as _csv
    with open(ROOT / "data/clean/water_reads.csv") as _f:
        _r = list(_csv.DictReader(_f))
    water_L_day = 1000 * sum(float(x["usage_m3"]) for x in _r) / sum(float(x["days"]) for x in _r)
    dhw_L_day = p["dhw_hot_fraction"] * water_L_day
    prof_sum = 0.06 + 0.03 + 0.015 + 0.04 + 0.04 + 0.017
    dhw_peak = dhw_L_day / 1000 / (prof_sum * 3600)

    # ------------------------------------------------------------ materials / constructions
    def mat(n, t, k, rho, cp, rough="MediumRough"):
        add("Material", n, roughness=rough, thickness=t, conductivity=k, density=rho, specific_heat=cp)

    def nomass(n, r):
        add("Material:NoMass", n, roughness="MediumRough", thermal_resistance=round(r, 4))

    mat("Gypsum 13mm", 0.0127, 0.16, 800, 1090, "Smooth")
    mat("OSB 11mm", 0.011, 0.12, 650, 1210)
    mat("Siding 10mm", 0.010, 0.11, 600, 1300)
    mat("Plywood 19mm", 0.019, 0.12, 540, 1210)
    # v0.5: internal-mass materials scaled by density, not area (area changes the radiant enclosure: see mass_dens_mult)
    mat("IM Gypsum 13mm", 0.0127, 0.16, 800 * p.get("mass_dens_mult", 1.0), 1090, "Smooth")
    mat("IM Plywood 19mm", 0.019, 0.12, 540 * p.get("mass_dens_mult", 1.0), 1210)
    mat("Concrete 200mm", 0.20, 1.40, 2240, 900)
    mat("Concrete 100mm", 0.10, 1.40, 2240, 900)
    mat("Shingles", 0.006, 0.16, 1100, 1260, "Rough")
    mat("Steel skin", 0.0008, 45, 7800, 500, "Smooth")
    mat("Basement wall insulation", 0.089, 0.089 / p["rsi_bsmt_int_ins"], 30, 1000)
    mat("Gravel", 0.15, 1.8, 2000, 840)
    mat("Brick 90mm", 0.09, 0.80, 1900, 800, "Rough")
    films = 0.15
    r_layers_wall = 0.0127 / 0.16 + 0.011 / 0.12 + 0.010 / 0.11
    for n, r in [("F2", p["rsi_wall_f2"]), ("Garage", p["rsi_wall_garage"])]:
        nomass(f"Wall ins {n}", r - films - r_layers_wall)
        add("Construction", f"Ext Wall {n}", outside_layer="Siding 10mm", layer_2="OSB 11mm",
            layer_3=f"Wall ins {n}", layer_4="Gypsum 13mm")
    # F1: brick veneer + 25 mm vented cavity (photos); same effective RSI as EnerGuide
    nomass("Brick cavity", 0.15)
    nomass("Wall ins F1", p["rsi_wall_f1"] - films - 0.09 / 0.80 - 0.15 - 0.0127 / 0.16 - 0.011 / 0.12)
    add("Construction", "Ext Wall F1", outside_layer="Brick 90mm", layer_2="Brick cavity", layer_3="OSB 11mm",
        layer_4="Wall ins F1", layer_5="Gypsum 13mm")
    # garage-to-house wall (interzone): reverse construction for the garage side
    add("Construction", "House-Garage Wall", outside_layer="Gypsum 13mm", layer_2="Wall ins Garage",
        layer_3="Gypsum 13mm")
    nomass("Header ins", p["rsi_header"] - films - 0.011 / 0.12 - 0.010 / 0.11)
    add("Construction", "Header", outside_layer="Siding 10mm", layer_2="OSB 11mm", layer_3="Header ins")
    nomass("Ceiling ins", p["rsi_ceiling"] - 0.2 - 0.0127 / 0.16)
    add("Construction", "Ceiling to Attic", outside_layer="Ceiling ins", layer_2="Gypsum 13mm")
    add("Construction", "Ceiling to Attic Rev", outside_layer="Gypsum 13mm", layer_2="Ceiling ins")
    nomass("Exposed floor ins", p["rsi_exposed_floor"] - 0.2 - 0.019 / 0.12)
    add("Construction", "Exposed Floor", outside_layer="Exposed floor ins", layer_2="Plywood 19mm")
    add("Construction", "Exposed Floor Rev", outside_layer="Plywood 19mm", layer_2="Exposed floor ins")
    nomass("Joist cavity", 0.18)
    add("Construction", "Interior Floor", outside_layer="Gypsum 13mm", layer_2="Joist cavity",
        layer_3="Plywood 19mm")
    add("Construction", "Interior Floor Rev", outside_layer="Plywood 19mm", layer_2="Joist cavity",
        layer_3="Gypsum 13mm")
    add("Construction", "Basement Ceiling", outside_layer="Plywood 19mm", layer_2="Joist cavity")
    add("Construction", "Basement Ceiling Rev", outside_layer="Joist cavity", layer_2="Plywood 19mm")
    add("Construction", "Foundation Wall", outside_layer="Concrete 200mm")
    add("Construction", "Foundation Wall Above Grade", outside_layer="Concrete 200mm",
        layer_2="Basement wall insulation")
    add("Construction", "Foundation Wall Above Grade Rev", outside_layer="Basement wall insulation",
        layer_2="Concrete 200mm")
    add("Construction", "Slab", outside_layer="Concrete 100mm")
    add("Construction", "Garage Slab", outside_layer="Concrete 100mm")
    add("Construction", "Roof", outside_layer="Shingles", layer_2="OSB 11mm")
    add("Construction", "Garage Wall", outside_layer="Siding 10mm", layer_2="OSB 11mm")
    # party wall (1986 semi): 200 mm hollow concrete block; above grade finished with gypsum on strapping.
    # Hollow normal-weight block RSI ~0.19 (ASHRAE Fundamentals) -> k_eff 1.05; neighbour side = film via OSC.
    mat("Concrete block 200mm", 0.19, 1.0, 1100, 880)
    nomass("Strapping air space", 0.17)
    add("Construction", "Party Wall", outside_layer="Concrete block 200mm", layer_2="Strapping air space",
        layer_3="Gypsum 13mm")
    add("Construction", "Party Wall Bsmt", outside_layer="Concrete block 200mm")
    add("Construction", "Party Gable", outside_layer="Gypsum 13mm", layer_2="Concrete 200mm",
        layer_3="Gypsum 13mm")
    for n, t in [("Neighbour", p["t_neighbour"]), ("Neighbour Bsmt", p["t_neighbour_bsmt"])]:
        add("SurfaceProperty:OtherSideCoefficients", n, combined_convective_radiative_film_coefficient=8.3,
            constant_temperature=t, constant_temperature_coefficient=1.0, external_dry_bulb_temperature_coefficient=0,
            ground_temperature_coefficient=0, wind_speed_coefficient=0, zone_air_temperature_coefficient=0)
    add("Construction", "Gable", outside_layer="Siding 10mm", layer_2="OSB 11mm")
    nomass("Door core", 0.98 - films)
    add("Construction", "Ext Door", outside_layer="Steel skin", layer_2="Door core", layer_3="Steel skin")
    add("Construction", "Interior Partition", outside_layer="IM Gypsum 13mm", layer_2="IM Gypsum 13mm")
    add("Construction", "Furniture", outside_layer="IM Plywood 19mm")
    for n, u, sh in [("Win New", p["win_u_new"], p["win_shgc_new"]), ("Win Side", p["win_u_side"], p["win_shgc_side"]),
                     ("Win Bsmt", p["win_u_bsmt"], p["win_shgc_bsmt"])]:
        add("WindowMaterial:SimpleGlazingSystem", n + " glz", u_factor=u, solar_heat_gain_coefficient=sh,
            visible_transmittance=0.6)
        add("Construction", n, outside_layer=n + " glz")

    # ------------------------------------------------------------ zones
    zones = {"Basement": Z_B, "Floor1": Z_1, "Floor2": Z_2, "Attic": Z_A, "Garage": 0.0}
    for z, zo in zones.items():
        add("Zone", z, x_origin=0, y_origin=0, z_origin=0, multiplier=1)

    def surf(name, typ, cons, zone, bc, pts, bc_obj=None, sun=True):
        f = dict(surface_type=typ, construction_name=cons, zone_name=zone,
                 outside_boundary_condition=bc, vertices=verts(pts))
        if bc_obj:
            f["outside_boundary_condition_object"] = bc_obj
        exposed = bc == "Outdoors"
        f["sun_exposure"] = "SunExposed" if exposed and sun else "NoSun"
        f["wind_exposure"] = "WindExposed" if exposed else "NoWind"
        add("BuildingSurface:Detailed", name, **f)

    def sub(name, typ, cons, host, pts):
        f = dict(surface_type=typ, construction_name=cons, building_surface_name=host, number_of_vertices=4)
        for i, (x, y, z) in enumerate(pts, 1):
            f.update({f"vertex_{i}_x_coordinate": round(x, 4), f"vertex_{i}_y_coordinate": round(y, 4),
                      f"vertex_{i}_z_coordinate": round(z, 4)})
        add("FenestrationSurface:Detailed", name, **f)

    def edges(poly):
        return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]

    # helpers to classify a footprint edge: party (x=0), garage-adjacent, or exterior
    def is_party(a, b):
        return a[0] == 0 and b[0] == 0

    def is_garage_adj(a, b):  # notch edges
        return {a, b} in ({(NX, 0), (NX, NY)}, {(NX, NY), (W, NY)})

    # --- Basement: Kiva below grade (Z_B..0), above-grade foundation strip (0..Z_AG), header (Z_AG..Z_1)
    add("Foundation:Kiva", "Basement Foundation", initial_indoor_air_temperature=18,
        interior_vertical_insulation_material_name="Basement wall insulation",
        interior_vertical_insulation_depth=p["bsmt_ins_depth_below_grade"],
        wall_height_above_grade=0.0, wall_depth_below_slab=0.0)
    add("Foundation:Kiva", "Garage Foundation", initial_indoor_air_temperature=10,
        wall_height_above_grade=0.0, wall_depth_below_slab=0.3)
    add("Foundation:Kiva:Settings", "Foundation:Kiva:Settings 1", soil_conductivity=p["soil_k"],
        soil_density=1842, soil_specific_heat=419, ground_solar_absorptivity=0.9,
        ground_thermal_absorptivity=0.9, ground_surface_roughness=0.03, far_field_width=40,
        deep_ground_boundary_condition="Autoselect", deep_ground_depth=40,
        minimum_cell_dimension=0.02, maximum_cell_growth_coefficient=1.5, simulation_timestep="Hourly")
    exp_perim = 0.0
    for i, (a, b) in enumerate(edges(FOOT_NOTCH)):
        L = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        if is_party(a, b):
            surf(f"Bsmt Party {i}", "Wall", "Party Wall Bsmt", "Basement", "OtherSideCoefficients",
                 wall_pts(a, b, Z_B, Z_1), bc_obj="Neighbour Bsmt")
            continue
        exp_perim += L
        surf(f"Bsmt Wall BG {i}", "Wall", "Foundation Wall", "Basement", "Foundation",
             wall_pts(a, b, Z_B, 0.0), bc_obj="Basement Foundation")
        if is_garage_adj(a, b):   # above-grade strip and header face the garage
            surf(f"Bsmt Wall AG {i}", "Wall", "Foundation Wall Above Grade", "Basement", "Surface",
                 wall_pts(a, b, 0.0, Z_1), bc_obj=f"Garage To Bsmt {i}")
        else:
            surf(f"Bsmt Wall AG {i}", "Wall", "Foundation Wall Above Grade", "Basement", "Outdoors",
                 wall_pts(a, b, 0.0, Z_AG))
            surf(f"Bsmt Header {i}", "Wall", "Header", "Basement", "Outdoors", wall_pts(a, b, Z_AG, Z_1))
    surf("Bsmt Slab", "Floor", "Slab", "Basement", "Foundation", down_pts(FOOT_NOTCH, Z_B),
         bc_obj="Basement Foundation")
    add("SurfaceProperty:ExposedFoundationPerimeter", "Bsmt Slab Perimeter", surface_name="Bsmt Slab",
        exposed_perimeter_calculation_method="TotalExposedPerimeter", total_exposed_perimeter=round(exp_perim, 3))
    surf("Bsmt Ceiling", "Ceiling", "Basement Ceiling", "Basement", "Surface", up_pts(FOOT_NOTCH, Z_1),
         bc_obj="F1 Floor")
    # basement windows: rear (NW) above-grade strip; rear edge is (W,D)->(0,D) = edge index 4
    for k, (x0, x1) in enumerate([(1.0, 1.76), (4.5, 5.26)]):
        sub(f"Bsmt Win {k + 1}", "Window", "Win Bsmt", "Bsmt Wall AG 4",
            [(x1, D, 0.53), (x1, D, 0.23), (x0, D, 0.23), (x0, D, 0.53)])

    # --- Floor 1
    for i, (a, b) in enumerate(edges(FOOT_NOTCH)):
        name = f"F1 Wall {i}"
        if is_party(a, b):
            surf(name, "Wall", "Party Wall", "Floor1", "OtherSideCoefficients", wall_pts(a, b, Z_1, Z_2),
                 bc_obj="Neighbour")
        elif is_garage_adj(a, b):
            surf(name, "Wall", "House-Garage Wall", "Floor1", "Surface", wall_pts(a, b, Z_1, Z_2),
                 bc_obj=f"Garage Adj {i}")
        else:
            surf(name, "Wall", "Ext Wall F1", "Floor1", "Outdoors", wall_pts(a, b, Z_1, Z_2))
    surf("F1 Floor", "Floor", "Basement Ceiling Rev", "Floor1", "Surface", down_pts(FOOT_NOTCH, Z_1),
         bc_obj="Bsmt Ceiling")
    surf("F1 Ceiling", "Ceiling", "Interior Floor", "Floor1", "Surface", up_pts(FOOT_NOTCH, Z_2),
         bc_obj="F2 Floor Main")
    # F1 windows/doors. Front edge 0: (0,0)->(NX,0); rear edge 4: (W,D)->(0,D)
    sub("F1 Kitchen Win", "Window", "Win New", "F1 Wall 0",
        [(1.0, 0, Z_1 + 1.99), (1.0, 0, Z_1 + 1.0), (2.2, 0, Z_1 + 1.0), (2.2, 0, Z_1 + 1.99)])
    sub("F1 Front Door", "Door", "Ext Door", "F1 Wall 0",
        [(2.75, 0, Z_1 + 2.03), (2.75, 0, Z_1 + 0.02), (3.66, 0, Z_1 + 0.02), (3.66, 0, Z_1 + 2.03)])
    sub("F1 Patio Door", "GlassDoor", "Win New", "F1 Wall 4",
        [(5.85, D, Z_1 + 2.08), (5.85, D, Z_1 + 0.02), (4.22, D, Z_1 + 0.02), (4.22, D, Z_1 + 2.08)])
    sub("F1 Living Win", "Window", "Win New", "F1 Wall 4",
        [(2.25, D, Z_1 + 1.71), (2.25, D, Z_1 + 0.9), (1.03, D, Z_1 + 0.9), (1.03, D, Z_1 + 1.71)])

    # --- Floor 2 (full rectangle; part of its floor is over the garage notch)
    for i, (a, b) in enumerate(edges(FOOT_RECT)):
        if is_party(a, b):
            surf(f"F2 Wall {i}", "Wall", "Party Wall", "Floor2", "OtherSideCoefficients", wall_pts(a, b, Z_2, Z_A),
                 bc_obj="Neighbour")
        else:
            surf(f"F2 Wall {i}", "Wall", "Ext Wall F2", "Floor2", "Outdoors", wall_pts(a, b, Z_2, Z_A))
    notch = [(NX, 0), (W, 0), (W, NY), (NX, NY)]
    surf("F2 Floor Main", "Floor", "Interior Floor Rev", "Floor2", "Surface", down_pts(FOOT_NOTCH, Z_2),
         bc_obj="F1 Ceiling")
    surf("F2 Floor Over Garage", "Floor", "Exposed Floor", "Floor2", "Surface", down_pts(notch, Z_2),
         bc_obj="Garage Ceiling")
    surf("F2 Ceiling", "Ceiling", "Ceiling to Attic", "Floor2", "Surface", up_pts(FOOT_RECT, Z_A),
         bc_obj="Attic Floor")
    # F2 windows: front edge 0 (0,0)->(W,0); rear edge 2 (W,D)->(0,D); NE edge 1 (W,0)->(W,D)
    for k, (x0, w, h) in enumerate([(0.8, 1.194, 1.175), (4.4, 1.194, 1.168)]):
        sub(f"F2 Front Win {k + 1}", "Window", "Win New", "F2 Wall 0",
            [(x0, 0, Z_2 + 0.9 + h), (x0, 0, Z_2 + 0.9), (x0 + w, 0, Z_2 + 0.9), (x0 + w, 0, Z_2 + 0.9 + h)])
    sub("F2 Master Win", "Window", "Win New", "F2 Wall 2",
        [(5.85, D, Z_2 + 2.07), (5.85, D, Z_2 + 0.9), (4.25, D, Z_2 + 0.9), (4.25, D, Z_2 + 2.07)])
    sub("Stair Win", "Window", "Win Side", "F2 Wall 1",
        [(W, 4.0, Z_2 + 0.86), (W, 4.0, Z_2 + 0.1), (W, 4.9, Z_2 + 0.1), (W, 4.9, Z_2 + 0.86)])

    # --- Attic: gable roof, ridge along x at mid-depth
    ym = D / 2
    surf("Attic Floor", "Floor", "Ceiling to Attic Rev", "Attic", "Surface", down_pts(FOOT_RECT, Z_A),
         bc_obj="F2 Ceiling")
    surf("Roof Front", "Roof", "Roof", "Attic", "Outdoors",
         [(0, ym, Z_A + RIDGE), (0, 0, Z_A), (W, 0, Z_A), (W, ym, Z_A + RIDGE)])
    surf("Roof Rear", "Roof", "Roof", "Attic", "Outdoors",
         [(W, ym, Z_A + RIDGE), (W, D, Z_A), (0, D, Z_A), (0, ym, Z_A + RIDGE)])
    surf("Gable NE", "Wall", "Gable", "Attic", "Outdoors",
         [(W, ym, Z_A + RIDGE), (W, 0, Z_A), (W, D, Z_A)])
    surf("Gable SW", "Wall", "Party Gable", "Attic", "Adiabatic",
         [(0, ym, Z_A + RIDGE), (0, D, Z_A), (0, 0, Z_A)])

    # --- Garage: x NX..GX1, y GY0..NY, z 0..Z_2 ; notch walls shared with house
    gfoot = [(NX, GY0), (GX1, GY0), (GX1, NY), (W, NY), (NX, NY)]
    surf("Garage Front", "Wall", "Garage Wall", "Garage", "Outdoors", wall_pts((NX, GY0), (GX1, GY0), 0, Z_2))
    surf("Garage NE", "Wall", "Garage Wall", "Garage", "Outdoors", wall_pts((GX1, GY0), (GX1, NY), 0, Z_2))
    surf("Garage Rear Stub", "Wall", "Garage Wall", "Garage", "Outdoors", wall_pts((GX1, NY), (W, NY), 0, Z_2))
    # walls facing the house: F1 notch edges are edge 1 ((NX,0)->(NX,NY)) and 2 ((NX,NY)->(W,NY))
    surf("Garage Adj 2", "Wall", "House-Garage Wall", "Garage", "Surface",
         wall_pts((W, NY), (NX, NY), Z_1, Z_2), bc_obj="F1 Wall 2")
    surf("Garage Adj 1", "Wall", "House-Garage Wall", "Garage", "Surface",
         wall_pts((NX, NY), (NX, 0), Z_1, Z_2), bc_obj="F1 Wall 1")
    surf("Garage To Bsmt 2", "Wall", "Foundation Wall Above Grade Rev", "Garage", "Surface",
         wall_pts((W, NY), (NX, NY), 0.0, Z_1), bc_obj="Bsmt Wall AG 2")
    surf("Garage To Bsmt 1", "Wall", "Foundation Wall Above Grade Rev", "Garage", "Surface",
         wall_pts((NX, NY), (NX, 0), 0.0, Z_1), bc_obj="Bsmt Wall AG 1")
    surf("Garage Side Fwd", "Wall", "Garage Wall", "Garage", "Outdoors",   # faces the front walk
         wall_pts((NX, 0), (NX, GY0), 0, Z_2))
    surf("Garage Ceiling", "Ceiling", "Exposed Floor Rev", "Garage", "Surface", up_pts(notch, Z_2),
         bc_obj="F2 Floor Over Garage")
    surf("Garage Roof", "Roof", "Roof", "Garage", "Outdoors",
         up_pts([(NX, GY0), (GX1, GY0), (GX1, NY), (W, NY), (W, 0), (NX, 0)], Z_2))
    surf("Garage Slab", "Floor", "Garage Slab", "Garage", "Foundation", down_pts(gfoot, 0.0),
         bc_obj="Garage Foundation")
    add("SurfaceProperty:ExposedFoundationPerimeter", "Garage Slab Perimeter", surface_name="Garage Slab",
        exposed_perimeter_calculation_method="TotalExposedPerimeter",
        total_exposed_perimeter=round((GX1 - NX) + (NY - GY0) + 0.46 + (0 - GY0), 3))

    # --- shading (building-relative coords). Assumed: NE neighbour 2-storey wall 3.0 m from our NE wall,
    # 7 m high, along our full depth; rear spruce ~11 m high, 4 m wide, 8 m behind the rear wall, centred
    # behind the NE half (photos: shades the upstairs rear window), 30% transmittance (conifer, year-round).
    add("Schedule:Constant", "Spruce Transmittance", schedule_type_limits_name="Fraction", hourly_value=0.3)
    nx = W + 3.0
    add("Shading:Building:Detailed", "NE Neighbour", number_of_vertices=4, vertices=verts(
        [(nx, -1.0, 7.0), (nx, -1.0, 0.0), (nx, D + 1.0, 0.0), (nx, D + 1.0, 7.0)]))
    add("Shading:Building:Detailed", "Rear Spruce", transmittance_schedule_name="Spruce Transmittance",
        number_of_vertices=4, vertices=verts([(3.0, D + 8.0, 11.0), (3.0, D + 8.0, 0.5),
                                              (7.0, D + 8.0, 0.5), (7.0, D + 8.0, 11.0)]))

    # fix: F1 wall 1/2 boundary objects named after garage surfaces
    m["BuildingSurface:Detailed"]["F1 Wall 1"]["outside_boundary_condition_object"] = "Garage Adj 1"
    m["BuildingSurface:Detailed"]["F1 Wall 2"]["outside_boundary_condition_object"] = "Garage Adj 2"

    # ------------------------------------------------------------ internal mass, gains, infiltration
    area = {"Basement": 47.4, "Floor1": 50.2, "Floor2": 50.2}
    for z, a in area.items():
        add("InternalMass", f"{z} Partitions", construction_name="Interior Partition", zone_or_zonelist_name=z,
            surface_area=round(1.5 * a * p["mass_mult"], 1))
        add("InternalMass", f"{z} Furniture", construction_name="Furniture", zone_or_zonelist_name=z,
            surface_area=round(1.0 * a * p["mass_mult"], 1))
        add("ElectricEquipment", f"{z} Plug", zone_or_zonelist_or_space_or_spacelist_name=z,
            schedule_name="Plug Gains W", design_level_calculation_method="EquipmentLevel",
            design_level=p["gains_split"][z], fraction_latent=0.05, fraction_radiant=0.3,
            fraction_lost=p["gains_fraction_lost"], end_use_subcategory="Plug and Lights")
    for z, n in [("Floor1", 0.6), ("Floor2", 0.4)]:
        add("People", f"{z} People", zone_or_zonelist_or_space_or_spacelist_name=z,
            number_of_people_schedule_name="Occupancy", number_of_people_calculation_method="People",
            number_of_people=p["people"] * n, fraction_radiant=0.3, activity_level_schedule_name="Activity")
    ela = p["infil_mult"] * p["ela_house_cm2"]
    for z, frac in [("Basement", 0.25), ("Floor1", 0.35), ("Floor2", 0.40)]:
        add("ZoneInfiltration:EffectiveLeakageArea", f"{z} ELA", zone_or_space_name=z, schedule_name="AlwaysOn",
            effective_air_leakage_area=round(ela * frac, 2), stack_coefficient=p["stack_coef"],
            wind_coefficient=p["wind_coef"])
    if p["damper_ela_cm2"] > 0:     # fireplace on the NE wall of Floor1; leaks only when the damper is open
        add("ZoneInfiltration:EffectiveLeakageArea", "Fireplace Damper ELA", zone_or_space_name="Floor1",
            schedule_name="Damper Open", effective_air_leakage_area=round(p["infil_mult"] * p["damper_ela_cm2"], 2),
            stack_coefficient=p["stack_coef"], wind_coefficient=p["wind_coef"])
    cfm = 0.000471947 * p["exhaust_mult"]
    for name, z, sch, f in [("Range Hood", "Floor1", "Range Hood", p["range_cfm"]),
                            ("Bath Fan", "Floor2", "Bath Fan", p["bath_cfm"]),
                            ("Dryer Exhaust", "Basement", "Dryer On", p["dryer_cfm"])]:
        add("ZoneVentilation:DesignFlowRate", name, zone_or_zonelist_or_space_or_spacelist_name=z, schedule_name=sch,
            design_flow_rate_calculation_method="Flow/Zone", design_flow_rate=round(f * cfm, 5),
            ventilation_type="Exhaust", fan_pressure_rise=0.0, fan_total_efficiency=1.0,   # fan power is in plug loads
            constant_term_coefficient=1, temperature_term_coefficient=0, velocity_term_coefficient=0,
            velocity_squared_term_coefficient=0, minimum_indoor_temperature=-100, maximum_indoor_temperature=100,
            delta_temperature=-100, minimum_outdoor_temperature=-100, maximum_outdoor_temperature=100,
            maximum_wind_speed=40)
    for z in ["Basement", "Floor1", "Floor2"]:
        add("ZoneAirBalance:OutdoorAir", f"{z} Air Balance", zone_name=z, air_balance_method="Quadrature",
            induced_outdoor_air_due_to_unbalanced_duct_leakage=0, induced_outdoor_air_schedule_name="AlwaysOn")
    # dryer: electricity on the meter, most of its heat leaves with the exhaust air
    add("ElectricEquipment", "Dryer", zone_or_zonelist_or_space_or_spacelist_name="Basement", schedule_name="Dryer W",
        design_level_calculation_method="EquipmentLevel", design_level=1.0, fraction_latent=0.0,
        fraction_radiant=0.1, fraction_lost=p["dryer_heat_lost"], end_use_subcategory="Dryer")
    for z, ach in [("Attic", p["attic_ach"]), ("Garage", p["garage_ach"])]:
        add("ZoneInfiltration:DesignFlowRate", f"{z} Infiltration", zone_or_zonelist_or_space_or_spacelist_name=z,
            schedule_name="AlwaysOn", design_flow_rate_calculation_method="AirChanges/Hour", air_changes_per_hour=ach,
            constant_term_coefficient=0, temperature_term_coefficient=0, velocity_term_coefficient=0.224,
            velocity_squared_term_coefficient=0)

    # ------------------------------------------------------------ HVAC (template, hard-sized)
    add("HVACTemplate:Thermostat", "Ecobee", heating_setpoint_schedule_name="Heating Setpoint",
        cooling_setpoint_schedule_name="Cooling Setpoint")
    for z in ["Floor1", "Floor2", "Basement"]:
        add("HVACTemplate:Zone:Unitary", f"{z} Unitary", zone_name=z, template_unitary_system_name="Furnace AC",
            template_thermostat_name="Ecobee",
            supply_air_maximum_flow_rate=round(p["supply_flow_m3s"] * p["flow_split"][z], 4),
            outdoor_air_method="Flow/Zone", outdoor_air_flow_rate_per_zone=0.0,
            baseboard_heating_type="None")
    add("HVACTemplate:System:Unitary", "Furnace AC", system_availability_schedule_name="AlwaysOn",
        control_zone_or_thermostat_location_name="Floor1", supply_fan_maximum_flow_rate=p["supply_flow_m3s"],
        supply_fan_operating_mode_schedule_name="FanCycling", supply_fan_total_efficiency=p["fan_eff"],
        supply_fan_delta_pressure=p["fan_dp_Pa"], supply_fan_motor_efficiency=0.85,
        supply_fan_motor_in_air_stream_fraction=1.0, cooling_coil_type="SingleSpeedDX",
        cooling_coil_availability_schedule_name="AlwaysOn", cooling_design_supply_air_temperature=12.8,
        cooling_coil_gross_rated_total_capacity=round(p["ac_cap_W"], 1),
        cooling_coil_gross_rated_sensible_heat_ratio=p["ac_shr"], cooling_coil_gross_rated_cop=p["ac_cop_gross"],
        heating_coil_type="Gas", heating_coil_availability_schedule_name="AlwaysOn",
        heating_design_supply_air_temperature=45, heating_coil_capacity=round(p["furnace_output_W"], 1),
        gas_heating_coil_efficiency=round(p["furnace_eff"], 4), gas_heating_coil_parasitic_electric_load=0,
        maximum_outdoor_air_flow_rate=0.0, minimum_outdoor_air_flow_rate=0.0, economizer_type="NoEconomizer",
        supply_fan_placement="BlowThrough", night_cycle_control="StayOff", heat_recovery_type="None",
        dehumidification_control_type="None", humidifier_type="None", return_fan="No")

    # ------------------------------------------------------------ ecobee sensor blending (EMS)
    # The ecobee controls on (1-w)*T_main + w*T_upstairs (w by hour, from data: 1.0 overnight 'Sleep').
    # EnergyPlus controls one zone, so shift the Floor1 setpoints by w*(T_F1 - T_F2) each timestep,
    # which drives the blended temperature to the ecobee setpoint (one-timestep lag on zone temps).
    if p["thermostat"] == "weighted":
        for n, key, var in [("TF1", "Floor1", "Zone Mean Air Temperature"), ("TF2", "Floor2", "Zone Mean Air Temperature"),
                            ("SchH", "Heating Setpoint", "Schedule Value"), ("SchC", "Cooling Setpoint", "Schedule Value"),
                            ("Wup", "Upstairs Weight", "Schedule Value")]:
            add("EnergyManagementSystem:Sensor", n, output_variable_or_output_meter_index_key_name=key,
                output_variable_or_output_meter_name=var)
        add("EnergyManagementSystem:Actuator", "F1HtgSP", actuated_component_unique_name="Floor1",
            actuated_component_type="Zone Temperature Control", actuated_component_control_type="Heating Setpoint")
        add("EnergyManagementSystem:Actuator", "F1ClgSP", actuated_component_unique_name="Floor1",
            actuated_component_type="Zone Temperature Control", actuated_component_control_type="Cooling Setpoint")
        lines = ["SET corr = Wup * (TF1 - TF2)",
                 "IF corr > 3.0", "SET corr = 3.0", "ELSEIF corr < -3.0", "SET corr = -3.0", "ENDIF",
                 "IF SchH > 11.0", "SET F1HtgSP = SchH + corr", "ELSE", "SET F1HtgSP = SchH", "ENDIF",
                 "IF SchC < 34.0", "SET F1ClgSP = SchC + corr", "ELSE", "SET F1ClgSP = SchC", "ENDIF",
                 "IF F1ClgSP < F1HtgSP + 1.0", "SET F1ClgSP = F1HtgSP + 1.0", "ENDIF"]
        add("EnergyManagementSystem:Program", "EcobeeBlend", lines=[{"program_line": l} for l in lines])
        add("EnergyManagementSystem:ProgramCallingManager", "EcobeeBlend Manager",
            energyplus_model_calling_point="BeginTimestepBeforePredictor", programs=[{"program_name": "EcobeeBlend"}])

    # ------------------------------------------------------------ DHW (stand-alone mixed tank)
    add("WaterHeater:Mixed", "DHW Tank", tank_volume=0.189, setpoint_temperature_schedule_name="DHW Setpoint",
        deadband_temperature_difference=5, maximum_temperature_limit=82, heater_control_type="Cycle",
        heater_maximum_capacity=round(p["dhw_cap_W"], 1), heater_fuel_type="NaturalGas",
        heater_thermal_efficiency=p["dhw_eff"], off_cycle_parasitic_fuel_consumption_rate=p["dhw_pilot_W"],
        off_cycle_parasitic_fuel_type="NaturalGas", off_cycle_parasitic_heat_fraction_to_tank=0.8,
        on_cycle_parasitic_fuel_consumption_rate=p["dhw_pilot_W"], on_cycle_parasitic_fuel_type="NaturalGas",
        on_cycle_parasitic_heat_fraction_to_tank=0.8, ambient_temperature_indicator="Zone",
        ambient_temperature_zone_name="Basement", off_cycle_loss_coefficient_to_ambient_temperature=p["dhw_ua_W_K"],
        off_cycle_loss_fraction_to_zone=1.0, on_cycle_loss_coefficient_to_ambient_temperature=p["dhw_ua_W_K"],
        on_cycle_loss_fraction_to_zone=1.0, peak_use_flow_rate=round(dhw_peak, 8),
        use_flow_rate_fraction_schedule_name="DHW Draw", end_use_subcategory="DHW")

    # ------------------------------------------------------------ outputs
    for i, (k, v) in enumerate([("*", "Zone Mean Air Temperature"), ("*", "Site Outdoor Air Drybulb Temperature"),
                                ("*", "Heating Coil NaturalGas Rate"), ("*", "Heating Coil Runtime Fraction"),
                                ("*", "Cooling Coil Runtime Fraction"), ("*", "Cooling Coil Electricity Rate"),
                                ("*", "Fan Electricity Rate"), ("*", "Water Heater NaturalGas Rate"),
                                ("*", "Zone Infiltration Air Change Rate"),
                                ("*", "Zone Combined Outdoor Air Changes per Hour"),
                                ("*", "Zone Thermostat Heating Setpoint Temperature")]):
        add("Output:Variable", f"Output:Variable {i}", key_value=k, variable_name=v, reporting_frequency="Hourly")
    for i, mtr in enumerate(["Electricity:Facility", "NaturalGas:Facility", "Heating:NaturalGas",
                             "WaterSystems:NaturalGas", "Cooling:Electricity", "Fans:Electricity"]):
        add("Output:Meter", f"Output:Meter {i}", key_name=mtr, reporting_frequency="Hourly")
    add("Output:Table:SummaryReports", "Output:Table:SummaryReports 1", reports=[{"report_name": "AllSummary"}])
    add("OutputControl:Table:Style", "OutputControl:Table:Style 1", column_separator="HTML")
    add("Output:SQLite", "Output:SQLite 1", option_type="SimpleAndTabular")
    add("Output:Diagnostics", "Output:Diagnostics 1", diagnostics=[{"key": "DisplayExtraWarnings"}])
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", default="cwec", choices=["cwec", "actual"])
    ap.add_argument("--out", default=str(ROOT / "model/house.epJSON"))
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(build(a.weather), indent=1))
    print("wrote", a.out)
