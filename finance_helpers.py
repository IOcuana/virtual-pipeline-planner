# finance_helpers.py — Scenarios LC/LB/Margin & transport breakdown (v15.6)

from __future__ import annotations

from typing import Optional, Dict, Tuple
import math
import streamlit as st

from constants import DAYS_PER_YEAR, ROADTRAIN_MAP, DOLLIES_PER_COMBO
from finance import capital_recovery_factor
from calcs import compute_calcs


def _get_float(key: str, default: float) -> float:
    try:
        return float(st.session_state.get(key, default))
    except Exception:
        return float(default)


def _get_int(key: str, default: int) -> int:
    try:
        return int(st.session_state.get(key, default))
    except Exception:
        return int(default)


def _planner_caps_for_use() -> Tuple[float, float]:
    a = _get_float("a_cap_usable_gj", _get_float("a_cap_calc_gj", 0.0))
    b = _get_float("b_cap_usable_gj", _get_float("b_cap_calc_gj", 0.0))
    return a, b

def _current_driver_shift_cap() -> float:
    """
    Map the selected fatigue regime to the driver shift cap (hours).
    """
    import streamlit as st
    regime = st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)")
    normalised = {
        "Standard Hours (Solo Drivers)": "Standard Hours (Solo Drivers)",
        "Basic Fatigue Management (BFM)": "Basic Fatigue Management (BFM)",
        "Advanced Fatigue Management (AFM)": "Advanced Fatigue Management (AFM)",
    }.get(regime, "Standard Hours (Solo Drivers)")
    return {
        "Standard Hours (Solo Drivers)": 12.0,
        "Basic Fatigue Management (BFM)": 14.0,
        "Advanced Fatigue Management (AFM)": 16.0,
    }.get(normalised, 12.0)

def compute_levelized_metrics_for_scenario(
    daily_energy_gj_override: Optional[float] = None,
    distance_km_override: Optional[float] = None,
    fill_rate_gjph_override: Optional[float] = None,
    unload_time_h_override: Optional[float] = None,
) -> Dict[str, float]:

    daily_gj_plan = float(
        daily_energy_gj_override
        if daily_energy_gj_override is not None
        else st.session_state.get("daily_energy_gj", 0.0)
    )
    utilisation_frac = float(st.session_state.get("utilisation_frac", 1.0))
    eff_daily_gj = float(
        st.session_state.get("effective_daily_gj", daily_gj_plan * utilisation_frac)
    )
    annual_energy_delivered_gj = eff_daily_gj * DAYS_PER_YEAR

    distance_km = float(
        distance_km_override
        if distance_km_override is not None
        else st.session_state.get("distance_km_oneway", 900.0)
    )
    speed_kmh = _get_float("speed_kmh", 75.0)
    hours_per_day = _get_float("hours_per_day", 24.0)

    combo = st.session_state.get("combo", "A-triple")
    a_count = int(ROADTRAIN_MAP.get(combo, {"A": 0}).get("A", 0))
    b_count = int(ROADTRAIN_MAP.get(combo, {"B": 0}).get("B", 0))

    a_cap_gj, b_cap_gj = _planner_caps_for_use()

    fill_rate_gjph = (
        float(fill_rate_gjph_override)
        if fill_rate_gjph_override is not None
        else _get_float("fill_rate_gjph", 41.67)
    )
    unload_time_h = (
        float(unload_time_h_override)
        if unload_time_h_override is not None
        else _get_float("unload_time_h", 12.0)
    )

    concurrent_bays_per_set = _get_int("concurrent_bays_per_set", 3)
    available_bays_A = _get_int("available_bays_A", 12)
    available_unload_bays_B = _get_int("available_unload_bays_B", 4)
    changeover_overhead_h = _get_float("changeover_overhead_h", 0.25)

    mode = st.session_state.get("mode", "DropAndPull")
    driver_shift_cap = _get_float("driver_shift_cap", 12.0)
    crew_change_overhead_h = _get_float("crew_change_overhead_h", 0.25)
    delay_buffer_min = _get_float("delay_buffer_min", 30.0)
    truck_util_target = _get_float("truck_util_target", 0.85)
    fatigue_regime = st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)")
    relay_location = st.session_state.get("relay_location", "No relay")

    calcs = compute_calcs(
        daily_energy_gj=daily_gj_plan,
        distance_km_oneway=distance_km,
        speed_kmh=speed_kmh,
        hours_per_day=hours_per_day,
        a_cap_gj=a_cap_gj,
        b_cap_gj=b_cap_gj,
        a_count=a_count,
        b_count=b_count,
        fill_rate_gjph=fill_rate_gjph,
        concurrent_bays_per_set=concurrent_bays_per_set,
        unload_time_h=unload_time_h,
        available_bays_A=available_bays_A,
        available_unload_bays_B=available_unload_bays_B,
        changeover_overhead_h=changeover_overhead_h,
        mode=mode,
        driver_shift_cap=driver_shift_cap,
        crew_change_overhead_h=crew_change_overhead_h,
        delay_buffer_min=delay_buffer_min,
        truck_util_target=truck_util_target,
        fatigue_regime=fatigue_regime,
        relay_location=relay_location,
    )

    active_trucks = int(calcs.get("active_trucks", 0))
    a_trailers = int(calcs.get("a_trailers", 0))
    b_trailers = int(calcs.get("b_trailers", 0))
    combos_in_circ = int(calcs.get("sets_in_circulation", 0))
    truck_km_per_day = float(calcs.get("truck_km_per_day", 0.0))
    trailer_km_per_day = float(calcs.get("trailer_km_per_day", 0.0))

    # Prefer the first enabled finance plan (e.g., Plan A) if available; fallback to prior session key / default 6%
    _plans = st.session_state.get("finance_plans", [])
    _enabled = [p for p in _plans if p.get("enabled", False)]
    if _enabled:
        interest_rate_percent = float(_enabled[0].get("rate_pct", 6.0))
    else:
        # Back-compat: allow the old key if user set it; otherwise use 6.0%
        interest_rate_percent = _get_float("interest_rate_percent", 6.0)
    interest_rate = interest_rate_percent / 100.0

    # Prefer the term (NPER months) from the first enabled finance plan; fallback to prior key / 60 months
    if _enabled:
        asset_life_months = int(float(_enabled[0].get("nper_months", 60)))
    else:
        asset_life_months = int(_get_float("asset_life_months", 60))

    crf = capital_recovery_factor(interest_rate, asset_life_months / 12.0)

    ms_comp = _get_float("ms_comp", 50_000_000.0)
    ms_utils = _get_float("ms_utils", 0.0)
    ms_gcond = _get_float("ms_gcond", 0.0)
    ms_bay_cost = _get_float("ms_bay_cost", 500_000.0)
    ms_bays_total = ms_bay_cost * available_bays_A
    ms_capex = ms_comp + ms_utils + ms_gcond + ms_bays_total

    ms_opex_pct_percent = _get_float("ms_opex_pct_percent", 5.0)
    ms_opex_year = ms_capex * (ms_opex_pct_percent / 100.0)

    ms_energy_method = st.session_state.get("ms_energy_method", "Fuel price ($/GJ fuel)")
    if str(ms_energy_method).startswith("Fuel"):
        ms_fuel_price_per_gj = _get_float("ms_fuel_price_per_gj", 0.0)
        ms_energy_intensity_gj_per_gj = _get_float("ms_energy_intensity_gj_per_gj", 0.02)
        ms_energy_cost_year = (
            annual_energy_delivered_gj * ms_energy_intensity_gj_per_gj * ms_fuel_price_per_gj
        )
    else:
        ms_elec_price_per_kwh = _get_float("ms_elec_price_per_kwh", 0.20)
        ms_kwh_per_gj = _get_float("ms_kwh_per_gj", 5.6)
        ms_energy_cost_year = annual_energy_delivered_gj * ms_kwh_per_gj * ms_elec_price_per_kwh

    ds_bay_cost = _get_float("ds_bay_cost", 100_000.0)
    ds_capex = available_unload_bays_B * ds_bay_cost
    ds_opex_pct_percent = _get_float("ds_opex_pct_percent", 5.0)
    ds_opex_year = ds_capex * (ds_opex_pct_percent / 100.0)

    truck_capex_unit = _get_float("truck_capex_unit", 550_000.0)
    a_trailer_capex_unit = _get_float("a_trailer_capex_unit", 800_000.0)
    b_trailer_capex_unit = _get_float("b_trailer_capex_unit", 650_000.0)
    dolly_capex_unit = _get_float("dolly_capex_unit", 50_000.0)
    dollies_per_combo = int(DOLLIES_PER_COMBO.get(combo, 0))
    total_dollies = int(math.ceil(combos_in_circ * dollies_per_combo))

    capex_total_all = (
        active_trucks * truck_capex_unit
        + (a_trailers * a_trailer_capex_unit + b_trailers * b_trailer_capex_unit + total_dollies * dolly_capex_unit)
        + ms_capex
        + ds_capex
    )

    if st.session_state.get("maint_basis_trk", "$/km") == "$/km":
        trk_maint_per_km = _get_float("trk_maint_per_km", 0.36)
        trk_maint_year = trk_maint_per_km * truck_km_per_day * DAYS_PER_YEAR
    else:
        trk_maint_year = _get_float("trk_maint_year", 300_000.0)

    if st.session_state.get("maint_basis_trl", "$/km") == "$/km":
        trl_maint_per_km = _get_float("trl_maint_per_km", 0.10)
        trl_maint_year = trl_maint_per_km * trailer_km_per_day * DAYS_PER_YEAR
    else:
        trl_maint_year = _get_float("trl_maint_year", 200_000.0)

    truck_km_per_litre = _get_float("truck_km_per_litre", 1.8)
    diesel_price_per_l = _get_float("diesel_price_per_l", 2.00)
    route_days = _get_float(key="route_days_per_year", default=365.0)
    litres_per_day = truck_km_per_day / max(1e-6, truck_km_per_litre)
    fuel_cost_year = litres_per_day * diesel_price_per_l * route_days

    wage_unit = st.session_state.get("wage_unit", "$/day")
    wage_per_driver = _get_float(key="wage_per_driver", default=600.0)
    fifo_per_driver_day = _get_float(key="fifo_per_driver_day", default=120.0)
    camp_per_driver_day = _get_float(key="camp_per_driver_day", default=150.0)
    working_days = _get_float(key="working_days_per_year", default=200.0)  # per-driver availability
    drivers_on_duty = float(calcs.get("drivers_on_duty", 0))
    relief_factor = _get_float(key="relief_factor", default=1.00)

    # Per-driver annual wage from selected unit
    driver_shift_cap = float(_current_driver_shift_cap())
    if wage_unit == "$/day":
        annual_wage_per_driver = wage_per_driver * working_days
    elif wage_unit == "$/hr":
        annual_wage_per_driver = wage_per_driver * driver_shift_cap * working_days
    else:  # "$/year"
        annual_wage_per_driver = wage_per_driver

    annual_fifo_camp_per_driver = (fifo_per_driver_day + camp_per_driver_day) * working_days
    per_driver_annual_total = annual_wage_per_driver + annual_fifo_camp_per_driver

    # Availability & relief → staffed headcount
    staff_drivers = math.ceil(drivers_on_duty * relief_factor * (365.0 / max(working_days, 1e-9)))

    driver_cost_year = per_driver_annual_total * staff_drivers

    insurance_year = _get_float("insurance_year", 300_000.0)
    misc_year = _get_float("misc_year", 150_000.0)
    reg_truck_per_year = _get_float("reg_truck_per_year", 10_000.0)
    reg_trailer_per_year = _get_float("reg_trailer_per_year", 2_000.0)
    registration_year = reg_truck_per_year * active_trucks + reg_trailer_per_year * (a_trailers + b_trailers)

    other_opex_year = insurance_year + misc_year + registration_year

    transport_ops_year = (
        trk_maint_year + trl_maint_year + fuel_cost_year + driver_cost_year + other_opex_year
    )

    infra_opex_year = ms_opex_year + ds_opex_year
    station_energy_year = ms_energy_cost_year
    annuitized_capex_year = capex_total_all * crf
    total_annual_cost = annuitized_capex_year + infra_opex_year + station_energy_year + transport_ops_year

    benefit_gas_per_gj = _get_float("benefit_gas_per_gj", _get_float("gas_revenue_per_gj", 0.0))
    carbon_benefit_per_gj = _get_float("carbon_benefit_per_gj", 0.0)
    other_benefit_per_gj = _get_float("other_benefit_per_gj", 0.0)
    lb_per_gj = max(0.0, benefit_gas_per_gj + carbon_benefit_per_gj + other_benefit_per_gj)

    denom_gj_year = max(annual_energy_delivered_gj, 1e-9)

    lc_per_gj = total_annual_cost / denom_gj_year
    transport_lc_per_gj = transport_ops_year / denom_gj_year
    margin_per_gj = lb_per_gj - lc_per_gj

    st.session_state["lc_per_gj"] = float(lc_per_gj)
    st.session_state["lb_per_gj"] = float(lb_per_gj)
    st.session_state["margin_per_gj"] = float(margin_per_gj)
    st.session_state["transport_lc_per_gj"] = float(transport_lc_per_gj)

    return {
        "lc_per_gj": float(lc_per_gj),
        "lb_per_gj": float(lb_per_gj),
        "margin_per_gj": float(margin_per_gj),
        "transport_lc_per_gj": float(transport_lc_per_gj),
    }
