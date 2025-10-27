# calcs.py — Phase 1 refactor (drop-in)
from __future__ import annotations

import math

import streamlit as st

from fatigue import required_breaks_time, solo_standard_shift_ok


def compute_calcs(
    daily_energy_gj: float,
    distance_km_oneway: float,
    speed_kmh: float,
    hours_per_day: float,
    a_cap_gj: float,
    b_cap_gj: float,
    a_count: int,
    b_count: int,
    fill_rate_gjph: float,
    concurrent_bays_per_set: int,
    unload_time_h: float,
    available_bays_A: int,
    available_unload_bays_B: int,
    changeover_overhead_h: float,
    mode: str,
    driver_shift_cap: float,
    crew_change_overhead_h: float,
    delay_buffer_min: float,
    truck_util_target: float,
    fatigue_regime: str,
    relay_location: str,
):
    trailers_per_set = a_count + b_count
    drive_time_h_oneway = distance_km_oneway / max(1e-6, speed_kmh)

    a_fill_time_h = (a_cap_gj / max(1e-6, fill_rate_gjph)) if a_count > 0 else 0.0
    b_fill_time_h = (b_cap_gj / max(1e-6, fill_rate_gjph)) if b_count > 0 else 0.0

    per_set_capacity_gj = a_count * a_cap_gj + b_count * b_cap_gj

    if concurrent_bays_per_set >= trailers_per_set and trailers_per_set > 0:
        fill_time_h = max(a_fill_time_h if a_count else 0.0, b_fill_time_h if b_count else 0.0)
    else:
        total_fill = a_count * a_fill_time_h + b_count * b_fill_time_h
        fill_time_h = total_fill / max(1, concurrent_bays_per_set)

    trailer_set_cycle_h = (
        fill_time_h + unload_time_h + 2 * drive_time_h_oneway + changeover_overhead_h
    )
    truck_cycle_drop_h = 2 * drive_time_h_oneway + crew_change_overhead_h

    sets_per_day = daily_energy_gj / max(1e-6, per_set_capacity_gj)

    trucks_drop = int(
        math.ceil(
            (sets_per_day * truck_cycle_drop_h / hours_per_day) / max(0.01, truck_util_target)
        )
    )
    trucks_through = int(
        math.ceil(
            (sets_per_day * trailer_set_cycle_h / hours_per_day) / max(0.01, truck_util_target)
        )
    )

    sets_in_circulation = int(math.ceil(sets_per_day * trailer_set_cycle_h / hours_per_day))
    a_trailers = sets_in_circulation * a_count
    b_trailers = sets_in_circulation * b_count
    total_trailers = a_trailers + b_trailers

    # Fatigue
    leg_drive = drive_time_h_oneway
    leg_breaks = required_breaks_time(leg_drive)
    rt_other_work = unload_time_h if mode == "ThroughRoad" else 0.0
    round_trip_driver_work = (
        (2 * leg_drive) + (2 * leg_breaks) + rt_other_work + crew_change_overhead_h
    )
    leg_can_fit = (leg_drive + leg_breaks) <= driver_shift_cap
    rt_compliance_code = solo_standard_shift_ok(round_trip_driver_work)

    cycle_time = truck_cycle_drop_h if mode == "DropAndPull" else trailer_set_cycle_h
    drivers_per_truck = max(1, int(math.ceil(cycle_time / driver_shift_cap)))

    if relay_location != "No relay":
        if relay_location == "Camp at A, B & Midpoint":
            half_leg_drive = drive_time_h_oneway / 2.0
            half_leg_breaks = required_breaks_time(half_leg_drive)
            leg_A_to_Mid_work = half_leg_drive + half_leg_breaks
            leg_Mid_to_B_work = (
                half_leg_drive + half_leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0)
            )
            leg_B_to_Mid_work = half_leg_drive + half_leg_breaks
            leg_Mid_to_A_work = half_leg_drive + half_leg_breaks
            codes = [
                solo_standard_shift_ok(x)
                for x in (
                    leg_A_to_Mid_work,
                    leg_Mid_to_B_work,
                    leg_B_to_Mid_work,
                    leg_Mid_to_A_work,
                )
            ]
            leg_can_fit = all(c != 2 for c in codes)
            rt_compliance_code = max(codes)
            drivers_per_truck = max(2, int(math.ceil(cycle_time / driver_shift_cap)))
        else:
            leg_AtoB_work = (
                leg_drive + leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0)
            )
            leg_BtoA_work = leg_drive + leg_breaks
            leg_AtoB_code = solo_standard_shift_ok(leg_AtoB_work)
            leg_BtoA_code = solo_standard_shift_ok(leg_BtoA_work)
            leg_can_fit = (leg_AtoB_code != 2) and (leg_BtoA_code != 2)
            rt_compliance_code = max(leg_AtoB_code, leg_BtoA_code)
            drivers_per_truck = max(2, int(math.ceil(cycle_time / driver_shift_cap)))
    else:
        if not leg_can_fit:
            drivers_per_truck = max(drivers_per_truck, 2)

    active_trucks = trucks_drop if mode == "DropAndPull" else trucks_through
    drivers_on_duty = drivers_per_truck * active_trucks
    rostered_drivers = int(
        math.ceil(drivers_on_duty * max(1.0, st.session_state.get("relief_factor", 1.3)))
    )

    # Bay utilization (A)
    bay_hours_per_day_A = daily_energy_gj / max(1e-6, fill_rate_gjph)
    if st.session_state.get("mother_swap_occupies_bay", False):
        bay_hours_per_day_A += sets_per_day * max(changeover_overhead_h, 0.0)
    avg_bays_needed_A = bay_hours_per_day_A / hours_per_day
    if available_bays_A <= 0:
        bay_utilization_A = float("inf") if avg_bays_needed_A > 0 else 0.0
        bay_flag_A = "NO BAYS DEFINED" if avg_bays_needed_A > 0 else "OK"
    else:
        bay_utilization_A = avg_bays_needed_A / available_bays_A
        bay_flag_A = "OVER 100% - add bays or reduce demand" if bay_utilization_A > 1.0 else "OK"

    # Unload bay utilization (B)
    try:
        _unload_rate_gjph = float(st.session_state.get("unload_rate_gjph", 0.0))
    except Exception:
        _unload_rate_gjph = 0.0
    unload_hours_per_day_B = daily_energy_gj / max(_unload_rate_gjph, 1e-9)
    if st.session_state.get("daughter_swap_occupies_bay", False):
        _swap_h = float(st.session_state.get("changeover_overhead_h_daughter", 0.0))
        unload_hours_per_day_B += sets_per_day * max(_swap_h, 0.0)
    avg_unload_bays_needed_B = unload_hours_per_day_B / hours_per_day
    if available_unload_bays_B <= 0:
        unload_utilization_B = float("inf") if avg_unload_bays_needed_B > 0 else 0.0
        unload_flag_B = "NO BAYS DEFINED" if avg_unload_bays_needed_B > 0 else "OK"
    else:
        unload_utilization_B = avg_unload_bays_needed_B / available_unload_bays_B
        unload_flag_B = (
            "OVER 100% - add bays or reduce unload time" if unload_utilization_B > 1.0 else "OK"
        )

    energy_per_truckhour = (
        (per_set_capacity_gj / (2 * drive_time_h_oneway))
        if mode == "DropAndPull"
        else (per_set_capacity_gj / max(1e-6, cycle_time))
    )

    truck_km_per_set = 2.0 * distance_km_oneway
    trailer_km_per_set = truck_km_per_set * trailers_per_set
    truck_km_per_day = sets_per_day * truck_km_per_set
    trailer_km_per_day = sets_per_day * trailer_km_per_set

    if not leg_can_fit or rt_compliance_code == 2:
        compliance_flag = 2
    elif rt_compliance_code == 1:
        compliance_flag = 1
    else:
        compliance_flag = 0

    return {
        "drive_time_h_oneway": drive_time_h_oneway,
        "a_fill_time_h": a_fill_time_h,
        "b_fill_time_h": b_fill_time_h,
        "per_set_capacity_gj": per_set_capacity_gj,
        "fill_time_h": fill_time_h,
        "trailer_set_cycle_h": trailer_set_cycle_h,
        "truck_cycle_drop_h": truck_cycle_drop_h,
        "sets_per_day": sets_per_day,
        "trucks_drop": trucks_drop,
        "trucks_through": trucks_through,
        "sets_in_circulation": sets_in_circulation,
        "a_trailers": a_trailers,
        "b_trailers": b_trailers,
        "total_trailers": total_trailers,
        "active_trucks": active_trucks,
        "drivers_on_duty": drivers_on_duty,
        "drivers_per_truck": drivers_per_truck,
        "rostered_drivers": rostered_drivers,
        "avg_bays_needed_A": avg_bays_needed_A,
        "bay_utilization_A": bay_utilization_A,
        "bay_flag_A": bay_flag_A,
        "avg_unload_bays_needed_B": avg_unload_bays_needed_B,
        "unload_utilization_B": unload_utilization_B,
        "unload_flag_B": unload_flag_B,
        "energy_per_truckhour": energy_per_truckhour,
        "round_trip_driver_work": round_trip_driver_work,
        "rt_compliance_code": rt_compliance_code,
        "compliance_flag": compliance_flag,
        "fatigue_regime_note": st.session_state.get(
            "fatigue_regime", "Standard Hours (Solo Drivers)"
        ),
        "relay_location": st.session_state.get("relay_location", "No relay"),
        "leg_breaks_h": leg_breaks,
        "leg_AtoB_work": (
            drive_time_h_oneway + leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0)
        ),
        "leg_BtoA_work": (drive_time_h_oneway + leg_breaks),
        "truck_km_per_day": truck_km_per_day,
        "trailer_km_per_day": trailer_km_per_day,
        "trailers_per_set": trailers_per_set,
        "unload_time_h": unload_time_h,
    }
