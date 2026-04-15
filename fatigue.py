# fatigue.py — Phase 1 refactor (drop-in)
from __future__ import annotations


def required_breaks_time(hours: float) -> float:
    t = 0.0
    if hours > 5.25:
        t += 0.25
    if hours > 8.0:
        t += 0.25
    if hours > 11.0:
        t += 0.50
    return t


def solo_standard_shift_ok(work_hours: float) -> int:
    if work_hours > 12.0:
        return 2
    if work_hours > 11.0:
        return 1
    return 0


# Display labels and mapping for the compliance map
RELAY_OPTIONS_DISPLAY = [
    "No relay",
    "Camp at Mother",
    "Midpoint relay",
    "Camp at Daughter",
    "Camp at Mother & Daughter",
    "Camp at Mother, Daughter & Midpoint",
]
RELAY_DISPLAY_TO_LEGACY = {
    "No relay": "No relay",
    "Camp at Mother": "Camp at A",
    "Midpoint relay": "Midpoint relay",
    "Camp at Daughter": "Camp at B",
    "Camp at Mother & Daughter": "Camp at Mother & Daughter",
    "Camp at Mother, Daughter & Midpoint": "Camp at A, B & Midpoint",
}


def required_breaks_time_map(hours: float) -> float:
    # kept separate to avoid cross-import circularity with Streamlit states
    return required_breaks_time(hours)


def solo_standard_shift_ok_map(work_hours: float) -> int:
    return solo_standard_shift_ok(work_hours)


def compliance_code_for(distance_km, speed_kmh, relay_option_display, mode, unload_time_h):
    relay_option = RELAY_DISPLAY_TO_LEGACY.get(relay_option_display, "No relay")
    leg_drive = distance_km / max(speed_kmh, 1e-9)

    def breaks(t):
        return required_breaks_time_map(t)

    if relay_option == "No relay":
        leg_can_fit = (leg_drive + breaks(leg_drive)) <= 12.0
        rt_other_work = unload_time_h if mode == "ThroughRoad" else 0.0
        rt_work = (2 * leg_drive) + (2 * breaks(leg_drive)) + rt_other_work
        rt_code = solo_standard_shift_ok_map(rt_work)
        if (not leg_can_fit) or rt_code == 2:
            return 2
        if rt_code == 1:
            return 1
        return 0
    if relay_option in (
        "Camp at A",
        "Midpoint relay",
        "Camp at B",
        "Camp at Mother & Daughter",
    ):
        leg_AtoB = leg_drive + breaks(leg_drive) + (unload_time_h if mode == "ThroughRoad" else 0.0)
        leg_BtoA = leg_drive + breaks(leg_drive)
        c1 = solo_standard_shift_ok_map(leg_AtoB)
        c2 = solo_standard_shift_ok_map(leg_BtoA)
        return max(c1, c2)
    if relay_option == "Camp at A, B & Midpoint":
        half_leg = leg_drive / 2.0
        b1 = breaks(half_leg)
        w_A_Mid = half_leg + b1
        w_Mid_B = half_leg + b1 + (unload_time_h if mode == "ThroughRoad" else 0.0)
        w_B_Mid = half_leg + b1
        w_Mid_A = half_leg + b1
        codes = [solo_standard_shift_ok_map(x) for x in (w_A_Mid, w_Mid_B, w_B_Mid, w_Mid_A)]
        return max(codes)
    return 2


def best_mode_for(distance_km, speed_kmh, mode, unload_time_h, borderline_ok: bool):
    codes = [
        compliance_code_for(distance_km, speed_kmh, r, mode, unload_time_h)
        for r in RELAY_OPTIONS_DISPLAY
    ]
    if 0 in codes:
        return codes.index(0), 0
    if borderline_ok and 1 in codes:
        return codes.index(1), 1
    if 1 in codes:
        return codes.index(1), 1
    return codes.index(2), 2
