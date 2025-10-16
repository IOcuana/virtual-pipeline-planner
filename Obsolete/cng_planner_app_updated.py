import math
import numpy as np
import pandas as pd
import streamlit as st

# Optional CoolProp for real-gas properties
try:
    from CoolProp.CoolProp import PropsSI  # type: ignore
    COOLPROP_OK = True
except Exception:
    COOLPROP_OK = False

# Optional Matplotlib for compliance map plotting (safety-checked)
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    MPL_OK = True
except Exception:
    MPL_OK = False

st.set_page_config(page_title="Virtual Pipeline Planner", layout="wide")

# ------------------------------
# Constants & helpers
# ------------------------------
R_UNIV = 8.314462618  # J/mol-K
MOLAR_MASS = {"Methane": 0.01604246, "Hydrogen": 0.00201588}  # kg/mol
M3_PER_FT3 = 0.028316846592  # m^3 per ft^3
FT3_PER_M3 = 1.0 / M3_PER_FT3
GJ_PER_MMBTU = 1.055056  # 1 MMBtu ≈ 1.055056 GJ

def density_kg_per_m3(T_K: float, P_Pa: float, gas_label: str) -> float:
    """Return density using CoolProp when available; otherwise ideal-gas fallback."""
    if COOLPROP_OK:
        try:
            return float(PropsSI("D", "T", T_K, "P", P_Pa, gas_label))
        except Exception:
            pass
    # Ideal-gas fallback
    molar_mass = MOLAR_MASS.get(gas_label, MOLAR_MASS["Methane"])
    return (P_Pa * molar_mass) / (R_UNIV * T_K)

# ----------------------------------------
# Road-train mapping (A = A-trailer, B = B-trailer)
# ----------------------------------------
ROADTRAIN_MAP = {
    "Semitrailer": {"A": 1, "B": 0},
    "B-double": {"A": 1, "B": 1},
    "A-double": {"A": 2, "B": 0},
    "B-triple": {"A": 1, "B": 2},
    "A-triple": {"A": 3, "B": 0},
    "AB-triple": {"A": 2, "B": 1},
    "AAB-quad": {"A": 3, "B": 1},
    "ABB-quad": {"A": 2, "B": 2},
    "BAB-quad": {"A": 2, "B": 2},
}

# ------------------------------
# Sidebar inputs
# ------------------------------
with st.sidebar:
    st.header("Operating Mode & Demand")
    mode = st.selectbox("Mode", ["DropAndPull", "ThroughRoad"])
    project_months = st.number_input("Project duration [months]", min_value=1, value=12, step=1)
    hours_per_day = st.number_input("Operating hours per day", min_value=1, max_value=24, value=24, step=1)

    daily_energy_gj = st.number_input(
        "Daily energy to transport (GJ/day)",
        min_value=0.0,
        value=10000.00,
        step=100.0,
        format="%.2f",
        help="Energy delivered per day in gigajoules."
    )

    # Reference condition for conversions (affects kg/day and display order)
    ref_choice = st.radio(
        "Conversion reference",
        ["Normal (20 °C, 1.01325 barₐ)", "Standard (15 °C, 1.01325 barₐ)"],
        index=1,
        help="Pick which condition the sidebar volumetric & mass conversions use."
    )
    st.session_state["ref_choice"] = ref_choice

    # Conversions block — uses current HV & gas from session_state if present, else sensible defaults
    default_hv = 50032000.0  # Methane LHV [J/kg]
    gas_for_conv = st.session_state.get("gas_type_selected", "Methane")
    hv_for_conv = float(st.session_state.get("HV_J_per_kg", default_hv))

    # Normal conditions: 20 °C, 1.01325 bar(a)
    Tn_K = 20.0 + 273.15
    Pn_Pa = 1.01325e5
    # Standard conditions: 15 °C, 1.01325 bar(a)
    Ts_K = 15.0 + 273.15
    Ps_Pa = 1.01325e5

    def rho_at(T_K, P_Pa):
        label = "Hydrogen" if gas_for_conv == "Hydrogen" else "Methane"
        return density_kg_per_m3(T_K, P_Pa, label)

    rho_N = rho_at(Tn_K, Pn_Pa)
    rho_S = rho_at(Ts_K, Ps_Pa)

    # Energy per unit volume at N and S (MJ/m3)
    MJ_per_m3_N = rho_N * (hv_for_conv / 1e6)  # (kg/m3) * (MJ/kg)
    MJ_per_m3_S = rho_S * (hv_for_conv / 1e6)

    # Volumetric flows (energy → volume at N/S)
    Nm3_per_day = (daily_energy_gj * 1000.0) / max(MJ_per_m3_N, 1e-12)
    Sm3_per_day = (daily_energy_gj * 1000.0) / max(MJ_per_m3_S, 1e-12)
    Nft3_per_day = Nm3_per_day * FT3_PER_M3
    Sft3_per_day = Sm3_per_day * FT3_PER_M3
    MMSCFD = Sft3_per_day / 1e6

    # Mass from chosen condition: kg/day = ρ × volumetric_flow
    if ref_choice.startswith("Standard"):
        kg_per_day = rho_S * Sm3_per_day
    else:
        kg_per_day = rho_N * Nm3_per_day

    # Energy alt units
    MMBTU_per_day = daily_energy_gj / GJ_PER_MMBTU

    # Display conversions (with kg/day tooltip)
    st.caption(f"≈ **{daily_energy_gj / 1000:.3f} TJ/day**")
    if ref_choice.startswith("Standard"):
        st.caption(f"≈ **{Sm3_per_day:,.0f} Sm³/day**, **{MMSCFD:,.3f} MMSCF/D**  _(S: 15 °C, 1.01325 barₐ)_")
        st.caption(f"≈ **{Nm3_per_day:,.0f} Nm³/day**, **{Nft3_per_day:,.0f} Nft³/day**  _(N: 20 °C, 1.01325 barₐ)_")
    else:
        st.caption(f"≈ **{Nm3_per_day:,.0f} Nm³/day**, **{Nft3_per_day:,.0f} Nft³/day**  _(N: 20 °C, 1.01325 barₐ)_")
        st.caption(f"≈ **{Sm3_per_day:,.0f} Sm³/day**, **{MMSCFD:,.3f} MMSCF/D**  _(S: 15 °C, 1.01325 barₐ)_")

    kg_tip = (
        "Mass derived from Standard conditions (15 °C, 1.01325 barₐ): ρ(S) × Sm³/day"
        if ref_choice.startswith("Standard")
        else "Mass derived from Normal conditions (20 °C, 1.01325 barₐ): ρ(N) × Nm³/day"
    )
    st.markdown(
        f"""<span title="{kg_tip}">≈ <b>{kg_per_day:,.0f} kg/day</b> <sup>ⓘ</sup></span>
             &nbsp; • &nbsp; <b>{MMBTU_per_day:,.0f} MMBTU/day</b>""",
        unsafe_allow_html=True
    )

    st.divider()

    st.header("Route")
    st.caption("A = Gas production location, B = Gas delivery location")
    distance_km_oneway = st.number_input("Distance A→B (km)", min_value=0.01, value=900.0, step=10.0)
    speed_kmh = st.number_input(
        "Average driving speed (km/h)",
        min_value=0.01,
        max_value=110.0,
        value=75.0,
        step=1.0
    )

    st.header("Road Train Combination")
    combo_names = list(ROADTRAIN_MAP.keys())
    # Default to A-triple
    default_idx = combo_names.index("A-triple") if "A-triple" in ROADTRAIN_MAP else 0
    combo = st.selectbox("Combination", combo_names, index=default_idx, help="A = A-trailer, B = B-trailer")
    a_count = ROADTRAIN_MAP[combo]["A"]
    b_count = ROADTRAIN_MAP[combo]["B"]
    trailers_per_set = a_count + b_count
    st.caption(f"Selected: **{combo}** → A: **{a_count}**, B: **{b_count}** (Total {trailers_per_set})")
    st.session_state["combo"] = combo  # so Planner/Scenarios can reference

    st.header("Terminal Ops (A & B)")
    fill_rate_gjph = st.number_input("Per-bay fill rate (GJ/h)", min_value=0.001, value=41.67, step=0.01, format="%.2f")
    concurrent_bays_per_set = st.number_input(
        "Concurrent bays per set (trailers filled in parallel)",
        min_value=1, max_value=8, value=min(3, trailers_per_set) if trailers_per_set > 0 else 1, step=1
    )
    unload_time_h = st.number_input("Unload time per set at B (h)", min_value=0.0, value=12.0, step=0.25)
    available_bays = st.number_input("Available filling bays at A (count)", min_value=0, value=12, step=1)
    changeover_overhead_h = st.number_input("Trailer swap/handling overhead per visit (h)", min_value=0.0, value=0.25, step=0.05)

    st.header("Drivers & Compliance")
    fatigue_regime = st.selectbox(
        "Fatigue regime",
        ["Standard Hours (Solo Drivers)", "Basic Fatigue Management (BFM) – placeholder", "Advanced Fatigue Management (AFM) – placeholder"],
        index=0
    )
    relay_location = st.selectbox(
        "Relay location",
        ["No relay", "Camp at A", "Midpoint relay", "Camp at B", "Camp at A & B", "Camp at A, B & Midpoint"],
        index=0,
        help=("Where drivers swap for legs. "
              "‘Camp at A & B’ alternates between end camps across shifts. "
              "‘Camp at A, B & Midpoint’ splits each direction into two shorter legs via a midpoint camp.")
    )
    st.session_state["fatigue_regime"] = fatigue_regime
    st.session_state["relay_location"] = relay_location

    driver_shift_cap = 12.0  # hours total in any 24h (Standard Hours)
    driver_min_cont_rest = 7.0  # not explicitly modeled in cycle builder

    crew_change_overhead_h = st.number_input("Crew change overhead at swap (h)", min_value=0.0, value=0.25, step=0.05)
    delay_buffer_min = st.number_input("Target compliance buffer per leg (min)", min_value=0, value=30, step=5)
    relief_factor = st.number_input("Driver relief factor", min_value=1.0, max_value=2.0, value=1.3, step=0.05)

    st.header("Utilization Target")
    truck_util_target = st.number_input("Truck utilization (0–1)", min_value=0.01, max_value=1.0, value=0.85, step=0.01)

# ---------------------------------------
# Tabs: Payload & Capacity | Planner | Scenarios
# ---------------------------------------
tabs = st.tabs(["Payload & Capacity", "Planner", "Scenarios"])

# ---------------------------------------
# PAYLOAD & CAPACITY (first tab)
# ---------------------------------------
with tabs[0]:
    st.subheader("Payload & Capacity (A- and B-trailers)")

    colp1, colp2 = st.columns(2, vertical_alignment="top")

    # ----------------- LEFT COLUMN: Thermofluid inputs -----------------
    with colp1:
        st.markdown("**Thermofluid inputs**")
        gas_type = st.selectbox("Gas Type", ["Hydrogen", "Methane"], index=1)
        st.session_state["gas_type_selected"] = gas_type  # expose to sidebar conversions

        PW_bar_g = st.number_input("Working Pressure [bar(g)]", value=300.0, min_value=0.0, step=1.0, key="phys_PW")

        Pmin_bar_g = st.number_input(
            "Minimum Cylinder Pressure [bar(g)]",
            min_value=0.0,
            max_value=PW_bar_g,  # enforce Pmin ≤ PW at the widget
            value=min(10.0, PW_bar_g),
            step=1.0,
            help="Pressure below which the trailer is considered empty for dispatch purposes."
        )
        if Pmin_bar_g > PW_bar_g + 1e-9:
            st.error("Minimum Cylinder Pressure cannot exceed Working Pressure.")
            Pmin_bar_g = PW_bar_g

        # Heating value basis (default = Gross)
        hv_options = ["Gross Heating Value [J/kg]", "Net Heating Value [J/kg]"]
        hv_basis = st.selectbox("Heating value basis", hv_options, index=0)
        hv_key = "Gross" if hv_basis.startswith("Gross") else "Net"
        st.session_state["hv_basis_key"] = hv_key

        # Gas temperature
        T_C = st.number_input("Gas temperature (°C)", min_value=-20.0, max_value=65.0, value=15.0, step=1.0, key="phys_T")

        # Auto-sync HV
        auto_sync_hv = st.checkbox(
            "Auto-sync heating value to Gas Type & Basis",
            value=True,
            help="When on, the HV value updates automatically when Gas Type or Basis changes."
        )

        DEFAULT_HV = {
            "Methane": {"Gross": 55571000.0,  "Net": 50032000.0},
            "Hydrogen": {"Gross": 141950000.0, "Net": 119910000.0},
        }

        if "last_gas_type" not in st.session_state:
            st.session_state["last_gas_type"] = gas_type
        if "last_hv_key" not in st.session_state:
            st.session_state["last_hv_key"] = hv_key
        if "HV_J_per_kg" not in st.session_state:
            st.session_state["HV_J_per_kg"] = DEFAULT_HV[gas_type][hv_key]

        if auto_sync_hv and (gas_type != st.session_state["last_gas_type"] or hv_key != st.session_state["last_hv_key"]):
            st.session_state["HV_J_per_kg"] = DEFAULT_HV[gas_type][hv_key]

        st.session_state["last_gas_type"] = gas_type
        st.session_state["last_hv_key"] = hv_key

        hv_col1, hv_col2 = st.columns([3, 1])
        with hv_col2:
            def _reset_hv():
                st.session_state["HV_J_per_kg"] = DEFAULT_HV[gas_type][hv_key]
            st.button("Reset", on_click=_reset_hv)
        with hv_col1:
            st.number_input(
                "Selected heating value [J/kg] (auto)",
                min_value=0.0,
                value=float(st.session_state.get("HV_J_per_kg", DEFAULT_HV[gas_type][hv_key])),
                step=1000.0,
                format="%.0f",
                key="HV_J_per_kg"
            )

    # ----------------- RIGHT COLUMN: Composition & volumes -----------------
    with colp2:
        st.markdown("**Composition & usable volume**")
        gas_purity = st.number_input(
            "Fuel gas purity (mole fraction 0–1)",
            value=1.00, min_value=0.0, max_value=1.0, step=0.01
        )

        # --- A-trailer ---
        st.markdown("**A-trailer cylinder set**")
        a_cyl_count = st.number_input("Number of cylinders per A-trailer", min_value=1, value=1, step=1)
        a_cyl_water_L = st.number_input(
            "Cylinder water capacity [Litres] (A)",
            min_value=0.01,
            value=41000.0,
            step=100.0,
            format="%.1f"
        )
        a_trailer_vol_m3 = (a_cyl_count * a_cyl_water_L) / 1000.0
        st.caption(f"Computed A-trailer internal gas volume: **{a_trailer_vol_m3:,.2f} m³**")

        st.markdown("---")

        # --- B-trailer ---
        st.markdown("**B-trailer cylinder set**")
        b_cyl_count = st.number_input("Number of cylinders per B-trailer", min_value=1, value=1, step=1)
        b_cyl_water_L = st.number_input(
            "Cylinder water capacity [Litres] (B)",
            min_value=0.01,
            value=17100.0,
            step=100.0,
            format="%.1f"
        )
        b_trailer_vol_m3 = (b_cyl_count * b_cyl_water_L) / 1000.0
        st.caption(f"Computed B-trailer internal gas volume: **{b_trailer_vol_m3:,.2f} m³**")

    # --- Thermo: compute capacities ---
    P_abs_bar    = st.session_state["phys_PW"] + 1.01325
    Pmin_abs_bar = Pmin_bar_g + 1.01325
    P_Pa   = P_abs_bar   * 1e5
    Pmin_Pa = Pmin_abs_bar * 1e5
    T_K = st.session_state["phys_T"] + 273.15

    gas_label = "Hydrogen" if gas_type == "Hydrogen" else "Methane"
    rho_full = density_kg_per_m3(T_K, P_Pa,   gas_label)
    rho_min  = density_kg_per_m3(T_K, Pmin_Pa, gas_label)

    HV_J_per_kg = float(st.session_state.get("HV_J_per_kg", DEFAULT_HV[gas_type][hv_key]))

    # Total capacity @ PW (energy)
    a_cap_calc_gj = rho_full * a_trailer_vol_m3 * gas_purity * (HV_J_per_kg / 1e9)
    b_cap_calc_gj = rho_full * b_trailer_vol_m3 * gas_purity * (HV_J_per_kg / 1e9)
    # Total capacity @ PW (mass)
    a_cap_calc_kg = (a_cap_calc_gj * 1e9) / max(HV_J_per_kg, 1e-12)
    b_cap_calc_kg = (b_cap_calc_gj * 1e9) / max(HV_J_per_kg, 1e-12)

    # Usable capacity (PW → Pmin) (energy)
    delta_rho = max(rho_full - rho_min, 0.0)
    a_cap_usable_gj = delta_rho * a_trailer_vol_m3 * gas_purity * (HV_J_per_kg / 1e9)
    b_cap_usable_gj = delta_rho * b_trailer_vol_m3 * gas_purity * (HV_J_per_kg / 1e9)
    # Usable capacity (mass)
    a_cap_usable_kg = (a_cap_usable_gj * 1e9) / max(HV_J_per_kg, 1e-12)
    b_cap_usable_kg = (b_cap_usable_gj * 1e9) / max(HV_J_per_kg, 1e-12)

    # --- Combination totals ---
    comb_key = st.session_state.get("combo", "A-triple")
    comb_counts = ROADTRAIN_MAP.get(comb_key, {"A": 0, "B": 0})
    comb_A = comb_counts["A"]
    comb_B = comb_counts["B"]

    # Combination totals @ PW
    comb_total_gj = comb_A * a_cap_calc_gj + comb_B * b_cap_calc_gj
    comb_total_kg = comb_A * a_cap_calc_kg + comb_B * b_cap_calc_kg

    # Combination usable (PW → Pmin)
    comb_usable_gj = comb_A * a_cap_usable_gj + comb_B * b_cap_usable_gj
    comb_usable_kg = comb_A * a_cap_usable_kg + comb_B * b_cap_usable_kg

    # Persist for planner (planner uses usable first, then total)
    st.session_state["a_cap_calc_gj"]   = a_cap_calc_gj
    st.session_state["b_cap_calc_gj"]   = b_cap_calc_gj
    st.session_state["a_cap_usable_gj"] = a_cap_usable_gj
    st.session_state["b_cap_usable_gj"] = b_cap_usable_gj

    # ---- Total capacity row
    st.subheader("Trailer Total Capacity")
    t1, t2, t3 = st.columns(3)
    t1.markdown(
        f"**A-trailer @ $P_{{W}}$**<br>"
        f"<span style='font-size:1.2em;'>{a_cap_calc_gj:,.2f}</span> GJ<br>"
        f"<span>{a_cap_calc_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )
    t2.markdown(
        f"**B-trailer @ $P_{{W}}$**<br>"
        f"<span style='font-size:1.2em;'>{b_cap_calc_gj:,.2f}</span> GJ<br>"
        f"<span>{b_cap_calc_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )
    t3.markdown(
        f"**{comb_key} total @ $P_{{W}}$**<br>"
        f"<span style='font-size:1.2em;'>{comb_total_gj:,.2f}</span> GJ<br>"
        f"<span>{comb_total_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )

    # ---- Usable capacity row
    st.subheader("Trailer Usable Capacity ($P_{W} \\rightarrow P_{min}$)")
    u1, u2, u3 = st.columns(3)
    u1.markdown(
        f"**A-trailer usable** ($P_{{W}} \\rightarrow P_{{min}}$)<br>"
        f"<span style='font-size:1.2em;'>{a_cap_usable_gj:,.2f}</span> GJ<br>"
        f"<span>{a_cap_usable_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )
    u2.markdown(
        f"**B-trailer usable** ($P_{{W}} \\rightarrow P_{{min}}$)<br>"
        f"<span style='font-size:1.2em;'>{b_cap_usable_gj:,.2f}</span> GJ<br>"
        f"<span>{b_cap_usable_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )
    u3.markdown(
        f"**{comb_key} usable** ($P_{{W}} \\rightarrow P_{{min}}$)<br>"
        f"<span style='font-size:1.2em;'>{comb_usable_gj:,.2f}</span> GJ<br>"
        f"<span>{comb_usable_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )

    if abs(st.session_state["phys_PW"] - Pmin_bar_g) < 1e-6:
        st.info("Usable capacity is zero because Working Pressure equals Minimum Cylinder Pressure.")

    st.caption("**Densities** — ρ(PW): {:.2f} kg/m³, ρ(Pmin): {:.2f} kg/m³{}".format(
        rho_full, rho_min, " (CoolProp)" if COOLPROP_OK else " (ideal gas)"
    ))

# ---------------------------------------
# Fatigue rules (Standard Hours – Solo)
# ---------------------------------------
def required_breaks_time(hours: float) -> float:
    """
    Cumulative break time required for a continuous work/drive segment:
      >5.25 h => +0.25 h (15 min)
      >8.00  h => +0.25 h (now 30 min total)
      >11.0  h => +0.50 h (now 60 min total)
    """
    t = 0.0
    if hours > 5.25:
        t += 0.25
    if hours > 8.0:
        t += 0.25
    if hours > 11.0:
        t += 0.50
    return t

def solo_standard_shift_ok(work_hours: float) -> int:
    """
    Returns compliance code for a single driver's shift:
      0 = OK (<= 12 h total)
      1 = Borderline ( >11 to <=12, must include at least 1 h breaks in the shift )
      2 = Non-compliant (>12 h)
    """
    if work_hours > 12.0:
        return 2
    if work_hours > 11.0:
        return 1
    return 0

# ---------------------------------------
# Core calculations
# ---------------------------------------
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
    available_bays: int,
    changeover_overhead_h: float,
    mode: str,
    driver_shift_cap: float,
    crew_change_overhead_h: float,
    delay_buffer_min: float,
    truck_util_target: float,
    fatigue_regime: str,
    relay_location: str
):
    trailers_per_set = a_count + b_count
    drive_time_h_oneway = distance_km_oneway / max(1e-6, speed_kmh)

    # Per-trailer fill times
    a_fill_time_h = (a_cap_gj / max(1e-6, fill_rate_gjph)) if a_count > 0 else 0.0
    b_fill_time_h = (b_cap_gj / max(1e-6, fill_rate_gjph)) if b_count > 0 else 0.0

    per_set_capacity_gj = a_count * a_cap_gj + b_count * b_cap_gj

    # Parallel vs sequential fill approximation
    if concurrent_bays_per_set >= trailers_per_set and trailers_per_set > 0:
        fill_time_h = max(a_fill_time_h if a_count else 0.0, b_fill_time_h if b_count else 0.0)
    else:
        total_fill = a_count * a_fill_time_h + b_count * b_fill_time_h
        fill_time_h = total_fill / max(1, concurrent_bays_per_set)

    # Full trailer-set cycle (for ThroughRoad)
    trailer_set_cycle_h = fill_time_h + unload_time_h + 2 * drive_time_h_oneway + changeover_overhead_h
    # Truck cycle when drop-and-pull (driver mostly drives) — keep changeover overhead
    truck_cycle_drop_h = 2 * drive_time_h_oneway + crew_change_overhead_h

    # Sets per day from demand
    sets_per_day = daily_energy_gj / max(1e-6, per_set_capacity_gj)

    trucks_drop = int(math.ceil((sets_per_day * truck_cycle_drop_h / hours_per_day) / max(0.01, truck_util_target)))
    trucks_through = int(math.ceil((sets_per_day * trailer_set_cycle_h / hours_per_day) / max(0.01, truck_util_target)))

    sets_in_circulation = int(math.ceil(sets_per_day * trailer_set_cycle_h / hours_per_day))
    a_trailers = sets_in_circulation * a_count
    b_trailers = sets_in_circulation * b_count
    total_trailers = a_trailers + b_trailers

    # ---- Fatigue: standard hours (solo) with relay strategies
    leg_drive = drive_time_h_oneway
    leg_breaks = required_breaks_time(leg_drive)

    # Base (no relay): one driver attempts the round trip
    rt_other_work = unload_time_h if mode == "ThroughRoad" else 0.0
    round_trip_driver_work = (2 * leg_drive) + (2 * leg_breaks) + rt_other_work + crew_change_overhead_h
    leg_can_fit = (leg_drive + leg_breaks) <= driver_shift_cap
    rt_compliance_code = solo_standard_shift_ok(round_trip_driver_work)

    # Defaults
    cycle_time = truck_cycle_drop_h if mode == "DropAndPull" else trailer_set_cycle_h
    drivers_per_truck = max(1, int(math.ceil(cycle_time / driver_shift_cap)))
    relay_msg = "No relay: one driver attempts round trip (swap added automatically if a leg cannot fit)."
    comp_msgs = []

    if relay_location != "No relay":
        if relay_location == "Camp at A, B & Midpoint":
            # Split each direction into two legs: A↔Mid and Mid↔B (and reverse).
            half_leg_drive = (drive_time_h_oneway / 2.0)
            half_leg_breaks = required_breaks_time(half_leg_drive)

            # A→Mid and Mid→A (pure drive + breaks)
            leg_A_to_Mid_work = half_leg_drive + half_leg_breaks
            leg_Mid_to_A_work = half_leg_drive + half_leg_breaks

            # Mid→B includes unload if ThroughRoad; B→Mid is pure drive
            leg_Mid_to_B_work = half_leg_drive + half_leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0)
            leg_B_to_Mid_work = half_leg_drive + half_leg_breaks

            codes = [
                solo_standard_shift_ok(leg_A_to_Mid_work),
                solo_standard_shift_ok(leg_Mid_to_B_work),
                solo_standard_shift_ok(leg_B_to_Mid_work),
                solo_standard_shift_ok(leg_Mid_to_A_work),
            ]
            # All sub-legs must fit ≤12 h
            leg_can_fit = all(c != 2 for c in codes)
            rt_compliance_code = max(codes)  # worst-case
            drivers_per_truck = max(2, int(math.ceil(cycle_time / driver_shift_cap)))
            relay_msg = ("Relay active (Camp at A, B & Midpoint): A↔Mid and Mid↔B legs are "
                         "handled by local drivers at each camp (B-side unload in ThroughRoad).")
            comp_msgs.append("Midpoint relay splits the route into two shorter legs per direction.")

            # Expose for UI summaries (approximate consolidated leg views)
            leg_AtoB_work = leg_A_to_Mid_work + (half_leg_drive + half_leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0))
            leg_BtoA_work = leg_B_to_Mid_work + (half_leg_drive + half_leg_breaks)

        else:
            # Other relay modes
            leg_AtoB_work = leg_drive + leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0)
            leg_BtoA_work = leg_drive + leg_breaks
            leg_AtoB_code = solo_standard_shift_ok(leg_AtoB_work)
            leg_BtoA_code = solo_standard_shift_ok(leg_BtoA_work)
            leg_can_fit = (leg_AtoB_code != 2) and (leg_BtoA_code != 2)
            rt_compliance_code = max(leg_AtoB_code, leg_BtoA_code)
            drivers_per_truck = max(2, int(math.ceil(cycle_time / driver_shift_cap)))

            if relay_location == "Camp at A & B":
                relay_msg = ("Relay active (Camp at A & B): drivers alternate stays between A and B camps across shifts; "
                             "each leg is handled by the local camp driver.")
            else:
                relay_msg = f"Relay active ({relay_location}): separate drivers cover A→B and B→A legs."

            comp_msgs.append("In relay modes, the A→B leg includes unloading time (ThroughRoad only).")
            if mode == "ThroughRoad":
                comp_msgs.append("B-side driver performs unloading at B.")
    else:
        # No relay: maybe need a second driver if even one leg won't fit in a 12h shift
        if not leg_can_fit:
            drivers_per_truck = max(drivers_per_truck, 2)  # need a relay/swap to cover each leg

    active_trucks = trucks_drop if mode == "DropAndPull" else trucks_through
    drivers_on_duty = drivers_per_truck * active_trucks
    rostered_drivers = int(math.ceil(drivers_on_duty * max(1.0, st.session_state.get("relief_factor", 1.3))))

    # Bay utilization
    bay_hours_per_day = daily_energy_gj / max(1e-6, fill_rate_gjph)
    avg_bays_needed = bay_hours_per_day / hours_per_day

    if available_bays <= 0:
        bay_utilization = float("inf") if avg_bays_needed > 0 else 0.0
        bay_flag = "NO BAYS DEFINED" if avg_bays_needed > 0 else "OK"
    else:
        bay_utilization = avg_bays_needed / available_bays
        bay_flag = "OVER 100% - add bays or reduce demand" if bay_utilization > 1.0 else "OK"

    # Overall compliance flag
    if not leg_can_fit or rt_compliance_code == 2:
        compliance_flag = 2
    elif rt_compliance_code == 1:
        compliance_flag = 1
    else:
        compliance_flag = 0

    energy_per_truckhour = (per_set_capacity_gj / (2 * drive_time_h_oneway)) if mode == "DropAndPull" \
        else (per_set_capacity_gj / max(1e-6, cycle_time))

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
        "avg_bays_needed": avg_bays_needed,
        "bay_utilization": bay_utilization,
        "bay_flag": bay_flag,
        "energy_per_truckhour": energy_per_truckhour,
        # Fatigue outputs
        "round_trip_driver_work": round_trip_driver_work,
        "rt_compliance_code": rt_compliance_code,
        "compliance_flag": compliance_flag,
        "fatigue_regime_note": st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)"),
        "relay_location": st.session_state.get("relay_location", "No relay"),
        "leg_breaks_h": leg_breaks,
        "leg_AtoB_work": (drive_time_h_oneway + leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0)),
        "leg_BtoA_work": (drive_time_h_oneway + leg_breaks),
    }

# Use Payload tab capacities (usable preferred)
a_cap_for_planner = float(st.session_state.get("a_cap_usable_gj", st.session_state.get("a_cap_calc_gj", 0.0)))
b_cap_for_planner = float(st.session_state.get("b_cap_usable_gj", st.session_state.get("b_cap_calc_gj", 0.0)))

# ---------------------------------------
# Helper for compliance map
# ---------------------------------------
RELAY_OPTIONS = ["No relay", "Camp at A", "Midpoint relay", "Camp at B", "Camp at A & B", "Camp at A, B & Midpoint"]

def required_breaks_time_map(hours: float) -> float:
    # same logic as required_breaks_time
    t = 0.0
    if hours > 5.25:
        t += 0.25
    if hours > 8.0:
        t += 0.25
    if hours > 11.0:
        t += 0.50
    return t

def solo_standard_shift_ok_map(work_hours: float) -> int:
    if work_hours > 12.0:
        return 2
    if work_hours > 11.0:
        return 1
    return 0

def compliance_code_for(distance_km, speed_kmh, relay_option, mode, unload_time_h):
    """Return 0=OK, 1=Borderline, 2=Non-compliant for given distance/speed/relay option."""
    leg_drive = distance_km / max(speed_kmh, 1e-9)
    def breaks(t): return required_breaks_time_map(t)

    if relay_option == "No relay":
        leg_can_fit = (leg_drive + breaks(leg_drive)) <= 12.0
        rt_other_work = unload_time_h if mode == "ThroughRoad" else 0.0
        rt_work = (2*leg_drive) + (2*breaks(leg_drive)) + rt_other_work
        rt_code = solo_standard_shift_ok_map(rt_work)
        if (not leg_can_fit) or rt_code == 2:
            return 2
        if rt_code == 1:
            return 1
        return 0

    if relay_option in ("Camp at A", "Midpoint relay", "Camp at B", "Camp at A & B"):
        leg_AtoB = leg_drive + breaks(leg_drive) + (unload_time_h if mode == "ThroughRoad" else 0.0)
        leg_BtoA = leg_drive + breaks(leg_drive)
        c1 = solo_standard_shift_ok_map(leg_AtoB)
        c2 = solo_standard_shift_ok_map(leg_BtoA)
        return max(c1, c2)

    if relay_option == "Camp at A, B & Midpoint":
        half_leg = (leg_drive / 2.0)
        b1 = breaks(half_leg)
        w_A_Mid = half_leg + b1
        w_Mid_A = half_leg + b1
        w_Mid_B = half_leg + b1 + (unload_time_h if mode == "ThroughRoad" else 0.0)
        w_B_Mid = half_leg + b1
        codes = [solo_standard_shift_ok_map(x) for x in (w_A_Mid, w_Mid_B, w_B_Mid, w_Mid_A)]
        return max(codes)

    return 2

def best_mode_for(distance_km, speed_kmh, mode, unload_time_h, borderline_ok: bool):
    """
    Return (best_index, status_code) where best_index is index into RELAY_OPTIONS
    preferring the least-intrusive option that yields acceptable status.
    If borderline_ok=False: acceptable is only OK(0).
    If borderline_ok=True: acceptable is OK(0) then Borderline(1).
    If none acceptable, pick the lowest status (1 if any, else 2) with earliest option.
    """
    codes = [compliance_code_for(distance_km, speed_kmh, r, mode, unload_time_h) for r in RELAY_OPTIONS]
    if 0 in codes:
        return codes.index(0), 0
    if borderline_ok and 1 in codes:
        return codes.index(1), 1
    if 1 in codes:
        return codes.index(1), 1
    return codes.index(2), 2

# ---------------------------------------
# Planner tab
# ---------------------------------------
with tabs[1]:
    st.title(f"{st.session_state.get('gas_type_selected', 'CNG')} Virtual Pipeline Planner")
    st.caption(
        "Planner uses **usable capacity (PW → Pmin)** when available; "
        "otherwise **total capacity @ PW**."
    )

    calcs = compute_calcs(
        daily_energy_gj=daily_energy_gj,
        distance_km_oneway=distance_km_oneway,
        speed_kmh=speed_kmh,
        hours_per_day=hours_per_day,
        a_cap_gj=a_cap_for_planner,
        b_cap_gj=b_cap_for_planner,
        a_count=ROADTRAIN_MAP[st.session_state.get("combo", "A-triple")]["A"],
        b_count=ROADTRAIN_MAP[st.session_state.get("combo", "A-triple")]["B"],
        fill_rate_gjph=fill_rate_gjph,
        concurrent_bays_per_set=concurrent_bays_per_set,
        unload_time_h=unload_time_h,
        available_bays=available_bays,
        changeover_overhead_h=changeover_overhead_h,
        mode=mode,
        driver_shift_cap=12.0,
        crew_change_overhead_h=st.session_state.get("crew_change_overhead_h", 0.25) if isinstance(st.session_state.get("crew_change_overhead_h", 0.25), (int,float)) else 0.25,
        delay_buffer_min=st.session_state.get("delay_buffer_min", 30) if isinstance(st.session_state.get("delay_buffer_min", 30), (int,float)) else 30,
        truck_util_target=truck_util_target,
        fatigue_regime=st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)") if isinstance(st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)"), str) else "Standard Hours (Solo Drivers)",
        relay_location=st.session_state.get("relay_location", "No relay") if isinstance(st.session_state.get("relay_location", "No relay"), str) else "No relay",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Fleet")
        st.metric("Active trucks", calcs["active_trucks"])
        st.metric("Drivers per truck (on duty)", calcs["drivers_per_truck"])
        st.metric("Drivers on duty / day", calcs["drivers_on_duty"])
        st.metric("Rostered drivers", calcs["rostered_drivers"])

    with c2:
        st.subheader("Trailers")
        comb_label = st.session_state.get("combo", "A-triple")
        plural_comb = f"{comb_label}s"
        st.metric(f"{plural_comb} in circulation", calcs["sets_in_circulation"])
        st.metric("A-trailers", calcs["a_trailers"])
        st.metric("B-trailers", calcs["b_trailers"])
        st.metric("Total trailers", calcs["total_trailers"])

    with c3:
        st.subheader("Ops")
        st.metric("Avg bays needed", f"{calcs['avg_bays_needed']:.2f}")
        util_pct = "∞%" if not math.isfinite(calcs["bay_utilization"]) else f"{100 * calcs['bay_utilization']:.1f}%"
        st.metric("Bay utilization", util_pct)
        st.write(f"**Bay status:** :{'red_circle' if (calcs['bay_flag'] != 'OK') else 'green_circle'}: {calcs['bay_flag']}")
        st.metric("Energy / truck-hour", f"{calcs['energy_per_truckhour']:.2f} GJ/h")

    # Compliance panel
    st.subheader("Driver Fatigue Compliance")
    tag = {0: "✅ OK", 1: "🟡 Borderline", 2: "🔴 Non-compliant"}[calcs["compliance_flag"]]
    st.write(f"**Status:** {tag}  •  **Regime:** {calcs['fatigue_regime_note']}  •  **Relay:** {calcs['relay_location']}")
    st.write(f"- One-way drive: **{calcs['drive_time_h_oneway']:.2f} h**, required breaks: **{calcs['leg_breaks_h']:.2f} h**.")
    if calcs['relay_location'] != "No relay":
        st.write(f"- A→B driver shift (incl. unload if ThroughRoad): **{calcs['leg_AtoB_work']:.2f} h**")
        st.write(f"- B→A driver shift: **{calcs['leg_BtoA_work']:.2f} h**")
        if calcs['relay_location'] == "Camp at A, B & Midpoint":
            st.write("- Route split: A↔Mid and Mid↔B legs; B-side sub-leg includes unloading in ThroughRoad.")
    else:
        st.write(f"- Round-trip driver work (incl. unload if ThroughRoad): **{calcs['round_trip_driver_work']:.2f} h**")

    # -----------------------------
    # Compliance Map (Distance vs Speed)
    # -----------------------------
    with st.expander("Compliance map (distance vs speed)"):
        st.caption("Shows the least-intrusive relay mode that achieves compliance for each distance–speed pair under your current settings.")
        borderline_ok = st.checkbox("Treat ‘Borderline’ as acceptable", value=False)

        # Grid ranges
        ccol1, ccol2, ccol3, ccol4 = st.columns(4)
        with ccol1:
            d_min = st.number_input("Min distance (km)", value=100.0, step=50.0)
        with ccol2:
            d_max = st.number_input("Max distance (km)", value=1500.0, step=50.0)
        with ccol3:
            v_min = st.number_input("Min speed (km/h)", value=60.0, step=5.0)
        with ccol4:
            v_max = st.number_input("Max speed (km/h)", value=100.0, step=5.0)

        if not MPL_OK:
            st.warning(
                "Matplotlib is not installed in this environment. "
                "Install it with `.venv\\Scripts\\python.exe -m pip install matplotlib`, "
                "or add `matplotlib>=3.9.0` to requirements.txt and reinstall. "
                "Compliance Map is disabled until matplotlib is available."
            )
        else:
            D = np.linspace(max(d_min, 1e-3), max(d_max, d_min + 1e-3), 41)
            V = np.linspace(max(v_min, 1e-3), max(v_max, v_min + 1e-3), 41)

            mode_idx_grid = np.zeros((V.size, D.size), dtype=int)
            status_grid   = np.zeros((V.size, D.size), dtype=int)

            for i, v in enumerate(V):
                for j, d in enumerate(D):
                    bi, sc = best_mode_for(d, v, mode, unload_time_h, borderline_ok)
                    mode_idx_grid[i, j] = bi
                    status_grid[i, j]   = sc

            fig, ax = plt.subplots(figsize=(8, 5))
            cmap = ListedColormap([
                "#2ca02c",  # No relay
                "#1f77b4",  # Camp at A
                "#ff7f0e",  # Midpoint relay
                "#9467bd",  # Camp at B
                "#8c564b",  # Camp at A & B
                "#d62728",  # Camp at A, B & Midpoint
            ])
            bounds = np.arange(len(RELAY_OPTIONS)+1) - 0.5
            norm = BoundaryNorm(bounds, cmap.N)

            im = ax.imshow(
                mode_idx_grid,
                origin="lower",
                aspect="auto",
                extent=[D.min(), D.max(), V.min(), V.max()],
                cmap=cmap,
                norm=norm
            )
            ax.set_xlabel("Distance A→B (km)")
            ax.set_ylabel("Average speed (km/h)")
            title_suffix = " (OK only)" if not borderline_ok else " (OK → Borderline)"
            ax.set_title("Best relay mode to achieve acceptable compliance" + title_suffix)

            # Current operating point marker colored by compliance
            cur_idx, cur_status = best_mode_for(distance_km_oneway, speed_kmh, mode, unload_time_h, borderline_ok)
            edge_color = {0: "#2ca02c", 1: "#ffbf00", 2: "#d62728"}[cur_status]  # green / amber / red
            ax.scatter(
                [distance_km_oneway], [speed_kmh],
                marker="o", s=90,
                edgecolors=edge_color, facecolors="none", linewidths=2, zorder=5
            )
            status_label = {0: "OK", 1: "Borderline", 2: "Non-compliant"}[cur_status]
            annot_text = f"{RELAY_OPTIONS[cur_idx]} • {status_label}"
            ax.annotate(
                annot_text,
                (distance_km_oneway, speed_kmh),
                xytext=(8, 8), textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=edge_color, lw=1),
                zorder=6
            )

            legend_handles = [Patch(color=cmap(i), label=RELAY_OPTIONS[i]) for i in range(len(RELAY_OPTIONS))]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)

            st.pyplot(fig)

# ---------------------------------------
# Scenarios tab
# ---------------------------------------
with tabs[2]:
    st.subheader("Scenarios (save/load)")
    scen_name = st.text_input("Scenario name", value="MyScenario")

    params_for_save = {
        "scenario": scen_name,
        "project_months": project_months,
        "combo": st.session_state.get("combo", "A-triple"),
        "mode": mode,
        "hours_per_day": hours_per_day,
        "daily_energy_gj": daily_energy_gj,
        "distance_km_oneway": distance_km_oneway,
        "speed_kmh": speed_kmh,
        "a_cap_gj": a_cap_for_planner,
        "b_cap_gj": b_cap_for_planner,
        "a_count": ROADTRAIN_MAP[st.session_state.get("combo", "A-triple")]["A"],
        "b_count": ROADTRAIN_MAP[st.session_state.get("combo", "A-triple")]["B"],
        "fill_rate_gjph": fill_rate_gjph,
        "concurrent_bays_per_set": concurrent_bays_per_set,
        "unload_time_h": unload_time_h,
        "available_bays": available_bays,
        "changeover_overhead_h": changeover_overhead_h,
        "driver_shift_cap": 12.0,
        "crew_change_overhead_h": st.session_state.get("crew_change_overhead_h", 0.25) if isinstance(st.session_state.get("crew_change_overhead_h", 0.25), (int,float)) else 0.25,
        "delay_buffer_min": st.session_state.get("delay_buffer_min", 30) if isinstance(st.session_state.get("delay_buffer_min", 30), (int,float)) else 30,
        "truck_util_target": truck_util_target,
        "gas_type": st.session_state.get("gas_type_selected", "Methane"),
        "hv_basis": st.session_state.get("hv_basis_key", "Gross"),
        "reference_condition": st.session_state.get("ref_choice", "Standard (15 °C, 1.01325 barₐ)"),
        "fatigue_regime": st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)"),
        "relay_location": st.session_state.get("relay_location", "No relay"),
    }

    c_left, c_right = st.columns(2)
    with c_left:
        if st.button("Save Scenario (append to scenarios.csv)"):
            df_row = pd.DataFrame([params_for_save])
            try:
                existing = pd.read_csv("scenarios.csv")
                combined = pd.concat([existing, df_row], ignore_index=True)
            except Exception:
                combined = df_row
            combined.to_csv("scenarios.csv", index=False)
            st.success("Scenario saved to scenarios.csv")

    with c_right:
        if st.button("Load scenarios.csv"):
            try:
                df = pd.read_csv("scenarios.csv")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Failed to load scenarios.csv: {e}")

st.caption("Relay strategies split legs across drivers (A→B includes unloading in ThroughRoad). Fatigue regime models Standard Hours (Solo). Sidebar kg/day uses the selected reference (N or S) via ρ×Q; capacities are configured in the Payload & Capacity tab.")
