# v13.9 — Instructions tab + summary ribbon across tabs
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

# Optional Matplotlib for plotting
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
M3_PER_FT3 = 0.028316846592
FT3_PER_M3 = 1.0 / M3_PER_FT3
GJ_PER_MMBTU = 1.055056
DAYS_PER_YEAR = 365.0

def density_kg_per_m3(T_K: float, P_Pa: float, gas_label: str) -> float:
    """Return density using CoolProp when available; otherwise ideal-gas fallback."""
    if COOLPROP_OK:
        try:
            return float(PropsSI("D", "T", T_K, "P", P_Pa, gas_label))
        except Exception:
            pass
    molar_mass = MOLAR_MASS.get(gas_label, MOLAR_MASS["Methane"])
    return (P_Pa * molar_mass) / (R_UNIV * T_K)

def capital_recovery_factor(i: float, n_years: float) -> float:
    """CRF = i(1+i)^n / ((1+i)^n - 1); if i≈0, return 1/n."""
    i = float(i)
    n_years = float(n_years)
    if n_years <= 0:
        return 0.0
    if abs(i) < 1e-9:
        return 1.0 / n_years
    p = (1.0 + i) ** n_years
    return i * p / (p - 1.0)

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
DOLLIES_PER_COMBO = {
    "Semitrailer": 0,
    "B-double": 0,
    "A-double": 1,
    "B-triple": 0,
    "A-triple": 2,
    "AB-triple": 1,
    "AAB-quad": 2,
    "ABB-quad": 1,
    "BAB-quad": 1,
}

# ------------------------------
# Sidebar inputs
# ------------------------------
with st.sidebar:
    st.header("Operating Mode & Demand")
    mode = st.selectbox("Mode", ["DropAndPull", "ThroughRoad"])
    project_months = st.number_input("Project duration [months]", min_value=1, value=12, step=1)
    project_years = project_months / 12.0
    hours_per_day = st.number_input("Operating hours per day", min_value=1, max_value=24, value=24, step=1)

    daily_energy_gj = st.number_input(
        "Daily energy to transport (GJ/day)",
        min_value=0.0,
        value=10000.00,
        step=100.0,
        format="%.2f",
        help="Energy delivered per day in gigajoules."
    )

    # Reference condition for conversions
    ref_choice = st.radio(
        "Conversion reference",
        ["Normal (20 °C, 1.01325 barₐ)", "Standard (15 °C, 1.01325 barₐ)"],
        index=1,
        help="Pick which condition the sidebar volumetric & mass conversions use."
    )
    st.session_state["ref_choice"] = ref_choice

    default_hv = 50032000.0  # Methane LHV [J/kg]
    gas_for_conv = st.session_state.get("gas_type_selected", "Methane")
    hv_for_conv = float(st.session_state.get("HV_J_per_kg", default_hv))

    # Normal: 20 °C, 1.01325 bar(a); Standard: 15 °C, 1.01325 bar(a)
    Tn_K, Pn_Pa = 293.15, 1.01325e5
    Ts_K, Ps_Pa = 288.15, 1.01325e5

    def rho_at(T_K, P_Pa):
        label = "Hydrogen" if gas_for_conv == "Hydrogen" else "Methane"
        return density_kg_per_m3(T_K, P_Pa, label)

    rho_N = rho_at(Tn_K, Pn_Pa)
    rho_S = rho_at(Ts_K, Ps_Pa)

    MJ_per_m3_N = rho_N * (hv_for_conv / 1e6)
    MJ_per_m3_S = rho_S * (hv_for_conv / 1e6)

    Nm3_per_day = (daily_energy_gj * 1000.0) / max(MJ_per_m3_N, 1e-12)
    Sm3_per_day = (daily_energy_gj * 1000.0) / max(MJ_per_m3_S, 1e-12)
    Nft3_per_day = Nm3_per_day * FT3_PER_M3
    Sft3_per_day = Sm3_per_day * FT3_PER_M3
    MMSCFD = Sft3_per_day / 1e6

    kg_per_day = rho_S * Sm3_per_day if ref_choice.startswith("Standard") else rho_N * Nm3_per_day
    MMBTU_per_day = daily_energy_gj / GJ_PER_MMBTU

    st.caption(f"≈ **{daily_energy_gj / 1000:.3f} TJ/day**")
    if ref_choice.startswith("Standard"):
        st.caption(f"≈ **{Sm3_per_day:,.0f} Sm³/day**, **{MMSCFD:,.3f} MMSCF/D**  _(S: 15 °C, 1.01325 barₐ)_")
        st.caption(f"≈ **{Nm3_per_day:,.0f} Nm³/day**, **{Nft3_per_day:,.0f} Nft³/day**  _(N: 20 °C, 1.01325 barₐ)_")
    else:
        st.caption(f"≈ **{Nm3_per_day:,.0f} Nm³/day**, **{Nft3_per_day:,.0f} Nft³/day**  _(N: 20 °C, 1.01325 barₐ)_")
        st.caption(f"≈ **{Sm3_per_day:,.0f} Sm³/day**, **{MMSCFD:,.3f} MMSCF/D**  _(S: 15 °C, 1.01325 barₐ)_")

    kg_tip = ("Mass derived from S: ρ(S) × Sm³/day" if ref_choice.startswith("Standard")
              else "Mass derived from N: ρ(N) × Nm³/day")
    st.markdown(
        f"""<span title="{kg_tip}">≈ <b>{kg_per_day:,.0f} kg/day</b> <sup>ⓘ</sup></span>
             &nbsp; • &nbsp; <b>{MMBTU_per_day:,.0f} MMBTU/day</b>""",
        unsafe_allow_html=True
    )

    st.divider()

    st.header("Route")
    st.caption("A = Gas production location, B = Gas delivery location")
    distance_km_oneway = st.number_input("Distance A→B (km)", min_value=0.01, value=900.0, step=10.0)
    speed_kmh = st.number_input("Average driving speed (km/h)", min_value=0.01, max_value=110.0, value=75.0, step=1.0)

    st.header("Road Train Combination")
    combo_names = list(ROADTRAIN_MAP.keys())
    default_idx = combo_names.index("A-triple") if "A-triple" in ROADTRAIN_MAP else 0
    combo = st.selectbox("Combination", combo_names, index=default_idx, help="A = A-trailer, B = B-trailer")
    a_count = ROADTRAIN_MAP[combo]["A"]
    b_count = ROADTRAIN_MAP[combo]["B"]
    trailers_per_set = a_count + b_count
    st.caption(f"Selected: **{combo}** → A: **{a_count}**, B: **{b_count}** (Total {trailers_per_set})")
    st.session_state["combo"] = combo

    st.header("Terminal Ops (A & B)")
    fill_rate_gjph = st.number_input("Per-bay fill rate (GJ/h)", min_value=0.001, value=41.67, step=0.01, format="%.2f")
    concurrent_bays_per_set = st.number_input(
        "Concurrent bays per set (trailers filled in parallel)",
        min_value=1, max_value=8, value=min(3, trailers_per_set) if trailers_per_set > 0 else 1, step=1
    )
    unload_time_h = st.number_input("Unload time per set at B (h)", min_value=0.0, value=12.0, step=0.25)
    available_bays_A = st.number_input("Available filling bays at A (count)", min_value=0, value=12, step=1)
    available_unload_bays_B = st.number_input("Available unloading bays at B (count)", min_value=0, value=4, step=1)
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
        help=("Where drivers swap for legs. ‘Camp at A & B’ alternates between end camps; "
              "‘A, B & Midpoint’ splits each direction into two shorter legs via a midpoint camp.")
    )
    st.session_state["fatigue_regime"] = fatigue_regime
    st.session_state["relay_location"] = relay_location

    driver_shift_cap = 12.0  # Standard Hours cap
    crew_change_overhead_h = st.number_input("Crew change overhead at swap (h)", min_value=0.0, value=0.25, step=0.05)
    delay_buffer_min = st.number_input("Target compliance buffer per leg (min)", min_value=0, value=30, step=5)
    relief_factor = st.number_input("Driver relief factor", min_value=1.0, max_value=2.0, value=1.3, step=0.05)

    st.header("Utilization Target")
    truck_util_target = st.number_input("Truck utilization (0–1)", min_value=0.01, max_value=1.0, value=0.85, step=0.01)

# ---------------------------------------
# Tabs (Instructions first)
# ---------------------------------------
tabs = st.tabs([
    "Instructions",
    "Payload & Capacity",
    "Planner",
    "Infrastructure & Transport Costs",
    "Benefits & Carbon",
    "Scenarios"
])

# ------------------------------
# Summary ribbon helper
# ------------------------------
def summary_ribbon():
    """Render a compact ribbon using session-state values if present."""
    lc = st.session_state.get("lc_per_gj", None)
    lb = st.session_state.get("lb_per_gj", None)
    def fmt(x):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        return f"${x:,.2f}/GJ"
    def badge(val, good_when_higher=True):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "⬜"
        if good_when_higher:
            return "🟢" if val > 0 else ("🟡" if abs(val) < 0.01 else "🔴")
        else:
            return "🟢" if val <= 0 else ("🟡" if val < 0.01 else "🔴")
    net = None
    if lc is not None and lb is not None and not (math.isnan(lc) or math.isnan(lb)):
        net = lb - lc
    st.markdown(
        f"""
<div style="border:1px solid #e5e5e5; border-radius:8px; padding:8px 12px; margin-bottom:10px;">
  <b>Summary:</b>
  &nbsp;&nbsp; Cost: <span>{badge(lc, good_when_higher=False)}</span> <b>{fmt(lc)}</b>
  &nbsp;&nbsp; Benefit: <span>{badge(lb, good_when_higher=True)}</span> <b>{fmt(lb)}</b>
  &nbsp;&nbsp; Net: <span>{badge(net, good_when_higher=True)}</span> <b>{fmt(net)}</b>
</div>
""",
        unsafe_allow_html=True
    )

# ---------------------------------------
# INSTRUCTIONS (first tab)
# ---------------------------------------
with tabs[0]:
    st.title("Instructions")
    summary_ribbon()
    st.markdown("""
### 1) Purpose
This app plans a **virtual pipeline** for compressed gas (Methane or Hydrogen) using modular road-train combinations.  
It estimates **fleet size**, **bay utilisation**, **driver compliance**, and **levelized cost/benefit** (incl. carbon).

---

### 2) Quick start
1. **Sidebar → Operating Mode & Demand:** enter daily GJ, distance, speed, operating hours.  
2. **Sidebar → Road Train Combination:** pick e.g. *A-triple* (A = A-trailer, B = B-trailer).  
3. **Payload & Capacity tab:** set pressures, cylinder sizes, gas type; review capacity (GJ & kg).  
4. **Planner tab:** see trucks, combinations in circulation, trailers, bays, compliance.  
5. **Infrastructure & Transport Costs tab:** enter CAPEX/OPEX, maintenance, fuel, drivers, etc.  
6. **Benefits & Carbon tab:** set gas revenue and carbon avoidance assumptions.  
7. **Scenarios tab:** save/load scenarios for comparison.

---

### 3) Tips
- **Usable capacity = from $P_{W} \\rightarrow P_{min}$**, total capacity is @ $P_W$.
- **Drivers & Compliance** follow NHVR *Standard Hours (Solo)* logic with relay options.
- Toggle **Normal (20 °C)** vs **Standard (15 °C)** conditions in the sidebar for conversions.
- **Dollies per combination** are auto-derived from the selection (A/B mix).
- Fill/Unload **bay utilisation** appears on the **Planner** tab.
- Levelized cost uses **CRF** (interest % and NPER months).

---

### 4) Outputs (where to look)
- **Payload & Capacity:** trailer & combination capacity (GJ, kg; total vs usable).  
- **Planner:** trucks, combinations in circulation, A/B trailers, bays, compliance, driver utilisation chart.  
- **Infrastructure & Transport Costs:** annuitised CAPEX, OPEX, transport ops → **Levelized Cost ($/GJ)**.  
- **Benefits & Carbon:** gas value + carbon avoidance → **Levelized Benefit ($/GJ)**.  
- **Scenarios:** save and reload parameter sets.
""")

# ---------------------------------------
# PAYLOAD & CAPACITY  (index 1)
# ---------------------------------------
with tabs[1]:
    st.subheader("Payload & Capacity (A- and B-trailers)")
    colp1, colp2 = st.columns(2, vertical_alignment="top")
    summary_ribbon()

    # LEFT: Thermofluid inputs
    with colp1:
        st.markdown("**Thermofluid inputs**")
        gas_type = st.selectbox("Gas Type", ["Hydrogen", "Methane"], index=1)
        st.session_state["gas_type_selected"] = gas_type

        PW_bar_g = st.number_input("Working Pressure [bar(g)]", value=300.0, min_value=0.0, step=1.0, key="phys_PW")
        Pmin_bar_g = st.number_input(
            "Minimum Cylinder Pressure [bar(g)]",
            min_value=0.0, max_value=PW_bar_g, value=min(10.0, PW_bar_g), step=1.0,
            help="Pressure below which the trailer is considered empty for dispatch purposes."
        )
        if Pmin_bar_g > PW_bar_g + 1e-9:
            st.error("Minimum Cylinder Pressure cannot exceed Working Pressure.")
            Pmin_bar_g = PW_bar_g

        hv_options = ["Gross Heating Value [J/kg]", "Net Heating Value [J/kg]"]
        hv_basis = st.selectbox("Heating value basis", hv_options, index=0)
        hv_key = "Gross" if hv_basis.startswith("Gross") else "Net"
        st.session_state["hv_basis_key"] = hv_key

        T_C = st.number_input("Gas temperature (°C)", min_value=-20.0, max_value=65.0, value=15.0, step=1.0, key="phys_T")

        auto_sync_hv = st.checkbox("Auto-sync heating value to Gas Type & Basis", value=True)

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

    # RIGHT: volumes
    with colp2:
        st.markdown("**Composition & usable volume**")
        gas_purity = st.number_input("Fuel gas purity (mole fraction 0–1)", value=1.00, min_value=0.0, max_value=1.0, step=0.01)

        st.markdown("**A-trailer cylinder set**")
        a_cyl_count = st.number_input("Number of cylinders per A-trailer", min_value=1, value=1, step=1)
        a_cyl_water_L = st.number_input("Cylinder water capacity [Litres] (A)", min_value=0.01, value=41000.0, step=100.0, format="%.1f")
        a_trailer_vol_m3 = (a_cyl_count * a_cyl_water_L) / 1000.0
        st.caption(f"Computed A-trailer internal gas volume: **{a_trailer_vol_m3:,.2f} m³**")
        st.markdown("---")

        st.markdown("**B-trailer cylinder set**")
        b_cyl_count = st.number_input("Number of cylinders per B-trailer", min_value=1, value=1, step=1)
        b_cyl_water_L = st.number_input("Cylinder water capacity [Litres] (B)", min_value=0.01, value=17100.0, step=100.0, format="%.1f")
        b_trailer_vol_m3 = (b_cyl_count * b_cyl_water_L) / 1000.0
        st.caption(f"Computed B-trailer internal gas volume: **{b_trailer_vol_m3:,.2f} m³**")

    # Thermo calcs
    P_abs_bar, Pmin_abs_bar = st.session_state["phys_PW"] + 1.01325, Pmin_bar_g + 1.01325
    P_Pa, Pmin_Pa = P_abs_bar * 1e5, Pmin_abs_bar * 1e5
    T_K = st.session_state["phys_T"] + 273.15
    gas_label = "Hydrogen" if gas_type == "Hydrogen" else "Methane"
    rho_full = density_kg_per_m3(T_K, P_Pa, gas_label)
    rho_min  = density_kg_per_m3(T_K, Pmin_Pa, gas_label)
    HV_J_per_kg = float(st.session_state.get("HV_J_per_kg", DEFAULT_HV[gas_type][hv_key]))

    a_cap_calc_gj = rho_full * a_trailer_vol_m3 * gas_purity * (HV_J_per_kg / 1e9)
    b_cap_calc_gj = rho_full * b_trailer_vol_m3 * gas_purity * (HV_J_per_kg / 1e9)
    a_cap_calc_kg = (a_cap_calc_gj * 1e9) / max(HV_J_per_kg, 1e-12)
    b_cap_calc_kg = (b_cap_calc_gj * 1e9) / max(HV_J_per_kg, 1e-12)

    delta_rho = max(rho_full - rho_min, 0.0)
    a_cap_usable_gj = delta_rho * a_trailer_vol_m3 * gas_purity * (HV_J_per_kg / 1e9)
    b_cap_usable_gj = delta_rho * b_trailer_vol_m3 * gas_purity * (HV_J_per_kg / 1e9)
    a_cap_usable_kg = (a_cap_usable_gj * 1e9) / max(HV_J_per_kg, 1e-12)
    b_cap_usable_kg = (b_cap_usable_gj * 1e9) / max(HV_J_per_kg, 1e-12)

    comb_key = st.session_state.get("combo", "A-triple")
    comb_counts = ROADTRAIN_MAP.get(comb_key, {"A": 0, "B": 0})
    comb_A, comb_B = comb_counts["A"], comb_counts["B"]

    comb_total_gj = comb_A * a_cap_calc_gj + comb_B * b_cap_calc_gj
    comb_total_kg = comb_A * a_cap_calc_kg + comb_B * b_cap_calc_kg
    comb_usable_gj = comb_A * a_cap_usable_gj + comb_B * b_cap_usable_gj
    comb_usable_kg = comb_A * a_cap_usable_kg + comb_B * b_cap_usable_kg

    # Persist for planner
    st.session_state["a_cap_calc_gj"]   = a_cap_calc_gj
    st.session_state["b_cap_calc_gj"]   = b_cap_calc_gj
    st.session_state["a_cap_usable_gj"] = a_cap_usable_gj
    st.session_state["b_cap_usable_gj"] = b_cap_usable_gj

    st.subheader("Trailer Total Capacity")
    t1, t2, t3 = st.columns(3)
    t1.markdown(f"**A-trailer @ $P_{{W}}$**<br><span style='font-size:1.2em;'>{a_cap_calc_gj:,.2f}</span> GJ<br><span>{a_cap_calc_kg:,.0f} kg</span>", unsafe_allow_html=True)
    t2.markdown(f"**B-trailer @ $P_{{W}}$**<br><span style='font-size:1.2em;'>{b_cap_calc_gj:,.2f}</span> GJ<br><span>{b_cap_calc_kg:,.0f} kg</span>", unsafe_allow_html=True)
    t3.markdown(f"**{comb_key} total @ $P_{{W}}$**<br><span style='font-size:1.2em;'>{comb_total_gj:,.2f}</span> GJ<br><span>{comb_total_kg:,.0f} kg</span>", unsafe_allow_html=True)

    st.subheader("Trailer Usable Capacity ($P_{W} \\rightarrow P_{min}$)")
    u1, u2, u3 = st.columns(3)
    u1.markdown(f"**A-trailer usable** ($P_{{W}} \\rightarrow P_{{min}}$)<br><span style='font-size:1.2em;'>{a_cap_usable_gj:,.2f}</span> GJ<br><span>{a_cap_usable_kg:,.0f} kg</span>", unsafe_allow_html=True)
    u2.markdown(f"**B-trailer usable** ($P_{{W}} \\rightarrow P_{{min}}$)<br><span style='font-size:1.2em;'>{b_cap_usable_gj:,.2f}</span> GJ<br><span>{b_cap_usable_kg:,.0f} kg</span>", unsafe_allow_html=True)
    u3.markdown(f"**{comb_key} usable** ($P_{{W}} \\rightarrow P_{{min}}$)<br><span style='font-size:1.2em;'>{comb_usable_gj:,.2f}</span> GJ<br><span>{comb_usable_kg:,.0f} kg</span>", unsafe_allow_html=True)

    if abs(st.session_state["phys_PW"] - Pmin_bar_g) < 1e-6:
        st.info("Usable capacity is zero because Working Pressure equals Minimum Cylinder Pressure.")
    st.caption("**Densities** — ρ(PW): {:.2f} kg/m³, ρ(Pmin): {:.2f} kg/m³{}".format(
        rho_full, rho_min, " (CoolProp)" if COOLPROP_OK else " (ideal gas)"
    ))

# ---------------------------------------
# Fatigue rules (Standard Hours – Solo)
# ---------------------------------------
def required_breaks_time(hours: float) -> float:
    t = 0.0
    if hours > 5.25: t += 0.25
    if hours > 8.0:  t += 0.25
    if hours > 11.0: t += 0.50
    return t

def solo_standard_shift_ok(work_hours: float) -> int:
    if work_hours > 12.0: return 2
    if work_hours > 11.0: return 1
    return 0

# ---------------------------------------
# Core calculations for planner
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
    available_bays_A: int,
    available_unload_bays_B: int,
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

    a_fill_time_h = (a_cap_gj / max(1e-6, fill_rate_gjph)) if a_count > 0 else 0.0
    b_fill_time_h = (b_cap_gj / max(1e-6, fill_rate_gjph)) if b_count > 0 else 0.0

    per_set_capacity_gj = a_count * a_cap_gj + b_count * b_cap_gj

    if concurrent_bays_per_set >= trailers_per_set and trailers_per_set > 0:
        fill_time_h = max(a_fill_time_h if a_count else 0.0, b_fill_time_h if b_count else 0.0)
    else:
        total_fill = a_count * a_fill_time_h + b_count * b_fill_time_h
        fill_time_h = total_fill / max(1, concurrent_bays_per_set)

    trailer_set_cycle_h = fill_time_h + unload_time_h + 2 * drive_time_h_oneway + changeover_overhead_h
    truck_cycle_drop_h = 2 * drive_time_h_oneway + crew_change_overhead_h

    sets_per_day = daily_energy_gj / max(1e-6, per_set_capacity_gj)

    trucks_drop = int(math.ceil((sets_per_day * truck_cycle_drop_h / hours_per_day) / max(0.01, truck_util_target)))
    trucks_through = int(math.ceil((sets_per_day * trailer_set_cycle_h / hours_per_day) / max(0.01, truck_util_target)))

    sets_in_circulation = int(math.ceil(sets_per_day * trailer_set_cycle_h / hours_per_day))
    a_trailers = sets_in_circulation * a_count
    b_trailers = sets_in_circulation * b_count
    total_trailers = a_trailers + b_trailers

    # Fatigue
    leg_drive = drive_time_h_oneway
    leg_breaks = required_breaks_time(leg_drive)
    rt_other_work = unload_time_h if mode == "ThroughRoad" else 0.0
    round_trip_driver_work = (2 * leg_drive) + (2 * leg_breaks) + rt_other_work + crew_change_overhead_h
    leg_can_fit = (leg_drive + leg_breaks) <= driver_shift_cap
    rt_compliance_code = solo_standard_shift_ok(round_trip_driver_work)

    cycle_time = truck_cycle_drop_h if mode == "DropAndPull" else trailer_set_cycle_h
    drivers_per_truck = max(1, int(math.ceil(cycle_time / driver_shift_cap)))

    if relay_location != "No relay":
        if relay_location == "Camp at A, B & Midpoint":
            half_leg_drive = (drive_time_h_oneway / 2.0)
            half_leg_breaks = required_breaks_time(half_leg_drive)
            leg_A_to_Mid_work = half_leg_drive + half_leg_breaks
            leg_Mid_to_B_work = half_leg_drive + half_leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0)
            leg_B_to_Mid_work = half_leg_drive + half_leg_breaks
            leg_Mid_to_A_work = half_leg_drive + half_leg_breaks
            codes = [solo_standard_shift_ok(x) for x in (leg_A_to_Mid_work, leg_Mid_to_B_work, leg_B_to_Mid_work, leg_Mid_to_A_work)]
            leg_can_fit = all(c != 2 for c in codes)
            rt_compliance_code = max(codes)
            drivers_per_truck = max(2, int(math.ceil(cycle_time / driver_shift_cap)))
        else:
            leg_AtoB_work = leg_drive + leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0)
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
    rostered_drivers = int(math.ceil(drivers_on_duty * max(1.0, st.session_state.get("relief_factor", 1.3))))

    # Bay utilization (A)
    bay_hours_per_day_A = daily_energy_gj / max(1e-6, fill_rate_gjph)
    avg_bays_needed_A = bay_hours_per_day_A / hours_per_day
    if available_bays_A <= 0:
        bay_utilization_A = float("inf") if avg_bays_needed_A > 0 else 0.0
        bay_flag_A = "NO BAYS DEFINED" if avg_bays_needed_A > 0 else "OK"
    else:
        bay_utilization_A = avg_bays_needed_A / available_bays_A
        bay_flag_A = "OVER 100% - add bays or reduce demand" if bay_utilization_A > 1.0 else "OK"

    # Unload bay utilization (B)
    unload_hours_per_day_B = sets_per_day * unload_time_h
    avg_unload_bays_needed_B = unload_hours_per_day_B / hours_per_day
    if available_unload_bays_B <= 0:
        unload_utilization_B = float("inf") if avg_unload_bays_needed_B > 0 else 0.0
        unload_flag_B = "NO BAYS DEFINED" if avg_unload_bays_needed_B > 0 else "OK"
    else:
        unload_utilization_B = avg_unload_bays_needed_B / available_unload_bays_B
        unload_flag_B = "OVER 100% - add bays or reduce unload time" if unload_utilization_B > 1.0 else "OK"

    energy_per_truckhour = (per_set_capacity_gj / (2 * drive_time_h_oneway)) if mode == "DropAndPull" \
        else (per_set_capacity_gj / max(1e-6, cycle_time))

    # Simple distance metrics
    truck_km_per_set = 2.0 * distance_km_oneway
    trailer_km_per_set = truck_km_per_set * trailers_per_set
    truck_km_per_day = sets_per_day * truck_km_per_set
    trailer_km_per_day = sets_per_day * trailer_km_per_set

    # Overall compliance flag
    if not leg_can_fit or rt_compliance_code == 2: compliance_flag = 2
    elif rt_compliance_code == 1:                 compliance_flag = 1
    else:                                          compliance_flag = 0

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
        "fatigue_regime_note": st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)"),
        "relay_location": st.session_state.get("relay_location", "No relay"),
        "leg_breaks_h": leg_breaks,
        "leg_AtoB_work": (drive_time_h_oneway + leg_breaks + (unload_time_h if mode == "ThroughRoad" else 0.0)),
        "leg_BtoA_work": (drive_time_h_oneway + leg_breaks),
        "truck_km_per_day": truck_km_per_day,
        "trailer_km_per_day": trailer_km_per_day,
        "trailers_per_set": trailers_per_set,
    }

# Planner uses usable capacity by default
a_cap_for_planner = float(st.session_state.get("a_cap_usable_gj", st.session_state.get("a_cap_calc_gj", 0.0)))
b_cap_for_planner = float(st.session_state.get("b_cap_usable_gj", st.session_state.get("b_cap_calc_gj", 0.0)))

# Helper for compliance map
RELAY_OPTIONS = ["No relay", "Camp at A", "Midpoint relay", "Camp at B", "Camp at A & B", "Camp at A, B & Midpoint"]
def required_breaks_time_map(hours: float) -> float:
    t = 0.0
    if hours > 5.25: t += 0.25
    if hours > 8.0:  t += 0.25
    if hours > 11.0: t += 0.50
    return t
def solo_standard_shift_ok_map(work_hours: float) -> int:
    if work_hours > 12.0: return 2
    if work_hours > 11.0: return 1
    return 0
def compliance_code_for(distance_km, speed_kmh, relay_option, mode, unload_time_h):
    leg_drive = distance_km / max(speed_kmh, 1e-9)
    def breaks(t): return required_breaks_time_map(t)
    if relay_option == "No relay":
        leg_can_fit = (leg_drive + breaks(leg_drive)) <= 12.0
        rt_other_work = unload_time_h if mode == "ThroughRoad" else 0.0
        rt_work = (2*leg_drive) + (2*breaks(leg_drive)) + rt_other_work
        rt_code = solo_standard_shift_ok_map(rt_work)
        if (not leg_can_fit) or rt_code == 2: return 2
        if rt_code == 1: return 1
        return 0
    if relay_option in ("Camp at A", "Midpoint relay", "Camp at B", "Camp at A & B"):
        leg_AtoB = leg_drive + breaks(leg_drive) + (unload_time_h if mode == "ThroughRoad" else 0.0)
        leg_BtoA = leg_drive + breaks(leg_drive)
        c1 = solo_standard_shift_ok_map(leg_AtoB); c2 = solo_standard_shift_ok_map(leg_BtoA)
        return max(c1, c2)
    if relay_option == "Camp at A, B & Midpoint":
        half_leg = (leg_drive / 2.0); b1 = breaks(half_leg)
        w_A_Mid = half_leg + b1; w_Mid_B = half_leg + b1 + (unload_time_h if mode == "ThroughRoad" else 0.0)
        w_B_Mid = half_leg + b1;  w_Mid_A = half_leg + b1
        codes = [solo_standard_shift_ok_map(x) for x in (w_A_Mid, w_Mid_B, w_B_Mid, w_Mid_A)]
        return max(codes)
    return 2
def best_mode_for(distance_km, speed_kmh, mode, unload_time_h, borderline_ok: bool):
    codes = [compliance_code_for(distance_km, speed_kmh, r, mode, unload_time_h) for r in RELAY_OPTIONS]
    if 0 in codes: return codes.index(0), 0
    if borderline_ok and 1 in codes: return codes.index(1), 1
    if 1 in codes: return codes.index(1), 1
    return codes.index(2), 2

# ---------------------------------------
# Planner tab (index 2)
# ---------------------------------------
with tabs[2]:
    st.title(f"{st.session_state.get('gas_type_selected', 'CNG')} Virtual Pipeline Planner")
    summary_ribbon()
    st.caption("Planner uses **usable capacity (PW → Pmin)** when available; otherwise **total capacity @ PW**.")

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
        available_bays_A=available_bays_A,
        available_unload_bays_B=available_unload_bays_B,
        changeover_overhead_h=changeover_overhead_h,
        mode=mode,
        driver_shift_cap=12.0,
        crew_change_overhead_h=st.session_state.get("crew_change_overhead_h", 0.25),
        delay_buffer_min=st.session_state.get("delay_buffer_min", 30),
        truck_util_target=truck_util_target,
        fatigue_regime=st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)"),
        relay_location=st.session_state.get("relay_location", "No relay"),
    )
    st.session_state["latest_calcs_cache"] = calcs

    c1, c2, c3, c4 = st.columns(4)
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
        st.subheader("Ops — Fill (A)")
        st.metric("Avg fill bays needed", f"{calcs['avg_bays_needed_A']:.2f}")
        utilA = "∞%" if not math.isfinite(calcs["bay_utilization_A"]) else f"{100 * calcs['bay_utilization_A']:.1f}%"
        st.metric("Fill bay utilisation", utilA)
        st.write(f"**Fill bay status:** :{'red_circle' if (calcs['bay_flag_A'] != 'OK') else 'green_circle'}: {calcs['bay_flag_A']}")
        st.metric("Energy / truck-hour", f"{calcs['energy_per_truckhour']:.2f} GJ/h")
    with c4:
        st.subheader("Ops — Unload (B)")
        st.metric("Avg unload bays needed", f"{calcs['avg_unload_bays_needed_B']:.2f}")
        utilB = "∞%" if not math.isfinite(calcs["unload_utilization_B"]) else f"{100 * calcs['unload_utilization_B']:.1f}%"
        st.metric("Unload bay utilisation", utilB)
        st.write(f"**Unload bay status:** :{'red_circle' if (calcs['unload_flag_B'] != 'OK') else 'green_circle'}: {calcs['unload_flag_B']}")

    # Driver utilisation graphic
    st.subheader("Driver utilisation (per-shift breakdown)")
    def _leg_segments(mode, drive_h, breaks_h, unload_h, overhead_h):
        return {"Drive": max(drive_h,0.0), "Breaks": max(breaks_h,0.0),
                "Unload": max(unload_h if mode=="ThroughRoad" else 0.0,0.0), "Overhead": max(overhead_h,0.0)}
    driver_cap = 12.0
    if st.session_state.get("relay_location", "No relay") == "No relay":
        segs = {"Round trip": {"Drive": 2*calcs["drive_time_h_oneway"],
                               "Breaks": 2*calcs["leg_breaks_h"],
                               "Unload": (unload_time_h if mode=="ThroughRoad" else 0.0),
                               "Overhead": st.session_state.get("crew_change_overhead_h",0.25)}}
    else:
        leg_drive, leg_breaks = calcs["drive_time_h_oneway"], calcs["leg_breaks_h"]
        half_over = 0.5*st.session_state.get("crew_change_overhead_h",0.25)
        segs = {"A→B": _leg_segments(mode, leg_drive, leg_breaks, unload_time_h, half_over),
                "B→A": _leg_segments(mode, leg_drive, leg_breaks, 0.0,          half_over)}
    if MPL_OK:
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        labels = list(segs.keys()); categories = ["Drive","Breaks","Unload","Overhead"]
        colors = {"Drive":"#1f77b4","Breaks":"#ff7f0e","Unload":"#2ca02c","Overhead":"#9467bd"}
        idx = np.arange(len(labels)); bottoms = np.zeros(len(labels))
        for cat in categories:
            vals = [segs[l][cat] for l in labels]
            ax2.bar(idx, vals, bottom=bottoms, label=cat, color=colors.get(cat,None))
            bottoms += np.array(vals)
        ax2.axhline(y=driver_cap, color="#d62728", linestyle="--", linewidth=1)
        for i, tot in enumerate(bottoms):
            pct = 100.0*tot/driver_cap
            ax2.text(i, tot+0.2, f"{tot:.2f} h ({pct:.0f}%)", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(idx); ax2.set_xticklabels(labels); ax2.set_ylim(0, max(driver_cap, max(bottoms)+2))
        ax2.set_ylabel("Hours per shift"); ax2.set_title("Driver utilisation vs 12 h cap (Standard Hours)")
        ax2.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.20))
        st.pyplot(fig2)
    else:
        st.warning("Matplotlib not installed — showing text breakdown instead.")
        for name, parts in segs.items():
            total = sum(parts.values()); pct = 100.0*total/driver_cap
            st.write(f"**{name}** — Total: {total:.2f} h ({pct:.0f}% of 12 h)")
            st.caption(f"Drive: {parts['Drive']:.2f} h • Breaks: {parts['Breaks']:.2f} h • Unload: {parts['Unload']:.2f} h • Overhead: {parts['Overhead']:.2f} h")

    # Compliance map
    with st.expander("Compliance map (distance vs speed)"):
        st.caption("Least-intrusive relay mode that achieves compliance at each distance–speed pair.")
        borderline_ok = st.checkbox("Treat ‘Borderline’ as acceptable", value=False)
        ccol1, ccol2, ccol3, ccol4 = st.columns(4)
        with ccol1: d_min = st.number_input("Min distance (km)", value=100.0, step=50.0)
        with ccol2: d_max = st.number_input("Max distance (km)", value=1500.0, step=50.0)
        with ccol3: v_min = st.number_input("Min speed (km/h)", value=60.0, step=5.0)
        with ccol4: v_max = st.number_input("Max speed (km/h)", value=100.0, step=5.0)
        if not MPL_OK:
            st.warning("Matplotlib is not installed. Add `matplotlib>=3.9.0` to requirements.txt and reinstall.")
        else:
            D = np.linspace(max(d_min, 1e-3), max(d_max, d_min + 1e-3), 41)
            V = np.linspace(max(v_min, 1e-3), max(v_max, v_min + 1e-3), 41)
            mode_idx_grid = np.zeros((V.size, D.size), dtype=int)
            for i, v in enumerate(V):
                for j, d in enumerate(D):
                    bi, _ = best_mode_for(d, v, mode, unload_time_h, borderline_ok)
                    mode_idx_grid[i, j] = bi
            fig, ax = plt.subplots(figsize=(8, 5))
            cmap = ListedColormap(["#2ca02c","#1f77b4","#ff7f0e","#9467bd","#8c564b","#d62728"])
            bounds = np.arange(len(RELAY_OPTIONS)+1) - 0.5; norm = BoundaryNorm(bounds, cmap.N)
            ax.imshow(mode_idx_grid, origin="lower", aspect="auto", extent=[D.min(), D.max(), V.min(), V.max()], cmap=cmap, norm=norm)
            ax.set_xlabel("Distance A→B (km)"); ax.set_ylabel("Average speed (km/h)")
            ax.set_title("Best relay mode to achieve acceptable compliance" + (" (OK → Borderline)" if borderline_ok else " (OK only)"))
            cur_idx, cur_status = best_mode_for(distance_km_oneway, speed_kmh, mode, unload_time_h, borderline_ok)
            edge_color = {0:"#2ca02c",1:"#ffbf00",2:"#d62728"}[cur_status]
            ax.scatter([distance_km_oneway],[speed_kmh], marker="o", s=90, edgecolors=edge_color, facecolors="none", linewidths=2, zorder=5)
            status_label = {0:"OK",1:"Borderline",2:"Non-compliant"}[cur_status]
            ax.annotate(f"{RELAY_OPTIONS[cur_idx]} • {status_label}", (distance_km_oneway, speed_kmh),
                        xytext=(8,8), textcoords="offset points", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=edge_color, lw=1), zorder=6)
            legend_handles = [Patch(color=cmap(i), label=RELAY_OPTIONS[i]) for i in range(len(RELAY_OPTIONS))]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)
            st.pyplot(fig)

# ---------------------------------------
# COSTS TAB — Infrastructure & Transport (index 3)
# ---------------------------------------
with tabs[3]:
    st.title("Infrastructure & Transport Costs")
    summary_ribbon()
    st.caption("Annuitisation via CRF to compute **Levelized Cost ($/GJ)**. CAPEX inputs show a helper in $ millions.")

    # Pull latest calcs for counts and km
    calcs_cost = st.session_state.get("latest_calcs_cache")
    if not calcs_cost:
        calcs_cost = compute_calcs(
            daily_energy_gj, st.session_state.get("distance_km_oneway", 900.0), st.session_state.get("speed_kmh", 75.0),
            st.session_state.get("hours_per_day", 24.0),
            a_cap_for_planner, b_cap_for_planner,
            ROADTRAIN_MAP[st.session_state.get("combo","A-triple")]["A"],
            ROADTRAIN_MAP[st.session_state.get("combo","A-triple")]["B"],
            st.session_state.get("fill_rate_gjph", 41.67),
            st.session_state.get("concurrent_bays_per_set", 3),
            st.session_state.get("unload_time_h", 12.0),
            st.session_state.get("available_bays_A", 12),
            st.session_state.get("available_unload_bays_B", 4),
            st.session_state.get("changeover_overhead_h", 0.25),
            st.session_state.get("mode", "DropAndPull"),
            12.0,
            st.session_state.get("crew_change_overhead_h",0.25),
            st.session_state.get("delay_buffer_min",30),
            st.session_state.get("truck_util_target",0.85),
            st.session_state.get("fatigue_regime","Standard Hours (Solo Drivers)"),
            st.session_state.get("relay_location","No relay")
        )

    sets_per_day = calcs_cost["sets_per_day"]
    truck_km_per_day = calcs_cost["truck_km_per_day"]
    trailer_km_per_day = calcs_cost["trailer_km_per_day"]
    active_trucks = calcs_cost["active_trucks"]
    a_trailers = calcs_cost["a_trailers"]
    b_trailers = calcs_cost["b_trailers"]
    combos_in_circ = calcs_cost["sets_in_circulation"]

    dollies_per_combo = DOLLIES_PER_COMBO.get(st.session_state.get("combo","A-triple"), 0)
    total_dollies = int(math.ceil(combos_in_circ * dollies_per_combo))

    annual_gj = daily_energy_gj * DAYS_PER_YEAR

    cA, cB = st.columns(2, vertical_alignment="top")

    # ------------- Infrastructure (Mother/Daughter + fleet CAPEX, OPEX) -------------
    with cA:
        st.subheader("Infrastructure Costs")
        st.markdown("**Mother Station – CAPEX ($)**")
        ms_comp = st.number_input("Compression plant CAPEX", min_value=0.0, value=5_000_000.0, step=50_000.0, format="%.0f")
        st.caption(f"= **${ms_comp/1_000_000:.3f} million**")
        ms_utils = st.number_input("Utilities CAPEX", min_value=0.0, value=500_000.0, step=10_000.0, format="%.0f")
        st.caption(f"= **${ms_utils/1_000_000:.3f} million**")
        ms_gcond = st.number_input("Gas conditioning CAPEX", min_value=0.0, value=750_000.0, step=10_000.0, format="%.0f")
        st.caption(f"= **${ms_gcond/1_000_000:.3f} million**")
        ms_bay_cost = st.number_input("Filling bay cost per bay", min_value=0.0, value=300_000.0, step=10_000.0, format="%.0f")
        st.caption(f"(per bay) = **${ms_bay_cost/1_000_000:.3f} million**")
        ms_bays_total = ms_bay_cost * st.session_state.get("available_bays_A", 12)
        st.caption(f"Filling bays total ({st.session_state.get('available_bays_A', 12)} ×): **${ms_bays_total/1_000_000:.3f} million**")
        ms_capex = ms_comp + ms_utils + ms_gcond + ms_bays_total

        st.markdown("**Mother Station – OPEX**")
        ms_opex_pct_percent = st.number_input("OPEX as % of Mother CAPEX (/year)", min_value=0.0, max_value=100.0, value=5.0, step=0.5, format="%.1f")
        ms_opex_year = ms_capex * (ms_opex_pct_percent / 100.0)

        st.markdown("**Mother Station – Fuel/Power for compression**")
        ms_energy_method = st.radio("Energy costing method", ["Fuel price ($/GJ fuel)", "Electricity price ($/kWh)"], index=0)
        if ms_energy_method.startswith("Fuel"):
            ms_fuel_price_per_gj = st.number_input("Fuel price ($/GJ)", min_value=0.0, value=0.0, step=0.5)  # default 0.0
            ms_energy_intensity_gj_per_gj = st.number_input("Station energy intensity (GJ fuel per GJ delivered)", min_value=0.0, value=0.02, step=0.005, format="%.3f")
            ms_energy_cost_year = annual_gj * ms_energy_intensity_gj_per_gj * ms_fuel_price_per_gj
        else:
            ms_elec_price_per_kwh = st.number_input("Electricity price ($/kWh)", min_value=0.0, value=0.20, step=0.01)
            ms_kwh_per_gj = st.number_input("Station energy intensity (kWh per GJ delivered)", min_value=0.0, value=5.6, step=0.1)
            ms_energy_cost_year = annual_gj * ms_kwh_per_gj * ms_elec_price_per_kwh

        st.markdown("---")
        st.markdown("**Daughter Station – CAPEX ($)**")
        st.caption(f"Unloading bays (count) is taken from the sidebar: **{st.session_state.get('available_unload_bays_B', 4)}**")
        ds_bay_cost = st.number_input("Unloading bay cost per bay", min_value=0.0, value=300_000.0, step=10_000.0, format="%.0f")
        st.caption(f"(per bay) = **${ds_bay_cost/1_000_000:.3f} million**")
        ds_capex = st.session_state.get('available_unload_bays_B', 4) * ds_bay_cost
        st.caption(f"Unloading bays total: **${ds_capex/1_000_000:.3f} million**")
        ds_opex_pct_percent = st.number_input("OPEX as % of Daughter CAPEX (/year)", min_value=0.0, max_value=100.0, value=5.0, step=0.5, format="%.1f")
        ds_opex_year = ds_capex * (ds_opex_pct_percent / 100.0)

        st.markdown("---")
        st.markdown("**Fleet CAPEX ($)**")
        truck_capex_unit = st.number_input("Truck CAPEX per unit", min_value=0.0, value=550_000.0, step=10_000.0, format="%.0f")
        st.caption(f"= **${truck_capex_unit/1_000_000:.3f} million per truck**")
        a_trailer_capex_unit = st.number_input("A-trailer CAPEX per unit", min_value=0.0, value=800_000.0, step=10_000.0, format="%.0f")
        st.caption(f"= **${a_trailer_capex_unit/1_000_000:.3f} million per A-trailer**")
        b_trailer_capex_unit = st.number_input("B-trailer CAPEX per unit", min_value=0.0, value=650_000.0, step=10_000.0, format="%.0f")
        st.caption(f"= **${b_trailer_capex_unit/1_000_000:.3f} million per B-trailer**")
        st.caption(f"Dollies per combination (auto): **{dollies_per_combo}**")
        dolly_capex_unit = st.number_input("Dolly CAPEX per unit (if used)", min_value=0.0, value=80_000.0, step=5_000.0, format="%.0f")
        st.caption(f"= **${dolly_capex_unit/1_000_000:.3f} million per dolly**")

        fleet_truck_capex = active_trucks * truck_capex_unit
        fleet_a_trailer_capex = a_trailers * a_trailer_capex_unit
        fleet_b_trailer_capex = b_trailers * b_trailer_capex_unit
        fleet_dolly_capex = total_dollies * dolly_capex_unit
        fleet_capex = fleet_truck_capex + fleet_a_trailer_capex + fleet_b_trailer_capex + fleet_dolly_capex

        st.caption(f"Fleet totals — Trucks: **${fleet_truck_capex/1_000_000:.3f}M**, "
                   f"A-trailers: **${fleet_a_trailer_capex/1_000_000:.3f}M**, "
                   f"B-trailers: **${fleet_b_trailer_capex/1_000_000:.3f}M**, "
                   f"Dollies: **${fleet_dolly_capex/1_000_000:.3f}M**")

    # ------------- Transport ops (service, fuel, drivers, insurance) -------------
    with cB:
        st.subheader("Transport Costs (annualised)")
        st.markdown("**Maintenance**")

        maint_basis_trk = st.radio("Truck maintenance basis", ["$/km", "$/year"], index=0, horizontal=True)
        if maint_basis_trk == "$/km":
            trk_maint_per_km = st.number_input("Truck maintenance ($/km)", min_value=0.0, value=0.35, step=0.05)
            trk_maint_year = trk_maint_per_km * truck_km_per_day * DAYS_PER_YEAR
        else:
            trk_maint_year = st.number_input("Truck maintenance ($/year)", min_value=0.0, value=300_000.0, step=10_000.0, format="%.0f")

        maint_basis_trl = st.radio("Trailer maintenance basis", ["$/km", "$/year"], index=0, horizontal=True)
        if maint_basis_trl == "$/km":
            trl_maint_per_km = st.number_input("Trailer maintenance ($/km)", min_value=0.0, value=0.10, step=0.02)
            trl_maint_year = trl_maint_per_km * trailer_km_per_day * DAYS_PER_YEAR
        else:
            trl_maint_year = st.number_input("Trailer maintenance ($/year)", min_value=0.0, value=200_000.0, step=10_000.0, format="%.0f")

        st.markdown("**Fuel**")
        truck_km_per_litre = st.number_input("Truck consumption (km/L)", min_value=0.01, value=1.7, step=0.1)
        diesel_price_per_l = st.number_input("Diesel price ($/L)", min_value=0.0, value=2.10, step=0.05)
        litres_per_day = truck_km_per_day / max(1e-6, truck_km_per_litre)
        fuel_cost_year = litres_per_day * diesel_price_per_l * DAYS_PER_YEAR

        st.markdown("**Drivers**")
        wage_per_driver_day = st.number_input("Wages per driver ($/day)", min_value=0.0, value=600.0, step=25.0)
        fifo_per_driver_day = st.number_input("FIFO costs per driver ($/day)", min_value=0.0, value=120.0, step=10.0)
        camp_per_driver_day = st.number_input("Camp accommodation per driver ($/day)", min_value=0.0, value=150.0, step=10.0)
        drivers_on_duty = calcs_cost["drivers_on_duty"]
        driver_cost_year = (wage_per_driver_day + fifo_per_driver_day + camp_per_driver_day) * drivers_on_duty * DAYS_PER_YEAR

        st.markdown("**Insurance & Misc.**")
        insurance_year = st.number_input("Insurance ($/year)", min_value=0.0, value=300_000.0, step=10_000.0, format="%.0f")
        misc_year = st.number_input("Miscellaneous ($/year)", min_value=0.0, value=150_000.0, step=10_000.0, format="%.0f")

        st.markdown("---")
        st.subheader("Finance / Levelization")
        interest_rate_percent = st.number_input("Interest rate (real, per year) [%]", min_value=0.0, max_value=100.0, value=8.0, step=0.5, format="%.1f")
        interest_rate = interest_rate_percent / 100.0
        asset_life_months = st.number_input("NPER (months) for CAPEX recovery", min_value=3, value=max(60, int(project_months)), step=1)
        asset_life_years = asset_life_months / 12.0
        crf = capital_recovery_factor(interest_rate, asset_life_years)

    # ---- Rollups
    infra_capex_total = (ms_comp + ms_utils + ms_gcond + ms_bays_total) + ds_capex + (fleet_truck_capex + fleet_a_trailer_capex + fleet_b_trailer_capex + fleet_dolly_capex)
    infra_opex_year = ms_opex_year + ds_opex_year
    station_energy_year = ms_energy_cost_year
    transport_ops_year = trk_maint_year + trl_maint_year + fuel_cost_year + driver_cost_year + insurance_year + misc_year

    annuitized_capex_year = infra_capex_total * crf
    total_annual_cost = annuitized_capex_year + infra_opex_year + station_energy_year + transport_ops_year

    levelized_cost_per_gj = (total_annual_cost / max(daily_energy_gj * DAYS_PER_YEAR, 1e-9)) if daily_energy_gj > 0 else float("nan")
    st.session_state["lc_per_gj"] = float(levelized_cost_per_gj)

    st.markdown("### Annualised Cost Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Annuitised CAPEX (/yr)", f"${annuitized_capex_year:,.0f}")
    s2.metric("OPEX + Station energy (/yr)", f"${(infra_opex_year + station_energy_year):,.0f}")
    s3.metric("Transport ops (/yr)", f"${transport_ops_year:,.0f}")
    st.metric("Total annual cost", f"${total_annual_cost:,.0f}")
    st.metric("Levelized Cost ($/GJ)", f"${levelized_cost_per_gj:,.2f}")

    # Re-render ribbon now that LC is known
    summary_ribbon()

# ---------------------------------------
# BENEFITS TAB — Gas sales & carbon avoidance (index 4)
# ---------------------------------------
with tabs[4]:
    st.title("Benefits & Carbon")
    summary_ribbon()
    st.caption("Compute **Total Levelized Benefit ($/GJ)** and explore CO₂-e for venting vs flaring vs capture/sale.")

    annual_gj = daily_energy_gj * DAYS_PER_YEAR

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Gas Sales / Value")
        gas_price_per_gj = st.number_input("Gas sales value ($/GJ delivered)", min_value=0.0, value=12.0, step=0.5)
        other_benefit_year = st.number_input("Other operational benefits ($/yr)", min_value=0.0, value=0.0, step=10_000.0, format="%.0f")
        gas_revenue_year = gas_price_per_gj * annual_gj

    with b2:
        st.subheader("Carbon Emissions & Avoidance")
        st.markdown("**Baseline handling scenario** (what would have happened without capture):")
        baseline = st.radio("Baseline", ["Venting", "Flaring"], index=0, help="Select the counterfactual practice the project avoids.")

        # Emission factors — placeholders; adjust as needed
        if st.session_state.get("gas_type_selected","Methane") == "Hydrogen":
            vent_co2e_t_per_gj_default = 0.00
            flare_co2_t_per_gj_default = 0.00
        else:
            vent_co2e_t_per_gj_default = 0.05   # tCO2e/GJ placeholder
            flare_co2_t_per_gj_default  = 0.002 # tCO2/GJ placeholder

        vent_co2e_t_per_gj = st.number_input("Venting CO₂-e factor (tCO₂-e/GJ)", min_value=0.0, value=vent_co2e_t_per_gj_default, step=0.001, format="%.3f")
        flare_co2_t_per_gj = st.number_input("Flaring CO₂ factor (tCO₂/GJ)",   min_value=0.0, value=flare_co2_t_per_gj_default,  step=0.001, format="%.3f")

        carbon_price = st.number_input("Carbon value ($/tCO₂-e)", min_value=0.0, value=30.0, step=1.0)

        # Annual baseline emissions (tCO2e/yr)
        if baseline == "Venting":
            baseline_tco2e_year = vent_co2e_t_per_gj * annual_gj
        else:
            baseline_tco2e_year = flare_co2_t_per_gj * annual_gj

        project_tco2e_year = 0.0  # capture/sale assumption (editable later)
        avoided_tco2e_year = max(baseline_tco2e_year - project_tco2e_year, 0.0)
        carbon_benefit_year = avoided_tco2e_year * carbon_price

    total_benefit_year = gas_revenue_year + carbon_benefit_year + other_benefit_year
    levelized_benefit_per_gj = (total_benefit_year / max(annual_gj, 1e-9)) if annual_gj > 0 else float("nan")
    st.session_state["lb_per_gj"] = float(levelized_benefit_per_gj)

    st.markdown("### Annual Benefit Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gas value (/yr)", f"${gas_revenue_year:,.0f}")
    c2.metric("Carbon avoidance (/yr)", f"${carbon_benefit_year:,.0f}")
    c3.metric("Other (/yr)", f"${other_benefit_year:,.0f}")
    st.metric("Total annual benefit", f"${total_benefit_year:,.0f}")
    st.metric("Levelized Benefit ($/GJ)", f"${levelized_benefit_per_gj:,.2f}")

    # Re-render ribbon now that LB is known
    summary_ribbon()

    with st.expander("Carbon calculations (details)"):
        st.write(f"Baseline: **{baseline}**")
        st.write(f"Baseline CO₂-e (t/yr): **{baseline_tco2e_year:,.0f}**")
        st.write(f"Project CO₂-e (t/yr): **{project_tco2e_year:,.0f}**")
        st.write(f"Avoided CO₂-e (t/yr): **{avoided_tco2e_year:,.0f}**")
        st.write(f"Carbon value ($/yr): **${carbon_benefit_year:,.0f}**")

# ---------------------------------------
# Scenarios tab (index 5)
# ---------------------------------------
with tabs[5]:
    st.subheader("Scenarios (save/load)")
    summary_ribbon()
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
        "available_bays_A": available_bays_A,
        "available_unload_bays_B": available_unload_bays_B,
        "changeover_overhead_h": changeover_overhead_h,
        "driver_shift_cap": 12.0,
        "crew_change_overhead_h": st.session_state.get("crew_change_overhead_h", 0.25),
        "delay_buffer_min": st.session_state.get("delay_buffer_min", 30),
        "truck_util_target": truck_util_target,
        "gas_type": st.session_state.get("gas_type_selected", "Methane"),
        "hv_basis": st.session_state.get("hv_basis_key", "Gross"),
        "reference_condition": st.session_state.get("ref_choice", "Standard (15 °C, 1.01325 barₐ)"),
        "fatigue_regime": st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)"),
        "relay_location": st.session_state.get("relay_location", "No relay"),
        # Latest $/GJ if already computed in this session:
        "lc_per_gj": st.session_state.get("lc_per_gj", None),
        "lb_per_gj": st.session_state.get("lb_per_gj", None),
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

# Footer
st.caption("v13.9 — Instructions tab + top summary ribbon. Costs/benefits persist in session_state so the ribbon is visible on all tabs.")
