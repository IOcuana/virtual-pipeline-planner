# v15_8 — Input diagnostics table + traceability to ribbon pills
import math

import numpy as np
import pandas as pd
import streamlit as st

# ⬇️ add these two lines right here
from calcs import compute_calcs
from finance import capital_recovery_factor
from finance_helpers import compute_levelized_metrics_for_scenario

# Optional CoolProp for real-gas properties — use the authoritative flag from physics
try:
    from physics import _COOLPROP_OK as COOLPROP_OK
except Exception:
    COOLPROP_OK = False

# Optional Matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    MPL_OK = True
except Exception:
    MPL_OK = False

st.set_page_config(page_title="Virtual Pipeline Planner", layout="wide")

from textwrap import dedent  # (keep once near here if not already imported)

# ---------------------------------------------------------------------
# Driver Utilisation Chart Helper
# ---------------------------------------------------------------------


# --- Helper: capacity to feed the planner (usable if available, else total @ PW) ---
def _get_caps_for_planner():
    a = float(st.session_state.get("a_cap_usable_gj", st.session_state.get("a_cap_calc_gj", 0.0)))
    b = float(st.session_state.get("b_cap_usable_gj", st.session_state.get("b_cap_calc_gj", 0.0)))
    # persist for cross-tab reuse
    st.session_state["a_cap_for_planner"] = a
    st.session_state["b_cap_for_planner"] = b
    return a, b


from constants import (
    DAYS_PER_YEAR,
    DOLLIES_PER_COMBO,
    FT3_PER_M3,
    GJ_PER_MMBTU,
    ROADTRAIN_MAP,
    STEER_TYRES_PER_TRUCK,
    DRIVE_TYRES_PER_TRUCK,
    TRAILER_TYRES_PER_A,
    TRAILER_TYRES_PER_B,
)

from fatigue import (
    RELAY_OPTIONS_DISPLAY,
    RELAY_DISPLAY_TO_LEGACY,
    best_mode_for,
    compliance_code_for,   # <-- used to score each relay option
)
from physics import density_kg_per_m3
from ui_helpers import (
    _inject_planner_expander_css,
    _inject_sidebar_expander_css,
    currency_commas,
    help_flyout,
    number_commas,
    render_driver_utilisation_section,
    summary_ribbon,
)

# ------------------------------
# Constants & helpers
# ------------------------------

def _current_driver_shift_cap() -> float:
    """
    Map the selected fatigue regime to the driver shift cap (hours).
    - Standard Hours (Solo Drivers): 12h
    - Basic Fatigue Management (BFM): 14h
    - Advanced Fatigue Management (AFM): 16h
    """
    regime = st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)")

    # Normalise common label variants (our sidebar adds "– placeholder")
    normalised = {
        "Standard Hours (Solo Drivers)": "Standard Hours (Solo Drivers)",
        "Basic Fatigue Management (BFM)": "Basic Fatigue Management (BFM)",
        "Basic Fatigue Management (BFM) – placeholder": "Basic Fatigue Management (BFM)",
        "Advanced Fatigue Management (AFM)": "Advanced Fatigue Management (AFM)",
        "Advanced Fatigue Management (AFM) – placeholder": "Advanced Fatigue Management (AFM)",
    }.get(regime, "Standard Hours (Solo Drivers)")

    return {
        "Standard Hours (Solo Drivers)": 12.0,
        "Basic Fatigue Management (BFM)": 14.0,
        "Advanced Fatigue Management (AFM)": 16.0,
    }.get(normalised, 12.0)


with st.sidebar:
    _inject_sidebar_expander_css()
    # 1) Operating Mode & Demand (default open)
    with st.expander("Operating Mode & Demand", expanded=True):
        mode = st.selectbox("Mode", ["DropAndPull", "ThroughRoad"])
        st.session_state["mode"] = mode
        project_months = st.number_input("Project duration [months]", min_value=1, value=60, step=1)
        project_years = project_months / 12.0
        hours_per_day = st.number_input(
            "Operating hours per day [h]",
            min_value=1,
            max_value=24,
            value=24,
            step=1,
            help="Defines the daily operating window for loading and delivery activities. Contributes to the **Utilisation & Duty Cycle** summary ribbon by setting the hours available per day for station and fleet operations.",
        )

        st.session_state["hours_per_day"] = float(hours_per_day)

        daily_energy_gj = number_commas(
            "Daily gas to transport [GJ/day]",
            key="daily_energy_gj",
            value=10000.00,
            decimals=2,
            help="Energy delivered per day in gigajoules.",
        )
        # --- NEW: Utilisation ---
        utilisation_pct = st.number_input(
            "Utilisation [% of plan]",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
            help="Share of the planned daily gas that is actually delivered over time (downtime, weather, etc.).",
        )
        utilisation_frac = utilisation_pct / 100.0

        # Keep "planned" daily as-is; compute an "expected" daily for revenue and other actuals
        effective_daily_gj = float(daily_energy_gj) * utilisation_frac

        # Make available to other tabs
        st.session_state["utilisation_frac"] = utilisation_frac
        st.session_state["effective_daily_gj"] = effective_daily_gj

        st.caption(
            "Notes: Revenue and other ‘actuals’ use **Daily gas to transport × Utilisation%**. "
            "Planned capacities (fleet sizing, fills, etc.) still use the full daily figure."
        )

        # Reference condition for conversions
        ref_choice = st.radio(
            "Conversion reference",
            ["Normal (20 °C, 1.01325 barₐ)", "Standard (15 °C, 1.01325 barₐ)"],
            index=1,
            help="Pick which condition the sidebar volumetric & mass conversions use.",
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

        kg_per_day = (
            rho_S * Sm3_per_day if ref_choice.startswith("Standard") else rho_N * Nm3_per_day
        )
        MMBTU_per_day = daily_energy_gj / GJ_PER_MMBTU

        st.caption(f"≈ **{daily_energy_gj / 1000:.3f} TJ/day**")
        if ref_choice.startswith("Standard"):
            st.caption(
                f"≈ **{Sm3_per_day:,.0f} Sm³/day**, **{MMSCFD:,.3f} MMSCF/D**  _(S: 15 °C, 1.01325 barₐ)_"
            )
            st.caption(
                f"≈ **{Nm3_per_day:,.0f} Nm³/day**, **{Nft3_per_day:,.0f} Nft³/day**  _(N: 20 °C, 1.01325 barₐ)_"
            )
        else:
            st.caption(
                f"≈ **{Nm3_per_day:,.0f} Nm³/day**, **{Nft3_per_day:,.0f} Nft³/day**  _(N: 20 °C, 1.01325 barₐ)_"
            )
            st.caption(
                f"≈ **{Sm3_per_day:,.0f} Sm³/day**, **{MMSCFD:,.3f} MMSCF/D**  _(S: 15 °C, 1.01325 barₐ)_"
            )

        kg_tip = (
            "Mass derived from S: ρ(S) × Sm³/day"
            if ref_choice.startswith("Standard")
            else "Mass derived from N: ρ(N) × Nm³/day"
        )
        st.markdown(
            f"""<span title="{kg_tip}">≈ <b>{kg_per_day:,.0f} kg/day</b> <sup>ⓘ</sup></span>
                 &nbsp; • &nbsp; <b>{MMBTU_per_day:,.0f} MMBTU/day</b>""",
            unsafe_allow_html=True,
        )
    # 2) Route (default open)
    with st.expander("Route", expanded=True):
        st.caption("Mother = Gas production location, Daughter = Gas delivery location.")
        distance_km_oneway = st.number_input("Distance Mother→Daughter [km]", min_value=0.01, value=900.0, step=10.0)
        speed_kmh = st.number_input("Average driving speed [km/h]", min_value=0.01, max_value=110.0, value=75.0,
                                    step=1.0)

        route_hours_per_day = st.number_input(
            "Operating hours per day (route) [h]", min_value=1, max_value=24, value=24, step=1,
            help="Maximum driving window per day. Reduce for daylight-only or restricted routes."
        )
        route_days_per_year = st.number_input(
            "Operating days per year (route) [days]", min_value=1, max_value=365, value=365, step=1,
            help="Annual driving availability for the route (e.g., holiday shutdowns)."
        )
        st.session_state["route_hours_per_day"] = float(route_hours_per_day)
        st.session_state["route_days_per_year"] = float(route_days_per_year)

        # Persist for cross-tab reuse
        st.session_state["distance_km_oneway"] = float(distance_km_oneway)
        st.session_state["speed_kmh"] = float(speed_kmh)

    # 3) Road Train Combination (collapsed)
    with st.expander("Road Train Combination", expanded=False):
        combo_names = list(ROADTRAIN_MAP.keys())
        default_idx = combo_names.index("A-triple") if "A-triple" in ROADTRAIN_MAP else 0
        combo = st.selectbox(
            "Combination", combo_names, index=default_idx, help="A = A-trailer, B = B-trailer"
        )
        a_count = ROADTRAIN_MAP[combo]["A"]
        b_count = ROADTRAIN_MAP[combo]["B"]
        trailers_per_set = a_count + b_count
        st.caption(
            f"Selected: **{combo}** → A: **{a_count}**, B: **{b_count}** (Total {trailers_per_set})"
        )
        st.session_state["combo"] = combo

    # 4) Terminal Ops (Mother) (collapsed)
    with st.expander("Terminal Ops (Mother)", expanded=False):
        fill_rate_gjph = st.number_input(
            "Per-bay fill rate [GJ/h]", min_value=0.001, value=41.67, step=0.01, format="%.2f"
        )
        concurrent_bays_per_set = st.number_input(
            "Concurrent bays per set (trailers filled in parallel)",
            min_value=1,
            max_value=8,
            value=min(3, trailers_per_set) if trailers_per_set > 0 else 1,
            step=1,
        )
        available_bays_A = st.number_input(
            "Available filling bays at Mother [count]", min_value=0, value=12, step=1
        )
        changeover_overhead_h = st.number_input(
            "Trailer swap / handling overhead per visit [h]",
            min_value=0.0,
            value=0.25,
            step=0.05,
            key="changeover_overhead_mother",
        )

        # Persist Mother ops for downstream tabs
        st.session_state["fill_rate_gjph"] = float(fill_rate_gjph)
        st.session_state["concurrent_bays_per_set"] = int(concurrent_bays_per_set)
        st.session_state["available_bays_A"] = int(available_bays_A)
        st.session_state["changeover_overhead_h"] = float(changeover_overhead_h)

        # NEW: Mother station operating window
        ms_hours_per_day = st.number_input(
            "Operating hours per day (Mother) [h]",
            min_value=1, max_value=24, value=24, step=1,
            help="Defines daily availability of the Mother Station. Defaults to 24/7."
        )
        ms_days_per_year = st.number_input(
            "Operating days per year (Mother) [days]",
            min_value=1, max_value=365, value=365, step=1,
            help="Annual availability of the Mother Station. Defaults to 24/7."
        )
        st.session_state["ms_hours_per_day"] = float(ms_hours_per_day)
        st.session_state["ms_days_per_year"] = float(ms_days_per_year)

    # 5) Terminal Ops (Daughter) (collapsed)
    with st.expander("Terminal Ops (Daughter)", expanded=False):

        # Inputs (ordered to match Mother)
        unload_rate_gjph = st.number_input(
            "Per-bay unload rate [GJ/h]",
            min_value=0.001,
            value=41.67,
            step=0.01,
            format="%.2f",
            key="unload_rate_gjph_v14",
        )
        unload_concurrent_bays = st.number_input(
            "Concurrent bays per set (trailers unloaded in parallel)",
            min_value=1,
            max_value=8,
            value=min(3, trailers_per_set) if trailers_per_set > 0 else 1,
            step=1,
            key="unload_concurrent_bays_v14",
        )
        available_unload_bays_B = st.number_input(
            "Available unloading bays at Daughter [count]",
            min_value=0,
            value=12,
            step=1,
        )
        changeover_overhead_h_d = st.number_input(
            "Trailer swap / handling at Daughter [h]",
            min_value=0.0,
            value=0.25,
            step=0.05,
            key="changeover_overhead_h_daughter",
        )
        ds_hours_per_day = st.number_input(
            "Operating hours per day (Daughter) [h]",
            min_value=1,
            max_value=24,
            value=24,
            step=1,
            help="Daily availability of the Daughter Station. Defaults to 24/7.",
        )
        ds_days_per_year = st.number_input(
            "Operating days per year (Daughter) [days]",
            min_value=1,
            max_value=365,
            value=365,
            step=1,
            help="Annual availability of the Daughter Station. Defaults to 24/7.",
        )

        # Persist Daughter ops for downstream tabs
        st.session_state["available_unload_bays_B"] = int(available_unload_bays_B)
        st.session_state["unload_rate_gjph"] = float(unload_rate_gjph)
        st.session_state["unload_concurrent_bays"] = int(unload_concurrent_bays)
        st.session_state["ds_hours_per_day"] = float(ds_hours_per_day)
        st.session_state["ds_days_per_year"] = float(ds_days_per_year)

        # Compute unload time per set at Daughter:
        # per-set energy (GJ) uses usable capacity when available; else total @ PW.
        comb_key = st.session_state.get("combo", "A-triple")
        a_count = ROADTRAIN_MAP[comb_key]["A"]
        b_count = ROADTRAIN_MAP[comb_key]["B"]
        a_gj = float(st.session_state.get("a_cap_usable_gj", st.session_state.get("a_cap_calc_gj", 0.0)))
        b_gj = float(st.session_state.get("b_cap_usable_gj", st.session_state.get("b_cap_calc_gj", 0.0)))
        per_set_capacity_gj_sidebar = a_count * a_gj + b_count * b_gj

        effective_unload_rate_gjph = float(unload_rate_gjph) * max(1, int(unload_concurrent_bays))
        unload_proc_time_h = per_set_capacity_gj_sidebar / max(effective_unload_rate_gjph, 1e-9)
        unload_time_h_calc = unload_proc_time_h + float(changeover_overhead_h_d)

        # Persist computed unload time for downstream calcs & displays
        st.session_state["unload_time_h"] = float(unload_time_h_calc)

    # 6) Drivers & Compliance (collapsed)
    with st.expander("Drivers & Compliance", expanded=False):
        fatigue_regime = st.selectbox(
            "Fatigue regime",
            [
                "Standard Hours (Solo Drivers)",
                "Basic Fatigue Management (BFM) – placeholder",
                "Advanced Fatigue Management (AFM) – placeholder",
            ],
            index=0,
        )
        # Persist current fatigue regime (relay strategy now lives on the Planner tab)
        st.session_state["fatigue_regime"] = fatigue_regime

        crew_change_overhead_h = st.number_input(
            "Crew change overhead at swap [h]", min_value=0.0, value=0.25, step=0.05
        )
        delay_buffer_min = st.number_input(
            "Target compliance buffer per leg [min]",
            min_value=0,
            value=30,
            step=5,
            help="Adds this many minutes to each road leg as a compliance margin; lengthens cycle time and affects fleet size and costs."
        )
        st.session_state["crew_change_overhead_h"] = float(crew_change_overhead_h)
        st.session_state["delay_buffer_min"] = float(delay_buffer_min)

        relief_factor = st.number_input(
            "Driver relief factor",
            min_value=1.0,
            max_value=2.0,
            value=1.00,
            step=0.05,
            key="relief_factor",
            help="Multiplier for extra roster coverage (leave, training, absenteeism). 1.00 = no relief; 1.15–1.25 typical for continuous ops."
        )

    # 7) Utilisation Target (collapsed)
    with st.expander("Utilisation Target", expanded=False):
        truck_util_target = st.number_input(
            "Truck duty utilisation target [0–1]",
            min_value=0.01,
            max_value=1.0,
            value=0.85,
            step=0.01,
            help="Target average on-duty utilisation per truck. Used to scale required fleet size — lower targets require more trucks to meet demand.",
            key="truck_util_target",
        )

# ---------------------------------------
# Tabs (Instructions first)
# ---------------------------------------
tabs = st.tabs(
    [
        "Instructions",
        "Payload & Capacity",
        "Planner",
        "Infrastructure & Transport Costs",
        "Benefits & Carbon",
        "Scenarios",
    ]
)


# ---------------------------------------
# INSTRUCTIONS (first tab)
# ---------------------------------------
with tabs[0]:
    st.markdown("## Instructions")
    summary_ribbon("24px")
    st.subheader("Definitions")
    st.markdown(
        """
**Operating modes**

- **DropAndPull** — Each driver drops off a loaded trailer set at the **Daughter** site, collects an empty set, and returns to the **Mother** site. Trailers are swapped and drivers cycle continuously.
- **ThroughRoad** — Drivers follow a through-route: the same loaded trailers are delivered and **unloaded** before returning to refill. Fewer swaps; potentially higher driving duty time.

**Summary Ribbon Pills** (shown at the top of each tab)

- **Levelized Cost [$/GJ]** — Total annualized cost (CAPEX + OPEX + energy + transport) divided by annual GJ delivered.
- **Levelized Benefit [$/GJ]** — Total annual benefit (gas revenue + carbon avoidance + other) divided by annual GJ delivered.
- **Margin [$/GJ]** — Levelized Benefit minus Levelized Cost.
- **Trucking Cost (excl. stations) [$/GJ]** — Transport-only costs (fleet, fuel, maintenance, drivers) per GJ delivered.

---


### 1) Purpose
This app plans a **virtual pipeline** for compressed gas (Methane or Hydrogen) using modular road-train combinations.  
It estimates **fleet size**, **bay utilisation**, **driver compliance**, and **levelized cost/benefit** (incl. carbon).

The app helps you:
- Size your fleet and infrastructure based on throughput requirements
- Understand cost drivers with detailed breakdowns and visualizations
- Evaluate economic viability and carbon benefits
- Compare scenarios and optimize design parameters

---

### 2) Quick start
1. **Sidebar → Operating Mode & Demand:** enter daily GJ, distance, speed, operating hours, and utilisation %.  
2. **Sidebar → Road Train Combination:** pick e.g. *A-triple* (Mother = Fill site, Daughter = Unload site).  
3. **Sidebar → Availability & Fatigue:** set route/station operating hours, fatigue regime, and relay options.
4. **Payload & Capacity tab:** set pressures, cylinder sizes, gas type; review capacity (GJ & kg).  
5. **Planner tab:** see trucks, combinations in circulation, trailers, bays, compliance, and driver utilisation.  
6. **Infrastructure & Transport Costs tab:** enter CAPEX/OPEX, maintenance, fuel, drivers, insurance, etc.  
7. **Benefits & Carbon tab:** set gas revenue and carbon avoidance assumptions.  
8. **Scenarios tab:** save/load scenarios for comparison and run sensitivity analyses.
9. **Instructions tab (below):** explore the Input Diagnostics table and Cost Breakdown pie charts for insights.

---

### 3) Tips
- **Usable capacity = from $P_{W} \\rightarrow P_{min}$**, total capacity is @ $P_W$.
- **Drivers & Compliance** follow NHVR *Standard Hours (Solo)* logic with relay options (No relay, Before/After Daughter, Midpoint).
- Toggle **Normal (20 °C)** vs **Standard (15 °C)** conditions in the sidebar for conversions.
- **Dollies per combination** are auto-derived from the selection (Mother/Daughter mix).
- Fill/Unload **bay utilisation** appears on the **Planner** tab with visual indicators (🟢 OK, 🔴 Warning).
- Levelized cost uses **CRF** (Capital Recovery Factor) based on interest rate % and loan term (NPER months).
- **Multi-plan finance:** Allocate different CAPEX items to different finance plans (up to 10) with varying rates and terms.
- **Timing conventions:** Two PV bases are shown (where available):
  - **Start-of-Operations (COD) basis** – pre-build CAPEX is compounded forward to the start of operations; OPEX and revenue are discounted over operating years from that point.
  - **Today (FID) basis** – the same streams are discounted an additional two years back to the investment-decision date (today).
- **OPEX breakdowns:** Mother and Daughter stations are displayed separately in Cost Summary. Transport OPEX is split into Fuel, Maintenance, Drivers, Insurance, and Miscellaneous.
- **Utilisation [% of plan]:** Adjusts expected delivered GJ and therefore all revenues and benefits; planner sizing still uses planned daily throughput.
- **Driver wage unit:** Toggle between $/day, $/hr, and $/year. FIFO and camp costs remain per day.
- **Tyre costs:** Automatically included in maintenance totals based on replacement intervals and fleet km.
- **Summary ribbon consistency:** All tabs show the same ribbon values, computed from current inputs using the finance_helpers module.

---

### 4) Outputs (where to look)
- **Instructions tab (this page):**
  - **Input Diagnostics table** — traces all user inputs to their impact on summary ribbon pills and other outputs.
  - **Cost Breakdown pie charts** — visual breakdown of Levelized Cost and Trucking Cost components.
  
- **Payload & Capacity tab:** 
  - Trailer & combination capacity (GJ, kg; total vs usable).
  - Gas properties at working conditions.
  - Payload weight calculations.
  
- **Planner tab:** 
  - Active trucks, combinations in circulation, Mother/Daughter trailers.
  - Bay requirements and utilization percentages.
  - Driver compliance codes and relay recommendations.
  - Driver utilisation chart showing duty cycles.
  
- **Infrastructure & Transport Costs tab:** 
  - CAPEX inputs (Mother/Daughter stations, fleet).
  - Multi-plan finance allocation.
  - OPEX inputs (stations, maintenance, fuel, drivers, insurance, registration).
  - Cost Summary with annuitized CAPEX, OPEX, and project totals.
  - Finance costs and repayments breakdown.
  - **Levelized Cost [$/GJ]** calculation.
  
- **Benefits & Carbon tab:** 
  - Gas sales revenue and carbon pricing.
  - Baseline vs project emissions comparison.
  - Carbon avoidance calculations (tCO₂-e/yr).
  - **Levelized Benefit [$/GJ]** calculation.
  - Economic context showing benefit components.
  
- **Scenarios tab:** 
  - Save and reload complete parameter sets.
  - Scenario comparison table with configurable outputs.
  - X/Y plotting with dual-axis support.
  - Independent variable sweep (Distance, Daily gas, Fill rate, Unload rate).
  - CSV export for offline analysis.

---

### 5) Summary Ribbon Pills (Key Outputs)

The four pills at the top of each tab provide at-a-glance project economics:

1. **Levelized Cost [$/GJ]** = (Annualized CAPEX + Infrastructure OPEX + Station Energy + Transport OPEX) ÷ Annual GJ delivered
2. **Levelized Benefit [$/GJ]** = (Gas Revenue + Carbon Benefit + Other Benefits) ÷ Annual GJ delivered  
3. **Margin [$/GJ]** = Levelized Benefit − Levelized Cost
4. **Trucking Cost [$/GJ]** = (Fleet CAPEX annualized + Fuel + Maintenance + Drivers + Other Transport OPEX) ÷ Annual GJ delivered

These values update automatically as you modify inputs across different tabs.
"""
    )
    
    # ---------------------------------------
    # DIAGNOSTICS TABLE
    # ---------------------------------------
    with st.expander("🔍 Input-to-Output Diagnostics", expanded=False):
        st.markdown("### Input Traceability Matrix")
        st.caption("This table shows how each user input contributes to the summary ribbon pills and other outputs.")
        
        # Build comprehensive input mapping
        # Format: (Input Label, Location, Ribbon Pills, Diagnostic, Other Outputs)
        diagnostics_data = [
            # ===== SIDEBAR INPUTS =====
            ("Mode", "Sidebar", "Levelized Cost, Trucking Cost", "Affects driver scheduling logic (DropAndPull vs ThroughRoad)", "Active trucks, Drivers per truck, Sets per day"),
            ("Project duration [months]", "Sidebar", "Levelized Cost", "Used in CRF calculation for annuitization", "NPV calculations, All project totals"),
            ("Operating hours per day [h]", "Sidebar", "Levelized Cost, Trucking Cost", "Affects bay availability and truck utilization", "Bay utilization, Truck utilization"),
            ("Daily gas to transport [GJ/day]", "Sidebar", "Levelized Cost, Levelized Benefit, Margin, Trucking Cost", "Primary throughput driver for all calculations", "Fleet size, Bay counts, Annual revenue, All $/GJ metrics"),
            ("Utilisation [% of plan]", "Sidebar", "Levelized Cost, Levelized Benefit, Margin", "Adjusts effective throughput for revenue/cost calculations", "Effective daily GJ, Annual revenue, Annual benefit"),
            ("Conversion reference", "Sidebar", "No contribution", "Display-only conversion between Normal/Standard conditions", "Volumetric conversions (Nm³, Sm³, ft³)"),
            ("Distance Mother→Daughter [km]", "Sidebar", "Levelized Cost, Trucking Cost", "Drives trucking time, fuel consumption, fleet size", "Truck km/day, Fuel cost, Active trucks, Sets per day"),
            ("Speed [km/h]", "Sidebar", "Levelized Cost, Trucking Cost", "Affects cycle time and fleet requirements", "Cycle time, Active trucks, Sets per day"),
            ("Road Train Combination", "Sidebar", "Levelized Cost, Trucking Cost", "Defines A/B trailer counts and dolly requirements", "Fleet composition, CAPEX, Dolly counts"),
            ("Fill rate [GJ/h]", "Sidebar", "Levelized Cost, Trucking Cost", "Determines Mother station bay utilization", "Fill time, Bay flag, Mother bays required"),
            ("Concurrent bays per set", "Sidebar", "Levelized Cost", "Affects filling parallelization at Mother", "Fill time per set, Bay utilization"),
            ("Unload time [h]", "Sidebar", "Levelized Cost, Trucking Cost", "Determines Daughter station bay utilization", "Unload bay flag, Daughter bays required, Cycle time"),
            ("Available Mother bays", "Sidebar", "Levelized Cost", "Capacity constraint for Mother station", "Bay utilization %, Bay flag"),
            ("Available Daughter bays", "Sidebar", "Levelized Cost", "Capacity constraint for Daughter station", "Unload bay utilization %, Unload flag"),
            ("Changeover overhead [h]", "Sidebar", "Levelized Cost, Trucking Cost", "Adds time per fill/unload cycle", "Cycle time, Sets per day"),
            ("Fatigue regime", "Sidebar", "Levelized Cost, Trucking Cost", "Sets driver shift cap (12h/14h/16h)", "Driver shift cap, Compliance code, Drivers per truck"),
            ("Relay location", "Sidebar", "Levelized Cost, Trucking Cost", "Affects driver compliance and scheduling", "Compliance code, Relay option, Drivers per truck"),
            ("Crew change overhead [h]", "Sidebar", "Levelized Cost, Trucking Cost", "Time for driver handover at relay point", "Cycle time with relay"),
            ("Delay buffer [min]", "Sidebar", "Levelized Cost, Trucking Cost", "Safety margin added to cycle calculations", "Effective cycle time"),
            ("Truck utilisation target", "Sidebar", "Levelized Cost, Trucking Cost", "Drives fleet sizing (inverse relationship)", "Active trucks, Truck utilization %"),
            
            # ===== PAYLOAD & CAPACITY TAB =====
            ("Gas Type", "Payload & Capacity tab", "Levelized Cost, Levelized Benefit, Margin, Trucking Cost", "Affects density, energy content, safety factors", "Capacity (GJ), Capacity (kg), All mass/energy calculations"),
            ("Heating Value basis", "Payload & Capacity tab", "Levelized Cost, Levelized Benefit, Margin, Trucking Cost", "LHV vs HHV affects energy calculations", "Capacity (GJ), Energy conversions"),
            ("Working Pressure (barₐ)", "Payload & Capacity tab", "Levelized Cost, Trucking Cost", "Defines usable gas mass per trailer", "Usable capacity (GJ), Usable capacity (kg)"),
            ("Min Pressure (barₐ)", "Payload & Capacity tab", "Levelized Cost, Trucking Cost", "Lower bound for usable capacity", "Usable capacity (GJ), Usable capacity (kg)"),
            ("Gas Temperature (°C)", "Payload & Capacity tab", "Levelized Cost, Trucking Cost", "Affects gas density and capacity", "Density, Capacity calculations"),
            ("Cylinder Volume A-trailer [L]", "Payload & Capacity tab", "Levelized Cost, Trucking Cost", "A-trailer capacity driver", "A-trailer capacity (GJ), A-trailer capacity (kg)"),
            ("Cylinder Volume B-trailer [L]", "Payload & Capacity tab", "Levelized Cost, Trucking Cost", "B-trailer capacity driver", "B-trailer capacity (GJ), B-trailer capacity (kg)"),
            ("Cylinders per A-trailer", "Payload & Capacity tab", "Levelized Cost, Trucking Cost", "A-trailer capacity multiplier", "A-trailer total volume, A-trailer capacity"),
            ("Cylinders per B-trailer", "Payload & Capacity tab", "Levelized Cost, Trucking Cost", "B-trailer capacity multiplier", "B-trailer total volume, B-trailer capacity"),
            ("Tare weight A-trailer [kg]", "Payload & Capacity tab", "Levelized Cost", "Affects payload calculations", "Payload weight, GVM compliance"),
            ("Tare weight B-trailer [kg]", "Payload & Capacity tab", "Levelized Cost", "Affects payload calculations", "Payload weight, GVM compliance"),
            
            # ===== INFRASTRUCTURE & TRANSPORT COSTS TAB =====
            ("Compression plant CAPEX [$]", "Infrastructure & Transport Costs tab", "Levelized Cost", "Mother station capital expenditure", "Annuitized CAPEX, Total CAPEX"),
            ("Utilities CAPEX [$]", "Infrastructure & Transport Costs tab", "Levelized Cost", "Mother station utilities capital", "Annuitized CAPEX, Total CAPEX"),
            ("Gas conditioning CAPEX [$]", "Infrastructure & Transport Costs tab", "Levelized Cost", "Mother station gas treatment capital", "Annuitized CAPEX, Total CAPEX"),
            ("Filling bay cost per bay [$]", "Infrastructure & Transport Costs tab", "Levelized Cost", "Mother station bay infrastructure", "Annuitized CAPEX, Total Mother CAPEX"),
            ("Mother Station OPEX", "Infrastructure & Transport Costs tab", "Levelized Cost", "Annual operating expenditure", "Total annual cost, OPEX breakdown"),
            ("Electricity price [$/kWh]", "Infrastructure & Transport Costs tab", "Levelized Cost", "Mother station energy cost", "Station energy [$/year], Energy OPEX"),
            ("Fuel price [$/L]", "Infrastructure & Transport Costs tab", "Levelized Cost", "Mother station backup/auxiliary fuel", "Fuel cost [$/year]"),
            ("Unloading bays cost per bay [$]", "Infrastructure & Transport Costs tab", "Levelized Cost", "Daughter station bay infrastructure", "Annuitized CAPEX, Total Daughter CAPEX"),
            ("Daughter Station OPEX", "Infrastructure & Transport Costs tab", "Levelized Cost", "Annual operating expenditure", "Total annual cost, OPEX breakdown"),
            ("Truck CAPEX per unit [$]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Prime mover capital cost", "Total truck CAPEX, Annuitized CAPEX"),
            ("A-trailer CAPEX per unit [$]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "A-trailer capital cost", "Total road-train CAPEX, Annuitized CAPEX"),
            ("B-trailer CAPEX per unit [$]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "B-trailer capital cost", "Total road-train CAPEX, Annuitized CAPEX"),
            ("Dolly CAPEX per unit [$]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Dolly capital cost", "Total road-train CAPEX, Annuitized CAPEX"),
            ("Daily CAPEX per unit [$]", "Infrastructure & Transport Costs tab", "Levelized Cost", "Other fleet equipment capital", "Total fleet CAPEX"),
            ("Truck maintenance basis", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "$/km or $/year basis selection", "Truck maintenance [$/year]"),
            ("Truck maintenance [$/km or $/year]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Maintenance cost input", "Total truck maintenance [$/year]"),
            ("Trailer maintenance basis", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "$/km or $/year basis selection", "Trailer maintenance [$/year]"),
            ("Trailer maintenance [$/km or $/year]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Maintenance cost input", "Total trailer maintenance [$/year]"),
            ("Steer Tyres [$/tyre]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Replacement cost per steer tyre", "Total tyre cost [$/year]"),
            ("Replace at [km/tyre]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Steer tyre replacement interval", "Tyre replacement frequency"),
            ("Drive Tyres [$/tyre]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Replacement cost per drive tyre", "Total tyre cost [$/year]"),
            ("Replace at [km/tyre]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Drive tyre replacement interval", "Tyre replacement frequency"),
            ("Trailer Tyres [$/tyre]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Replacement cost per trailer tyre", "Total tyre cost [$/year]"),
            ("Replace at [km/tyre]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Trailer tyre replacement interval", "Tyre replacement frequency"),
            ("Truck consumption [km/L]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Fuel efficiency", "Total trucking fuel [$/year]"),
            ("Diesel price [$/L]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Fuel price", "Total trucking fuel [$/year]"),
            ("Wage unit", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "$/day or $/year basis selection", "Total driver costs [$/year]"),
            ("Wages per driver [$/day or $/year]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Driver compensation", "Average annual driver wages/salary [$/year]"),
            ("FIFO costs per driver [$/day]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Fly-in fly-out accommodation", "FIFO costs per driver [$/day]"),
            ("Camp accommodation per driver [$/day]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Camp accommodation cost", "Camp accommodation per driver [$/day]"),
            ("Working days per year [days]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Annual working days", "Total driver costs [$/year]"),
            ("Annual registration per truck [$/year]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Fleet registration fees", "Total annual truck fleet registration costs [$/year]"),
            ("Annual registration per trailer [$/year]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Trailer registration fees", "Total annual trailer fleet registration costs [$/year]"),
            ("Insurance [$/year]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Fleet insurance cost", "Insurance [$/year]"),
            ("Miscellaneous [$/year]", "Infrastructure & Transport Costs tab", "Levelized Cost, Trucking Cost", "Other transport OPEX", "Miscellaneous [$/year]"),
            ("Finance Plan (rate, NPER)", "Infrastructure & Transport Costs tab", "Levelized Cost", "Interest rate and loan term for CRF", "Capital Recovery Factor, Annuitized CAPEX"),
            
            # ===== BENEFITS & CARBON TAB =====
            ("Revenue - sales gas [$/GJ delivered]", "Benefits & Carbon tab", "Levelized Benefit, Margin", "Gas sales revenue", "Total gas revenue [$/yr], Levelized Benefit"),
            ("Baseline", "Benefits & Carbon tab", "Levelized Benefit, Margin", "Reference emissions case", "Baseline CO₂-e (t/yr), Avoided CO₂-e"),
            ("GWP basis", "Benefits & Carbon tab", "Levelized Benefit, Margin", "Methane warming potential (GWP100 or GWP20)", "Baseline emissions, Avoided emissions"),
            ("Flare efficiency [%]", "Benefits & Carbon tab", "Levelized Benefit, Margin", "Combustion efficiency for flaring", "Flaring EF [tCO₂/GJ], Baseline emissions"),
            ("Carbon price [$/tCO₂-e]", "Benefits & Carbon tab", "Levelized Benefit, Margin", "Value of avoided emissions", "Carbon avoidance [$/yr], Levelized Benefit"),
            ("Include transport emissions", "Benefits & Carbon tab", "Levelized Benefit, Margin", "Accounts for trucking CO₂", "Project CO₂-e (t/yr), Net avoided emissions"),
            ("Diesel EF [kgCO₂-e/L]", "Benefits & Carbon tab", "Levelized Benefit, Margin", "Diesel emission factor", "Transport emissions (t/yr)"),
            ("Other benefit [$/year]", "Benefits & Carbon tab", "Levelized Benefit, Margin", "Additional benefits not captured elsewhere", "Total annual benefit [$/yr]"),
        ]
        
        # Create DataFrame
        import pandas as pd
        df_diag = pd.DataFrame(diagnostics_data, columns=[
            "Input Label",
            "Location",
            "Ribbon Pills",
            "Diagnostic",
            "Other Outputs"
        ])
        
        # Display the table with sorting capability
        st.dataframe(
            df_diag,
            use_container_width=True,
            hide_index=True,
            height=600,
        )
        
        # Summary statistics
        st.markdown("---")
        st.markdown("#### Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_inputs = len(df_diag)
        lc_inputs = len(df_diag[df_diag["Ribbon Pills"].str.contains("Levelized Cost", na=False)])
        lb_inputs = len(df_diag[df_diag["Ribbon Pills"].str.contains("Levelized Benefit", na=False)])
        tc_inputs = len(df_diag[df_diag["Ribbon Pills"].str.contains("Trucking Cost", na=False)])
        no_contrib = len(df_diag[df_diag["Ribbon Pills"] == "No contribution"])
        
        col1.metric("Total Inputs", total_inputs)
        col2.metric("→ Levelized Cost", lc_inputs)
        col3.metric("→ Levelized Benefit", lb_inputs)
        col4.metric("→ Trucking Cost", tc_inputs)
        
        st.caption(f"**{no_contrib}** inputs have no direct contribution to ribbon pills (typically display-only settings).")
        
        # Download button
        csv = df_diag.to_csv(index=False)
        st.download_button(
            label="📥 Download Diagnostics as CSV",
            data=csv,
            file_name="input_diagnostics_v15_8.csv",
            mime="text/csv",
        )
    
    # ---------------------------------------
    # COST BREAKDOWN PIE CHARTS
    # ---------------------------------------
    with st.expander("📊 Cost Breakdown Analysis", expanded=False):
        st.markdown("### Summary Ribbon Pill Breakdown")
        st.caption("Visual breakdown of the components contributing to Levelized Cost and Trucking Cost.")
        
        # Try to get current cost breakdown values from session state
        # These should have been calculated by the Infrastructure & Transport Costs tab
        try:
            # Get the latest metrics
            _metrics = compute_levelized_metrics_for_scenario()
            lc_per_gj = float(_metrics.get("lc_per_gj", 0.0))
            transport_lc_per_gj = float(_metrics.get("transport_lc_per_gj", 0.0))
        except Exception:
            lc_per_gj = float(st.session_state.get("lc_per_gj", 0.0))
            transport_lc_per_gj = float(st.session_state.get("transport_lc_per_gj", 0.0))
        
        if not MPL_OK:
            st.warning("Matplotlib is not available. Install matplotlib to view cost breakdown charts.")
        else:
            # Get effective annual GJ for all calculations
            eff_daily = float(st.session_state.get("effective_daily_gj", 
                              st.session_state.get("daily_energy_gj", 10000.0)))
            annual_gj = eff_daily * DAYS_PER_YEAR
            
            # ===== LEVELIZED COST PIE CHART (MORE GRANULAR) =====
            st.markdown("#### Levelized Cost [$/GJ] - Detailed Breakdown")
            
            try:
                # Get more granular cost components
                annualized_capex = float(st.session_state.get("annuitized_capex_year", 0.0))
                infra_opex = float(st.session_state.get("infra_opex_year", 0.0))
                station_energy = float(st.session_state.get("station_energy_year", 0.0))
                
                # Get transport components individually for more detail
                fuel_cost = float(st.session_state.get("fuel_cost_year_calc", 0.0))
                trk_maint = float(st.session_state.get("trk_maint_year_calc", 0.0))
                trl_maint = float(st.session_state.get("trl_maint_year_calc", 0.0))
                driver_cost = float(st.session_state.get("driver_cost_year_calc", 0.0))
                other_opex = float(st.session_state.get("other_opex_year_calc", 0.0))
                
                if annual_gj > 0:
                    # Build granular breakdown
                    lc_components = {
                        'CAPEX (annualized)': annualized_capex / annual_gj,
                        'Infrastructure OPEX': infra_opex / annual_gj,
                        'Station Energy': station_energy / annual_gj,
                        'Fuel': fuel_cost / annual_gj,
                        'Truck Maintenance': trk_maint / annual_gj,
                        'Trailer Maintenance': trl_maint / annual_gj,
                        'Drivers': driver_cost / annual_gj,
                        'Registration/Insurance/Misc': other_opex / annual_gj,
                    }
                    
                    # Filter out zero values
                    lc_components = {k: v for k, v in lc_components.items() if v > 0.001}
                    
                    if lc_components:
                        fig1, ax1 = plt.subplots(figsize=(12, 8))
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#FFEAA7', '#DFE6E9', '#74B9FF']
                        
                        wedges, texts, autotexts = ax1.pie(
                            lc_components.values(),
                            labels=lc_components.keys(),
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors[:len(lc_components)],
                            textprops={'fontsize': 11}
                        )
                        
                        # Make percentage text bold and white
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                            autotext.set_fontsize(12)
                        
                        ax1.axis('equal')
                        
                        # Add legend with values
                        legend_labels = [f'{k}: ${v:.2f}/GJ ({v/lc_per_gj*100:.1f}%)' 
                                       for k, v in lc_components.items()]
                        ax1.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), 
                                  fontsize=10)
                        
                        plt.title(f'Total Levelized Cost: ${lc_per_gj:.2f}/GJ', 
                                fontsize=14, fontweight='bold', pad=20)
                        st.pyplot(fig1)
                        plt.close()
                    else:
                        st.info("No cost data available yet. Visit the Infrastructure & Transport Costs tab to generate data.")
                else:
                    st.info("No throughput data available. Set Daily gas to transport in the sidebar.")
                    
            except Exception as e:
                st.info("Cost breakdown not yet calculated. Visit the Infrastructure & Transport Costs tab to see detailed breakdown.")
            
            st.markdown("---")
            
            # ===== TRUCKING COST PIE CHART (MORE GRANULAR) =====
            st.markdown("#### Trucking Cost (excl. stations) [$/GJ] - Detailed Breakdown")
            
            try:
                # Get granular transport cost components
                trucking_capex_annual = float(st.session_state.get("trucking_capex_annual", 0.0))
                fuel_cost = float(st.session_state.get("fuel_cost_year_calc", 0.0))
                trk_maint = float(st.session_state.get("trk_maint_year_calc", 0.0))
                trl_maint = float(st.session_state.get("trl_maint_year_calc", 0.0))
                driver_cost = float(st.session_state.get("driver_cost_year_calc", 0.0))
                registration = float(st.session_state.get("registration_year_calc", 0.0))
                insurance = float(st.session_state.get("insurance_year_calc", 0.0))
                misc = float(st.session_state.get("misc_year_calc", 0.0))
                
                if annual_gj > 0:
                    # Build trucking cost breakdown
                    tc_components = {
                        'Fleet CAPEX (annualized)': trucking_capex_annual / annual_gj,
                        'Fuel': fuel_cost / annual_gj,
                        'Truck Maintenance': trk_maint / annual_gj,
                        'Trailer Maintenance': trl_maint / annual_gj,
                        'Drivers': driver_cost / annual_gj,
                        'Registration': registration / annual_gj,
                        'Insurance': insurance / annual_gj,
                        'Miscellaneous': misc / annual_gj,
                    }
                    
                    # Filter out zero/negligible values
                    tc_components = {k: v for k, v in tc_components.items() if v > 0.001}
                    
                    if tc_components:
                        fig2, ax2 = plt.subplots(figsize=(12, 8))
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#FFEAA7', '#DFE6E9', '#74B9FF']
                        
                        wedges, texts, autotexts = ax2.pie(
                            tc_components.values(),
                            labels=tc_components.keys(),
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors[:len(tc_components)],
                            textprops={'fontsize': 11}
                        )
                        
                        # Make percentage text bold and white
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                            autotext.set_fontsize(12)
                        
                        ax2.axis('equal')
                        
                        # Add legend with values
                        legend_labels = [f'{k}: ${v:.2f}/GJ ({v/transport_lc_per_gj*100:.1f}%)' 
                                       for k, v in tc_components.items()]
                        ax2.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                                  fontsize=10)
                        
                        plt.title(f'Total Trucking Cost: ${transport_lc_per_gj:.2f}/GJ', 
                                fontsize=14, fontweight='bold', pad=20)
                        st.pyplot(fig2)
                        plt.close()
                    else:
                        st.info("No transport cost data available yet. Visit the Infrastructure & Transport Costs tab to generate data.")
                else:
                    st.info("No throughput data available. Set Daily gas to transport in the sidebar.")
                    
            except Exception as e:
                st.info("Transport cost breakdown not yet calculated. Visit the Infrastructure & Transport Costs tab to see detailed breakdown.")
        
        st.markdown("---")
        st.caption("**Note:** These charts show the breakdown of costs at the current input values. "
                   "Modify inputs in the sidebar and Infrastructure & Transport Costs tab to update the breakdown. "
                   "The charts will update after visiting the Infrastructure & Transport Costs tab.")

# ---------------------------------------
# PAYLOAD & CAPACITY  (index 1)
# ---------------------------------------
with tabs[1]:
    st.markdown("## Payload & Capacity (A- and B-trailers)")
    summary_ribbon("24px")

    colp1, colp2 = st.columns(2, vertical_alignment="top")

    # LEFT: Thermofluid inputs
    with colp1:
        st.markdown("**Thermofluid inputs**")
        gas_type = st.selectbox("Gas Type", ["Hydrogen", "Methane"], index=1)
        st.session_state["gas_type_selected"] = gas_type

        PW_bar_g = st.number_input(
            "Working Pressure [bar(g)]", value=300.0, min_value=0.0, step=1.0, key="phys_PW"
        )
        Pmin_bar_g = st.number_input(
            "Minimum Cylinder Pressure [bar(g)]",
            min_value=0.0,
            max_value=PW_bar_g,
            value=min(10.0, PW_bar_g),
            step=1.0,
            help="Pressure below which the trailer is considered empty for dispatch purposes.",
        )
        if Pmin_bar_g > PW_bar_g + 1e-9:
            st.error("Minimum Cylinder Pressure cannot exceed Working Pressure.")
            Pmin_bar_g = PW_bar_g

        hv_options = ["Gross Heating Value [J/kg]", "Net Heating Value [J/kg]"]
        hv_basis = st.selectbox("Heating value basis", hv_options, index=0)
        hv_key = "Gross" if hv_basis.startswith("Gross") else "Net"
        st.session_state["hv_basis_key"] = hv_key

        T_C = st.number_input(
            "Gas temperature [°C]",
            min_value=-20.0,
            max_value=65.0,
            value=15.0,
            step=1.0,
            key="phys_T",
        )

        auto_sync_hv = st.checkbox(
            "Auto-sync heating value to Gas Type & Basis",
            value=True,
            help="When checked, the heating value updates automatically based on the selected Gas Type and Basis. Uncheck to enter a custom value manually."
        )

        DEFAULT_HV = {
            "Methane": {"Gross": 55571000.0, "Net": 50032000.0},
            "Hydrogen": {"Gross": 141950000.0, "Net": 119910000.0},
        }

        if "last_gas_type" not in st.session_state:
            st.session_state["last_gas_type"] = gas_type
        if "last_hv_key" not in st.session_state:
            st.session_state["last_hv_key"] = hv_key
        if "HV_J_per_kg" not in st.session_state:
            st.session_state["HV_J_per_kg"] = DEFAULT_HV[gas_type][hv_key]

        if auto_sync_hv and (
            gas_type != st.session_state["last_gas_type"]
            or hv_key != st.session_state["last_hv_key"]
        ):
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
                key="HV_J_per_kg",
            )
            # --- Densities display (moved here per v14 Section 12) ---
            # Compute densities at PW and Pmin using current inputs
            P_abs_bar_loc = st.session_state["phys_PW"] + 1.01325
            Pmin_abs_bar_loc = Pmin_bar_g + 1.01325
            P_Pa_loc, Pmin_Pa_loc = P_abs_bar_loc * 1e5, Pmin_abs_bar_loc * 1e5
            T_K_loc = st.session_state["phys_T"] + 273.15
            gas_label_loc = "Hydrogen" if gas_type == "Hydrogen" else "Methane"
            rho_full_loc = density_kg_per_m3(T_K_loc, P_Pa_loc, gas_label_loc)
            rho_min_loc = density_kg_per_m3(T_K_loc, Pmin_Pa_loc, gas_label_loc)
            coolprop_tag = (
                "<span style='font-style:italic; color:#666;'> (CoolProp)</span>"
                if COOLPROP_OK
                else "<span style='font-style:italic; color:#666;'> (ideal gas)</span>"
            )
            st.markdown(
                f"""<div style='font-size:1.25rem; line-height:1.25; margin-top:0.25rem;'>
                <b>Densities —</b><br>
                ρ(P<sub>W</sub>): {rho_full_loc:.2f} [kg/m³]<br>
                ρ(P<sub>min</sub>): {rho_min_loc:.2f} [kg/m³]{coolprop_tag}
                </div>""",
                unsafe_allow_html=True,
            )

    # RIGHT: volumes
    with colp2:
        st.markdown("**Composition & usable volume**")
        gas_purity = st.number_input(
            "Fuel gas purity (mole fraction 0–1)",
            value=1.00,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )

        st.markdown("**A-trailer cylinder set**")
        a_cyl_count = st.number_input(
            "Number of cylinders per A-trailer", min_value=1, value=1, step=1
        )
        a_cyl_water_L = st.number_input(
            "Cylinder water capacity [Litres] (A)",
            min_value=0.01,
            value=41000.0,
            step=100.0,
            format="%.1f",
        )
        a_trailer_vol_m3 = (a_cyl_count * a_cyl_water_L) / 1000.0
        st.caption(f"Computed A-trailer internal gas volume: **{a_trailer_vol_m3:,.2f} m³**")
        st.markdown("---")

        st.markdown("**B-trailer cylinder set**")
        b_cyl_count = st.number_input(
            "Number of cylinders per B-trailer", min_value=1, value=1, step=1
        )
        b_cyl_water_L = st.number_input(
            "Cylinder water capacity [Litres] (B)",
            min_value=0.01,
            value=17100.0,
            step=100.0,
            format="%.1f",
        )
        b_trailer_vol_m3 = (b_cyl_count * b_cyl_water_L) / 1000.0
        st.caption(f"Computed B-trailer internal gas volume: **{b_trailer_vol_m3:,.2f} m³**")

    # Thermo calcs
    P_abs_bar, Pmin_abs_bar = st.session_state["phys_PW"] + 1.01325, Pmin_bar_g + 1.01325
    P_Pa, Pmin_Pa = P_abs_bar * 1e5, Pmin_abs_bar * 1e5
    T_K = st.session_state["phys_T"] + 273.15
    gas_label = "Hydrogen" if gas_type == "Hydrogen" else "Methane"
    rho_full = density_kg_per_m3(T_K, P_Pa, gas_label)
    rho_min = density_kg_per_m3(T_K, Pmin_Pa, gas_label)
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
    st.session_state["a_cap_calc_gj"] = a_cap_calc_gj
    st.session_state["b_cap_calc_gj"] = b_cap_calc_gj
    st.session_state["a_cap_usable_gj"] = a_cap_usable_gj
    st.session_state["b_cap_usable_gj"] = b_cap_usable_gj

    st.subheader("Trailer Total Capacity")
    t1, t2, t3 = st.columns(3)
    t1.markdown(
        f"**A-trailer @ $P_{{W}}$**<br><span style='font-size:1.2em;'>{a_cap_calc_gj:,.2f}</span> GJ<br><span>{a_cap_calc_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )
    t2.markdown(
        f"**B-trailer @ $P_{{W}}$**<br><span style='font-size:1.2em;'>{b_cap_calc_gj:,.2f}</span> GJ<br><span>{b_cap_calc_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )
    t3.markdown(
        f"**{comb_key} total @ $P_{{W}}$**<br><span style='font-size:1.2em;'>{comb_total_gj:,.2f}</span> GJ<br><span>{comb_total_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )

    st.subheader("Trailer Usable Capacity ($P_{W} \\rightarrow P_{min}$)")
    u1, u2, u3 = st.columns(3)
    u1.markdown(
        f"**A-trailer usable** ($P_{{W}} \\rightarrow P_{{min}}$)<br><span style='font-size:1.2em;'>{a_cap_usable_gj:,.2f}</span> GJ<br><span>{a_cap_usable_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )
    u2.markdown(
        f"**B-trailer usable** ($P_{{W}} \\rightarrow P_{{min}}$)<br><span style='font-size:1.2em;'>{b_cap_usable_gj:,.2f}</span> GJ<br><span>{b_cap_usable_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )
    u3.markdown(
        f"**{comb_key} usable** ($P_{{W}} \\rightarrow P_{{min}}$)<br><span style='font-size:1.2em;'>{comb_usable_gj:,.2f}</span> GJ<br><span>{comb_usable_kg:,.0f} kg</span>",
        unsafe_allow_html=True,
    )

    if abs(st.session_state["phys_PW"] - Pmin_bar_g) < 1e-6:
        st.info(
            "Usable capacity is zero because Working Pressure equals Minimum Cylinder Pressure."
        )
    # (moved densities caption per v14 Section 12))

# ---------------------------------------
# Fatigue rules (Standard Hours – Solo)
# ---------------------------------------
unload_time_h = st.session_state.get("unload_time_h", 12.0)
with tabs[2]:
    st.markdown(f"## {st.session_state.get('gas_type_selected', 'CNG')} Virtual Pipeline Planner")
    summary_ribbon("24px")

    _inject_planner_expander_css()

    help_flyout(
        "planner_overview",
        "Planner balances **capacity, travel time, and terminal ops** to size trucks, trailers, and bays. ‘Utilisation’ reflects duty-time vs allowable hours.",
    )
    st.caption(
        "Planner uses **usable capacity (PW → Pmin)** when available; otherwise **total capacity @ PW**."
    )
    a_cap_for_planner, b_cap_for_planner = _get_caps_for_planner()

    # --- Drivers & Relay Strategy (advised, ranked, filtered) ---
    # --- Drivers & Relay Strategy (advised, ranked, filtered, with BEST tag) ---
    st.subheader("Drivers & Relay Strategy")

    # Convenience getters
    _dist = float(st.session_state.get("distance_km_oneway", 900.0))
    _vavg = float(st.session_state.get("speed_kmh", 75.0))
    _mode = st.session_state.get("mode", "DropAndPull")
    _unld = float(st.session_state.get("unload_time_h", 12.0))


    # Helper: collapse Mother/Daughter to one display label
    def _collapse(label: str) -> str:
        return "One camp at terminal" if label in ("Camp at Mother", "Camp at Daughter") else label


    # --- Step 1: Compliance scoring for all options
    scored = []
    for opt in RELAY_OPTIONS_DISPLAY:
        code = compliance_code_for(_dist, _vavg, opt, _mode, _unld)  # 0,1,2 = compliant, borderline, not feasible
        scored.append((opt, code, _collapse(opt)))

    # Collapse duplicates (Mother/Daughter) keeping best compliance code
    agg = {}  # disp_label -> (best_code, representative_original_label)
    for orig_label, code, disp_label in scored:
        if disp_label not in agg or code < agg[disp_label][0]:
            agg[disp_label] = (code, orig_label)

    # Rank by feasibility first (0,1,2), then simplicity
    _simplicity_order = {"No relay": 0, "One camp at terminal": 1, "Midpoint relay": 2}
    ordered = sorted(agg.items(), key=lambda kv: (kv[1][0], _simplicity_order.get(kv[0], 99), kv[0]))

    # Filter out ❌ not-feasible unless everything is ❌
    viable_items = [(disp, val) for disp, val in ordered if val[0] < 2] or ordered

    # --- Step 2: Among viable options, compute Trucking Cost (/GJ) and Drivers to find "best"
    from finance_helpers import compute_levelized_metrics_for_scenario  # returns transport_lc_per_gj
    from calcs import compute_calcs


    def _gf(key, default=0.0):
        try:
            return float(st.session_state.get(key, default))
        except Exception:
            return float(default)


    # Inputs needed for compute_calcs scoring of drivers
    daily_energy_gj = _gf("daily_energy_gj", 0.0)
    hours_per_day = _gf("hours_per_day", 24.0)
    a_count = int(st.session_state.get("a_count", 1))
    b_count = int(st.session_state.get("b_count", 0))
    fill_rate_gjph = _gf("fill_rate_gjph", 20.0)
    concurrent_bays_per_set = int(st.session_state.get("concurrent_bays_per_set", 1))
    available_bays_A = int(st.session_state.get("available_bays_A", 2))
    available_unload_bays_B = int(st.session_state.get("available_unload_bays_B", 2))
    changeover_overhead_h = _gf("changeover_overhead_h", 0.0)
    crew_change_overhead_h = _gf("crew_change_overhead_h", 0.25)
    delay_buffer_min = _gf("delay_buffer_min", 30.0)
    truck_util_target = _gf("truck_util_target", 1.0)

    # Driver shift cap from fatigue regime (12/14/16 h)
    fatigue_regime = st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)")
    _norm = {
        "Standard Hours (Solo Drivers)": 12.0,
        "Basic Fatigue Management (BFM)": 14.0,
        "Advanced Fatigue Management (AFM)": 16.0,
    }.get(fatigue_regime, 12.0)
    driver_shift_cap = _norm

    # We already have a_cap_for_planner, b_cap_for_planner above this block
    a_cap_gj = float(a_cap_for_planner)
    b_cap_gj = float(b_cap_for_planner)

    # Evaluate each viable option
    option_metrics = []  # (disp_label, trucking_cost_per_gj, drivers_on_duty)
    _prev = st.session_state.get("relay_location", "No relay")
    for disp, (code, orig_label) in viable_items:
        legacy = RELAY_DISPLAY_TO_LEGACY.get(orig_label, "No relay")

        # Temporarily set relay for metric functions
        st.session_state["relay_location"] = legacy

        # Trucking cost (/GJ)
        _metrics = compute_levelized_metrics_for_scenario()
        trucking_cost_gj = float(_metrics.get("transport_lc_per_gj", 0.0))

        # Drivers on duty (from calcs)
        _calcs = compute_calcs(
            daily_energy_gj=daily_energy_gj,
            distance_km_oneway=_dist,
            speed_kmh=_vavg,
            hours_per_day=hours_per_day,
            a_cap_gj=a_cap_gj,
            b_cap_gj=b_cap_gj,
            a_count=a_count,
            b_count=b_count,
            fill_rate_gjph=fill_rate_gjph,
            concurrent_bays_per_set=concurrent_bays_per_set,
            unload_time_h=_unld,
            available_bays_A=available_bays_A,
            available_unload_bays_B=available_unload_bays_B,
            changeover_overhead_h=changeover_overhead_h,
            mode=_mode,
            driver_shift_cap=driver_shift_cap,
            crew_change_overhead_h=crew_change_overhead_h,
            delay_buffer_min=delay_buffer_min,
            truck_util_target=truck_util_target,
            fatigue_regime=fatigue_regime,
            relay_location=legacy,
        )
        drivers_on_duty = int(_calcs.get("drivers_on_duty", 0))

        option_metrics.append((disp, trucking_cost_gj, drivers_on_duty))

    # Restore previous relay selection
    st.session_state["relay_location"] = _prev

    # Choose BEST: min Trucking Cost, then min drivers
    if option_metrics:
        best_disp, _, _ = min(option_metrics, key=lambda t: (t[1], t[2]))
    else:
        best_disp = viable_items[0][0]

    # Build UI labels (add ✅/⚠ and ⭐ best)
    _tag = {0: "✅ compliant", 1: "⚠ borderline", 2: "❌ not feasible"}
    viable_display = []
    for disp, (code, _) in viable_items:
        star = " ⭐ best" if disp == best_disp else ""
        viable_display.append(f"{disp} – {_tag[code]}{star}")

    # Compute global recommendation from pure compliance (unchanged)
    reco_idx_raw, reco_status = best_mode_for(_dist, _vavg, _mode, _unld, False)
    reco_disp = _collapse(RELAY_OPTIONS_DISPLAY[reco_idx_raw])
    default_index = next((i for i, (disp, _) in enumerate(viable_items) if disp == reco_disp), 0)

    # UI selectbox
    relay_display_sel = st.selectbox(
        "Relay strategy (advised)",
        viable_display,
        index=default_index,
        help=(
            "Feasible strategies only. ‘⭐ best’ minimises Trucking Cost and then drivers. "
            "The recommended default is based on compliance simplicity; you can override."
        ),
    )

    # Parse selection → collapsed label
    _selected_disp = relay_display_sel.split(" – ")[0]

    # Map collapsed label back to an original label and then to legacy key
    _orig_label = agg[_selected_disp][1]
    st.session_state["relay_location"] = RELAY_DISPLAY_TO_LEGACY.get(_orig_label, "No relay")

    # Small hint for compliance recommendation
    _status_label = {0: "OK", 1: "Borderline", 2: "Non-compliant"}.get(reco_status, "OK")
    st.caption(f"Recommended (compliance): **{reco_disp}** — {_status_label}")

    # Show a small recommendation hint
    _status_label = {0: "OK", 1: "Borderline", 2: "Non-compliant"}.get(reco_status, "OK")
    st.caption(f"Recommended: **{reco_disp}** — {_status_label}")

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
        driver_shift_cap=_current_driver_shift_cap(),
        crew_change_overhead_h=st.session_state.get("crew_change_overhead_h", 0.25),
        delay_buffer_min=st.session_state.get("delay_buffer_min", 30.0),
        truck_util_target=truck_util_target,
        fatigue_regime=st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)"),
        relay_location=st.session_state.get("relay_location", "No relay"),

        # NEW availability params
        route_hours_per_day=float(st.session_state.get("route_hours_per_day", 24.0)),
        route_days_per_year=float(st.session_state.get("route_days_per_year", 365.0)),
        ms_hours_per_day=float(st.session_state.get("ms_hours_per_day", 24.0)),
        ms_days_per_year=float(st.session_state.get("ms_days_per_year", 365.0)),
        ds_hours_per_day=float(st.session_state.get("ds_hours_per_day", 24.0)),
        ds_days_per_year=float(st.session_state.get("ds_days_per_year", 365.0)),
    )

    # Store results in session state (AFTER the call)
    st.session_state["calcs_latest"] = calcs
    st.session_state["latest_calcs_cache"] = calcs

    c1, c2, cT, c3, c4 = st.columns(5)
    with c1:
        st.subheader("Fleet")
        st.metric("Active trucks", calcs["active_trucks"])
        st.metric("Drivers per truck (on duty)", calcs["drivers_per_truck"])
        st.metric("Drivers on duty / day", calcs["drivers_on_duty"])

        D_oper = 365.0
        D_driver = float(st.session_state.get("working_days_per_year", 200.0))
        staff_drivers = int(math.ceil(calcs["drivers_on_duty"] * (D_oper / max(D_driver, 1e-9))))
        st.metric("Total fleet drivers on staff", staff_drivers)
        help_flyout(
            key="staff_drivers",
            text="Headcount on payroll = Drivers on duty × (365 ÷ Working days per year)."
        )

    with c2:
        st.subheader("Trailers")
        comb_label = st.session_state.get("combo", "A-triple")
        plural_comb = f"{comb_label}s"
        st.metric(f"{plural_comb} in circulation", calcs["sets_in_circulation"])
        st.metric("A-trailers", calcs["a_trailers"])
        st.metric("B-trailers", calcs["b_trailers"])
        st.metric("Total trailers", calcs["total_trailers"])

        combo_name = st.session_state.get("combo", "A-triple")
        dollies_per_combo = DOLLIES_PER_COMBO.get(combo_name, 0)
        total_dollies = int(math.ceil(calcs["sets_in_circulation"] * dollies_per_combo))
        st.metric("Total dollies", total_dollies)
    with cT:
        st.subheader("Ops — Transport")

        # Totals already computed by planner calcs
        active_trucks = int(calcs.get("active_trucks", 0))
        fleet_km_day = float(calcs.get("truck_km_per_day", 0.0))  # fleet total km/day
        fleet_km_year = fleet_km_day * DAYS_PER_YEAR
        fleet_km_month = fleet_km_year / 12.0

        # Per-truck averages
        denom = max(active_trucks, 1)
        avg_truck_km_day = fleet_km_day / denom
        avg_truck_km_month = fleet_km_month / denom
        avg_truck_km_year = fleet_km_year / denom

        # Combination (road-train set) travel — numerically same as fleet totals in this model
        combo_name = st.session_state.get("combo", "A-triple")
        combo_km_day = fleet_km_day
        combo_km_year = fleet_km_year
        combo_km_month = fleet_km_month

        # ---- Fleet totals
        st.metric("Total fleet travel [km/year]", f"{fleet_km_year:,.0f}")
        st.metric("Total fleet travel [km/month]", f"{fleet_km_month:,.0f}")
        st.metric("Total fleet travel [km/day]", f"{fleet_km_day:,.0f}")

        # ---- Per-truck averages
        st.metric("Average truck travel [km/truck/year]", f"{avg_truck_km_year:,.0f}")
        st.metric("Average truck travel [km/truck/month]", f"{avg_truck_km_month:,.0f}")
        st.metric("Average truck travel [km/truck/day]", f"{avg_truck_km_day:,.0f}")

        # ---- Combination (named)
        st.metric(f"Total {combo_name} travel [km/year]", f"{combo_km_year:,.0f}")
        st.metric(f"Average {combo_name} travel [km/month]", f"{combo_km_month:,.0f}")
        st.metric(f"Average {combo_name} travel [km/day]", f"{combo_km_day:,.0f}")

    with c3:
        st.subheader("Ops — Fill (Mother)")
        st.metric("Avg fill bays needed", f"{calcs['avg_bays_needed_A']:.2f}")
        utilA = (
            "∞%"
            if not math.isfinite(calcs["bay_utilization_A"])
            else f"{100 * calcs['bay_utilization_A']:.1f}%"
        )
        st.metric("Fill bay utilisation", utilA)
        st.write(
            f"**Fill bay status:** :{'red_circle' if (calcs['bay_flag_A'] != 'OK') else 'green_circle'}: {calcs['bay_flag_A']}"
        )
        gas_label = st.session_state.get("gas_type_selected", "Methane")
        st.metric(f"{gas_label} rate [GJ/h]", f"{calcs['energy_per_truckhour']:.2f} GJ/h")
        st.metric("Average fill time for combination [h]", f"{calcs['fill_time_h']:.2f}")
        # --- Gas collected (Mother) ---
        project_years_for_rollup = float(project_months) / 12.0
        total_gj_project = float(daily_energy_gj) * DAYS_PER_YEAR * project_years_for_rollup
        avg_gj_year = float(daily_energy_gj) * DAYS_PER_YEAR
        avg_gj_month = avg_gj_year / 12.0
        avg_gj_day = float(daily_energy_gj)

        st.metric("Total gas collected [GJ/project]", f"{total_gj_project:,.0f}")
        st.metric("Average total gas collected [GJ/year]", f"{avg_gj_year:,.0f}")
        st.metric("Average total gas collected [GJ/month]", f"{avg_gj_month:,.0f}")
        st.metric("Average total gas collected [GJ/day]", f"{avg_gj_day:,.0f}")

        # --- Expected (utilised) view, alongside planned ---
        eff_daily = float(st.session_state.get("effective_daily_gj", daily_energy_gj))
        project_years_for_rollup = float(project_months) / 12.0

        exp_day = eff_daily
        exp_year = exp_day * DAYS_PER_YEAR
        exp_month = exp_year / 12.0
        exp_proj = exp_year * project_years_for_rollup

        st.markdown("**(Expected @ utilisation)**")
        st.metric("Total gas collected [GJ/project]", f"{exp_proj:,.0f}")
        st.metric("Average total gas collected [GJ/year]", f"{exp_year:,.0f}")
        st.metric("Average total gas collected [GJ/month]", f"{exp_month:,.0f}")
        st.metric("Average total gas collected [GJ/day]", f"{exp_day:,.0f}")

    with c4:
        st.subheader("Ops — Unload (Daughter)")
        st.metric("Avg unload bays needed", f"{calcs['avg_unload_bays_needed_B']:.2f}")
        utilB = (
            "∞%"
            if not math.isfinite(calcs["unload_utilization_B"])
            else f"{100 * calcs['unload_utilization_B']:.1f}%"
        )
        st.metric("Unload bay utilisation", utilB)
        st.write(
            f"**Unload bay status:** :{'red_circle' if (calcs['unload_flag_B'] != 'OK') else 'green_circle'}: {calcs['unload_flag_B']}"
        )
        st.metric("Average unload time for combination [h]", f"{calcs['unload_time_h']:.2f}")
        # --- Gas delivered (Daughter) ---
        project_years_for_rollup = float(project_months) / 12.0
        total_gj_project = float(daily_energy_gj) * DAYS_PER_YEAR * project_years_for_rollup
        avg_gj_year = float(daily_energy_gj) * DAYS_PER_YEAR
        avg_gj_month = avg_gj_year / 12.0
        avg_gj_day = float(daily_energy_gj)

        st.metric("Total gas delivered [GJ/project]", f"{total_gj_project:,.0f}")
        st.metric("Average total gas delivered [GJ/year]", f"{avg_gj_year:,.0f}")
        st.metric("Average total gas delivered [GJ/month]", f"{avg_gj_month:,.0f}")
        st.metric("Average total gas delivered [GJ/day]", f"{avg_gj_day:,.0f}")

        # --- Expected (utilised) view, alongside planned ---
        eff_daily = float(st.session_state.get("effective_daily_gj", daily_energy_gj))
        project_years_for_rollup = float(project_months) / 12.0

        exp_day = eff_daily
        exp_year = exp_day * DAYS_PER_YEAR
        exp_month = exp_year / 12.0
        exp_proj = exp_year * project_years_for_rollup

        st.markdown("**(Expected @ utilisation)**")
        st.metric("Total gas delivered [GJ/project]", f"{exp_proj:,.0f}")
        st.metric("Average total gas delivered [GJ/year]", f"{exp_year:,.0f}")
        st.metric("Average total gas delivered [GJ/month]", f"{exp_month:,.0f}")
        st.metric("Average total gas delivered [GJ/day]", f"{exp_day:,.0f}")

    # Driver utilisation graphic
    with st.expander("Driver utilisation (per-shift breakdown)", expanded=False):
        help_flyout(
            "driver_util_calc",
            f"Breakdown of a **single shift**: drive, regulated breaks, unload (ThroughRoad), and overhead. Compared against the **{_current_driver_shift_cap():.0f} h cap**.",

        )
        render_driver_utilisation_section()

    # Compliance map

    with st.expander("Compliance map (distance–speed)", expanded=False):
        st.caption(
            "Least-intrusive relay mode that achieves compliance at each distance–speed pair."
        )
        borderline_ok = st.checkbox(
            "Treat ‘Borderline’ as acceptable",
            value=False,
            help="These controls affect map display only—they do not change calculations or cost results."
        )
        ccol1, ccol2, ccol3, ccol4 = st.columns(4)
        with ccol1:
            d_min = st.number_input("Min distance [km]", value=100.0, step=50.0)
        with ccol2:
            d_max = st.number_input("Max distance [km]", value=1500.0, step=50.0)
        with ccol3:
            v_min = st.number_input("Min speed [km/h]", value=60.0, step=5.0)
        with ccol4:
            v_max = st.number_input("Max speed [km/h]", value=100.0, step=5.0)
        if not MPL_OK:
            st.warning(
                "Matplotlib is not installed. Add `matplotlib>=3.9.0` to requirements.txt and reinstall."
            )
        else:
            D = np.linspace(max(d_min, 1e-3), max(d_max, d_min + 1e-3), 41)
            V = np.linspace(max(v_min, 1e-3), max(v_max, v_min + 1e-3), 41)
            mode_idx_grid = np.zeros((V.size, D.size), dtype=int)
            for i, v in enumerate(V):
                for j, d in enumerate(D):
                    bi, _ = best_mode_for(d, v, mode, unload_time_h, borderline_ok)
                    mode_idx_grid[i, j] = bi
            fig, ax = plt.subplots(figsize=(8, 5))
            cmap = ListedColormap(
                ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#d62728"]
            )
            bounds = np.arange(len(RELAY_OPTIONS_DISPLAY) + 1) - 0.5
            norm = BoundaryNorm(bounds, cmap.N)
            ax.imshow(
                mode_idx_grid,
                origin="lower",
                aspect="auto",
                extent=[D.min(), D.max(), V.min(), V.max()],
                cmap=cmap,
                norm=norm,
            )
            ax.set_xlabel("Distance Mother→Daughter [km]")
            ax.set_ylabel("Average speed [km/h]")
            ax.set_title(
                "Best relay mode to achieve acceptable compliance"
                + (" (OK → Borderline)" if borderline_ok else " (OK only)")
            )
            cur_idx, cur_status = best_mode_for(
                distance_km_oneway, speed_kmh, mode, unload_time_h, borderline_ok
            )
            edge_color = {0: "#2ca02c", 1: "#ffbf00", 2: "#d62728"}[cur_status]
            ax.scatter(
                [distance_km_oneway],
                [speed_kmh],
                marker="o",
                s=90,
                edgecolors=edge_color,
                facecolors="none",
                linewidths=2,
                zorder=5,
            )
            status_label = {0: "OK", 1: "Borderline", 2: "Non-compliant"}[cur_status]
            ax.annotate(
                f"{RELAY_OPTIONS_DISPLAY[cur_idx]} • {status_label}",
                (distance_km_oneway, speed_kmh),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=edge_color, lw=1),
                zorder=6,
            )
            legend_handles = [
                Patch(color=cmap(i), label=RELAY_OPTIONS_DISPLAY[i])
                for i in range(len(RELAY_OPTIONS_DISPLAY))
            ]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)
            st.pyplot(fig)

    # ---------------------------------------
    # COSTS TAB — Infrastructure & Transport (index 3)
    # ---------------------------------------
with tabs[3]:
    # Main section heading (match other tabs: heading first)
    st.markdown("## Infrastructure & Transport Costs")
    
    # Pre-compute levelized metrics for the ribbon using current session values
    try:
        _metrics = compute_levelized_metrics_for_scenario()
        st.session_state["lc_per_gj"] = float(_metrics.get("lc_per_gj", 0.0))
        st.session_state["lb_per_gj"] = float(_metrics.get("lb_per_gj", 0.0))
        st.session_state["transport_lc_per_gj"] = float(_metrics.get("transport_lc_per_gj", 0.0))
        margin_per_gj = float(_metrics.get("lb_per_gj", 0.0)) - float(_metrics.get("lc_per_gj", 0.0))
        st.session_state["margin_per_gj"] = margin_per_gj
    except Exception:
        # If compute fails, keep existing session state values
        pass
    
    summary_ribbon("24px")

    cA, cB = st.columns(2, vertical_alignment="top")

    def _plan_select_inline(key, default_idx=0):

        plans_list = st.session_state.get(
            "finance_plans",
            [{"label": "Plan A", "rate_pct": 6.0, "nper_months": 60, "enabled": True}],
        )
        labels = [p.get("label", "") for p in plans_list if p.get("enabled", True)]
        if not labels:
            labels = ["(no plans enabled)"]
            idx = 0
        else:
            idx = min(default_idx, len(labels) - 1)
        return st.selectbox("Plan", labels, index=idx, key=key)


    st.caption(
        "Annuitisation via CRF to compute **Levelized Cost [$/GJ]**. CAPEX inputs show a helper in $ millions."
    )

    # Compute latest calcs for counts and km (uses fresh A/B capacities)
    # Always refresh the "for planner" capacities (cheap and idempotent)
    a_cap_for_planner, b_cap_for_planner = _get_caps_for_planner()

    # Pull latest calcs for counts and km (cache may be used elsewhere)
    calcs_cost = compute_calcs(
        daily_energy_gj,
        st.session_state.get("distance_km_oneway", 900.0),
        st.session_state.get("speed_kmh", 75.0),
        st.session_state.get("hours_per_day", 24.0),
        a_cap_for_planner,
        b_cap_for_planner,
        ROADTRAIN_MAP[st.session_state.get("combo", "A-triple")]["A"],
        ROADTRAIN_MAP[st.session_state.get("combo", "A-triple")]["B"],
        st.session_state.get("fill_rate_gjph", 41.67),
        st.session_state.get("concurrent_bays_per_set", 3),
        st.session_state.get("unload_time_h", 12.0),
        st.session_state.get("available_bays_A", 12),
        st.session_state.get("available_unload_bays_B", 4),
        st.session_state.get("changeover_overhead_h", 0.25),
        st.session_state.get("mode", "DropAndPull"),
        _current_driver_shift_cap(),  # ✅ correct cap substitution
        st.session_state.get("crew_change_overhead_h", 0.25),
        st.session_state.get("delay_buffer_min", 30),
        st.session_state.get("truck_util_target", 0.85),
        st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)"),
        st.session_state.get("relay_location", "No relay"),

        # NEW availability params
        route_hours_per_day=float(st.session_state.get("route_hours_per_day", 24.0)),
        route_days_per_year=float(st.session_state.get("route_days_per_year", 365.0)),
        ms_hours_per_day=float(st.session_state.get("ms_hours_per_day", 24.0)),
        ms_days_per_year=float(st.session_state.get("ms_days_per_year", 365.0)),
        ds_hours_per_day=float(st.session_state.get("ds_hours_per_day", 24.0)),
        ds_days_per_year=float(st.session_state.get("ds_days_per_year", 365.0)),
    )

    sets_per_day = calcs_cost["sets_per_day"]
    truck_km_per_day = calcs_cost["truck_km_per_day"]
    trailer_km_per_day = calcs_cost["trailer_km_per_day"]
    active_trucks = calcs_cost["active_trucks"]
    a_trailers = calcs_cost["a_trailers"]
    b_trailers = calcs_cost["b_trailers"]
    combos_in_circ = calcs_cost["sets_in_circulation"]

    dollies_per_combo = DOLLIES_PER_COMBO.get(st.session_state.get("combo", "A-triple"), 0)
    total_dollies = int(math.ceil(combos_in_circ * dollies_per_combo))

    # Use expected daily gas (utilisation applied), fall back to planned if not set
    eff_daily = float(st.session_state.get("effective_daily_gj", daily_energy_gj))
    annual_gj = eff_daily * DAYS_PER_YEAR

    cA, cB = st.columns(2, vertical_alignment="top")

    # ------------- Infrastructure (Mother/Daughter + fleet CAPEX, OPEX) -------------
    with cA:
        st.subheader("Infrastructure Costs")
        st.markdown("**Mother Station – CAPEX ($)**")
        _cols = st.columns([3, 2])
        with _cols[0]:
            ms_comp = st.number_input(
                "Compression plant CAPEX [$]",
                min_value=0.0,
                value=50_000_000.0,
                step=50_000.0,
                format="%.0f",
                key="ms_comp",
            )
        with _cols[1]:
            tag_ms_comp = _plan_select_inline("tag_ms_comp")

        st.caption(f"= **${ms_comp/1_000_000:.3f} million**")
        _cols = st.columns([3, 2])
        with _cols[0]:
            ms_utils = st.number_input(
                "Utilities CAPEX [$]",
                min_value=0.0,
                value=0.0,
                step=10_000.0,
                format="%.0f",
                key="ms_utils",
            )
        with _cols[1]:
            tag_ms_utils = _plan_select_inline("tag_ms_utils")

        st.caption(f"= **${ms_utils / 1_000_000:.3f} million**")

        _cols = st.columns([3, 2])
        with _cols[0]:
            ms_gcond = st.number_input(
                "Gas conditioning CAPEX [$]",
                min_value=0.0,
                value=0.0,
                step=10_000.0,
                format="%.0f",
                key="ms_gcond",
            )
        with _cols[1]:
            tag_ms_gcond = _plan_select_inline("tag_ms_gcond")

        st.caption(f"= **${ms_gcond / 1_000_000:.3f} million**")

        _cols = st.columns([3, 2])
        with _cols[0]:
            ms_bay_cost = st.number_input(
                "Filling bay cost per bay [$]",
                min_value=0.0,
                value=500_000.0,
                step=10_000.0,
                format="%.0f",
                key="ms_bay_cost",
            )
        with _cols[1]:
            tag_ms_bays = _plan_select_inline("tag_ms_bays")

        st.caption(f"(per bay) = **${ms_bay_cost/1_000_000:.3f} million**")
        ms_bays_total = ms_bay_cost * st.session_state.get("available_bays_A", 12)
        st.caption(
            f"Filling bays total ({st.session_state.get('available_bays_A', 12)} ×): **${ms_bays_total/1_000_000:.3f} million**"
        )
        ms_capex = ms_comp + ms_utils + ms_gcond + ms_bays_total

        st.markdown("**Mother Station – OPEX**")
        ms_opex_pct_percent = st.number_input(
            "OPEX as % of Mother CAPEX (/year)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5,
            format="%.1f",
            key="ms_opex_pct_percent",
        )

        help_flyout(
            "opex_mother",
            "Annual **Mother OPEX** estimated as a simple percentage of Mother CAPEX. This covers operations, maintenance and site overheads.",
        )
        ms_opex_year = ms_capex * (ms_opex_pct_percent / 100.0)

        st.markdown("**Mother Station – Fuel/Power for compression**")
        ms_energy_method = st.radio(
            "Energy costing method",
            ["Fuel price ($/GJ fuel)", "Electricity price [$ / kWh]"],
            index=0,
            key="ms_energy_method",
        )

        if ms_energy_method.startswith("Fuel"):
            ms_fuel_price_per_gj = st.number_input(
                "Fuel price [$ / GJ]",
                min_value=0.0,
                value=0.0,
                step=0.5,
                key="ms_fuel_price_per_gj",
            )
            ms_energy_intensity_gj_per_gj = st.number_input(
                "Station energy intensity [GJ/GJ delivered]",
                min_value=0.0,
                value=0.02,
                step=0.005,
                format="%.3f",
                key="ms_energy_intensity_gj_per_gj",
            )

            ms_energy_cost_year = annual_gj * ms_energy_intensity_gj_per_gj * ms_fuel_price_per_gj
        else:
            ms_elec_price_per_kwh = st.number_input(
                "Electricity price [$ / kWh]",
                min_value=0.0,
                value=0.20,
                step=0.01,
                key="ms_elec_price_per_kwh",
            )
            ms_kwh_per_gj = st.number_input(
                "Station energy intensity [kWh/GJ delivered]",
                min_value=0.0,
                value=5.6,
                step=0.1,
                key="ms_kwh_per_gj",
            )

            ms_energy_cost_year = annual_gj * ms_kwh_per_gj * ms_elec_price_per_kwh
        help_flyout(
            "ms_energy_intensity",
            "**Station energy intensity** is the energy used by the Mother station to compress gas per unit energy delivered (e.g., 0.02 GJ of fuel per GJ delivered, or 5.6 kWh per GJ).",
        )

        st.markdown("---")
        st.markdown("**Daughter Station – CAPEX ($)**")
        st.caption(
            f"Unloading bays (count) is taken from the sidebar (Daughter): **{st.session_state.get('available_unload_bays_B', 4)}**"
        )
        _cols = st.columns([3, 2])
        with _cols[0]:
            ds_bay_cost = st.number_input(
                "Unloading bay cost per bay [$]",
                min_value=0.0,
                value=100_000.0,
                step=10_000.0,
                format="%.0f",
                key="ds_bay_cost",
            )
        with _cols[1]:
            tag_ds_bays = _plan_select_inline("tag_ds_bays")

        st.caption(f"(per bay) = **${ds_bay_cost/1_000_000:.3f} million**")
        ds_capex = st.session_state.get("available_unload_bays_B", 4) * ds_bay_cost
        st.caption(f"Unloading bays total: **${ds_capex/1_000_000:.3f} million**")
        ds_opex_pct_percent = st.number_input(
            "OPEX as % of Daughter CAPEX (/year)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5,
            format="%.1f",
            key="ds_opex_pct_percent",
        )

        help_flyout(
            "opex_daughter",
            "Annual **Daughter OPEX** estimated as a percentage of Daughter CAPEX (unloading bays and related infrastructure).",
        )
        ds_opex_year = ds_capex * (ds_opex_pct_percent / 100.0)

        st.markdown("---")
        st.markdown("**Fleet CAPEX ($)**")
        _cols = st.columns([3, 2])
        with _cols[0]:
            truck_capex_unit = st.number_input(
                "Truck CAPEX per unit [$]",
                min_value=0.0,
                value=550_000.0,
                step=10_000.0,
                format="%.0f",
                key="truck_capex_unit",
            )
        with _cols[1]:
            tag_truck_unit = _plan_select_inline("tag_truck_unit")

        st.caption(f"= **${truck_capex_unit/1_000_000:.3f} million per truck**")
        _cols = st.columns([3, 2])
        with _cols[0]:
            a_trailer_capex_unit = st.number_input(
                "A-trailer CAPEX per unit [$]",
                min_value=0.0,
                value=800_000.0,
                step=10_000.0,
                format="%.0f",
                key="a_trailer_capex_unit",
            )
        with _cols[1]:
            tag_a_trailer_unit = _plan_select_inline("tag_a_trailer_unit")

        st.caption(f"= **${a_trailer_capex_unit/1_000_000:.3f} million per A-trailer**")
        _cols = st.columns([3, 2])
        with _cols[0]:
            b_trailer_capex_unit = st.number_input(
                "B-trailer CAPEX per unit [$]",
                min_value=0.0,
                value=650_000.0,
                step=10_000.0,
                format="%.0f",
                key="b_trailer_capex_unit",
            )
        with _cols[1]:
            tag_b_trailer_unit = _plan_select_inline("tag_b_trailer_unit")

        st.caption(f"= **${b_trailer_capex_unit/1_000_000:.3f} million per B-trailer**")
        st.caption(f"Dollies per combination (auto): **{dollies_per_combo}**")
        _cols = st.columns([3, 2])
        with _cols[0]:
            dolly_capex_unit = st.number_input(
                "Dolly CAPEX per unit [$] (if used)",
                min_value=0.0,
                value=50_000.0,
                step=5_000.0,
                format="%.0f",
                key="dolly_capex_unit",
            )
        with _cols[1]:
            tag_dolly_unit = _plan_select_inline("tag_dolly_unit")

        st.caption(f"= **${dolly_capex_unit/1_000_000:.3f} million per dolly**")

        fleet_truck_capex = active_trucks * truck_capex_unit
        fleet_a_trailer_capex = a_trailers * a_trailer_capex_unit
        fleet_b_trailer_capex = b_trailers * b_trailer_capex_unit
        fleet_dolly_capex = total_dollies * dolly_capex_unit
        fleet_capex = (
            fleet_truck_capex + fleet_a_trailer_capex + fleet_b_trailer_capex + fleet_dolly_capex
        )

        st.caption(
            f"Fleet totals — Trucks: **${fleet_truck_capex/1_000_000:.3f}M**, "
            f"A-trailers: **${fleet_a_trailer_capex/1_000_000:.3f}M**, "
            f"B-trailers: **${fleet_b_trailer_capex/1_000_000:.3f}M**, "
            f"Dollies: **${fleet_dolly_capex/1_000_000:.3f}M**"
        )

    # ------------- Transport ops (service, fuel, drivers, insurance) -------------
    with cB:
        st.subheader("Transport Costs (annualised)")
        st.markdown("**Maintenance**")

        # --- Core maintenance (existing) ---
        maint_basis_trk = st.radio(
            "Truck maintenance basis",
            ["$/km", "$/year"],
            index=0,
            horizontal=True,
            key="maint_basis_trk",
        )

        if maint_basis_trk == "$/km":
            trk_maint_per_km = st.number_input(
                "Truck maintenance [$ / km]",
                min_value=0.0,
                value=0.36,
                step=0.05,
                key="trk_maint_per_km",
            )
            trk_maint_year = trk_maint_per_km * truck_km_per_day * DAYS_PER_YEAR
        else:
            trk_maint_year = st.number_input(
                "Truck maintenance [$ / year]",
                min_value=0.0,
                value=300_000.0,
                step=10_000.0,
                format="%.0f",
                key="trk_maint_year",
            )

        maint_basis_trl = st.radio(
            "Trailer maintenance basis",
            ["$/km", "$/year"],
            index=0,
            horizontal=True,
            key="maint_basis_trl",
        )

        if maint_basis_trl == "$/km":
            trl_maint_per_km = st.number_input(
                "Trailer maintenance [$ / km]",
                min_value=0.0,
                value=0.10,
                step=0.02,
                key="trl_maint_per_km",
            )
            trl_maint_year = trl_maint_per_km * trailer_km_per_day * DAYS_PER_YEAR
        else:
            trl_maint_year = st.number_input(
                "Trailer maintenance [$ / year]",
                min_value=0.0,
                value=200_000.0,
                step=10_000.0,
                format="%.0f",
                key="trl_maint_year",
            )

        # --- NEW: Tyres (adds to maintenance totals) ---
        st.markdown("**Tyres**")

        # Replacement rules come from constants.py (imported above)

        col_st1, col_st2 = st.columns([2, 1], vertical_alignment="center")
        with col_st1:
            tyre_steer_price = st.number_input(
                "Steer Tyres [$/Tyre]",
                min_value=0.0,
                value=float(st.session_state.get("tyre_steer_price", 900.0)),
                step=10.0,
                format="%.0f",
                key="tyre_steer_price",
            )
        with col_st2:
            tyre_steer_replace_km = st.number_input(
                "Replace at [km/steer tyre]",
                min_value=1.0,
                value=float(st.session_state.get("tyre_steer_replace_km", 120_000.0)),
                step=1_000.0,
                format="%.0f",
                key="tyre_steer_replace_km",
            )

        col_dt1, col_dt2 = st.columns([2, 1], vertical_alignment="center")
        with col_dt1:
            tyre_drive_price = st.number_input(
                "Drive Tyres [$/Tyre]",
                min_value=0.0,
                value=float(st.session_state.get("tyre_drive_price", 650.0)),
                step=10.0,
                format="%.0f",
                key="tyre_drive_price",
            )
        with col_dt2:
            tyre_drive_replace_km = st.number_input(
                "Replace at [km/drive tyre]",
                min_value=1.0,
                value=float(st.session_state.get("tyre_drive_replace_km", 180_000.0)),
                step=1_000.0,
                format="%.0f",
                key="tyre_drive_replace_km",
            )

        col_tt1, col_tt2 = st.columns([2, 1], vertical_alignment="center")
        with col_tt1:
            tyre_trailer_price = st.number_input(
                "Trailer Tyres [$/Tyre]",
                min_value=0.0,
                value=float(st.session_state.get("tyre_trailer_price", 550.0)),
                step=10.0,
                format="%.0f",
                key="tyre_trailer_price",
            )
        with col_tt2:
            tyre_trailer_replace_km = st.number_input(
                "Replace at [km/trailer tyre]",
                min_value=1.0,
                value=float(st.session_state.get("tyre_trailer_replace_km", 180_000.0)),
                step=1_000.0,
                format="%.0f",
                key="tyre_trailer_replace_km",
            )

        # Fleet travel (per year)
        fleet_truck_km_year = float(truck_km_per_day) * DAYS_PER_YEAR
        fleet_trailer_km_year = float(trailer_km_per_day) * DAYS_PER_YEAR

        # Tyre consumption model:
        # Tyres consumed per year = (km_per_year / km_per_tyre) * tyres_per_vehicle
        # For trucks, we use fleet_truck_km_year; for trailers, fleet_trailer_km_year.
        truck_tyre_cost_year = (
                tyre_steer_price * (
                    fleet_truck_km_year * (STEER_TYRES_PER_TRUCK / max(tyre_steer_replace_km, 1e-9)))
                + tyre_drive_price * (
                            fleet_truck_km_year * (DRIVE_TYRES_PER_TRUCK / max(tyre_drive_replace_km, 1e-9)))
        )

        # Both A- and B-trailers use 12 each (per the rule set)
        # trailer_km figure already sums km over all trailers; multiply by tyres per trailer once.
        # Both A- and B-trailers use the same tyre count (12 each by default).
        # Since trailer_km aggregates across all trailers, multiplying by tyres-per-trailer
        # gives the correct fleet consumption rate.
        trailer_tyre_cost_year = (
                tyre_trailer_price * (
                    fleet_trailer_km_year * (TRAILER_TYRES_PER_A / max(tyre_trailer_replace_km, 1e-9)))
        )

        # Roll tyre costs into maintenance totals
        trk_maint_year = float(trk_maint_year) + float(truck_tyre_cost_year)
        trl_maint_year = float(trl_maint_year) + float(trailer_tyre_cost_year)

        # Optional: show a quick breakdown note for transparency
        note_col1, note_col2 = st.columns(2)
        with note_col1:
            st.metric("Truck tyre cost [$/year]", f"${truck_tyre_cost_year:,.0f}")
        with note_col2:
            st.metric("Trailer tyre cost [$/year]", f"${trailer_tyre_cost_year:,.0f}")

        # --- Fuel (unchanged) ---
        st.markdown("**Fuel**")
        truck_km_per_litre = st.number_input(
            "Truck consumption [km/L]",
            min_value=0.01,
            value=1.8,
            step=0.1,
            key="truck_km_per_litre",
        )

        diesel_price_per_l = st.number_input(
            "Diesel price [$ / L]",
            min_value=0.0,
            value=2.00,
            step=0.05,
            key="diesel_price_per_l",
        )

        litres_per_day = truck_km_per_day / max(1e-6, truck_km_per_litre)
        fuel_cost_year = litres_per_day * diesel_price_per_l * DAYS_PER_YEAR

        st.markdown("**Drivers**")
        # --- Wages: unit toggle ($/day, $/hr, $/year), oncosts %, and working days ---
        # Persist prior selection/values
        st.session_state.setdefault("wage_unit", "$/day")
        st.session_state.setdefault("wage_per_driver", 600.0)       # value in the CURRENT unit
        st.session_state.setdefault("working_days_per_year", 200.0)
        st.session_state.setdefault("oncosts_pct", 0.0)

        # Wage unit (add $/hr)
        unit_options = ["$/day", "$/hr", "$/year"]
        current_unit = st.session_state.get("wage_unit", "$/day")
        unit_index = unit_options.index(current_unit) if current_unit in unit_options else 0
        unit_selected = st.radio("Wage unit", unit_options, index=unit_index, horizontal=True)

        # Convert stored wage value when unit changes (use working days & fatigue shift-cap hours/day)
        if unit_selected != current_unit:
            v = float(st.session_state.get("wage_per_driver", 600.0))
            days = float(st.session_state.get("working_days_per_year", 365.24))
            shift_cap = float(_current_driver_shift_cap())

            # Convert the old value to a canonical annual wage, then into the new unit
            if current_unit == "$/day":
                annual_from_old = v * days
            elif current_unit == "$/hr":
                annual_from_old = v * shift_cap * days
            else:  # "$/year"
                annual_from_old = v

            if unit_selected == "$/day":
                v_new = annual_from_old / max(days, 1e-9)
            elif unit_selected == "$/hr":
                v_new = annual_from_old / max(shift_cap * days, 1e-9)
            else:  # "$/year"
                v_new = annual_from_old

            st.session_state["wage_per_driver"] = float(v_new)
            st.session_state["wage_unit"] = unit_selected

        # Wage + Oncosts side by side (like the CAPEX/Plan layout)
        _cols = st.columns([3, 1])
        with _cols[0]:
            wage_label = f"Wages per driver [{st.session_state['wage_unit']}]"
            wage_val = currency_commas(
                wage_label,
                key="wage_per_driver",
                value=st.session_state.get("wage_per_driver", 600.0),
                decimals=0,
            )
        with _cols[1]:
            oncosts_pct = st.number_input(
                "Oncosts [%]",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get("oncosts_pct", 0.0)),
                step=0.5,
                format="%.1f",
                help="Percentage loading applied to wages (e.g., 33% → wage × 1.33).",
                key="oncosts_pct",
            )

        oncosts_factor = 1.0 + (oncosts_pct / 100.0)

        # FIFO & Camp remain [$/day]
        fifo_per_driver_day = st.number_input(
            "FIFO costs per driver [$/day]", min_value=0.0, value=120.0, step=10.0,
            key="fifo_per_driver_day",
        )
        camp_per_driver_day = st.number_input(
            "Camp accommodation per driver [$/day]", min_value=0.0, value=150.0, step=10.0,
            key="camp_per_driver_day",
        )

        # Working days per year
        working_days = st.number_input(
            "Working days per year [days]",
            min_value=1.0,
            value=float(st.session_state.get("working_days_per_year", 200.0)),
            step=1.0,
            format="%.2f",
            key="working_days_per_year",
            help="Average number of days each driver is available per year (after leave, training, RDOs). Used to calculate total driver headcount and costs."
        )

        # Helpful note: show the average ANNUAL wage per driver (wages only, excl. oncosts)
        _shift_cap = float(_current_driver_shift_cap())
        if st.session_state.get("wage_unit") == "$/day":
            annual_wage_excl_oncosts = float(wage_val) * working_days
        elif st.session_state.get("wage_unit") == "$/hr":
            annual_wage_excl_oncosts = float(wage_val) * _shift_cap * working_days
        else:  # "$/year"
            annual_wage_excl_oncosts = float(wage_val)

        # Line 1: match the simple format used for registration notes
        # Escape $ to avoid LaTeX formatting
        st.markdown(
            f"Average annual driver wages/salary [\\$/year]: \\${annual_wage_excl_oncosts:,.0f}",
            unsafe_allow_html=False,
        )

        st.markdown(
            "Note: this only includes Wages per driver and not oncosts. "
            "Oncosts factor into Total driver costs as part of OPEX totals.",
            unsafe_allow_html=False,
        )

        # Compute annual driver cost
        drivers_on_duty = calcs_cost["drivers_on_duty"]
        D_oper = 365.0
        D_driver = float(st.session_state.get("working_days_per_year", 200.0))
        drivers_total = int(math.ceil(drivers_on_duty * (D_oper / max(D_driver, 1e-9))))

        annual_wage_with_oncosts = annual_wage_excl_oncosts * oncosts_factor
        annual_fifo_camp = (fifo_per_driver_day + camp_per_driver_day) * working_days
        driver_cost_year = (annual_wage_with_oncosts + annual_fifo_camp) * drivers_total

        st.markdown("**Insurance & Misc.**")
        insurance_year = st.number_input(
            "Insurance [$ / year]",
            min_value=0.0,
            value=300_000.0,
            step=10_000.0,
            format="%.0f",
            key="insurance_year",
        )
        misc_year = st.number_input(
            "Miscellaneous [$ / year]",
            min_value=0.0,
            value=150_000.0,
            step=10_000.0,
            format="%.0f",
            key="misc_year",
        )

        st.markdown("**Registration**")
        reg_truck_per_year = st.number_input(
            "Annual registration per truck [$ / year]",
            min_value=0.0,
            value=10_000.0,
            step=500.0,
            format="%.0f",
            key="reg_truck_per_year",
        )
        reg_trailer_per_year = st.number_input(
            "Annual registration per trailer [$ / year]",
            min_value=0.0,
            value=2_000.0,
            step=100.0,
            format="%.0f",
            key="reg_trailer_per_year",
        )

        # Counts from planner calcs (already available above)
        registration_year = (
                reg_truck_per_year * active_trucks
                + reg_trailer_per_year * (a_trailers + b_trailers)
        )

        # NEW: Breakdown notes for visibility
        total_truck_reg_year = reg_truck_per_year * active_trucks
        total_trailer_reg_year = reg_trailer_per_year * (a_trailers + b_trailers)

        st.markdown(
            f"<p>Total annual truck fleet registration costs [$/year]: "
            f"<strong>${total_truck_reg_year:,.0f}</strong></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p>Total annual trailer fleet registration costs [$/year]: "
            f"<strong>${total_trailer_reg_year:,.0f}</strong></p>",
            unsafe_allow_html=True,
        )

        # Group “residual” transport OPEX here for clarity
        other_opex_year = insurance_year + misc_year + registration_year

        st.markdown("---")
    st.subheader("Finance / Levelization")

    # --- Multi-plan finance: up to 10 plans, each with Label, Rate% (real/yr), NPER (months) ---
    with st.expander("Finance plans (up to 10) & CAPEX tagging", expanded=True):
        # Initialise plans structure once
        if "finance_plans" not in st.session_state:
            st.session_state["finance_plans"] = [
                {"label": "Plan A", "rate_pct": 6.0, "nper_months": max(60, int(project_months)), "enabled": True}
            ]
        plans = st.session_state["finance_plans"]
        # One-time migration: if older sessions seeded 8.0% by default, shift them to 6.0%
        if not st.session_state.get("finance_plans_migrated_to_6", False):
            for p in plans:
                if float(p.get("rate_pct", 6.0)) == 8.0:
                    p["rate_pct"] = 6.0
            st.session_state["finance_plans_migrated_to_6"] = True

        # Editor for up to 10 rows
        max_rows = 10
        for i in range(max_rows):
            if i >= len(plans):
                plans.append(
                    {"label": f"Plan {chr(65 + i)}", "rate_pct": 6.0, "nper_months": max(60, int(project_months)),
                     "enabled": False})
            cols = st.columns([4, 3, 3, 2])
            with cols[0]:
                plans[i]["enabled"] = st.checkbox(f"Enable", value=plans[i]["enabled"], key=f"fin_enable_{i}")
                plans[i]["label"] = st.text_input("Label", value=plans[i]["label"], key=f"fin_label_{i}")
            with cols[1]:
                plans[i]["rate_pct"] = st.number_input(
                    "Rate [%/yr]",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(plans[i]["rate_pct"] if "rate_pct" in plans[i] else 6.0),
                    step=0.25,
                    format="%.2f",
                    key=f"fin_rate_{i}",
                )

            with cols[2]:
                plans[i]["nper_months"] = st.number_input(
                    "NPER [months]", min_value=3, value=int(plans[i]["nper_months"]), step=1, key=f"fin_nper_{i}"
                )
            with cols[3]:
                st.markdown("&nbsp;")  # spacer
        st.session_state["finance_plans"] = plans  # save back

        # ---- Tag each CAPEX bucket to a plan label ----
        # Buckets available below (already computed earlier in this tab):
        #   capex_truck_total, capex_roadtrain_total, capex_mother_total, capex_daughter_total
    # ---- Plan labels will now be chosen adjacent to each CAPEX input (below) ----
    enabled_labels = [p["label"] for p in plans if p["enabled"]]
    if not enabled_labels:
        st.warning("Enable at least one finance plan.")

    def _plan_select_inline(key, default_idx=0):
        idx = min(default_idx, max(len(enabled_labels) - 1, 0))
        return st.selectbox("Plan", enabled_labels if enabled_labels else [""], index=idx, key=key)

    # --- Build helper dicts for per-plan CRFs and per-plan CAPEX allocations ---
    from collections import defaultdict

    plan_params = {}
    for p in st.session_state["finance_plans"]:
        if not p["enabled"]:
            continue
        # annual real rate i, years = months/12 → CRF per plan
        i_yr = float(p["rate_pct"]) / 100.0
        n_years = float(p["nper_months"]) / 12.0
        plan_params[p["label"]] = {
            "i_yr": i_yr,
            "n_years": n_years,
            "crf_year": capital_recovery_factor(i_yr, n_years),  # uses finance.py helper
        }

    # ---- CAPEX breakdown (define BEFORE we map to plans) ----
    capex_truck_total = float(fleet_truck_capex)
    capex_roadtrain_total = float(fleet_a_trailer_capex + fleet_b_trailer_capex + fleet_dolly_capex)
    capex_mother_total = float(ms_capex)
    capex_daughter_total = float(ds_capex)
    capex_total_all = capex_truck_total + capex_roadtrain_total + capex_mother_total + capex_daughter_total

    # Map each CAPEX INPUT (or its implied total) to the selected plan label
    capex_by_plan = defaultdict(float)

    # Mother Station items
    capex_by_plan[tag_ms_comp] += float(ms_comp)
    capex_by_plan[tag_ms_utils] += float(ms_utils)
    capex_by_plan[tag_ms_gcond] += float(ms_gcond)
    capex_by_plan[tag_ms_bays] += float(ms_bays_total)  # per-bay cost × filling bays

    # Daughter Station items
    capex_by_plan[tag_ds_bays] += float(ds_capex)  # per-bay cost × unloading bays

    # Fleet items (unit cost × units)
    capex_by_plan[tag_truck_unit] += float(fleet_truck_capex)
    capex_by_plan[tag_a_trailer_unit] += float(fleet_a_trailer_capex)
    capex_by_plan[tag_b_trailer_unit] += float(fleet_b_trailer_capex)
    capex_by_plan[tag_dolly_unit] += float(fleet_dolly_capex)

    # Annualised CAPEX per plan (sum over all buckets assigned to that plan)
    annuitized_capex_year_by_plan = {}
    for lbl, cap in capex_by_plan.items():
        params = plan_params.get(lbl)
        if not params:
            continue
        annuitized_capex_year_by_plan[lbl] = cap * params["crf_year"]

    # Total across all plans (replaces single-plan `annuitized_capex_year`)
    annuitized_capex_year = sum(annuitized_capex_year_by_plan.values())

    # ---- Rollups
    infra_capex_total = (
        (ms_comp + ms_utils + ms_gcond + ms_bays_total)
        + ds_capex
        + (fleet_truck_capex + fleet_a_trailer_capex + fleet_b_trailer_capex + fleet_dolly_capex)
    )
    infra_opex_year = ms_opex_year + ds_opex_year
    station_energy_year = ms_energy_cost_year
    transport_ops_year = (
            trk_maint_year
            + trl_maint_year
            + fuel_cost_year
            + driver_cost_year
            + other_opex_year
    )
    
    # Store transport cost components in session state for pie chart access
    # Use _calc suffix to avoid conflicts with widget keys
    st.session_state["fuel_cost_year_calc"] = float(fuel_cost_year)
    st.session_state["trk_maint_year_calc"] = float(trk_maint_year)
    st.session_state["trl_maint_year_calc"] = float(trl_maint_year)
    st.session_state["driver_cost_year_calc"] = float(driver_cost_year)
    st.session_state["registration_year_calc"] = float(registration_year)
    st.session_state["insurance_year_calc"] = float(insurance_year)
    st.session_state["misc_year_calc"] = float(misc_year)
    st.session_state["other_opex_year_calc"] = float(other_opex_year)

    # (annuitized_capex_year is already computed from per-plan CRFs above)
    # CAPEX breakdown was defined above; keep infra_capex_total for totals

    # Split annuity into principal and interest components for current year
    # ---- Per-plan principal/interest/monthly rollups ----
    def _month_rate_from_annual(i_yr: float) -> float:
        return (1.0 + i_yr) ** (1.0 / 12.0) - 1.0


    annual_interest_payment = 0.0
    annual_principal_payment = 0.0
    monthly_principal_payment = 0.0
    monthly_interest_payment = 0.0
    monthly_payment = 0.0
    total_repayment_over_term = 0.0
    total_interest_over_term = 0.0
    total_principal_over_term = 0.0

    for lbl, cap in capex_by_plan.items():
        params = plan_params.get(lbl)
        if not params or cap <= 0:
            continue
        i_yr = params["i_yr"]
        n_months = int(round(params["n_years"] * 12.0))
        i_m = _month_rate_from_annual(i_yr)
        if n_months > 0 and i_m > 0.0:
            p = (1.0 + i_m) ** n_months
            crf_m = i_m * p / (p - 1.0)
        else:
            crf_m = 1.0 / max(n_months, 1)

        # Annual split (simple: cap * i + (cap*CRF_yr - interest))
        ann_pay = cap * params["crf_year"]
        ann_int = cap * i_yr
        ann_prin = max(ann_pay - ann_int, 0.0)
        annual_interest_payment += ann_int
        annual_principal_payment += ann_prin

        # Monthly split
        mpay = cap * crf_m
        mint = cap * i_m
        mprin = max(mpay - mint, 0.0)
        monthly_payment += mpay
        monthly_interest_payment += mint
        monthly_principal_payment += mprin

        # Totals over term
        total_principal_over_term += cap
        total_repayment_over_term += mpay * n_months
        total_interest_over_term += max(mpay * n_months - cap, 0.0)

    total_annual_cost = (
            annuitized_capex_year + infra_opex_year + station_energy_year + transport_ops_year
    )
    
    # Store cost components in session state for pie chart access
    st.session_state["annuitized_capex_year"] = float(annuitized_capex_year)
    st.session_state["infra_opex_year"] = float(infra_opex_year)
    st.session_state["station_energy_year"] = float(station_energy_year)
    st.session_state["transport_ops_year"] = float(transport_ops_year)
    st.session_state["total_annual_cost"] = float(total_annual_cost)

    # Use EFFECTIVE throughput (utilisation-adjusted) for LC, to match Benefits basis
    _eff_daily_lc = float(st.session_state.get("effective_daily_gj", daily_energy_gj))

    levelized_cost_per_gj = (
        (total_annual_cost / max(_eff_daily_lc * DAYS_PER_YEAR, 1e-9))
        if _eff_daily_lc > 0
        else float("nan")
    )

    # Project-duration rollups (use sidebar Project duration [months])
    project_years_for_rollup = float(project_months) / 12.0

    # Mother Station OPEX over project
    ms_facility_project = ms_opex_year * project_years_for_rollup
    ms_energy_project = ms_energy_cost_year * project_years_for_rollup
    ms_opex_project = ms_facility_project + ms_energy_project

    # NEW: Daughter Station OPEX over project
    ds_opex_project = ds_opex_year * project_years_for_rollup

    # Transport OPEX over project (four requested items)
    fuel_cost_project = fuel_cost_year * project_years_for_rollup
    trk_maint_project = trk_maint_year * project_years_for_rollup
    trl_maint_project = trl_maint_year * project_years_for_rollup
    driver_cost_project = driver_cost_year * project_years_for_rollup

    registration_project = registration_year * project_years_for_rollup
    insurance_project = insurance_year * project_years_for_rollup
    misc_project = misc_year * project_years_for_rollup
    other_opex_project = registration_project + insurance_project + misc_project

    transport_opex_project = (
            fuel_cost_project + trk_maint_project + trl_maint_project + driver_cost_project + other_opex_project
    )

    # Don't overwrite session state here - let compute_levelized_metrics_for_scenario() handle it
    # st.session_state["lc_per_gj"] = float(levelized_cost_per_gj)

    st.markdown("### Cost Summary [currency]")

    # Left: CAPEX breakdown (millions) with Principal/Interest lines
    s1, s2, sD, s3 = st.columns(4)

    with s1:
        st.markdown(f"**Total CAPEX [$ million]: {capex_total_all / 1_000_000:,.3f}**")
        st.markdown(
            f"""<div style='line-height:1.45; margin-top:2px;'>
              <div style='padding-left:12px;'>Total trucks CAPEX [$ million]: <b>{capex_truck_total / 1_000_000:,.3f}</b></div>
              <div style='padding-left:12px;'>Total road-train CAPEX [$ million]: <b>{capex_roadtrain_total / 1_000_000:,.3f}</b></div>
              <div style='padding-left:12px;'>Total Mother Station CAPEX [$ million]: <b>{capex_mother_total / 1_000_000:,.3f}</b></div>
              <div style='padding-left:12px;'>Total Daughter Station CAPEX [$ million]: <b>{capex_daughter_total / 1_000_000:,.3f}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with s2:
        # MOTHER — YEAR
        mother_opex_year_total = ms_opex_year + ms_energy_cost_year
        st.markdown(
            f"<div style='line-height:1.2;'>"
            f"<span style='font-weight:600;'>Total Mother Station OPEX [$/year]:</span> "
            f"<span style='font-weight:700;'>${mother_opex_year_total:,.0f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div style='line-height:1.45; margin-top:2px;'>
            <div style='padding-left:12px;'>Total Facility OPEX [$/year]: <b>${ms_opex_year:,.0f}</b></div>
            <div style='padding-left:12px;'>Total Fuel / Electricity Cost [$/year]: <b>${ms_energy_cost_year:,.0f}</b></div>
            <div style='border-top:1px solid #e5e5e5; margin:8px 0 8px 0;'></div>
            </div>""",
            unsafe_allow_html=True,
        )

        # MOTHER — PROJECT
        st.markdown(
            f"""<div style='line-height:1.45; margin-top:8px; padding-bottom:8px;'>
              <div style='font-weight:700;'>Total Mother Station OPEX [$/project]: ${ms_opex_project:,.0f}</div>
              <div style='padding-left:12px;'>Total Facility OPEX [$/project]: <b>${ms_facility_project:,.0f}</b></div>
              <div style='padding-left:12px;'>Total Fuel / Electricity Cost [$/project]: <b>${ms_energy_project:,.0f}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with sD:
        # DAUGHTER — mirrors Mother block (no energy line yet)
        st.markdown(
            f"<div style='line-height:1.2;'>"
            f"<span style='font-weight:600;'>Total Daughter Station OPEX [$/year]:</span> "
            f"<span style='font-weight:700;'>${ds_opex_year:,.0f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div style='line-height:1.45; margin-top:2px;'>
            <div style='padding-left:12px;'>Total Facility OPEX [$/year]: <b>${ds_opex_year:,.0f}</b></div>
            <div style='padding-left:12px;'>Total Fuel / Electricity Cost [$/year]: <b>$0</b></div>
            <div style='border-top:1px solid #e5e5e5; margin:8px 0 8px 0;'></div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div style='line-height:1.45; margin-top:8px; padding-bottom:8px;'>
              <div style='font-weight:700;'>Total Daughter Station OPEX [$/project]: ${ds_opex_project:,.0f}</div>
              <div style='padding-left:12px;'>Total Facility OPEX [$/project]: <b>${ds_opex_project:,.0f}</b></div>
              <div style='padding-left:12px;'>Total Fuel / Electricity Cost [$/project]: <b>$0</b></div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with s3:
        st.markdown(f"**Total Transport OPEX [\$/year]: \${transport_ops_year:,.0f}**")
        st.markdown(
            f"""<div style='line-height:1.45; margin-top:2px;'>
            <div style='padding-left:12px;'>Total trucking fuel [$ /year]: <b>${fuel_cost_year:,.0f}</b></div>
            <div style='padding-left:12px;'>Total truck maintenance [$ /year]: <b>${trk_maint_year:,.0f}</b></div>
            <div style='padding-left:12px;'>Total trailer maintenance [$ /year]: <b>${trl_maint_year:,.0f}</b></div>
            <div style='padding-left:12px;'>Total driver costs [$ /year]: <b>${driver_cost_year:,.0f}</b></div>
            <div style='padding-left:12px;'>Total other OPEX costs [$ /year]: <b>${other_opex_year:,.0f}</b></div>
            <div style="border-top:1px solid #e5e5e5; margin:8px 0 8px 0;"></div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ---- PROJECT totals (Transport) ----

        st.markdown(
            dedent(
                f"""\
            <div style='line-height:1.45; margin-top:8px; padding-bottom:8px;'>
              <div style='font-weight:700;'>Total Transport OPEX [$/project]: ${transport_opex_project:,.0f}</div>
              <div style='padding-left:12px;'>Total trucking fuel [$/project]: <b>${fuel_cost_project:,.0f}</b></div>
              <div style='padding-left:12px;'>Total truck maintenance [$/project]: <b>${trk_maint_project:,.0f}</b></div>
              <div style='padding-left:12px;'>Total trailer maintenance [$/project]: <b>${trl_maint_project:,.0f}</b></div>
              <div style='padding-left:12px;'>Total driver costs [$/project]: <b>${driver_cost_project:,.0f}</b></div>
              <div style='padding-left:12px;'>Total other OPEX costs [$/project]: <b>${other_opex_project:,.0f}</b></div>
            </div>"""
            ),
            unsafe_allow_html=True,
        )

        # ---- Trucking-only Levelized Cost [$/GJ] (excludes stations CAPEX/OPEX) ----
        # Uses same CRF you apply elsewhere (interest and N years are embedded in CRF).
        # trucking_annualized = annualized fleet CAPEX + annual transport OPEX
        # Safe defaults in case aggregation fails
        trucking_capex_annual = 0.0
        trucking_annualized = 0.0

        try:
            # Annualized fleet CAPEX using the per-input plan tags
            fleet_capex_by_plan = defaultdict(float)
            fleet_capex_by_plan[tag_truck_unit] += float(fleet_truck_capex)
            fleet_capex_by_plan[tag_a_trailer_unit] += float(fleet_a_trailer_capex)
            fleet_capex_by_plan[tag_b_trailer_unit] += float(fleet_b_trailer_capex)
            fleet_capex_by_plan[tag_dolly_unit] += float(fleet_dolly_capex)

            for lbl, cap in fleet_capex_by_plan.items():
                params = plan_params.get(lbl)
                if not params or cap <= 0:
                    continue
                trucking_capex_annual += cap * params["crf_year"]

            # Transport OPEX (already split above as fuel/maint/drivers/insurance/misc)
            transport_opex_annual = float(transport_ops_year)

            # Annualized trucking-only total (CAPEX ann. + OPEX)
            trucking_annualized = float(trucking_capex_annual) + transport_opex_annual
            
            # Store trucking CAPEX annualized in session state for pie chart access
            st.session_state["trucking_capex_annual"] = float(trucking_capex_annual)

        except Exception:
            # Keep last good values if CAPEX/OPEX aggregation hiccups
            pass


        try:
            # Use EFFECTIVE annual energy (utilisation-adjusted), consistent with LC & Benefits
            _eff_daily_transport = float(st.session_state.get("effective_daily_gj", daily_energy_gj))
            annual_energy_delivered_gj = _eff_daily_transport * DAYS_PER_YEAR

            # Only update when we have valid inputs; otherwise keep prior value (avoid writing 0.0)
            # Don't overwrite session state here - let compute_levelized_metrics_for_scenario() handle it
            # if trucking_annualized > 0 and annual_energy_delivered_gj > 0:
            #     st.session_state["transport_lc_per_gj"] = trucking_annualized / annual_energy_delivered_gj

        except Exception:
            # On transient errors, keep prior session values (last good) and still show the ribbon
            pass

        # Spacer to ensure clear gap under the project block (kept inside the column)
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Finance Costs and Repayments")

    # Optional: quick per-plan table for transparency
    with st.expander("Per-plan finance breakdown", expanded=False):
        import pandas as pd

        rows = []
        for lbl, cap in capex_by_plan.items():
            params = plan_params.get(lbl)
            if not params or cap <= 0:
                continue
            i_yr = params["i_yr"];
            n_years = params["n_years"];
            crf_yr = params["crf_year"]
            ann_pay = cap * crf_yr
            ann_int = cap * i_yr
            ann_prin = max(ann_pay - ann_int, 0.0)
            rows.append({
                "Plan": lbl,
                "Tagged CAPEX [$]": cap,
                "Rate [%/yr]": i_yr * 100.0,
                "Term [years]": n_years,
                "Annual principal [$]": ann_prin,
                "Annual interest [$]": ann_int,
                "Annual total [$]": ann_pay,
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.format({
                    "Tagged CAPEX [$]": "{:,.0f}",
                    "Rate [%/yr]": "{:.2f}",
                    "Term [years]": "{:.2f}",
                    "Annual principal [$]": "{:,.0f}",
                    "Annual interest [$]": "{:,.0f}",
                    "Annual total [$]": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No CAPEX buckets tagged to enabled plans yet.")

    finance_details_html = dedent(
        f"""\
    <div style="line-height:1.45; margin-top:2px;">

    <div style="font-weight:600; margin:6px 0 2px 0;">Annual repayments</div>
    <div style="padding-left:12px;">Annual Principal Payment [$ million/year]: <b>{annual_principal_payment / 1_000_000:,.3f}</b></div>
    <div style="padding-left:12px;">Annual Interest Payment [$ million/year]: <b>{annual_interest_payment / 1_000_000:,.3f}</b></div>
    <div style="padding-left:12px;">Total Annual Repayments [$ million/year]: <b>{annuitized_capex_year / 1_000_000:,.3f}</b></div>

    <div style="font-weight:600; margin:10px 0 2px 0;">Monthly repayments</div>
    <div style="padding-left:12px;">Monthly Principal Payment [$/month]: <b>{monthly_principal_payment:,.0f}</b></div>
    <div style="padding-left:12px;">Monthly Interest Payment [$/month]: <b>{monthly_interest_payment:,.0f}</b></div>
    <div style="padding-left:12px;">Total Monthly Repayments [$/month]: <b>{monthly_payment:,.0f}</b></div>

    <div style="font-weight:600; margin:10px 0 2px 0;">Total repayments over term of finance</div>
    <div style="padding-left:12px;">Total Principal Payment [$]: <b>{total_principal_over_term:,.0f}</b></div>
    <div style="padding-left:12px;">Total Interest Payment [$]: <b>{total_interest_over_term:,.0f}</b></div>
    <div style="padding-left:12px;">Total Repayment [$]: <b>{total_repayment_over_term:,.0f}</b></div>

    </div>
    """
    )
    st.markdown(finance_details_html, unsafe_allow_html=True)

    # ===== NPV & IRR — Project Economics =====
    st.markdown("---")
    st.markdown("### NPV & IRR (project economics)")

    # Inputs
    npv_disc_rate_pct = st.number_input(
        "Discount rate for NPV [%]", min_value=0.0, max_value=100.0, value=10.0, step=0.5
    )
    npv_disc_rate = npv_disc_rate_pct / 100.0

    st.caption(
        f"Project duration (from sidebar): **{project_months} months** "
        f"(≈ **{project_months / 12.0:.2f} years**)"
    )
    proj_years = float(project_months) / 12.0
    n_full_years = int(round(proj_years))  # simple annual model

    st.markdown("**Pre-project CAPEX allocation** (100% must be spent before operations)")
    capex_pct_y2 = st.number_input(
        "Year −2 CAPEX allocation [%]", min_value=0.0, max_value=100.0, value=20.0, step=1.0
    )
    capex_pct_y1 = st.number_input(
        "Year −1 CAPEX allocation [%]", min_value=0.0, max_value=100.0, value=80.0, step=1.0
    )

    # Enforce/normalise to 100% (keep user intent, but protect math)
    pct_sum = max(capex_pct_y2 + capex_pct_y1, 1e-9)
    if abs(pct_sum - 100.0) > 1e-6:
        st.warning("Year −2 % + Year −1 % ≠ 100%. Values will be proportionally scaled to 100%.")
    scale = 100.0 / pct_sum

    capex_total = float(capex_total_all)  # already computed above in this tab
    capex_y2 = -capex_total * (capex_pct_y2 * scale / 100.0)  # negative = outflow
    capex_y1 = -capex_total * (capex_pct_y1 * scale / 100.0)

    # --- Annual operating numbers (using utilisation-adjusted throughput) ---
    eff_daily = float(st.session_state.get("effective_daily_gj", daily_energy_gj))
    annual_gj = eff_daily * DAYS_PER_YEAR
    st.session_state["annual_energy_delivered_gj"] = float(annual_gj)

    # Annual revenue stream (includes utilisation) — single source of truth is Benefits & Carbon
    sales_price_per_gj = float(st.session_state.get("sales_gas_value_per_gj", 12.0))
    st.caption(
        f"Revenue price is set on **Benefits & Carbon → Gas Sales / Value**. "
        f"Current: **${sales_price_per_gj:.2f}/GJ**"
    )

    # Annual revenue stream (includes utilisation)
    revenue_year = sales_price_per_gj * annual_gj

    st.caption(
        f"Planned daily gas: {float(daily_energy_gj):,.0f} GJ/day  •  "
        f"Expected @ utilisation: {eff_daily:,.0f} GJ/day"
    )

    annual_opex = float(infra_opex_year + station_energy_year + transport_ops_year)

    # Build a simple YEARLY cashflow vector:
    # index 0 -> Year −2, 1 -> Year −1, then 1..N project years
    cashflows = [capex_y2, capex_y1] + [float(revenue_year - annual_opex)] * n_full_years

    # NPV at the start of operations (t=0).
    # For t<0 (−2, −1), discount exponent is negative → PV0 = CF * (1+r)^(+years_before)
    def npv_at_start(r, cfs):
        pv = 0.0
        for i, cf in enumerate(cfs):
            t = i - 2  # i=0 -> Year −2, i=1 -> Year −1, i=2 -> first operating year
            if t < 0:
                # Pre-build CAPEX brought forward to start of operations
                pv += cf * ((1.0 + r) ** abs(t))
            else:
                # Operating cash flows assumed at END of each year
                pv += cf / ((1.0 + r) ** (t + 1))
        return pv

    npv_value = npv_at_start(npv_disc_rate, cashflows)

    # IRR solver (robust bisection on standard NPV with t=0 at Year −2; IRR is timing-offset invariant)
    def npv_standard(r, cfs):
        """Standard NPV with t=0 at first element of cfs."""
        return sum(cf / ((1.0 + r) ** t) for t, cf in enumerate(cfs))

    def irr_bisection(cfs, lo=-0.99, hi=1.5, tol=1e-7, iters=200):
        f_lo = npv_standard(lo, cfs)
        f_hi = npv_standard(hi, cfs)
        # Expand interval if same sign
        expand = 0
        while f_lo * f_hi > 0 and expand < 25:
            lo -= 0.5
            hi += 0.5
            f_lo = npv_standard(lo, cfs)
            f_hi = npv_standard(hi, cfs)
            expand += 1
        if f_lo * f_hi > 0:
            return float("nan")
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f_mid = npv_standard(mid, cfs)
            if abs(f_mid) < tol:
                return mid
            if f_lo * f_mid <= 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return 0.5 * (lo + hi)

    irr_value = irr_bisection(cashflows)  # returns a rate (e.g., 0.147 = 14.7%)

    # Display
    c_npva, c_npvb, c_npvc = st.columns(3)
    with c_npva:
        st.metric(
            f"NPV ({npv_disc_rate_pct:.1f}%) [currency at start of operations]",
            f"${npv_value:,.0f}",
        )
    with c_npvb:
        st.metric("IRR [%]", f"{(irr_value * 100.0):.2f}%")
    with c_npvc:
        st.metric("Annual net operating cash flow [$/yr]", f"${(revenue_year - annual_opex):,.0f}")
    st.caption(
        "Timing note: CAPEX outflows (Years −2 and −1) are compounded forward to the start of operations (t = 0). "
        "Operating revenues and OPEX are discounted as end-of-year cash flows during the project."
    )

    # ---- Component NPVs (at the selected discount rate) ----
    st.markdown("#### Component NPVs")

    # Helpers
    def pv_at_start_of_ops_prebuild(amount_y2, amount_y1, r):
        return amount_y2 * (1.0 + r) ** 2 + amount_y1 * (1.0 + r)

    def pv_annuity_per_year(a, n_years, r):
        if n_years <= 0:
            return 0.0
        if r == 0.0:
            return a * n_years
        # ordinary annuity PV at t=0 (first cash flow at end of Year 1)
        return a * (1.0 - (1.0 + r) ** (-n_years)) / r / (1.0 + r)

    # Normalised split (already computed above)
    pY2 = (capex_pct_y2 * scale) / 100.0
    pY1 = (capex_pct_y1 * scale) / 100.0

    # --- CAPEX components
    capex_truck_pv = pv_at_start_of_ops_prebuild(
        -capex_truck_total * pY2, -capex_truck_total * pY1, npv_disc_rate
    )
    capex_roadtrain_pv = pv_at_start_of_ops_prebuild(
        -capex_roadtrain_total * pY2, -capex_roadtrain_total * pY1, npv_disc_rate
    )
    capex_mother_pv = pv_at_start_of_ops_prebuild(
        -capex_mother_total * pY2, -capex_mother_total * pY1, npv_disc_rate
    )
    capex_daughter_pv = pv_at_start_of_ops_prebuild(
        -capex_daughter_total * pY2, -capex_daughter_total * pY1, npv_disc_rate
    )

    # --- OPEX & REVENUE components
    mother_opex_year = float(ms_opex_year + ms_energy_cost_year)
    transport_opex_year = float(transport_ops_year)
    revenue_year_stream = float(revenue_year)
    carbon_benefit_year = float(st.session_state.get("carbon_benefit_year", 0.0))

    mother_opex_pv = -pv_annuity_per_year(mother_opex_year, n_full_years, npv_disc_rate)
    daughter_opex_pv = -pv_annuity_per_year(ds_opex_year, n_full_years, npv_disc_rate)
    transport_opex_pv = -pv_annuity_per_year(transport_opex_year, n_full_years, npv_disc_rate)
    revenue_pv = pv_annuity_per_year(revenue_year_stream, n_full_years, npv_disc_rate)
    carbon_benefit_pv = pv_annuity_per_year(carbon_benefit_year, n_full_years, npv_disc_rate)

    # --- Display in 4 columns
    c1_pv, c2_pv, c3_pv, c4_pv = st.columns(4)
    with c1_pv:
        st.metric("Total truck CAPEX [$]", f"{capex_truck_pv:,.0f}")
        st.metric("Total mother station OPEX [$]", f"{mother_opex_pv:,.0f}")
    with c2_pv:
        st.metric("Total road train CAPEX [$]", f"{capex_roadtrain_pv:,.0f}")
        st.metric("Total transport OPEX [$]", f"{transport_opex_pv:,.0f}")
    with c3_pv:
        st.metric("Total mother station CAPEX [$]", f"{capex_mother_pv:,.0f}")
        st.metric("Total revenue - sales gas [$]", f"{revenue_pv:,.0f}")
    with c4_pv:
        st.metric("Total daughter station CAPEX [$]", f"{capex_daughter_pv:,.0f}")
        st.metric("Total daughter station OPEX [$]", f"{daughter_opex_pv:,.0f}")
        st.metric("Total flare/vent avoidance [$]", f"{carbon_benefit_pv:,.0f}")

    st.caption(
        "Notes: CAPEX values are PV at start of operations using your Year −2/−1 split. "
        "OPEX streams are annual costs (negative cash flow). "
        "Revenue and carbon are positive streams discounted across project years."
    )

    # --- Optional Dual PV Comparison (FID vs COD) ---
    st.markdown("---")
    st.markdown("#### CAPEX Present Values — Comparison (Start-of-Operations vs Investment Decision Basis)")

    # 1️⃣ COD basis (as used in current app)
    def pv_cod(amount_y2, amount_y1, r):
        """Compound forward to start of operations (COD basis)."""
        return amount_y2 * (1 + r) ** 2 + amount_y1 * (1 + r)


    # 2️⃣ FID basis (discount back to investment decision date)
    def pv_fid(amount_y2, amount_y1, r):
        """Discount back to investment decision (FID basis)."""
        return amount_y2 / ((1 + r) ** 2) + amount_y1 / (1 + r)


    cod_vals = {
        "Truck CAPEX": pv_cod(-capex_truck_total * pY2, -capex_truck_total * pY1, npv_disc_rate),
        "Road-train CAPEX": pv_cod(-capex_roadtrain_total * pY2, -capex_roadtrain_total * pY1, npv_disc_rate),
        "Mother Station CAPEX": pv_cod(-capex_mother_total * pY2, -capex_mother_total * pY1, npv_disc_rate),
        "Daughter Station CAPEX": pv_cod(-capex_daughter_total * pY2, -capex_daughter_total * pY1, npv_disc_rate),
    }

    fid_vals = {
        "Truck CAPEX": pv_fid(-capex_truck_total * pY2, -capex_truck_total * pY1, npv_disc_rate),
        "Road-train CAPEX": pv_fid(-capex_roadtrain_total * pY2, -capex_roadtrain_total * pY1, npv_disc_rate),
        "Mother Station CAPEX": pv_fid(-capex_mother_total * pY2, -capex_mother_total * pY1, npv_disc_rate),
        "Daughter Station CAPEX": pv_fid(-capex_daughter_total * pY2, -capex_daughter_total * pY1, npv_disc_rate),
    }

    # --- Display two stacked sections
    st.markdown("##### Present Value @ Commercial Operation (COD Basis)")
    st.caption(
        "_Each pre-build CAPEX component is grown forward to the point operations commence (COD). "
        "This aligns with the IRR and other cashflows referenced to the start of operations._"
    )

    cols_cod = st.columns(4)
    for (label, val), col in zip(cod_vals.items(), cols_cod):
        with col:
            st.metric(label, f"{val:,.0f}")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Present Value @ Investment Decision (Today Basis)")
    st.caption(
        "_Each pre-build CAPEX component is discounted back to the investment decision date (today). "
        "This aligns with corporate discounted-cashflow and feasibility valuations._"
    )

    cols_fid = st.columns(4)
    for (label, val), col in zip(fid_vals.items(), cols_fid):
        with col:
            st.metric(label, f"{val:,.0f}")

    # --- Revenue & OPEX PV on the same timing bases (COD vs Today/FID) ---

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Revenue & OPEX Present Values (Start-of-Operations vs Today Basis)")

    # COD basis: you already compute PVs for streams starting at the beginning of operations
    rev_cod = pv_annuity_per_year(revenue_year_stream, n_full_years, npv_disc_rate)
    # annual_opex combines infra_opex_year + station_energy_year + transport_ops_year (already computed)
    opex_cod = -pv_annuity_per_year(annual_opex, n_full_years, npv_disc_rate)

    # FID/Today basis: discount those COD-basis PVs back the pre-build years (−2, −1)
    years_to_today = 2  # Year −2 and Year −1 before ops start
    shift_factor = (1.0 + npv_disc_rate) ** years_to_today
    rev_fid = rev_cod / shift_factor
    opex_fid = opex_cod / shift_factor

    # Explain what’s going on (light, italic note)
    st.caption(
        "_Streams that occur during operations (revenue and OPEX) are PV’d from the start of operations on the COD basis. "
        "For the Today (FID) basis they are then discounted a further two years back to the investment decision date._"
    )

    # Two columns per basis for quick comparison
    col_rev_cod, col_rev_fid, col_opex_cod, col_opex_fid = st.columns(4)
    with col_rev_cod:
        st.metric("Revenue PV (Start-of-Operations basis)", f"${rev_cod:,.0f}")
    with col_rev_fid:
        st.metric("Revenue PV (Today / FID basis)", f"${rev_fid:,.0f}")
    with col_opex_cod:
        st.metric("OPEX PV (Start-of-Operations basis)", f"${opex_cod:,.0f}")
    with col_opex_fid:
        st.metric("OPEX PV (Today / FID basis)", f"${opex_fid:,.0f}")
    # --- OPEX breakdown PVs: Stations vs Transport (COD vs Today/FID) ---

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### OPEX Breakdown PVs — Stations vs Transport (Start-of-Operations vs Today Basis)")

    # Annual streams (positive magnitudes for costs; we’ll show PVs as negative)
    mother_annual_opex = float(ms_opex_year + ms_energy_cost_year)  # facility + energy (Mother)
    daughter_annual_opex = float(ds_opex_year)  # facility only (Daughter)
    stations_annual_opex = mother_annual_opex + daughter_annual_opex  # total stations OPEX
    transport_annual_opex = float(transport_ops_year)  # maintenance + fuel + drivers + misc

    # PV @ start-of-operations (COD basis)
    stations_opex_pv_cod = -pv_annuity_per_year(stations_annual_opex, n_full_years, npv_disc_rate)
    transport_opex_pv_cod = -pv_annuity_per_year(transport_annual_opex, n_full_years, npv_disc_rate)

    # Shift to Today (FID) basis: discount an extra two pre-build years (−2, −1)
    years_to_today = 2
    shift_factor = (1.0 + npv_disc_rate) ** years_to_today
    stations_opex_pv_fid = stations_opex_pv_cod / shift_factor
    transport_opex_pv_fid = transport_opex_pv_cod / shift_factor

    # Render: 4 columns — stations/transport × COD/FID
    c_st_cod, c_st_fid, c_tr_cod, c_tr_fid = st.columns(4)
    with c_st_cod:
        st.metric("Stations OPEX PV (Start-of-Operations basis)", f"{stations_opex_pv_cod:,.0f}")
    with c_st_fid:
        st.metric("Stations OPEX PV (Today / FID basis)", f"{stations_opex_pv_fid:,.0f}")
    with c_tr_cod:
        st.metric("Transport OPEX PV (Start-of-Operations basis)", f"{transport_opex_pv_cod:,.0f}")
    with c_tr_fid:
        st.metric("Transport OPEX PV (Today / FID basis)", f"{transport_opex_pv_fid:,.0f}")

    st.caption(
        "_Stations OPEX = Mother facility + Mother energy + Daughter facility. "
        "Transport OPEX = trucking fuel, maintenance, drivers, insurance & misc._"
    )

    # --- OPEX breakdown PVs: Mother vs Daughter (COD vs Today/FID) ---

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("##### OPEX Breakdown PVs — Mother vs Daughter (Start-of-Operations vs Today Basis)")

    mother_opex_pv_cod = -pv_annuity_per_year(mother_annual_opex, n_full_years, npv_disc_rate)
    daughter_opex_pv_cod = -pv_annuity_per_year(daughter_annual_opex, n_full_years, npv_disc_rate)

    mother_opex_pv_fid = mother_opex_pv_cod / shift_factor
    daughter_opex_pv_fid = daughter_opex_pv_cod / shift_factor

    m_cod, m_fid, d_cod, d_fid = st.columns(4)
    with m_cod:
        st.metric("Mother OPEX PV (Start-of-Operations basis)", f"{mother_opex_pv_cod:,.0f}")
    with m_fid:
        st.metric("Mother OPEX PV (Today / FID basis)", f"{mother_opex_pv_fid:,.0f}")
    with d_cod:
        st.metric("Daughter OPEX PV (Start-of-Operations basis)", f"{daughter_opex_pv_cod:,.0f}")
    with d_fid:
        st.metric("Daughter OPEX PV (Today / FID basis)", f"{daughter_opex_pv_fid:,.0f}")

    st.caption(
        "_Mother OPEX = facility + compression energy. "
        "Daughter OPEX = unloading facility only (no energy line at present)._"
    )

    # --- Transport OPEX breakdown PVs (Fuel, Maintenance, Drivers, Insurance/Misc) ---

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "##### Transport OPEX Breakdown PVs — Fuel, Maintenance, Drivers, Insurance/Misc (Start-of-Operations vs Today Basis)")

    # Annual components (positive magnitudes; PVs displayed as negatives = costs)
    transport_components = [
        ("Fuel", float(fuel_cost_year)),
        ("Truck maintenance", float(trk_maint_year)),
        ("Trailer maintenance", float(trl_maint_year)),
        ("Drivers", float(driver_cost_year)),
        ("Insurance", float(insurance_year)),
        ("Miscellaneous", float(misc_year)),
    ]

    # Compute PVs for both timing bases
    comp_pvs = []
    for label, annual_val in transport_components:
        pv_cod = -pv_annuity_per_year(annual_val, n_full_years, npv_disc_rate)
        pv_fid = pv_cod / shift_factor  # shift back 2 pre-build years to Today/FID basis
        comp_pvs.append((label, pv_cod, pv_fid))

    # Render in two rows of 3 columns (label with COD vs FID)
    for row_start in range(0, len(comp_pvs), 3):
        cols = st.columns(3)
        for (label, pv_cod, pv_fid), col in zip(comp_pvs[row_start:row_start + 3], cols):
            with col:
                st.markdown(f"**{label}**")
                st.metric("Start-of-Operations basis", f"{pv_cod:,.0f}")
                st.metric("Today / FID basis", f"{pv_fid:,.0f}")

    st.caption("_Negative PV values indicate costs. Start-of-Operations basis discounts operating years from COD; "
               "Today / FID basis applies an additional two years of discounting to the same streams._")

    with st.expander("Cash flow (annual) — details", expanded=False):
        years_labels = ["Year −2", "Year −1"] + [f"Year {i + 1}" for i in range(n_full_years)]
        df_cf = pd.DataFrame({"Year": years_labels, "Cash flow [$]": cashflows})
        st.dataframe(df_cf, use_container_width=True)

# ---------------------------------------
# BENEFITS TAB — Gas sales & carbon avoidance (index 4)
# ---------------------------------------
with tabs[4]:
    st.markdown("## Benefits & Carbon")
    # Compute current levelized metrics for summary ribbon
    lc_per_gj = float(st.session_state.get("lc_per_gj", 0.0))
    lb_per_gj = float(st.session_state.get("lb_per_gj", 0.0))
    margin_per_gj = lb_per_gj - lc_per_gj
    st.session_state["margin_per_gj"] = margin_per_gj
    summary_ribbon("24px")

    st.caption(
        "Compute **Total Levelized Benefit [$/GJ]** and explore CO₂-e for venting vs flaring vs capture/sale."
    )

    eff_daily = float(st.session_state.get("effective_daily_gj", daily_energy_gj))
    annual_gj = eff_daily * DAYS_PER_YEAR

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Gas Sales / Value")
        gas_price_per_gj = st.number_input(
            "Revenue - sales gas [$/GJ delivered]",
            min_value=0.0,
            value=12.0,
            step=0.5,
            key="sales_gas_value_per_gj",
        )
        st.caption("This is the single source of truth; other tabs mirror this value.")

        other_benefit_year = st.number_input(
            r"Other operational benefits [\$/yr]",
            min_value=0.0,
            value=0.0,
            step=10_000.0,
            format="%.0f",
        )
        # ✅ keep Scenarios & finance consistent with this input
        st.session_state["other_benefit_year"] = float(other_benefit_year)

        gas_revenue_year = gas_price_per_gj * annual_gj

    with b2:
        st.subheader("Carbon Emissions & Avoidance")
        st.markdown("**Baseline handling scenario** (what would have happened without capture):")
        baseline = st.radio(
            "Baseline",
            ["Venting", "Flaring"],
            index=1,
            help="Select the counterfactual practice the project avoids.",
        )
        help_flyout(
            "benefits_baseline",
            "Choose the **counterfactual** the project avoids. If the baseline is *Venting*, avoided CO₂-e is higher than *Flaring*, which oxidises methane to CO₂.",
        )

        # --- Emission factors & carbon price (presets + advanced) ---
        gas_is_h2 = st.session_state.get("gas_type_selected", "Methane") == "Hydrogen"

        # --- Persistence (optional): load carbon settings from URL ---
        _qp = st.experimental_get_query_params()
        preset_from_url = _qp.get("carbon_preset", [""])[0] == "1"
        baseline_from_url = _qp.get("baseline", [None])[0]
        gwp_from_url = _qp.get("gwp", [None])[0]
        flare_from_url = _qp.get("flare_eff", [None])[0]
        vent_qp = _qp.get("vent_ef", [None])[0]
        flare_qp = _qp.get("flare_ef", [None])[0]
        price_qp = _qp.get("carbon_price", [None])[0]

        # Seed session defaults from URL (if present) BEFORE widgets render
        if baseline_from_url in ["Venting", "Flaring"]:
            st.session_state["baseline_default_from_url"] = baseline_from_url
        if gwp_from_url in ["AR4 (25)", "AR5 (28)"]:
            st.session_state["gwp_choice"] = gwp_from_url
        if flare_from_url:
            try: st.session_state["flare_eff_pct"] = int(flare_from_url)
            except: pass
        if vent_qp:
            try: st.session_state["vent_co2e_t_per_gj"] = float(vent_qp)
            except: pass
        if flare_qp:
            try: st.session_state["flare_co2_t_per_gj"] = float(flare_qp)
            except: pass
        if price_qp:
            try: st.session_state["carbon_price_per_t"] = float(price_qp)
            except: pass

        adv = st.expander("Advanced carbon settings", expanded=False)
        with adv:
            col_adv1, col_adv2, col_adv3 = st.columns([1, 1, 1])
            with col_adv1:
                gwp_choice = st.selectbox(
                    "Methane GWP (CO₂-e basis)",
                    ["AR4 (25)", "AR5 (28)"],
                    index=1,
                    key="gwp_choice",
                    help="Choose the CH₄ global warming potential basis used to interpret venting CO₂-e.",
                )
                gwp_val = 25.0 if gwp_choice.startswith("AR4") else 28.0
            with col_adv2:
                flare_eff_pct = st.slider(
                    "Flare destruction efficiency [%]",
                    min_value=85, max_value=99, value=st.session_state.get("flare_eff_pct", 98), step=1,
                    key="flare_eff_pct",
                    help="Fraction of methane oxidised to CO₂ during flaring.",
                )
            with col_adv3:
                use_presets = st.checkbox(
                    "Use recommended presets",
                    value=st.session_state.get("use_presets_key", not gas_is_h2),
                    key="use_presets_key",
                    help="Tick to auto-fill sensible defaults for methane (venting & flaring). You can still override below."
                )

            # Optional: remember settings in the URL so they persist between sessions/shares
            remember = st.checkbox(
                "Remember these carbon settings in the URL",
                value=preset_from_url,
                help="Saves baseline, GWP, flare efficiency, EF values and carbon price into the page URL."
            )
            if remember:
                try:
                    st.experimental_set_query_params(
                        carbon_preset="1",
                        baseline=str(st.session_state.get("baseline_default_from_url", baseline)),
                        gwp=str(st.session_state.get("gwp_choice", "AR5 (28)")),
                        flare_eff=str(st.session_state.get("flare_eff_pct", 98)),
                        vent_ef=str(st.session_state.get("vent_co2e_t_per_gj", 0.0)),
                        flare_ef=str(st.session_state.get("flare_co2_t_per_gj", 0.0)),
                        carbon_price=str(st.session_state.get("carbon_price_per_t", 35.0)),
                    )
                except Exception:
                    pass

        # Recommended methane presets (you can override in the inputs below)
        if gas_is_h2:
            vent_default = 0.0
            flare_default = 0.0
            carbon_default = 0.0
        else:
            # Venting CO₂-e factor [tCO₂-e/GJ] — methane: recommended ~1.89 tCO₂-e/GJ on AR5(28).
            # Scale for AR4(25) linearly to keep sensitivity intuitive.
            vent_default_ar5 = 1.89
            vent_default = vent_default_ar5 if gwp_val == 28.0 else vent_default_ar5 * (25.0 / 28.0)

            # Flaring CO₂ factor [tCO₂/GJ] — recommended ~0.056 tCO₂/GJ; adjust by selected flare efficiency
            base_flare = 0.056
            flare_default = base_flare * (flare_eff_pct / 100.0)

            # Carbon price default
            carbon_default = 35.0

        # Inputs with explicit keys (so the reset button can set them)
        c_row1, c_row2, c_row3, c_row_btn = st.columns([1.1, 1.1, 1.0, 0.9])
        with c_row1:
            vent_co2e_t_per_gj = st.number_input(
                "Venting CO₂-e factor [tCO₂-e/GJ]",
                min_value=0.0,
                value=float(vent_default if st.session_state.get("use_presets_key", True) else (0.0 if gas_is_h2 else 0.05)),
                step=0.001,
                format="%.3f",
                key="vent_co2e_t_per_gj",
                help="CO₂-e per GJ of gas if baseline is venting methane."
            )
        with c_row2:
            flare_co2_t_per_gj = st.number_input(
                "Flaring CO₂ factor [tCO₂/GJ]",
                min_value=0.0,
                value=float(flare_default if st.session_state.get("use_presets_key", True) else (0.0 if gas_is_h2 else 0.002)),
                step=0.001,
                format="%.3f",
                key="flare_co2_t_per_gj",
                help="Direct CO₂ per GJ of gas if baseline is flaring methane (post-combustion CO₂)."
            )
        with c_row3:
            carbon_price = st.number_input(
                r"Carbon value [\$/tCO₂-e]",
                min_value=0.0,
                value=float(carbon_default),
                step=1.0,
                key="carbon_price_per_t",
                help="Market or internal shadow price applied to avoided tCO₂-e."
            )

        # ---- One-click reset to recommended methane presets ----
        def _reset_carbon_presets():
            # Recompute current recommended defaults (respecting latest GWP & flare efficiency)
            if gas_is_h2:
                _vent = 0.0
                _flare = 0.0
                _cprice = 0.0
            else:
                _vent_ar5 = 1.89
                _vent = _vent_ar5 if gwp_val == 28.0 else _vent_ar5 * (25.0 / 28.0)
                _flare = 0.056 * (flare_eff_pct / 100.0)
                _cprice = 35.0
            st.session_state["vent_co2e_t_per_gj"] = float(_vent)
            st.session_state["flare_co2_t_per_gj"] = float(_flare)
            st.session_state["carbon_price_per_t"] = float(_cprice)
            st.session_state["use_presets_key"] = True

        with c_row_btn:
            st.button("Reset methane presets", on_click=_reset_carbon_presets, help="Apply recommended methane defaults (GWP & flare efficiency aware).")

        # Pull back values (ensures downstream uses the possibly-reset session values)
        vent_co2e_t_per_gj = float(st.session_state.get("vent_co2e_t_per_gj", vent_default))
        flare_co2_t_per_gj = float(st.session_state.get("flare_co2_t_per_gj", flare_default))
        carbon_price = float(st.session_state.get("carbon_price_per_t", carbon_default))

        # Annual baseline emissions (tCO2e/yr)
        baseline_tco2e_year = (vent_co2e_t_per_gj if baseline == "Venting" else flare_co2_t_per_gj) * annual_gj

        # Optional: include project-side emissions (e.g., diesel) to avoid over-crediting
        with st.expander("Project-side emissions (optional)"):
            include_transport = st.checkbox("Include transport diesel emissions", value=False)
            project_tco2e_year = 0.0
            if include_transport:
                cal = st.session_state.get("latest_calcs_cache", {})
                fleet_km_day = float(cal.get("truck_km_per_day", 0.0))
                km_per_l = float(st.session_state.get("truck_km_per_litre", 1.8))
                l_per_day = fleet_km_day / max(km_per_l, 1e-9)
                # Diesel EF ≈ 2.68 kg CO₂ / L  →  tCO₂ = kg/1000
                project_tco2e_year += (l_per_day * 2.68 / 1000.0) * DAYS_PER_YEAR
                st.caption(f"Included transport emissions: ~{project_tco2e_year:,.0f} tCO₂-e/yr")

        avoided_tco2e_year = max(baseline_tco2e_year - project_tco2e_year, 0.0)
        carbon_benefit_year = avoided_tco2e_year * carbon_price
        st.session_state["carbon_benefit_year"] = float(carbon_benefit_year)

        # Safeguard Mechanism status (threshold = 100,000 tCO₂-e/yr)
        safeguard_threshold = 100_000.0
        pct_of_threshold = (avoided_tco2e_year / safeguard_threshold) * 100.0 if safeguard_threshold else 0.0
        avoided_intensity_t_per_gj = avoided_tco2e_year / max(annual_gj, 1e-9)

        if avoided_tco2e_year >= safeguard_threshold:
            st.warning(
                f"Safeguard Mechanism: **Above threshold** — avoided ~{avoided_tco2e_year:,.0f} tCO₂-e/yr "
                f"({pct_of_threshold:,.1f}% of 100,000). Avoided intensity ≈ {avoided_intensity_t_per_gj:.3f} tCO₂-e/GJ."
            )
        else:
            st.info(
                f"Safeguard Mechanism: **OK** — avoided ~{avoided_tco2e_year:,.0f} tCO₂-e/yr "
                f"({pct_of_threshold:,.1f}% of 100,000). Avoided intensity ≈ {avoided_intensity_t_per_gj:.3f} tCO₂-e/GJ."
            )

        # --- Baseline vs Project emissions intensity comparison (display only) ---
        baseline_intensity = baseline_tco2e_year / max(annual_gj, 1e-9)
        project_intensity = project_tco2e_year / max(annual_gj, 1e-9)
        reduction_pct = (1.0 - (project_intensity / max(baseline_intensity, 1e-9))) * 100.0 if baseline_tco2e_year > 0 else 0.0

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            st.metric("Baseline intensity", f"{baseline_intensity:.3f} tCO₂-e/GJ")
        with col_b:
            delta_abs = project_intensity - baseline_intensity
            st.metric("Project intensity", f"{project_intensity:.3f} tCO₂-e/GJ", f"{delta_abs:+.3f} Δ")
        with col_c:
            st.metric("Reduction", f"{reduction_pct:,.1f}%")

    total_benefit_year = gas_revenue_year + carbon_benefit_year + other_benefit_year
    levelized_benefit_per_gj = (
        (total_benefit_year / max(annual_gj, 1e-9)) if annual_gj > 0 else float("nan")
    )
    st.session_state["lb_per_gj"] = float(levelized_benefit_per_gj)

    st.markdown("### Annual Benefit Summary [currency]")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total gas revenue [$/yr]", f"${gas_revenue_year:,.0f}")
    c2.metric("Carbon avoidance [$/yr]", f"${carbon_benefit_year:,.0f}")
    c3.metric("Other [$/yr]", f"${other_benefit_year:,.0f}")
    st.metric("Total annual benefit [$/yr]", f"${total_benefit_year:,.0f}")
    st.metric("Levelized Benefit [$/GJ]", f"${levelized_benefit_per_gj:,.2f}")
    st.caption("Includes gas value + carbon avoidance (+ other, if any), divided by annual GJ delivered.")
    help_flyout(
        "levelized_benefit",
        "**Levelized Benefit** = (Gas value + Carbon avoidance + Other)/Annual GJ delivered. Units **$/GJ**.",
    )

    with st.expander("Carbon calculations (details)"):
        st.write(f"Baseline: **{baseline}**")
        st.write(f"GWP basis: **{gwp_choice}**  •  Flare efficiency: **{(0 if gas_is_h2 else flare_eff_pct)}%**")
        st.write(f"Venting EF [tCO₂-e/GJ]: **{vent_co2e_t_per_gj:.3f}**")
        st.write(f"Flaring EF [tCO₂/GJ]: **{flare_co2_t_per_gj:.3f}**")
        st.write(f"Baseline CO₂-e (t/yr): **{baseline_tco2e_year:,.0f}**")
        st.write(f"Project CO₂-e (t/yr): **{project_tco2e_year:,.0f}**")
        st.write(f"Avoided CO₂-e (t/yr): **{avoided_tco2e_year:,.0f}**")
        st.write(f"Carbon value [$/tCO₂-e]: **${carbon_price:,.0f}**")
        st.write(f"Carbon avoidance [$/yr]: **${carbon_benefit_year:,.0f}**")
        st.caption("Notes: Avoided CO₂-e is the difference between baseline and project-side emissions. "
                   "If transport emissions are included, they reduce the avoided total before pricing.")

# --- Make per-GJ benefit components available to Scenarios & finance_helpers ---
annual_gj = float(st.session_state.get("effective_daily_gj", daily_energy_gj)) * DAYS_PER_YEAR
gas_value_per_gj = float(st.session_state.get("sales_gas_value_per_gj", 0.0))
other_benefit_year_local = float(st.session_state.get("other_benefit_year", other_benefit_year))

st.session_state["benefit_gas_per_gj"] = gas_value_per_gj
st.session_state["carbon_benefit_per_gj"] = float(carbon_benefit_year / max(annual_gj, 1e-9))
st.session_state["other_benefit_per_gj"] = float(other_benefit_year_local / max(annual_gj, 1e-9))

# --- Economic context display (for reviewers/auditors) ---
carbon_value_per_gj = float(st.session_state["carbon_benefit_per_gj"])
total_benefit_per_gj = carbon_value_per_gj + gas_value_per_gj + float(st.session_state["other_benefit_per_gj"])
st.info(
    f"**Economic context** — Carbon: ${carbon_value_per_gj:,.2f}/GJ, "
    f"Gas: ${gas_value_per_gj:,.2f}/GJ "
    f"(Total benefit: ${total_benefit_per_gj:,.2f}/GJ). "
    "These values feed Levelized Benefit and Margin."
)

# ---------------------------------------
# Scenarios tab (index 5)
# ---------------------------------------
with tabs[5]:
    st.markdown("## Scenarios (save/load)")
    summary_ribbon("24px")

    scen_name = st.text_input("Scenario name", value="MyScenario")

    a_cap_for_planner, b_cap_for_planner = _get_caps_for_planner()
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
        "driver_shift_cap": _current_driver_shift_cap(),
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

    # -------------------------------
    # Scenario Analysis (Independent variable sweeps)
    # -------------------------------
    st.divider()
    st.subheader("Scenario analysis")

    left_col, right_col = st.columns([1, 1])

    # --- Left: Independent variable (single-choice) ---
    with left_col:
        st.markdown("#### Independent variable")

        indep_options = [
            "Daily gas to transport [GJ/day]",
            "Distance Mother→Daughter [km]",
            "Per-bay fill rate [GJ/h]",
            "Per-bay unload rate [GJ/h]",
            # add more here later...
        ]

        _default_idx = indep_options.index(
            "Distance Mother→Daughter [km]") if "Distance Mother→Daughter [km]" in indep_options else 0

        indep_var = st.radio(
            "Choose one",
            indep_options,
            index=_default_idx,
            key="scen_indep_var",
            horizontal=False,
        )

    # --- Right: Outputs to calculate (multi-select) ---
    # Master list of outputs the Results table (and plot) can compute
    all_outputs = [
        "Levelized Cost [$/GJ]",
        "Levelized Benefit [$/GJ]",
        "Margin [$/GJ]",
        "Trucking Cost (excl. stations) [$/GJ]",
        "Active trucks",
        "A-triples in circulation",
        "Drivers per truck (on duty)",
        "Drivers on duty / day",
    ]

    with right_col:
        # Header + control live in the SAME column so they stay together
        st.markdown("#### Select Outputs", help="Choose which columns to compute for each row.")
        st.markdown("""
        <style>
          /* Keep the heading and the multiselect tight and aligned */
          div[data-testid="stVerticalBlock"] > div:has(.stMultiSelect) { margin-top: -0.5rem; }
          .stMultiSelect [data-baseweb="tag"] {
            background-color: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--primary-color) !important;
          }
          .stMultiSelect [data-baseweb="tag"] svg {
            color: var(--text-color) !important;
            fill: var(--text-color) !important;
          }
        </style>
        """, unsafe_allow_html=True)

        selected_outputs = st.multiselect(
            "Select outputs",
            options=all_outputs,
            default=[
                "Levelized Cost [$/GJ]",
                "Levelized Benefit [$/GJ]",
                "Margin [$/GJ]",
                "Trucking Cost (excl. stations) [$/GJ]",
            ],
            key="scen_selected_outputs"
        )

    # Track current vs previous selection (we'll compare later, after helpers are defined)
    st.session_state.setdefault("scen_prev_selected_outputs", list(selected_outputs))
    st.session_state["scen_cur_selected_outputs"] = list(selected_outputs)

    # ---- Storage for scenario rows (per session) ----
    if "scenario_rows" not in st.session_state:
        st.session_state["scenario_rows"] = []  # list[dict]
    if "scenario_cols" not in st.session_state:
        # left-most column is always the independent variable label + value
        st.session_state["scenario_cols"] = ["Independent variable", "Value"] + list(all_outputs)


    # ---- Helper: safe getter for session keys with fallbacks ----
    def _get_any(keys, default=None):
        for k in keys:
            if k in st.session_state:
                return st.session_state.get(k)
        return default


    # Compute current pills (used when we can’t recompute fully)
    def _current_pills_snapshot():
        lc = _get_any(["lc_per_gj"], None)
        lb = _get_any(["lb_per_gj"], None)
        margin = None
        if lc is not None and lb is not None:
            try:
                margin = float(lb) - float(lc)
            except Exception:
                margin = None
        transport_lc = _get_any(["transport_lc_per_gj"], None)
        return lc, lb, margin, transport_lc

    # Attempt to compute operational outputs from known session keys
    # (duplicate removed — use the global _current_driver_shift_cap())

    def _operational_snapshot():

        """
        Pull operational counts from the most recent planner results.
        We store the whole dict in session_state["latest_calcs_cache"].
        """
        calcs = st.session_state.get("latest_calcs_cache") or {}

        active_trucks = calcs.get("active_trucks")
        a_triples = calcs.get("sets_in_circulation")  # road-train sets in circulation
        drv_per_truck = calcs.get("drivers_per_truck")
        drivers_on_duty = calcs.get("drivers_on_duty")

        # Fallbacks if the cache is missing (older sessions)
        if active_trucks is None:
            active_trucks = _get_any(["active_trucks", "num_active_trucks", "fleet_active_trucks"], None)
        if a_triples is None:
            a_triples = _get_any(["roadtrains_active", "a_triples_in_circulation", "num_roadtrains"], None)
        if drv_per_truck is None:
            drv_per_truck = _get_any(["drivers_per_truck_on_duty", "drivers_per_truck", "drivers_per_truck_target"],
                                     None)
        if drivers_on_duty is None:
            drivers_on_duty = _get_any(["drivers_on_duty_per_day", "drivers_on_duty", "total_drivers_on_duty"], None)

        # If drivers_on_duty is still None but we have trucks & drivers/truck, derive it
        if drivers_on_duty is None and active_trucks is not None and drv_per_truck is not None:
            try:
                drivers_on_duty = float(active_trucks) * float(drv_per_truck)
            except Exception:
                pass

        return active_trucks, a_triples, drv_per_truck, drivers_on_duty

    # ---- Independent variable value input & Add-row action ----
    st.markdown("#### Add values and build a comparison table")

    # Default values seeded from current session where possible
    _default_daily_gj = float(_get_any(["effective_daily_gj", "daily_energy_gj"], 10000.0))
    _default_dist_km = float(
        _get_any(["distance_km_oneway", "distance_km", "md_distance_km", "mother_daughter_distance_km"], 50.0))

    # Show a value input that matches the chosen independent variable
    if indep_var == "Daily gas to transport [GJ/day]":
        indep_value = st.number_input(
            "Enter daily gas [GJ/day] for this row",
            min_value=0.0,
            value=float(_default_daily_gj),
            step=100.0,
            key="scen_indep_value",
            help="This does not change other tabs; it's only used to calculate this row.",
        )
    elif indep_var == "Distance Mother→Daughter [km]":
        indep_value = st.number_input(
            "Enter distance [km] for this row",
            min_value=0.0,
            value=float(_default_dist_km),
            step=5.0,
            key="scen_indep_value",
            help="This does not change other tabs; it's only used to calculate this row.",
        )
    else:
        # Sensible defaults for rate-based IVs
        default_val = 0.0
        if indep_var == "Per-bay fill rate [GJ/h]":
            default_val = float(st.session_state.get("fill_rate_gjph", 41.67))
        elif indep_var == "Per-bay unload rate [GJ/h]":
            default_val = float(st.session_state.get("unload_rate_gjph", 41.67))

        indep_value = st.number_input(
            "Enter value for selected independent variable",
            min_value=0.0,
            value=default_val,
            step=1.0,
            key="scen_indep_value",
        )


    # ---- Calculate outputs for the new row (non-destructive to app state) ----
    def _calc_outputs_for_row(indep_name: str, indep_val: float, outputs: list[str]) -> dict:
        """
        Calculator for scenario rows using the SAME math as the Costs tab.
        - Uses finance_helpers.compute_levelized_metrics_for_scenario(...) for LC/LB/Margin/Trucking LC.
        - Leaves operational counts (trucks, drivers, etc.) as snapshots (non-destructive).
        """
        row = {"Independent variable": indep_name, "Value": indep_val}
        # --- Per-row overrides (define once and reuse for finance + ops) ---
        # Fill-rate override
        if indep_name == "Per-bay fill rate [GJ/h]":
            fill_rate_gjph_row = float(indep_val)
        else:
            fill_rate_gjph_row = float(st.session_state.get("fill_rate_gjph", 41.67))

        # Unload-time override (convert rate -> time per set + changeover)
        if indep_name == "Per-bay unload rate [GJ/h]":
            comb_key = st.session_state.get("combo", "A-triple")
            a_count = ROADTRAIN_MAP[comb_key]["A"]
            b_count = ROADTRAIN_MAP[comb_key]["B"]
            a_gj = float(st.session_state.get("a_cap_usable_gj", st.session_state.get("a_cap_calc_gj", 0.0)))
            b_gj = float(st.session_state.get("b_cap_usable_gj", st.session_state.get("b_cap_calc_gj", 0.0)))
            per_set_capacity_gj = a_count * a_gj + b_count * b_gj
            rate_gjph = float(indep_val)
            eff_rate = rate_gjph * max(1, int(st.session_state.get("unload_concurrent_bays", 1)))
            unload_time_h_row = per_set_capacity_gj / max(eff_rate, 1e-9) + float(
                st.session_state.get("changeover_overhead_h_daughter", 0.25)
            )
        else:
            unload_time_h_row = float(st.session_state.get("unload_time_h", 12.0))

        # 1) Levelized metrics (match Costs tab exactly)
        try:
            if indep_name == "Distance Mother→Daughter [km]":
                m = compute_levelized_metrics_for_scenario(distance_km_override=float(indep_val))

            elif indep_name == "Daily gas to transport [GJ/day]":
                m = compute_levelized_metrics_for_scenario(daily_energy_gj_override=float(indep_val))

            elif indep_name == "Per-bay fill rate [GJ/h]":
                try:
                    m = compute_levelized_metrics_for_scenario(
                        fill_rate_gjph_override=float(fill_rate_gjph_row)
                    )
                except TypeError:
                    m = compute_levelized_metrics_for_scenario()

            elif indep_name == "Per-bay unload rate [GJ/h]":
                try:
                    m = compute_levelized_metrics_for_scenario(
                        unload_time_h_override=float(unload_time_h_row)
                    )
                except TypeError:
                    m = compute_levelized_metrics_for_scenario()


            else:
                # Default: no override
                m = compute_levelized_metrics_for_scenario()

        except Exception:
            # Fallback to current pills if anything transient fails
            lc, lb, margin, transport_lc = _current_pills_snapshot()
            m = {
                "lc_per_gj": None if lc is None else float(lc),
                "lb_per_gj": None if lb is None else float(lb),
                "margin_per_gj": None if margin is None else float(margin),
                "transport_lc_per_gj": None if transport_lc is None else float(transport_lc),
            }

        # 2) Operational metrics for THIS row (recompute with the same overrides used for LC/LB)
        try:
            # Start from current UI values, then override the chosen independent variable
            daily_gj_plan = float(st.session_state.get("daily_energy_gj", 0.0))
            distance_km = float(st.session_state.get("distance_km_oneway", 900.0))
            if indep_name == "Distance Mother→Daughter [km]":
                distance_km = float(indep_val)
            elif indep_name == "Daily gas to transport [GJ/day]":
                daily_gj_plan = float(indep_val)
            # fill_rate_gjph_row and unload_time_h_row are precomputed above for this row

            # Capacities and combination
            a_cap_gj, b_cap_gj = _get_caps_for_planner()
            combo = st.session_state.get("combo", "A-triple")
            a_count = ROADTRAIN_MAP[combo]["A"]
            b_count = ROADTRAIN_MAP[combo]["B"]

            cal = compute_calcs(
                daily_energy_gj=daily_gj_plan,
                distance_km_oneway=distance_km,
                speed_kmh=float(st.session_state.get("speed_kmh", 75.0)),
                hours_per_day=float(st.session_state.get("hours_per_day", 24.0)),
                a_cap_gj=a_cap_gj,
                b_cap_gj=b_cap_gj,
                a_count=a_count,
                b_count=b_count,
                fill_rate_gjph=fill_rate_gjph_row,
                concurrent_bays_per_set=int(st.session_state.get("concurrent_bays_per_set", 3)),
                unload_time_h=unload_time_h_row,
                available_bays_A=int(st.session_state.get("available_bays_A", 12)),
                available_unload_bays_B=int(st.session_state.get("available_unload_bays_B", 4)),
                changeover_overhead_h=float(st.session_state.get("changeover_overhead_h", 0.25)),
                mode=st.session_state.get("mode", "DropAndPull"),
                driver_shift_cap=_current_driver_shift_cap(),
                crew_change_overhead_h=float(st.session_state.get("crew_change_overhead_h", 0.25)),
                delay_buffer_min=float(st.session_state.get("delay_buffer_min", 30.0)),
                truck_util_target=float(st.session_state.get("truck_util_target", 0.85)),
                fatigue_regime=st.session_state.get("fatigue_regime", "Standard Hours (Solo Drivers)"),
                relay_location=st.session_state.get("relay_location", "No relay"),

                # NEW availability params
                route_hours_per_day=float(st.session_state.get("route_hours_per_day", 24.0)),
                route_days_per_year=float(st.session_state.get("route_days_per_year", 365.0)),
                ms_hours_per_day=float(st.session_state.get("ms_hours_per_day", 24.0)),
                ms_days_per_year=float(st.session_state.get("ms_days_per_year", 365.0)),
                ds_hours_per_day=float(st.session_state.get("ds_hours_per_day", 24.0)),
                ds_days_per_year=float(st.session_state.get("ds_days_per_year", 365.0)),
            )

            # ✅ Persist this row’s results for the Warnings section (instead of the last Planner run)
            st.session_state["latest_row_calcs"] = cal

            op_trucks = cal.get("active_trucks")
            op_atriples = cal.get("sets_in_circulation")  # A-triples/sets in circulation for THIS row
            op_drv_per_truck = cal.get("drivers_per_truck")
            op_drivers_day = cal.get("drivers_on_duty")
        except Exception:
            # Fallback to the latest snapshot if recompute fails for any reason
            op_trucks, op_atriples, op_drv_per_truck, op_drivers_day = _operational_snapshot()

        # 3) Write requested outputs
        for out in outputs:
            val = "N/A"
            try:
                if out == "Levelized Cost [$/GJ]":
                    val = None if m.get("lc_per_gj") is None else float(m["lc_per_gj"])
                elif out == "Levelized Benefit [$/GJ]":
                    val = None if m.get("lb_per_gj") is None else float(m["lb_per_gj"])
                elif out == "Margin [$/GJ]":
                    val = None if m.get("margin_per_gj") is None else float(m["margin_per_gj"])
                elif out == "Trucking Cost (excl. stations) [$/GJ]":
                    val = None if m.get("transport_lc_per_gj") is None else float(m["transport_lc_per_gj"])
                elif out == "Active trucks":
                    val = None if op_trucks is None else float(op_trucks)
                elif out == "A-triples in circulation":
                    val = None if op_atriples is None else float(op_atriples)
                elif out == "Drivers per truck (on duty)":
                    val = None if op_drv_per_truck is None else float(op_drv_per_truck)
                elif out == "Drivers on duty / day":
                    val = None if op_drivers_day is None else float(op_drivers_day)
            except Exception:
                pass

            # Pretty formatting per cell
            if isinstance(val, float):
                if "[$/GJ]" in out:
                    row[out] = f"${val:,.2f}/GJ"
                else:
                    row[out] = f"{val:,.2f}" if not float(val).is_integer() else f"{int(val)}"
            else:
                row[out] = val

        # <-- return AFTER the loop so all selected outputs are filled
        return row

    # --- Auto/Manual refresh of existing rows when outputs selection changes ---
    def _parse_indep_val(v):
        s = str(v).replace("$", "").replace("/GJ", "").replace(",", "").strip()
        try:
            return float(s)
        except Exception:
            return 0.0


    def _recompute_all_rows_for_current_selection():
        cur_sel = list(st.session_state.get("scen_cur_selected_outputs", []))
        rows = st.session_state.get("scenario_rows", [])
        new_rows = []
        for r in rows:
            indep = r.get("Independent variable", "Distance Mother→Daughter [km]")
            val = _parse_indep_val(r.get("Value", 0))
            new_rows.append(_calc_outputs_for_row(indep, float(val), cur_sel))
        st.session_state["scenario_rows"] = new_rows


    # Auto-update when selection changes
    _prev = st.session_state.get("scen_prev_selected_outputs")
    _cur = st.session_state.get("scen_cur_selected_outputs")
    if _prev is not None and _cur is not None and _prev != _cur:
        # Snapshot plot widget state so the rerender after recompute preserves choices
        _prev_y1 = st.session_state.get("scen_plot_y1")
        _prev_y2 = st.session_state.get("scen_plot_y2")
        _prev_y2_on = st.session_state.get("scen_plot_enable_y2", False)

        _recompute_all_rows_for_current_selection()
        st.session_state["scen_prev_selected_outputs"] = list(_cur)

        # Restore plot widget state
        if _prev_y1 is not None:
            st.session_state["scen_plot_y1"] = _prev_y1
        if _prev_y2 is not None:
            st.session_state["scen_plot_y2"] = _prev_y2
        st.session_state["scen_plot_enable_y2"] = bool(_prev_y2_on)

    # (no-op placeholder removed)

    # --- Add row button ---
    # Force a blue primary button for this page (robust override vs theme)
    # NOTE (styling): Streamlit sometimes overrides primary button colors based on theme.
    # To keep our buttons consistently BLUE across versions/themes, we set a brute-force CSS
    # override on ALL .stButton > button elements. We avoid type="primary" so the theme
    # doesn’t repaints them red. If Streamlit changes DOM structure in a future release,
    # update the selector below (start from ".stButton > button" and inspect with dev tools).
    st.markdown("""
    <style>
    .stButton > button {
      background-color: #0a66c2 !important;   /* LinkedIn blue */
      border-color: #0a66c2 !important;
      color: #ffffff !important;
      background-image: none !important;
      box-shadow: none !important;
      border-radius: 6px !important;
      font-weight: 600 !important;
      min-width: 9rem !important;
    }
    .stButton > button:hover { filter: brightness(0.92); }
    </style>
    """, unsafe_allow_html=True)

    # Three buttons on one row: Add row | Clear table | Update table
    add_cols = st.columns([2, 2, 2])

    with add_cols[0]:
        if st.button("➕ Add row", key="scen_add_row",
                     help="Add this value to the results table"):
            if len(selected_outputs) == 0:
                st.warning("Select at least one output to calculate.")
            else:
                new_row = _calc_outputs_for_row(indep_var, float(indep_value), selected_outputs)
                st.session_state["scenario_rows"].append(new_row)

    with add_cols[1]:
        if st.button("✎ Clear table", key="scen_clear_rows",
                     help="Remove all rows from the results table"):
            st.session_state["scenario_rows"] = []

    with add_cols[2]:
        if st.button("↻ Update table outputs", key="scen_refresh_rows",
                     help="Recompute all rows for the current Selected outputs"):
            # Snapshot current plot widget state so a rerender won't reset them
            _prev_y1 = st.session_state.get("scen_plot_y1")
            _prev_y2 = st.session_state.get("scen_plot_y2")
            _prev_y2_on = st.session_state.get("scen_plot_enable_y2", False)

            _recompute_all_rows_for_current_selection()

            # Restore plot widget state (don’t force choices to be valid here; UI code already guards)
            if _prev_y1 is not None:
                st.session_state["scen_plot_y1"] = _prev_y1
            if _prev_y2 is not None:
                st.session_state["scen_plot_y2"] = _prev_y2
            st.session_state["scen_plot_enable_y2"] = bool(_prev_y2_on)

    # --- Results table ---
    st.markdown("#### Results")
    if len(st.session_state["scenario_rows"]) == 0:
        st.info("No rows yet. Choose an independent variable, enter a value, and click **Add row**.")
    else:
        # Build a consistent column order: Independent variable, Value, then selected outputs (only those chosen now)
        cols = ["Independent variable", "Value"] + selected_outputs

        # Normalize rows to the current selection
        normalized_rows = []
        for r in st.session_state["scenario_rows"]:
            normalized = {c: r.get(c, "N/A") for c in cols}
            normalized_rows.append(normalized)

        # Put the delete control INSIDE the table, left-most
        import pandas as pd

        df = pd.DataFrame(normalized_rows)
        df.insert(0, "🗑️", False)  # left-most checkbox column

        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "🗑️": st.column_config.CheckboxColumn("Delete", help="Tick to delete this row"),
            },
            num_rows="fixed",
        )

        # Apply deletions (remove rows where 🗑️ == True)
        if st.button("Apply deletions", key="scen_apply_deletions"):
            keep_mask = ~edited["🗑️"].fillna(False)
            kept = edited.loc[keep_mask, cols].to_dict(orient="records")
            st.session_state["scenario_rows"] = kept
            st.rerun()
        # ---- Warnings section (Planner-style) ----
        st.markdown("#### ⚠️ Warnings")

        # Use last computed scenario or planner results for consistency
        calcs_warn = st.session_state.get("latest_row_calcs", st.session_state.get("latest_calcs_cache", {}))
        fill_flag = calcs_warn.get("bay_flag_A", "N/A")
        unload_flag = calcs_warn.get("unload_flag_B", "N/A")

        fill_icon = ":green_circle:" if fill_flag == "OK" else ":red_circle:"
        unload_icon = ":green_circle:" if unload_flag == "OK" else ":red_circle:"

        st.write(f"**Fill bay status:** {fill_icon} {fill_flag}")
        st.write(f"**Unload bay status:** {unload_icon} {unload_flag}")

        # ---- Plot section (from Results table) ----
        st.markdown("#### Plot (from Results)")
        if len(st.session_state.get("scenario_rows", [])) == 0:
            st.info("No rows to plot. Add some rows to the **Results** table first.")
        else:
            # Rebuild a normalized DataFrame from the stored rows
            import pandas as pd

            rows = st.session_state["scenario_rows"]
            # Build a deterministic column order:
            # 1) meta columns, 2) current Selected outputs (in the same order as the pills),
            # 3) any extra columns that might exist in stored rows (append, stable order)
            pref_order = ["Independent variable", "Value"]
            selected_list = list(selected_outputs)

            # Collect columns from the stored rows while keeping encounter order
            seen = set()
            encounter = []
            for r in rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        encounter.append(k)

            extras = [c for c in encounter if c not in (pref_order + selected_list)]
            cols_plot = pref_order + selected_list + extras

            df_plot = pd.DataFrame([{c: r.get(c, "N/A") for c in cols_plot} for r in rows])


            # Helper: parse numeric cells like "$12.34/GJ", "1,234.56", or "N/A" -> float or NaN
            def _to_float(val):
                if val is None:
                    return float("nan")
                s = str(val).strip()
                if s in ("", "N/A", "NA", "nan", "None"):
                    return float("nan")
                s = s.replace("$", "").replace("/GJ", "").replace(",", "").strip()
                try:
                    return float(s)
                except Exception:
                    return float("nan")

            # Choose Y candidates (exclude the first two meta columns)
            y_candidates = [c for c in cols_plot if c not in ("Independent variable", "Value")]
            if not y_candidates:
                st.info("No plottable columns found yet — add outputs to the Results table.")
            else:
                # Stable index helper
                def _safe_index(options, name, fallback=0):
                    try:
                        return options.index(name)
                    except Exception:
                        return fallback

                # Y1 (always present)
                y1_default = st.session_state.get("scen_plot_y1", "Levelized Cost [$/GJ]")
                y1_name = st.selectbox(
                    "Y1 (left axis)",
                    y_candidates,
                    index=_safe_index(y_candidates, y1_default),
                    key="scen_plot_y1",
                )

                # Y2 controls: ALWAYS render both the checkbox and the selectbox.
                enable_y2 = st.checkbox(
                    "Add Y2 (right axis)",
                    value=st.session_state.get("scen_plot_enable_y2", False),
                    key="scen_plot_enable_y2",
                )

                # Options for Y2 must exclude Y1
                y2_all_options = [c for c in y_candidates if c != y1_name]
                y2_default = st.session_state.get("scen_plot_y2", "Levelized Benefit [$/GJ]")

                # If we have at least two y-candidates overall, keep the Y2 select enabled.
                y2_disabled = (not enable_y2) or (len(y_candidates) <= 1)

                # If a previous Y2 exists and is still valid (and != Y1), prefer it.
                y2_prev = st.session_state.get("scen_plot_y2")
                if (not y2_disabled) and y2_prev and (y2_prev not in y2_all_options) and (y2_prev in y_candidates) and (
                        y2_prev != y1_name):
                    y2_all_options = [y2_prev] + [c for c in y2_all_options if c != y2_prev]

                # Preserve previous Y2 selection even if temporarily missing
                y2_prev = st.session_state.get("scen_plot_y2")
                if y2_prev and y2_prev not in y2_all_options and not y2_disabled:
                    y2_all_options = [y2_prev] + [c for c in y2_all_options if c != y2_prev]

                y2_name = st.selectbox(
                    "Y2 (right axis)",
                    y2_all_options if y2_all_options else ["(no other columns)"],
                    index=_safe_index(y2_all_options, y2_default) if y2_all_options else 0,
                    key="scen_plot_y2",
                    disabled=y2_disabled,
                    help=None if not y2_disabled else "Enable Y2 and select another column (Y2 must differ from Y1).",
                )
                if y2_disabled or y2_name == "(no other columns)":
                    y2_name = None

                # Build numeric series
                x_vals = df_plot["Value"].map(_to_float).to_numpy()
                y1_vals = df_plot[y1_name].map(_to_float).to_numpy()
                y2_vals = df_plot[y2_name].map(_to_float).to_numpy() if y2_name else None

                # Sort by X for prettier lines
                import numpy as np

                order = np.argsort(x_vals)
                x_vals = x_vals[order]
                y1_vals = y1_vals[order]
                if y2_vals is not None:
                    y2_vals = y2_vals[order]

                if not MPL_OK:
                    st.warning(
                        "Matplotlib is not installed. Add `matplotlib>=3.9.0` to requirements.txt and reinstall."
                    )
                else:
                    fig, ax1 = plt.subplots(figsize=(8, 4.5))
                    ln1 = ax1.plot(x_vals, y1_vals, marker="o", markersize=8, linewidth=2)[0]
                    ax1.set_xlabel(df_plot["Independent variable"].iloc[0] if not df_plot.empty else "X")
                    ax1.set_ylabel(y1_name)
                    ax1.grid(True, alpha=0.25)

                    lines = [ln1]
                    labels = [y1_name]

                    # Y2 if enabled
                    if y2_vals is not None:
                        ax2 = ax1.twinx()
                        ln2 = ax2.plot(x_vals, y2_vals, marker="s", markersize=8, linestyle="--", linewidth=2)[0]
                        ax2.set_ylabel(y2_name)
                        lines.append(ln2)
                        labels.append(y2_name)

                    ax1.set_title("Scenario Results — X/Y Plot")
                    ax1.legend(lines, labels, loc="best")
                    st.pyplot(fig)

    # (Optional helper) keep this inside the Scenarios tab with the Save button
    with c_right:
        if st.button("Load scenarios.csv"):
            try:
                df = pd.read_csv("scenarios.csv")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load scenarios.csv: {e}")

    # Footer
    st.caption("v15_8 — Input diagnostics table + traceability to ribbon pills.")