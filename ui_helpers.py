# ui_helpers.py — helpers and summary ribbon (with trucking tooltip)
import math
from textwrap import dedent
import streamlit as st

# -------------------------
# Simple numeric inputs
# -------------------------
def number_commas(label, key=None, value=0.0, decimals=0, help=None):
    """Numeric input with fixed decimals (Streamlit doesn't support comma formatting in the box).
    Returns the numeric value.
    """
    fmt = f"%.{decimals}f"
    step = 10 ** (-decimals) if decimals > 0 else 1.0
    return st.number_input(label, key=key, value=float(value), step=step, format=fmt, help=help)


def currency_commas(label, key=None, value=0.0, decimals=0, help=None):
    """Currency-style input; currently just a number_input with fixed decimals."""
    return number_commas(label, key=key, value=value, decimals=decimals, help=help)


# -------------------------
# CSS injectors (lightweight)
# -------------------------
def _inject_sidebar_expander_css():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] details > summary {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_planner_expander_css():
    st.markdown(
        """
        <style>
        details > summary {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Small help caption
# -------------------------
def help_flyout(key, text):
    st.caption(text)


# -------------------------
# Driver utilisation section (placeholder renderer)
# -------------------------
def render_driver_utilisation_section():
    st.caption("Driver utilisation breakdown is shown here in the full version.")


# -------------------------
# Summary ribbon
# -------------------------
def _pill(label, value, unit="", bg="#eef2ff", fg="#1e3a8a", tip=None):
    """Render a colored 'pill' with label and value.
    Accepts 3 to 6 args: (label, value, [unit], [bg], [fg], [tip]).
    If 'tip' is provided, it is added as a tooltip (title attribute) and a small ⓘ marker.
    """
    try:
        v = float(value)
        text = f"{label}: ${v:,.2f}{unit}"
    except Exception:
        text = f"{label}: n/a"

    # Optional tooltip
    title_attr = f' title="{tip}"' if tip else ""
    if tip:
        text = f"{text} <sup>ⓘ</sup>"

    return (
        f"<span class='vp-pill'{title_attr} "
        f"style='display:inline-block; padding:4px 10px; margin:0 6px 6px 0; "
        f"border-radius:12px; background:{bg}; color:{fg}; font-weight:600;'>"
        f"{text}</span>"
    )


def summary_ribbon(font_size="22px"):
    """Top-of-tab ribbon with Levelized Cost/Benefit/Margin and Trucking Cost (excl. stations).

    On the very first app render (session flag 'app_boot' == True), show zeros in all pills
    to avoid jarring, mismatched initial values across tabs. After the first rerun (any input change),
    the flag is flipped to False in cng_planner_app.py and real values are shown.
    """
    # Are we on the first render?
    first_run = bool(st.session_state.get("app_boot", True))

    if first_run:
        lc = 0.0
        lb = 0.0
        margin = 0.0
        transport_lc = 0.0
    else:
        lc = float(st.session_state.get("lc_per_gj", 0.0))
        lb = float(st.session_state.get("lb_per_gj", 0.0))
        margin = lb - lc
        # persist for other tabs when not on first run
        st.session_state["margin_per_gj"] = margin
        transport_lc = float(st.session_state.get("transport_lc_per_gj", 0.0))

    # Colors
    cost_bg, cost_fg = "#eef2ff", "#1e3a8a"
    ben_bg, ben_fg = "#ecfdf5", "#065f46"
    if margin >= 0:
        mar_bg, mar_fg = "#ecfdf5", "#065f46"
    else:
        mar_bg, mar_fg = "#fee2e2", "#991b1b"
    trk_bg, trk_fg = "#fff7ed", "#9a3412"  # orange

    trucking_tip = (
        "Transport-only cost per GJ delivered, including fleet CAPEX (annualized), "
        "fuel, maintenance, drivers, registration, insurance, and miscellaneous OPEX. "
        "Excludes Mother and Daughter station CAPEX/OPEX."
    )

    html = f"""
    <div class='vp-ribbon' style='font-size:{font_size}; line-height:1; margin:8px 0 16px 0;'>
      <div class='vp-row1' style='display:flex; flex-wrap:wrap;'>
        {_pill("Levelized Cost", lc, "/GJ", cost_bg, cost_fg)}
        {_pill("Levelized Benefit", lb, "/GJ", ben_bg, ben_fg)}
        {_pill("Margin", margin, "/GJ", mar_bg, mar_fg)}
      </div>
      <div class='vp-row2' style='display:flex; flex-wrap:wrap; margin-top:6px;'>
        {_pill("Trucking Cost (excl. stations)", transport_lc, "/GJ", trk_bg, trk_fg, tip=trucking_tip)}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

