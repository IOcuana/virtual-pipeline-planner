# ui_helpers.py — helpers and summary ribbon
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
def _pill(label, value, unit="", bg="#eef2ff", fg="#1e3a8a"):
    """Render a colored 'pill' with label and value.
    Accepts 3 to 5 args: (label, value, [unit], [bg], [fg]).
    """
    try:
        v = float(value)
        text = f"{label}: ${v:,.2f}{unit}"
    except Exception:
        text = f"{label}: n/a"
    return (
        f"<span class='vp-pill' "
        f"style='display:inline-block; padding:4px 10px; margin:0 6px 6px 0; "
        f"border-radius:12px; background:{bg}; color:{fg}; font-weight:600;'>"
        f"{text}</span>"
    )


def summary_ribbon(font_size="22px"):
    """Top-of-tab ribbon with Levelized Cost/Benefit/Margin and Trucking Cost (excl. stations)."""
    lc = float(st.session_state.get("lc_per_gj", 0.0))
    lb = float(st.session_state.get("lb_per_gj", 0.0))
    margin = lb - lc
    st.session_state["margin_per_gj"] = margin  # persist for other tabs

    # Trucking-only LC (excludes station CAPEX/OPEX). Always show (even if 0).
    transport_lc = float(st.session_state.get("transport_lc_per_gj", 0.0))

    # Colors
    cost_bg, cost_fg = "#eef2ff", "#1e3a8a"
    ben_bg, ben_fg = "#ecfdf5", "#065f46"
    if margin >= 0:
        mar_bg, mar_fg = "#ecfdf5", "#065f46"
    else:
        mar_bg, mar_fg = "#fee2e2", "#991b1b"
    trk_bg, trk_fg = "#fff7ed", "#9a3412"  # orange

    html = f"""
    <div class='vp-ribbon' style='font-size:{font_size}; line-height:1; margin:8px 0 16px 0;'>
      <div class='vp-row1' style='display:flex; flex-wrap:wrap;'>
        {_pill("Levelized Cost", lc, "/GJ", cost_bg, cost_fg)}
        {_pill("Levelized Benefit", lb, "/GJ", ben_bg, ben_fg)}
        {_pill("Margin", margin, "/GJ", mar_bg, mar_fg)}
      </div>
      <div class='vp-row2' style='display:flex; flex-wrap:wrap; margin-top:6px;'>
        {_pill("Trucking Cost (excl. stations)", transport_lc, "/GJ", trk_bg, trk_fg)}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
