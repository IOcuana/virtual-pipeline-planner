# Phase‑1 Refactor Pack

This pack extracts *helpers and core calculations* from your monolithic Streamlit app into small modules.

## Files

- `constants.py` — unit constants and road‑train maps.
- `ui_helpers.py` — CSS injectors, flyout helper, and comma‑formatting inputs.
- `fatigue.py` — Standard Hours helpers and compliance map utilities.
- `calcs.py` — `compute_calcs(...)` (pure function), uses `fatigue` for breaks/limits.

## How to use

1. Drop all four `.py` files next to your main app file.
2. In your main app, add these imports at the top:

```python
from constants import (R_UNIV, MOLAR_MASS, M3_PER_FT3, FT3_PER_M3, GJ_PER_MMBTU, DAYS_PER_YEAR,
                       ROADTRAIN_MAP, DOLLIES_PER_COMBO)
from ui_helpers import (_inject_sidebar_expander_css, _inject_sidebar_inline_reset_css, inline_number_with_reset,
                        _inject_flyout_css, help_flyout, _inject_planner_expander_css,
                        _fmt_millions, _fmt_with_commas, _parse_commas, number_commas, currency_commas)
from fatigue import (required_breaks_time, solo_standard_shift_ok,
                     RELAY_OPTIONS_DISPLAY, RELAY_DISPLAY_TO_LEGACY, best_mode_for)
from calcs import compute_calcs
```

3. **Remove** the duplicated definitions from your main app:
   - constants block, ROADTRAIN_MAP/DOLLIES_PER_COMBO
   - all CSS injectors & number/currency comma helpers
   - `required_breaks_time`, `solo_standard_shift_ok`, and the compliance‑map helpers
   - `compute_calcs` function

4. Fix the over‑indent in `_render_driver_utilisation_section()` so the first `st.subheader(...)` sits one indent under `def`.

That’s it. Run `quick_format.bat` to reformat everything after the move.
