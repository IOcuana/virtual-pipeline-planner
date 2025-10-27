# physics.py — gas property helpers (self-contained)

try:
    from CoolProp.CoolProp import PropsSI  # type: ignore

    _COOLPROP_OK = True
except Exception:
    _COOLPROP_OK = False

from constants import MOLAR_MASS, R_UNIV


def density_kg_per_m3(T_K: float, P_Pa: float, gas_label: str) -> float:
    """Return gas density (kg/m3) using CoolProp when available; otherwise ideal-gas fallback.
    gas_label should match keys in MOLAR_MASS (e.g., "Methane", "Hydrogen").
    """
    if _COOLPROP_OK:
        try:
            return float(PropsSI("D", "T", T_K, "P", P_Pa, gas_label))
        except Exception:
            pass  # fall through to ideal gas

    molar_mass = MOLAR_MASS.get(gas_label, MOLAR_MASS.get("Methane"))
    if molar_mass is None:
        raise ValueError(f"Unknown gas_label: {gas_label}")
    # Ideal gas: rho = P*M / (R*T)
    return (P_Pa * molar_mass) / (R_UNIV * T_K)
