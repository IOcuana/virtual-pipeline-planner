# physics.py — gas property helpers (self-contained)

try:
    from CoolProp.CoolProp import PropsSI  # type: ignore

    _COOLPROP_OK = True
except Exception:
    _COOLPROP_OK = False

from constants import MOLAR_MASS, R_UNIV, HHV_PURE_J_PER_KG

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

def density_mixture_kg_per_m3(T_K: float, P_Pa: float, mole_fracs: dict) -> float:
    """
    Return mixture density [kg/m³] for a gas mixture defined by mole fractions.

    mole_fracs example:
    {
        "Methane": 0.90,
        "Ethane": 0.04,
        "Propane": 0.03,
        "Butane": 0.01,
        "Nitrogen": 0.02,
        "CO2": 0.00,
    }
    """
    # 1) Try CoolProp with a proper mixture string
    if _COOLPROP_OK:
        parts = []
        for comp, y in mole_fracs.items():
            if y <= 0.0:
                continue
            # Map our CO2 key to CoolProp's "CarbonDioxide"
            if comp == "CO2":
                cp_name = "CarbonDioxide"
            else:
                cp_name = comp
            parts.append(f"{cp_name}[{y}]")

        if parts:
            try:
                mixture_string = "&".join(parts)
                return float(PropsSI("D", "T", T_K, "P", P_Pa, mixture_string))
            except Exception:
                # Fall back to ideal-gas mixture below
                pass

    # 2) Ideal-gas fallback using mixture-averaged molar mass
    total_y = sum(y for y in mole_fracs.values() if y > 0.0)
    if total_y <= 0.0:
        # If something is wrong, just use Methane as a conservative fallback
        M_mix = MOLAR_MASS["Methane"]
    else:
        num = 0.0
        for comp, y in mole_fracs.items():
            if y <= 0.0:
                continue
            M = MOLAR_MASS.get(comp)
            if M is None:
                continue
            num += y * M
        if num <= 0.0:
            M_mix = MOLAR_MASS["Methane"]
        else:
            M_mix = num / total_y

    # ρ = P * M_mix / (R * T)
    return (P_Pa * M_mix) / (R_UNIV * T_K)


from constants import MOLAR_MASS, HHV_PURE_J_PER_KG

def mixture_hhv_J_per_kg(mole_fracs: dict, basis: str = "Net") -> float:
    """
    Compute mixture HHV [J/kg mixture] from mole fractions.
    Keys in mole_fracs should match MOLAR_MASS and HHV_PURE_J_PER_KG[basis]
    (e.g. 'Methane', 'Ethane', 'Propane', 'Butane', 'Nitrogen', 'CO2').

    Inerts (N2, CO2) naturally contribute 0 energy via HHV_PURE_J_PER_KG.
    """
    hv_map = HHV_PURE_J_PER_KG[basis]
    numerator = 0.0   # J/mol of mixture
    denominator = 0.0 # kg/mol of mixture

    for comp, y in mole_fracs.items():
        if y <= 0.0:
            continue
        if comp not in MOLAR_MASS:
            # Skip unknown components gracefully
            continue
        M = MOLAR_MASS[comp]  # kg/mol
        hhv_comp = hv_map.get(comp, 0.0)  # J/kg

        # Per mole of mixture: mass = y * M, energy = y * HHV * M
        denominator += y * M
        numerator += y * hhv_comp * M

    if denominator <= 0.0:
        return 0.0

    return numerator / denominator  # J/kg mixture


def mixture_hhv_J_per_kg(mole_fracs: dict, basis: str = "Net") -> float:
    """
    Compute mixture HHV [J/kg mixture] from mole fractions.
    Inerts (N2, CO2) naturally give 0 energy.
    """
    hv_map = HHV_PURE_J_PER_KG[basis]
    numerator = 0.0  # J/mol
    denominator = 0.0  # kg/mol

    for comp, y in mole_fracs.items():
        if y <= 0.0:
            continue
        M = MOLAR_MASS[comp]  # kg/mol
        hhv_comp = hv_map.get(comp, 0.0)  # J/kg

        denominator += y * M
        numerator += y * hhv_comp * M  # J/mol

    if denominator <= 0:
        return 0.0

    return numerator / denominator  # J/kg
