# Core constants, unit conversions, and combination maps

R_UNIV = 8.314462618  # J/mol-K
MOLAR_MASS = {
    "Methane": 0.01604246,
    "Ethane": 0.03006904,
    "Propane": 0.04409562,
    "Butane": 0.0581222,
    "Nitrogen": 0.0280134,
    "CO2": 0.0440095,
    "Hydrogen": 0.00201588,
    "Ammonia": 0.0170305,
}  # kg/mol
HHV_PURE_J_PER_KG = {
    "Gross": {
        "Methane": 55_571_000.0,
        "Ethane": 51_900_000.0,
        "Propane": 50_350_000.0,
        "Butane": 49_500_000.0,
        # Inerts: no energy contribution
        "Nitrogen": 0.0,
        "CO2": 0.0,
    },
    "Net": {
        "Methane": 50_032_000.0,
        "Ethane": 47_400_000.0,
        "Propane": 46_360_000.0,
        "Butane": 45_700_000.0,
        "Nitrogen": 0.0,
        "CO2": 0.0,
    },
}
M3_PER_FT3 = 0.028316846592
FT3_PER_M3 = 1.0 / M3_PER_FT3
GJ_PER_MMBTU = 1.055056
DAYS_PER_YEAR = 365.0

# Tyre configuration (per vehicle)
STEER_TYRES_PER_TRUCK = 2
DRIVE_TYRES_PER_TRUCK = 12
TRAILER_TYRES_PER_A = 12
TRAILER_TYRES_PER_B = 12

# Road-train mapping (A = A-trailer, B = B-trailer)
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
