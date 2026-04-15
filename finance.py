# finance.py — Financial calculation helpers
def capital_recovery_factor(i: float, n_years: float) -> float:
    """Calculate Capital Recovery Factor (CRF).

    CRF = i(1+i)^n / ((1+i)^n - 1)

    Args:
        i: Annual interest rate (as decimal, e.g., 0.08 for 8%)
        n_years: Asset life in years

    Returns:
        Capital recovery factor (annualization multiplier)

    Special case: If i ≈ 0, returns 1/n_years
    """
    i = float(i)
    n_years = float(n_years)

    if n_years <= 0:
        return 0.0

    if abs(i) < 1e-9:  # Near-zero interest rate
        return 1.0 / n_years

    p = (1.0 + i) ** n_years
    return i * p / (p - 1.0)
