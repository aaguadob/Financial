"""
iv_calculator.py
----------------
Calculates Implied Volatility (IV) from market option prices using
Newton-Raphson and Brent's method as fallback.

Supports both call and put IV calculation.

Usage:
    python iv_calculator.py
    python iv_calculator.py --stock 100 --strike 105 --call_price 3.5 --maturity 0.25 --risk_free 0.05
    python iv_calculator.py --stock 100 --strike 95  --put_price  2.1 --maturity 0.25 --risk_free 0.05
"""

import argparse
import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq


# ── Black-Scholes pricing ─────────────────────────────────────────────────────

def _d1_d2(S, K, r, sigma, T, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S, K, r, sigma, T, q=0.0) -> float:
    d1, d2 = _d1_d2(S, K, r, sigma, T, q)
    return (S * np.exp(-q * T) * stats.norm.cdf(d1)
            - K * np.exp(-r * T) * stats.norm.cdf(d2))


def bs_put(S, K, r, sigma, T, q=0.0) -> float:
    d1, d2 = _d1_d2(S, K, r, sigma, T, q)
    return (K * np.exp(-r * T) * stats.norm.cdf(-d2)
            - S * np.exp(-q * T) * stats.norm.cdf(-d1))


def bs_vega(S, K, r, sigma, T, q=0.0) -> float:
    """Vega is the same for calls and puts — used in Newton-Raphson."""
    d1, _ = _d1_d2(S, K, r, sigma, T, q)
    return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)


# ── IV solvers ────────────────────────────────────────────────────────────────

def iv_newton(
    market_price: float,
    S: float, K: float, r: float, T: float,
    option_type: str = "call",
    q: float = 0.0,
    sigma0: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """
    Newton-Raphson IV solver. Fast when it converges.
    Returns None if it fails to converge (fallback to Brent).
    """
    pricing_fn = bs_call if option_type == "call" else bs_put
    sigma = sigma0

    for i in range(max_iter):
        price  = pricing_fn(S, K, r, sigma, T, q)
        vega   = bs_vega(S, K, r, sigma, T, q)
        diff   = price - market_price

        if abs(vega) < 1e-10:
            return None  # degenerate — hand off to Brent

        sigma -= diff / vega

        if sigma <= 0:
            return None  # gone negative — hand off to Brent

        if abs(diff) < tol:
            return sigma

    return None  # did not converge


def iv_brent(
    market_price: float,
    S: float, K: float, r: float, T: float,
    option_type: str = "call",
    q: float = 0.0,
    low: float = 1e-6,
    high: float = 10.0,
) -> float | None:
    """
    Brent's method IV solver. Slower but globally convergent.
    Searches sigma in [low, high] i.e. 0% to 1000% vol.
    """
    pricing_fn = bs_call if option_type == "call" else bs_put

    f_low  = pricing_fn(S, K, r, low,  T, q) - market_price
    f_high = pricing_fn(S, K, r, high, T, q) - market_price

    if f_low * f_high > 0:
        return None  # market price outside BS range — intrinsic or arbitrage

    try:
        return brentq(
            lambda sigma: pricing_fn(S, K, r, sigma, T, q) - market_price,
            low, high, xtol=1e-6, maxiter=500,
        )
    except ValueError:
        return None


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """
    Main IV entry point.
    Tries Newton-Raphson first (fast), falls back to Brent (robust).

    Parameters
    ----------
    market_price : observed mid-price of the option
    S            : current stock / underlying price
    K            : strike price
    r            : annualised risk-free rate (e.g. 0.05 for 5%)
    T            : time to expiry in years (e.g. 30/365)
    option_type  : 'call' or 'put'
    q            : continuous dividend yield (default 0)

    Returns
    -------
    float : implied volatility (annualised, e.g. 0.25 = 25%)

    Raises
    ------
    ValueError if IV cannot be computed (e.g. price below intrinsic value)
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if T <= 0:
        raise ValueError("T (time to expiry) must be positive")
    if market_price <= 0:
        raise ValueError("market_price must be positive")

    # intrinsic value check
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price < intrinsic * 0.999:
        raise ValueError(
            f"market_price {market_price:.4f} is below intrinsic value "
            f"{intrinsic:.4f} — arbitrage condition, IV undefined"
        )

    iv = iv_newton(market_price, S, K, r, T, option_type, q)
    if iv is None:
        iv = iv_brent(market_price, S, K, r, T, option_type, q)
    if iv is None:
        raise ValueError(
            f"Could not compute IV for {option_type} — "
            f"price={market_price}, S={S}, K={K}, T={T:.4f}, r={r}"
        )
    return iv


# ── pretty printer ────────────────────────────────────────────────────────────

def print_iv_report(
    market_price: float,
    S: float, K: float, r: float, T: float,
    option_type: str = "call",
    q: float = 0.0,
):
    iv = implied_vol(market_price, S, K, r, T, option_type, q)

    moneyness = S / K
    if moneyness > 1.02:
        m_label = "ITM call / OTM put"
    elif moneyness < 0.98:
        m_label = "OTM call / ITM put"
    else:
        m_label = "ATM"

    pricing_fn = bs_call if option_type == "call" else bs_put
    bs_price = pricing_fn(S, K, r, iv, T, q)

    print(f"\n── Implied Volatility Report ──")
    print(f"  Option type   : {option_type.upper()}")
    print(f"  Spot (S)      : {S:.4f}")
    print(f"  Strike (K)    : {K:.4f}")
    print(f"  Moneyness S/K : {moneyness:.4f}  ({m_label})")
    print(f"  Maturity (T)  : {T:.4f} years  ({T*365:.1f} days)")
    print(f"  Risk-free (r) : {r:.2%}")
    print(f"  Dividend (q)  : {q:.2%}")
    print(f"  Market price  : {market_price:.4f}")
    print(f"  BS price @ IV : {bs_price:.4f}  (error: {abs(bs_price - market_price):.2e})")
    print(f"  ─────────────────────────────")
    print(f"  Implied Vol   : {iv:.4f}  ({iv:.2%})")
    return iv


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implied Volatility Calculator")
    parser.add_argument("--stock",      type=float, default=100.0)
    parser.add_argument("--strike",     type=float, default=100.0)
    parser.add_argument("--call_price", type=float, default=None)
    parser.add_argument("--put_price",  type=float, default=None)
    parser.add_argument("--maturity",   type=float, default=30/365,
                        help="Time to expiry in years (e.g. 30/365)")
    parser.add_argument("--risk_free",  type=float, default=0.05)
    parser.add_argument("--dividend",   type=float, default=0.0)
    args = parser.parse_args()

    if args.call_price is not None:
        print_iv_report(args.call_price, args.stock, args.strike,
                        args.risk_free, args.maturity, "call", args.dividend)

    if args.put_price is not None:
        print_iv_report(args.put_price, args.stock, args.strike,
                        args.risk_free, args.maturity, "put", args.dividend)

    if args.call_price is None and args.put_price is None:
        # demo
        print("── Demo: ATM call, S=100, K=100, T=30d, r=5%, market_price=3.50 ──")
        print_iv_report(3.50, 100.0, 100.0, 0.05, 30/365, "call")

        print("\n── Demo: OTM put, S=100, K=95, T=30d, r=5%, market_price=1.20 ──")
        print_iv_report(1.20, 100.0, 95.0, 0.05, 30/365, "put")