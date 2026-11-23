"""
Synthetic option chains with a known, arbitrage-free volatility surface.

Prices come from a power-law SSVI surface (Gatheral and Jacquier, 2014),
which is free of butterfly and calendar arbitrage for the default parameters,
plus quote noise and a bid-ask spread. The chain has the same columns as a
real snapshot, so the whole pipeline (clean, forward, implied vols, SVI fits,
arbitrage checks) can be tested offline against a known answer. Arbitrage can
be injected on purpose to test the checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bs import black_price
from .svi import SVIParams, power_law_phi, ssvi_to_raw, svi_implied_vol

DEFAULT_DAYS = (7, 14, 30, 60, 91, 182, 365)


def atm_vol_term_structure(T):
    """Upward-sloping ATM volatility: 15% at very short maturities, about 19% at two years."""
    return 0.15 + 0.04 * (1 - np.exp(-np.asarray(T, dtype=float) / 0.5))


def ssvi_slices(expiry_days=DEFAULT_DAYS, rho: float = -0.7, eta: float = 1.0,
                gamma: float = 0.5) -> dict:
    """Raw SVI parameters of each maturity slice of the SSVI surface, keyed by T in years."""
    out = {}
    for d in expiry_days:
        T = d / 365.0
        theta = float(atm_vol_term_structure(T) ** 2 * T)
        out[T] = ssvi_to_raw(theta, rho, power_law_phi(theta, eta, gamma))
    return out


def strike_grid(spot: float, T: float, n: int = 31) -> np.ndarray:
    """Strikes from about 4.5 ATM standard deviations below spot to 3 above, rounded to 0.5."""
    sd = float(atm_vol_term_structure(T)) * np.sqrt(T)
    k = np.linspace(-4.5 * sd, 3.0 * sd, n)
    return np.unique(np.round(spot * np.exp(k) * 2) / 2)


def synthetic_chain(spot: float = 100.0, r: float = 0.04, q: float = 0.015,
                    expiry_days=DEFAULT_DAYS, noise_vol: float = 0.002,
                    rel_spread: float = 0.03, seed: int = 0, inject: str | None = None):
    """
    Build a chain with the columns of a real snapshot (expiry, days, type,
    strike, bid, ask, last, volume, open_interest, yahoo_iv, spot).

    inject: None, "calendar" (the 91-day slice's variance is cut below the
    60-day slice's) or "butterfly" (the 30-day slice is replaced by a smile
    with a negative density). Returns (chain, truth) where truth maps T to the
    generating SVIParams.
    """
    rng = np.random.default_rng(seed)
    truth = ssvi_slices(expiry_days)
    Ts = sorted(truth)
    if inject == "calendar":
        T60, T91 = Ts[3], Ts[4]
        p = truth[T60]
        truth[T91] = SVIParams(a=p.a * 0.75, b=p.b * 0.75, rho=p.rho, m=p.m, sigma=p.sigma)
    elif inject == "butterfly":
        T30 = Ts[2]
        p = truth[T30]
        truth[T30] = SVIParams(a=p.a, b=1.9, rho=-0.98, m=p.m, sigma=0.002)
    elif inject is not None:
        raise ValueError(f"unknown injection {inject!r}")

    today = pd.Timestamp("2026-01-02")
    rows = []
    for d, T in zip(expiry_days, Ts):
        F = spot * np.exp((r - q) * T)
        df = np.exp(-r * T)
        expiry = (today + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for K in strike_grid(spot, T):
            K = float(K)
            iv = float(svi_implied_vol(np.log(K / F), T, truth[T])) + rng.normal(0, noise_vol)
            iv = max(iv, 0.01)
            for kind in ("C", "P"):
                mid = float(black_price(F, K, T, iv, df, kind))
                half = max(0.005, rel_spread / 2 * mid)
                bid = round(max(mid - half, 0.0), 4) if mid > 0.02 else 0.0
                ask = round(mid + half, 4)
                rows.append({"expiry": expiry, "days": d, "type": kind, "strike": K,
                             "bid": bid, "ask": ask, "last": round(mid, 4), "volume": 100.0,
                             "open_interest": 1000.0, "yahoo_iv": iv, "spot": spot})
    return pd.DataFrame(rows), truth
