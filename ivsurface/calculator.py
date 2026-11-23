"""
The calculator's option-ageing path.

Version 1 computed time to expiry as T - (today - date) for each historical
date, so time to expiry RISED along the chart (from about zero to T) and the
oldest third of the window was priced at expiry. Here an option is bought on
the first day of the window with a fixed expiry date, and its time to expiry
falls to zero as the dates move forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bs import bs_price


def option_through_time(spot: pd.Series, strike: float | None, maturity_years: float,
                        r: float, sigma: float, kind: str = "call", q: float = 0.0) -> pd.DataFrame:
    """
    Value an option bought on spot.index[0] (strike defaults to that day's
    close, i.e. at the money) with expiry maturity_years later, on every date
    up to expiry. Returns spot, time to expiry, option value and intrinsic value.
    """
    spot = spot.dropna().sort_index()
    start = spot.index[0]
    expiry = start + pd.Timedelta(days=round(maturity_years * 365))
    path = spot[spot.index <= expiry].to_frame("spot")
    K = float(path["spot"].iloc[0]) if strike is None else float(strike)
    path["T"] = np.maximum((expiry - path.index).days.values / 365.0, 0.0)
    path["option"] = bs_price(path["spot"].values, K, path["T"].values, r, sigma, kind, q)
    w = 1.0 if str(kind).lower().startswith("c") else -1.0
    path["intrinsic"] = np.maximum(w * (path["spot"].values - K), 0.0)
    path.attrs.update({"strike": K, "expiry": expiry})
    return path


def synthetic_path(days: int = 400, s0: float = 100.0, sigma: float = 0.2, seed: int = 3) -> pd.Series:
    """A geometric Brownian motion path on business days, for offline use."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-02", periods=days)
    rets = (0.06 - 0.5 * sigma ** 2) / 252 + sigma / np.sqrt(252) * rng.standard_normal(days)
    return pd.Series(s0 * np.exp(np.cumsum(rets)), index=idx, name="synthetic")
