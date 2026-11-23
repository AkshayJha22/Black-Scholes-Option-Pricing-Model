"""
Black-Scholes-Merton pricing, Greeks and implied volatility.

Conventions: S spot, K strike, T time to expiry in years, r continuously
compounded risk-free rate, q continuous dividend yield, sigma annualised
volatility, kind "call" or "put" (any case; "c"/"p" also accepted). Every
function is vectorised with NumPy and handles the limits T -> 0 and
sigma -> 0, where the price tends to the discounted intrinsic value of the
forward, max(+-(S e^{-qT} - K e^{-rT}), 0).

Black-76 versions (black_price, black_implied_vol) work on a forward F and a
discount factor, which is how listed chains are priced: the forward and the
discount factor are read off put-call parity, so no dividend assumption is
needed.

Version 1 of this project had four bugs that the tests now pin down:
bs_delta('CALL') returned the put's delta, zero time or zero volatility
raised ZeroDivisionError, an invalid option type returned None, and the
implied-volatility solver failed with a cryptic error outside the
no-arbitrage bounds.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_CALL = {"call", "c"}
_PUT = {"put", "p"}


def option_sign(kind) -> np.ndarray:
    """+1 for calls, -1 for puts. Raises ValueError for anything else."""
    kinds = np.atleast_1d(np.asarray(kind, dtype=object))
    out = np.empty(kinds.shape, dtype=float)
    for i, k in np.ndenumerate(kinds):
        key = str(k).strip().lower()
        if key in _CALL:
            out[i] = 1.0
        elif key in _PUT:
            out[i] = -1.0
        else:
            raise ValueError(f"option type must be 'call' or 'put', got {k!r}")
    return out if np.ndim(kind) else out[0]


def _arrays(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


def bs_price(S, K, T, r, sigma, kind="call", q=0.0):
    """Black-Scholes-Merton price; the sigma -> 0 and T -> 0 limits are exact."""
    S, K, T, r, sigma, q = _arrays(S, K, T, r, sigma, q)
    w = option_sign(kind)
    disc_r, disc_q = np.exp(-r * np.maximum(T, 0.0)), np.exp(-q * np.maximum(T, 0.0))
    degenerate = (T <= 0) | (sigma <= 0)
    vol_sqrt_t = np.where(degenerate, 1.0, sigma * np.sqrt(np.maximum(T, 0.0)))
    d1 = (np.log(S / K) + (r - q) * T + 0.5 * vol_sqrt_t ** 2) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    px = w * (S * disc_q * norm.cdf(w * d1) - K * disc_r * norm.cdf(w * d2))
    limit = np.maximum(w * (S * disc_q - K * disc_r), 0.0)
    return np.where(degenerate, limit, px)[()]


def bs_greeks(S, K, T, r, sigma, kind="call", q=0.0) -> dict:
    """
    Delta, gamma, vega (per 1.00 of volatility), theta (per year, the change
    as calendar time passes) and rho (per 1.00 of rate). In the sigma -> 0 or
    T -> 0 limits the Greeks are those of the deterministic payoff.
    """
    S, K, T, r, sigma, q = _arrays(S, K, T, r, sigma, q)
    w = option_sign(kind)
    Tp = np.maximum(T, 0.0)
    disc_r, disc_q = np.exp(-r * Tp), np.exp(-q * Tp)
    degenerate = (T <= 0) | (sigma <= 0)
    srt = np.where(degenerate, 1.0, sigma * np.sqrt(Tp))
    d1 = (np.log(S / K) + (r - q) * Tp + 0.5 * srt ** 2) / srt
    d2 = d1 - srt
    pdf = norm.pdf(d1)
    delta = w * disc_q * norm.cdf(w * d1)
    gamma = disc_q * pdf / (S * srt)
    vega = S * disc_q * pdf * np.sqrt(Tp)
    theta = (-S * disc_q * pdf * sigma / (2 * np.sqrt(np.where(degenerate, 1.0, Tp)))
             - w * r * K * disc_r * norm.cdf(w * d2) + w * q * S * disc_q * norm.cdf(w * d1))
    rho = w * K * Tp * disc_r * norm.cdf(w * d2)

    itm = w * (S * disc_q - K * disc_r) > 0
    zero = np.zeros_like(delta)
    delta = np.where(degenerate, np.where(itm, w * disc_q, 0.0), delta)
    gamma = np.where(degenerate, zero, gamma)
    vega = np.where(degenerate, zero, vega)
    theta = np.where(degenerate, np.where(itm, w * (q * S * disc_q - r * K * disc_r), 0.0), theta)
    rho = np.where(degenerate, np.where(itm, w * K * Tp * disc_r, 0.0), rho)
    return {"delta": delta[()], "gamma": gamma[()], "vega": vega[()],
            "theta": theta[()], "rho": rho[()]}


def bs_delta(S, K, T, r, sigma, kind="call", q=0.0):
    return bs_greeks(S, K, T, r, sigma, kind, q)["delta"]


def price_bounds(S, K, T, r, kind="call", q=0.0):
    """No-arbitrage bounds (lower, upper) on a European option's price."""
    S, K, T, r, q = _arrays(S, K, T, r, q)
    w = option_sign(kind)
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    lower = np.maximum(w * (S * disc_q - K * disc_r), 0.0)
    upper = np.where(w > 0, S * disc_q, K * disc_r)
    return lower[()], upper[()]


def implied_vol(price, S, K, T, r, kind="call", q=0.0, lo=1e-4, hi=5.0,
                tol=1e-10, max_iter=100):
    """
    Implied volatility, vectorised. Newton steps on vega, kept inside a
    bisection bracket [lo, hi] that shrinks every iteration, so it always
    converges when a solution exists. Prices outside the no-arbitrage bounds
    (or needing a volatility outside [lo, hi]) return NaN rather than raising.
    """
    price, S, K, T, r, q = np.broadcast_arrays(*_arrays(price, S, K, T, r, q))
    w = np.broadcast_to(option_sign(kind), price.shape)
    kinds = np.where(w > 0, "call", "put")
    lower, upper = price_bounds(S, K, T, r, kinds, q)
    lo_arr = np.full(price.shape, lo)
    hi_arr = np.full(price.shape, hi)
    ok = (price > lower) & (price < upper) & (T > 0)
    ok &= bs_price(S, K, T, r, lo_arr, kinds, q) <= price
    ok &= bs_price(S, K, T, r, hi_arr, kinds, q) >= price

    # Brenner-Subrahmanyam start, clipped into the bracket
    sigma = np.clip(np.sqrt(2 * np.pi / np.maximum(T, 1e-12)) * price / np.maximum(S, 1e-12),
                    lo * 2, hi / 2)
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, kinds, q) - price
        done = np.abs(diff) < tol
        if np.all(done | ~ok):
            break
        lo_arr = np.where(diff < 0, sigma, lo_arr)
        hi_arr = np.where(diff > 0, sigma, hi_arr)
        vega = bs_greeks(S, K, T, r, sigma, kinds, q)["vega"]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = sigma - diff / vega
        inside = (newton > lo_arr) & (newton < hi_arr) & np.isfinite(newton)
        sigma = np.where(done, sigma, np.where(inside, newton, 0.5 * (lo_arr + hi_arr)))
    return np.where(ok, sigma, np.nan)[()]


def implied_vol_brent(price, S, K, T, r, kind="call", q=0.0, lo=1e-4, hi=5.0) -> float:
    """Scalar implied volatility by Brent's method (version 1's approach), with bounds checks."""
    lower, upper = price_bounds(S, K, T, r, kind, q)
    if not (lower < price < upper) or T <= 0:
        return float("nan")
    f = lambda s: float(bs_price(S, K, T, r, s, kind, q)) - price
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-12))


# ---------------------------------------------------------------------------
# Black-76 on a forward (how listed chains are priced)
# ---------------------------------------------------------------------------

def _rate_from_df(T, df):
    T, df = _arrays(T, df)
    return np.where(T > 0, -np.log(df) / np.where(T > 0, T, 1.0), 0.0)


def black_price(F, K, T, sigma, df=1.0, kind="call"):
    """Black-76: df * w * (F N(w d1) - K N(w d2)), d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt T).
    Equivalent to Black-Scholes-Merton with S = F and r = q, the rate implied by df."""
    rate = _rate_from_df(T, df)
    return bs_price(F, K, T, rate, sigma, kind, q=rate)


def black_implied_vol(price, F, K, T, df=1.0, kind="call", **kw):
    """Implied volatility of a price quoted on a forward F with discount factor df."""
    rate = _rate_from_df(T, df)
    return implied_vol(price, F, K, T, rate, kind, q=rate, **kw)
