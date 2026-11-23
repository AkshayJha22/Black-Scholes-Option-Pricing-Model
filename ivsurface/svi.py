"""
SVI smiles and static-arbitrage checks.

Raw SVI (Gatheral, 2004) writes total implied variance w = sigma_BS^2 * T as
a function of log-moneyness k = ln(K / F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

a sets the level, b the angle between the wings, rho the tilt (negative for
equity skew), m shifts the smile left or right and sigma rounds the vertex.
The wings are straight lines in k, with slopes b(1 - rho) on the left and
b(1 + rho) on the right; Lee's moment formula caps those slopes at 2.

Static arbitrage (no free money from options on one underlying):
  * butterfly: call prices must be convex in strike, i.e. the risk-neutral
    density must be non-negative. For a smile w(k) the density is
    proportional to g(k) (Gatheral and Jacquier, 2014):
        g(k) = (1 - k w'/(2w))^2 - (w'^2 / 4) (1/w + 1/4) + w''/2,
    so g(k) >= 0 everywhere means no butterfly arbitrage;
  * calendar: total variance must not decrease with maturity at any fixed
    log-moneyness, w(k, T2) >= w(k, T1) for T2 > T1.

SSVI (Gatheral and Jacquier, 2014) is a family of whole surfaces that is free
of both arbitrages under simple parameter conditions; ssvi_to_raw turns one
SSVI slice into raw SVI, which is how the synthetic test chains are built.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_dict(self) -> dict:
        return asdict(self)


def svi_total_variance(k, p: SVIParams):
    k = np.asarray(k, dtype=float)
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma * p.sigma))


def svi_implied_vol(k, T: float, p: SVIParams):
    return np.sqrt(np.maximum(svi_total_variance(k, p), 1e-12) / T)


def svi_derivatives(k, p: SVIParams):
    """w, w' and w'' of raw SVI with respect to k."""
    k = np.asarray(k, dtype=float)
    x = k - p.m
    root = np.sqrt(x * x + p.sigma * p.sigma)
    w = p.a + p.b * (p.rho * x + root)
    w1 = p.b * (p.rho + x / root)
    w2 = p.b * p.sigma * p.sigma / root ** 3
    return w, w1, w2


def butterfly_g(k, p: SVIParams):
    """Gatheral-Jacquier g(k); the risk-neutral density is proportional to it."""
    w, w1, w2 = svi_derivatives(k, p)
    w = np.maximum(w, 1e-12)
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 * w1 / 4) * (1 / w + 0.25) + w2 / 2


def risk_neutral_density(k, T: float, p: SVIParams):
    """Density of log-moneyness k under the pricing measure implied by the smile."""
    w = np.maximum(svi_total_variance(k, p), 1e-12)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    return butterfly_g(k, p) / np.sqrt(2 * np.pi * w) * np.exp(-0.5 * d2 * d2)


def ssvi_to_raw(theta: float, rho: float, phi: float) -> SVIParams:
    """
    One SSVI slice, w(k) = theta/2 * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2)),
    in raw SVI form. theta is ATM total variance.
    """
    return SVIParams(a=theta / 2 * (1 - rho * rho), b=theta * phi / 2, rho=rho,
                     m=-rho / phi, sigma=np.sqrt(1 - rho * rho) / phi)


def power_law_phi(theta: float, eta: float = 1.0, gamma: float = 0.5) -> float:
    """Gatheral-Jacquier power-law SSVI curvature, phi(theta) = eta / (theta^gamma (1+theta)^(1-gamma))."""
    return eta / (theta ** gamma * (1 + theta) ** (1 - gamma))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

@dataclass
class SliceFit:
    T: float
    params: SVIParams
    rmse_vol: float          # root-mean-square error in implied volatility
    n: int
    k_range: tuple

    def implied_vol(self, k):
        return svi_implied_vol(k, self.T, self.params)


def fit_svi(k, iv, T: float, weights=None) -> SliceFit:
    """
    Fit raw SVI to one expiry's implied volatilities by least squares in
    volatility units (soft-L1 loss to tame stray quotes), from several
    starting points. Bounds: b >= 0, |rho| < 1, sigma > 0, and the wing
    slopes b(1 +- rho) <= 2 (Lee) plus a non-negative minimum variance are
    enforced by a penalty.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[ok], iv[ok]
    wts = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)[ok]
    w_mkt = iv * iv * T
    w_atm = float(np.interp(0.0, k, w_mkt)) if k.min() < 0 < k.max() else float(np.median(w_mkt))

    k_check = np.linspace(k.min(), k.max(), 60)

    def resid(x):
        p = SVIParams(*x)
        w = svi_total_variance(k, p)
        model_iv = np.sqrt(np.maximum(w, 1e-10) / T)
        pen = [max(0.0, p.b * (1 + abs(p.rho)) - 2.0) * 10,                      # Lee bound
               max(0.0, -(p.a + p.b * p.sigma * np.sqrt(1 - p.rho ** 2))) * 100]  # min variance >= 0
        g = butterfly_g(k_check, p)                                              # no butterfly
        pen = np.concatenate([pen, np.maximum(0.0, -g) * 0.5])                   # arbitrage in range
        return np.concatenate([(model_iv - iv) * wts, pen])

    # the smile's vertex m is kept inside the quoted range, so the fit cannot
    # explain the data with one wing and leave the other to explode
    k_lo, k_hi = float(k.min()), float(k.max())
    lo = [-w_atm * 2 - 0.05, 0.0, -0.999, k_lo, 1e-4]
    hi = [w_atm * 2 + 0.05, 5.0, 0.999, k_hi, 3.0]
    best = None
    span = k_hi - k_lo
    for m0 in (k_lo + 0.4 * span, k_lo + 0.6 * span, k_lo + 0.8 * span):
        for s0 in (0.05, 0.2):
            for rho0 in (-0.6, 0.0):
                x0 = [max(lo[0] + 1e-6, w_atm * 0.5), max(0.05, w_atm * 2), rho0, m0, s0]
                x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
                sol = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.01,
                                    max_nfev=4000)
                if best is None or sol.cost < best.cost:
                    best = sol
    p = SVIParams(*[float(v) for v in best.x])
    model = svi_implied_vol(k, T, p)
    rmse = float(np.sqrt(np.mean((model - iv) ** 2)))
    return SliceFit(T=T, params=p, rmse_vol=rmse, n=int(len(k)), k_range=(float(k.min()), float(k.max())))


# ---------------------------------------------------------------------------
# Static-arbitrage checks
# ---------------------------------------------------------------------------

def butterfly_check(fit: SliceFit, k_grid=None, tol: float = 1e-6) -> dict:
    """
    Minimum of g(k) over the quoted log-moneyness range; negative means a
    negative density there, i.e. butterfly arbitrage. The minimum over a range
    extended by half its width on each side is reported too, for the
    extrapolated wings.
    """
    lo, hi = fit.k_range
    k = np.linspace(lo, hi, 801) if k_grid is None else np.asarray(k_grid)
    g = butterfly_g(k, fit.params)
    span = hi - lo
    g_ext = butterfly_g(np.linspace(lo - 0.5 * span, hi + 0.5 * span, 1201), fit.params)
    bad = g < -tol
    return {"min_g": float(g.min()), "k_at_min": float(k[np.argmin(g)]),
            "share_negative": float(bad.mean()), "arbitrage_free": bool(not bad.any()),
            "min_g_extrapolated": float(g_ext.min())}


def calendar_check(fits: list, k_grid=None, tol_vol_pts: float = 0.1) -> list:
    """
    For consecutive maturities, the largest drop in total variance at any
    log-moneyness on a common grid (inside both slices' quoted ranges).
    A drop is calendar arbitrage; it is flagged when it is worth more than
    tol_vol_pts volatility points at the later maturity, about the size of
    fitting noise and far inside typical bid-ask spreads.
    """
    fits = sorted(fits, key=lambda f: f.T)
    out = []
    for f1, f2 in zip(fits[:-1], fits[1:]):
        lo = max(f1.k_range[0], f2.k_range[0])
        hi = min(f1.k_range[1], f2.k_range[1])
        k = np.linspace(lo, hi, 401) if k_grid is None else np.asarray(k_grid)
        w1, w2 = svi_total_variance(k, f1.params), svi_total_variance(k, f2.params)
        drop = w1 - w2
        i = int(np.argmax(drop))
        worst = float(drop[i])
        # the volatility change at T2 that would remove the drop
        vol_pts = 100 * (np.sqrt(max(w1[i], 1e-12) / f2.T) - np.sqrt(max(w2[i], 1e-12) / f2.T))
        out.append({"T1": f1.T, "T2": f2.T, "max_variance_drop": worst,
                    "drop_vol_pts": float(max(vol_pts, 0.0)), "k_at_worst": float(k[i]),
                    "arbitrage_free": bool(worst <= 0 or vol_pts <= tol_vol_pts)})
    return out


def quoted_butterflies(strikes, bids, asks, tol: float = 1e-9) -> dict:
    """
    Executable butterfly check on quotes of one option type and expiry. For
    adjacent strikes K1 < K2 < K3, the butterfly long lam*C(K1) + (1-lam)*C(K3),
    short C(K2), with lam = (K3 - K2) / (K3 - K1), pays off non-negatively, so
    it can never cost less than zero. Buying the wings at the ask and selling
    the body at the bid, a negative cost is a real arbitrage; a negative cost
    at mid prices only is usually quote noise.
    """
    K = np.asarray(strikes, dtype=float)
    b = np.asarray(bids, dtype=float)
    a = np.asarray(asks, dtype=float)
    order = np.argsort(K)
    K, b, a = K[order], b[order], a[order]
    m = (a + b) / 2
    lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
    exec_cost = lam * a[:-2] + (1 - lam) * a[2:] - b[1:-1]
    mid_cost = lam * m[:-2] + (1 - lam) * m[2:] - m[1:-1]
    return {"triples": int(len(lam)), "executable_arbitrages": int((exec_cost < -tol).sum()),
            "mid_price_violations": int((mid_cost < -tol).sum()),
            "worst_executable_cost": float(exec_cost.min()) if len(lam) else 0.0}
