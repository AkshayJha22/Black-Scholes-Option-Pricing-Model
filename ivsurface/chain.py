"""
Option chains: loading, cleaning, forwards, implied volatilities and the fitted surface.

Pipeline for one chain:
  1. clean: drop quotes with no bid, crossed or locked markets, very wide
     spreads and expiries outside a maturity window; use mid prices;
  2. forward and discount factor per expiry. European put-call parity says
     C - P = df * (F - K) at every strike. Listed equity options (SPY's
     included) are American: early exercise makes in-the-money options dearer
     and tilts C - P, so a free regression of C - P on K returns a discount
     factor above 1. With a risk-free rate (the 13-week T-bill stored with the
     snapshot), df = exp(-r T) and only the forward is read off parity, at the
     strikes nearest the money, as in Cboe's VIX method. Without a rate the
     regression is used, with df capped at 1;
  3. implied volatilities (Black-76 on the forward) of out-of-the-money
     options only (puts below the forward, calls above), which are the liquid
     ones and carry no early-exercise premium worth speaking of;
  4. an SVI fit per expiry and static-arbitrage checks across the surface.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .bs import black_implied_vol
from .svi import SliceFit, butterfly_check, calendar_check, fit_svi, quoted_butterflies, svi_implied_vol

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COLUMNS = ["expiry", "days", "type", "strike", "bid", "ask", "last", "volume",
           "open_interest", "yahoo_iv", "spot"]


def load_snapshot(ticker: str = "SPY", data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict]:
    chain = pd.read_csv(data_dir / f"{ticker}_chain_snapshot.csv")
    meta_path = data_dir / f"{ticker}_chain_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("tbill_13w") is not None and "rate" not in chain.columns:
        chain["rate"] = float(meta["tbill_13w"])
    return chain, meta


def fetch_live(ticker: str, max_days: int = 400) -> tuple[pd.DataFrame, dict]:
    """A live chain from Yahoo Finance (needs internet), in the snapshot's columns."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].dropna().iloc[-1])
    today = pd.Timestamp.today().normalize()
    rows = []
    for exp in tk.options:
        days = (pd.Timestamp(exp) - today).days
        if days < 1 or days > max_days:
            continue
        ch = tk.option_chain(exp)
        for kind, frame in (("C", ch.calls), ("P", ch.puts)):
            for _, r in frame.iterrows():
                rows.append({"expiry": exp, "days": days, "type": kind, "strike": float(r["strike"]),
                             "bid": float(r.get("bid", np.nan)), "ask": float(r.get("ask", np.nan)),
                             "last": float(r.get("lastPrice", np.nan)),
                             "volume": float(r.get("volume", np.nan)),
                             "open_interest": float(r.get("openInterest", np.nan)),
                             "yahoo_iv": float(r.get("impliedVolatility", np.nan)), "spot": spot})
    if not rows:
        raise RuntimeError(f"No option chain returned for {ticker!r}.")
    out = pd.DataFrame(rows)
    rate = None
    try:
        irx = yf.Ticker("^IRX").history(period="10d")["Close"].dropna()
        rate = float(irx.iloc[-1]) / 100.0 if len(irx) else None
    except Exception:
        rate = None
    if rate is not None:
        out["rate"] = rate
    return out, {"ticker": ticker, "spot": spot, "tbill_13w": rate,
                 "as_of_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")}


def clean(chain: pd.DataFrame, min_days: int = 2, max_days: int = 400,
          max_rel_spread: float = 0.5) -> pd.DataFrame:
    """Keep two-sided, not-too-wide quotes and add mid prices."""
    c = chain.copy()
    c = c[(c["days"] >= min_days) & (c["days"] <= max_days)]
    c = c[(c["bid"] > 0) & (c["ask"] > c["bid"])].copy()
    c["mid"] = (c["bid"] + c["ask"]) / 2
    c["rel_spread"] = (c["ask"] - c["bid"]) / c["mid"]
    return c[c["rel_spread"] <= max_rel_spread].reset_index(drop=True)


def forward_and_discount(slice_: pd.DataFrame, spot: float, rate: float | None = None) -> tuple[float, float, int]:
    """
    Forward and discount factor of one expiry from put-call parity,
    C - P = df * (F - K) (European).

    With a risk-free rate, df = exp(-rate * T) and the forward is the median of
    K + (C - P) / df over the six strikes nearest to where C - P changes sign,
    where early exercise is worth least (American options make in-the-money
    calls and puts dearer). Without one, C - P is regressed on K over strikes
    within 10% of spot; the slope is -df, capped at 1, and the forward is read
    near the money at that df. Returns (forward, df, strikes used).
    """
    calls = slice_[slice_["type"] == "C"].set_index("strike")["mid"]
    puts = slice_[slice_["type"] == "P"].set_index("strike")["mid"]
    both = pd.concat([calls, puts], axis=1, keys=["c", "p"]).dropna()
    if both.empty:
        return float(spot), 1.0, 0
    T = float(slice_["days"].iloc[0]) / 365.0

    def forward_near_money(df: float, n: int = 6) -> tuple[float, int]:
        diff = both["c"] - both["p"]
        k_star = float(diff.abs().idxmin())
        near = both.iloc[np.argsort(np.abs(both.index.values - k_star))[:n]]
        return float(np.median(near.index.values + (near["c"] - near["p"]).values / df)), len(near)

    if rate is not None and np.isfinite(rate):
        df = float(np.exp(-rate * T))
        F, n = forward_near_money(df)
        return F, df, n
    near = both[(both.index > 0.9 * spot) & (both.index < 1.1 * spot)]
    if len(near) < 3:
        near = both
    if len(near) >= 2:
        K = near.index.values.astype(float)
        slope, intercept = np.polyfit(K, (near["c"] - near["p"]).values, 1)
        df = -slope
        if 0.8 < df <= 1.0:
            return float(intercept / df), float(df), int(len(near))
        if 1.0 < df <= 1.1:          # American early exercise tilts the line: cap at 1
            F, n = forward_near_money(1.0)
            return F, 1.0, n
    F, n = forward_near_money(1.0)
    return F, 1.0, n


def implied_vols(chain: pd.DataFrame, spot: float | None = None, rate: float | None = None) -> pd.DataFrame:
    """Implied volatility of every out-of-the-money quote, with its forward and log-moneyness.
    rate: risk-free rate for discounting; defaults to the chain's "rate" column if it has one."""
    spot = float(chain["spot"].iloc[0]) if spot is None else spot
    if rate is None and "rate" in chain.columns and chain["rate"].notna().any():
        rate = float(chain["rate"].dropna().iloc[0])
    out = []
    for expiry, g in chain.groupby("expiry"):
        days = int(g["days"].iloc[0])
        T = days / 365.0
        F, df, n_parity = forward_and_discount(g, spot, rate)
        otm = g[((g["type"] == "P") & (g["strike"] < F)) | ((g["type"] == "C") & (g["strike"] >= F))].copy()
        if otm.empty:
            continue
        kinds = np.where(otm["type"] == "C", "call", "put")
        otm["iv"] = black_implied_vol(otm["mid"].values, F, otm["strike"].values, T, df, kinds)
        otm["T"], otm["forward"], otm["df"], otm["parity_points"] = T, F, df, n_parity
        otm["k"] = np.log(otm["strike"] / F)
        out.append(otm)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    return res[np.isfinite(res["iv"])].reset_index(drop=True)


@dataclass
class Surface:
    fits: dict                      # expiry -> SliceFit
    summary: pd.DataFrame           # one row per expiry
    calendar: list                  # consecutive-maturity checks
    ivs: pd.DataFrame               # the implied volatilities the fits used


def delta_strike_k(fit: SliceFit, delta: float, kind: str) -> float:
    """Log-moneyness whose forward delta is +delta (call) or -delta (put) on the fitted smile."""
    T = fit.T

    def fwd_delta(k):
        s = float(svi_implied_vol(k, T, fit.params)) * np.sqrt(T)
        d1 = -k / s + s / 2
        return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1

    sd = float(svi_implied_vol(0.0, T, fit.params)) * np.sqrt(T)
    if kind == "call":
        return brentq(lambda k: fwd_delta(k) - delta, 0.0, 6 * sd)
    return brentq(lambda k: fwd_delta(k) + delta, -8 * sd, 0.0)


def fit_surface(ivs: pd.DataFrame, min_points: int = 8, k_window: float = 4.0) -> Surface:
    """
    Fit SVI to every expiry with enough quotes. Quotes further than k_window
    ATM standard deviations from the forward are left out of the fit (deep
    wings are mostly stale or one-sided).
    """
    fits, rows = {}, []
    for expiry, g in ivs.groupby("expiry"):
        T = float(g["T"].iloc[0])
        atm_guess = float(np.interp(0.0, g.sort_values("k")["k"], g.sort_values("k")["iv"]))
        g = g[np.abs(g["k"]) <= k_window * atm_guess * np.sqrt(T)]
        if len(g) < min_points:
            continue
        fit = fit_svi(g["k"].values, g["iv"].values, T)
        fits[expiry] = fit
        bf = butterfly_check(fit)
        quotes = {kind: quoted_butterflies(q["strike"], q["bid"], q["ask"])
                  for kind, q in g.groupby("type") if len(q) >= 3}
        try:
            rr = 100 * (float(svi_implied_vol(delta_strike_k(fit, 0.25, "put"), T, fit.params))
                        - float(svi_implied_vol(delta_strike_k(fit, 0.25, "call"), T, fit.params)))
        except ValueError:
            rr = float("nan")
        rows.append({"expiry": expiry, "days": int(round(T * 365)), "T": T, "options": fit.n,
                     "forward": float(g["forward"].iloc[0]),
                     "atm_iv": float(svi_implied_vol(0.0, T, fit.params)),
                     "skew_25d_put_minus_call": rr, "rmse_vol_pts": 100 * fit.rmse_vol,
                     **{f"svi_{k}": v for k, v in fit.params.as_dict().items()},
                     "butterfly_min_g": bf["min_g"], "butterfly_ok": bf["arbitrage_free"],
                     "quoted_butterfly_arbitrages": sum(v["executable_arbitrages"] for v in quotes.values()),
                     "quoted_butterfly_mid_violations": sum(v["mid_price_violations"] for v in quotes.values())})
    summary = pd.DataFrame(rows).sort_values("T").reset_index(drop=True) if rows else pd.DataFrame()
    cal = calendar_check(list(fits.values())) if len(fits) > 1 else []
    if len(summary):
        flags = {round(c["T2"] * 365): c["arbitrage_free"] for c in cal}
        summary["calendar_ok_vs_previous"] = [flags.get(d, True) for d in summary["days"]]
    return Surface(fits=fits, summary=summary, calendar=cal, ivs=ivs)


def build_surface(chain: pd.DataFrame, min_days: int = 2, max_days: int = 400,
                  rate: float | None = None) -> Surface:
    """clean -> implied vols -> SVI fits -> arbitrage checks."""
    return fit_surface(implied_vols(clean(chain, min_days, max_days), rate=rate))
