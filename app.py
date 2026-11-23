"""
app.py: Implied Volatility Surface Explorer (Streamlit entry point).

Tabs:
  * Volatility surface: a listed option chain (the bundled SPY snapshot, a
    synthetic chain with a known answer, or a live chain from Yahoo) turned
    into implied volatilities, SVI smiles per expiry, a fitted surface and
    static-arbitrage checks.
  * Calculator: Black-Scholes-Merton prices, Greeks, an implied-volatility
    solver, price heatmaps and an option valued from purchase to expiry.

All maths lives in the ivsurface package; this file only does UI.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ivsurface import plots
from ivsurface.bs import bs_greeks, bs_price, implied_vol, price_bounds
from ivsurface.calculator import option_through_time, synthetic_path
from ivsurface.chain import build_surface, fetch_live, load_snapshot
from ivsurface.synthetic import synthetic_chain

st.set_page_config(page_title="Implied Volatility Surface Explorer", page_icon="📈", layout="wide")
CFG = {"displayModeBar": False}


@st.cache_data(show_spinner=False)
def get_chain(source: str, ticker: str):
    if source == "snapshot":
        chain, meta = load_snapshot("SPY")
        return chain, meta
    if source == "synthetic":
        chain, _ = synthetic_chain(seed=7)
        return chain, {"ticker": "SYNTHETIC", "spot": float(chain["spot"].iloc[0]),
                       "as_of_utc": "generated from a known SSVI surface"}
    return fetch_live(ticker)


@st.cache_data(show_spinner=False)
def get_surface(chain: pd.DataFrame, min_days: int, max_days: int):
    return build_surface(chain, min_days, max_days)


@st.cache_data(show_spinner=False)
def get_history(ticker: str, years: float) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, period=f"{int(np.ceil(years * 1.2 + 0.5))}y", auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("no data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    return s


st.title("Implied Volatility Surface Explorer")
tab_surface, tab_calc, tab_about = st.tabs(["Volatility surface", "Calculator", "How it works"])

# ---------------------------------------------------------------------------
# Volatility surface
# ---------------------------------------------------------------------------
with tab_surface:
    c1, c2, c3 = st.columns([2, 1, 1])
    source = c1.radio("Option chain", ["snapshot", "synthetic", "live"], horizontal=True,
                      format_func={"snapshot": "SPY snapshot (bundled)",
                                   "synthetic": "Synthetic demo chain",
                                   "live": "Live chain (Yahoo)"}.get)
    ticker = c2.text_input("Ticker (live)", "SPY", disabled=source != "live").strip().upper() or "SPY"
    min_days, max_days = c3.slider("Days to expiry", 1, 730, (2, 400))

    try:
        with st.spinner("Loading the chain and fitting smiles..."):
            chain, meta = get_chain(source, ticker)
            surface = get_surface(chain, min_days, max_days)
    except Exception as exc:  # live data unavailable
        st.error(f"Could not load a live chain ({exc}). Showing the bundled SPY snapshot instead.")
        chain, meta = get_chain("snapshot", "SPY")
        surface = get_surface(chain, min_days, max_days)

    s = surface.summary
    if s.empty:
        st.warning("No expiry had enough clean quotes to fit. Widen the maturity window.")
        st.stop()

    atm30 = float(np.interp(30, s["days"], s["atm_iv"]))
    k = st.columns(6)
    k[0].metric("Underlying", f"{meta.get('ticker', ticker)} {float(chain['spot'].iloc[0]):,.2f}")
    k[1].metric("Expiries fitted", f"{len(s)}")
    k[2].metric("Quotes used", f"{int(s['options'].sum()):,}")
    k[3].metric("30-day ATM vol", f"{atm30:.1%}")
    k[4].metric("Median fit error", f"{s['rmse_vol_pts'].median():.2f} vol pts")
    flags = int((~s["butterfly_ok"]).sum() + (~s["calendar_ok_vs_previous"]).sum())
    k[5].metric("Arbitrage flags", f"{flags}", help="SVI butterfly (negative density) plus calendar "
                "(variance falling with maturity by more than 0.1 vol points)")
    st.caption(f"Chain as of {meta.get('as_of_utc', 'n/a')} UTC. Forwards and discount factors come "
               "from put-call parity; implied vols from out-of-the-money mid prices.")

    exps = list(s["expiry"])
    default = [exps[i] for i in np.unique(np.linspace(0, len(exps) - 1, min(6, len(exps))).astype(int))]
    chosen = st.multiselect("Expiries to plot", exps, default=default)
    if chosen:
        st.plotly_chart(plots.fig_smiles(surface, chosen), width="stretch", config=CFG)
    cl, cr = st.columns(2)
    cl.plotly_chart(plots.fig_term_structure(surface), width="stretch", config=CFG)
    dens_exp = cr.selectbox("Risk-neutral density for", exps, index=min(len(exps) - 1, exps.index(default[len(default) // 2])))
    cr.plotly_chart(plots.fig_density(surface, dens_exp), width="stretch", config=CFG)
    if len(s) > 1:
        st.plotly_chart(plots.fig_surface(surface), width="stretch", config=CFG)

    st.markdown("#### Fits and static-arbitrage checks by expiry")
    show = s[["expiry", "days", "options", "forward", "atm_iv", "skew_25d_put_minus_call", "rmse_vol_pts",
              "svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma", "butterfly_min_g", "butterfly_ok",
              "calendar_ok_vs_previous", "quoted_butterfly_arbitrages"]].copy()
    st.dataframe(show.style.format({"forward": "{:.2f}", "atm_iv": "{:.2%}", "skew_25d_put_minus_call": "{:.2f}",
                                    "rmse_vol_pts": "{:.2f}", "svi_a": "{:.5f}", "svi_b": "{:.4f}",
                                    "svi_rho": "{:.3f}", "svi_m": "{:.4f}", "svi_sigma": "{:.4f}",
                                    "butterfly_min_g": "{:.4f}"}), width="stretch", hide_index=True)

    with st.expander("Quotes and Greeks for one expiry"):
        e = st.selectbox("Expiry", exps, key="greeks_exp")
        q = surface.ivs[surface.ivs["expiry"] == e].sort_values("strike").copy()
        rate = -np.log(q["df"].iloc[0]) / q["T"].iloc[0]
        g = bs_greeks(q["forward"].values, q["strike"].values, q["T"].values, rate, q["iv"].values,
                      np.where(q["type"] == "C", "call", "put"), q=rate)
        for name in ("delta", "gamma", "vega", "theta"):
            q[name] = g[name]
        q["vega"] /= 100
        q["theta"] /= 365
        st.dataframe(q[["type", "strike", "bid", "ask", "mid", "iv", "delta", "gamma", "vega", "theta"]]
                     .style.format({"iv": "{:.2%}", "delta": "{:.3f}", "gamma": "{:.5f}",
                                    "vega": "{:.3f}", "theta": "{:.3f}"}), width="stretch", hide_index=True)
        st.caption("Greeks use Black-76 on the forward at each quote's own implied vol; vega per vol point, "
                   "theta per calendar day.")

# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------
with tab_calc:
    c = st.columns(6)
    S = c[0].number_input("Spot", min_value=0.01, value=100.0, step=1.0)
    K = c[1].number_input("Strike", min_value=0.01, value=100.0, step=1.0)
    T = c[2].number_input("Years to expiry", min_value=0.0, value=1.0, step=0.05, format="%.3f")
    r = c[3].number_input("Risk-free rate", min_value=-0.05, max_value=0.5, value=0.05, step=0.005, format="%.3f")
    q = c[4].number_input("Dividend yield", min_value=0.0, max_value=0.5, value=0.0, step=0.005, format="%.3f")
    sigma = c[5].number_input("Volatility", min_value=0.0, max_value=5.0, value=0.30, step=0.01, format="%.3f")

    call, put = float(bs_price(S, K, T, r, sigma, "call", q)), float(bs_price(S, K, T, r, sigma, "put", q))
    m = st.columns(3)
    m[0].metric("Call", f"{call:,.4f}")
    m[1].metric("Put", f"{put:,.4f}")
    parity = call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))
    m[2].metric("Put-call parity residual", f"{parity:.1e}")
    gc, gp = bs_greeks(S, K, T, r, sigma, "call", q), bs_greeks(S, K, T, r, sigma, "put", q)
    greeks = pd.DataFrame({"Call": gc, "Put": gp}).T
    greeks["vega (per vol pt)"] = greeks.pop("vega") / 100
    greeks["theta (per day)"] = greeks.pop("theta") / 365
    greeks["rho (per rate pt)"] = greeks.pop("rho") / 100
    st.dataframe(greeks.style.format("{:.4f}"), width="stretch")

    st.markdown("#### Implied volatility from a price")
    i1, i2, i3 = st.columns([1, 1, 2])
    mkt = i1.number_input("Market price", min_value=0.0, value=round(call, 4), step=0.1, format="%.4f")
    kind = i2.radio("Type", ["call", "put"], horizontal=True)
    lower, upper = price_bounds(S, K, T, r, kind, q)
    iv = implied_vol(mkt, S, K, T, r, kind, q)
    if np.isfinite(iv):
        i3.success(f"Implied volatility {float(iv):.4%}")
    else:
        i3.warning(f"No volatility gives this price: a {kind} must trade strictly between "
                   f"{float(lower):.4f} and {float(upper):.4f} (no-arbitrage bounds).")

    st.markdown("#### Price heatmaps")
    h1, h2 = st.columns(2)
    spot_lo, spot_hi = h1.slider("Spot range", 1.0, 400.0, (max(1.0, 0.5 * S), 1.5 * S))
    vol_lo, vol_hi = h2.slider("Volatility range", 0.01, 1.5, (0.10, 0.60))
    st.plotly_chart(plots.fig_price_heatmaps(K, T, r, q, np.linspace(spot_lo, spot_hi, 10),
                                             np.linspace(vol_lo, vol_hi, 10)), width="stretch", config=CFG)
    if T > 0 and sigma > 0:
        st.plotly_chart(plots.fig_greeks_vs_spot(K, T, r, sigma, q, kind), width="stretch", config=CFG)

    st.markdown("#### One option from purchase to expiry")
    p1, p2, p3, p4 = st.columns(4)
    path_source = p1.radio("Underlying", ["synthetic", "live"], horizontal=True,
                           format_func={"synthetic": "Synthetic path", "live": "Real history (Yahoo)"}.get)
    hist_ticker = p2.text_input("Ticker", "AAPL", disabled=path_source != "live").strip().upper() or "AAPL"
    maturity = p3.number_input("Option life (years)", min_value=0.05, max_value=3.0, value=0.5, step=0.05)
    path_kind = p4.radio("Option", ["call", "put"], horizontal=True, key="path_kind")
    try:
        spot_hist = get_history(hist_ticker, maturity) if path_source == "live" else synthetic_path()
    except Exception:
        st.info("Live history unavailable: using a synthetic path.")
        spot_hist = synthetic_path()
    start_at = spot_hist.index[-1] - pd.Timedelta(days=round(maturity * 365))
    spot_hist = spot_hist[spot_hist.index >= start_at] if path_source == "live" else spot_hist
    path = option_through_time(spot_hist, None, maturity, r, max(sigma, 1e-4), path_kind, q)
    st.plotly_chart(plots.fig_option_through_time(path), width="stretch", config=CFG)
    st.caption(f"Bought at the money (strike {path.attrs['strike']:,.2f}) on {path.index[0]:%d %b %Y}, "
               f"expiring {path.attrs['expiry']:%d %b %Y}; constant volatility {sigma:.0%}. Time to expiry "
               "falls to zero along the chart, so the option's time value decays into its intrinsic value.")

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(r"""
### From quotes to a surface

1. **Clean.** Keep two-sided quotes (bid above zero, ask above bid) with a
   relative spread under 50%, inside the chosen maturity window. Use mid prices.
2. **Forward and discount factor per expiry.** Put-call parity says
   $C - P = DF\,(F - K)$ for every strike, so a straight-line fit of $C - P$
   against $K$ near the money gives $DF$ (minus the slope) and $F$. No rate or
   dividend assumption is needed.
3. **Implied volatilities** of out-of-the-money options (puts below the
   forward, calls above) by Black-76, with Newton steps inside a shrinking
   bisection bracket.
4. **SVI per expiry.** Total variance $w = \sigma^2 T$ against log-moneyness
   $k = \ln(K/F)$:
   $w(k) = a + b\left(\rho (k - m) + \sqrt{(k-m)^2 + \sigma^2}\right)$.
   Fitted in volatility units with a robust loss; the vertex $m$ is kept
   inside the quoted range and the fit is penalised for negative density.
5. **Static-arbitrage checks.**
   *Butterfly:* the risk-neutral density is proportional to
   $g(k) = \left(1 - \frac{k w'}{2w}\right)^2 - \frac{w'^2}{4}\left(\frac{1}{w} + \frac14\right) + \frac{w''}{2}$,
   which must be non-negative. Quoted butterflies are also checked at
   executable prices (wings at the ask, body at the bid).
   *Calendar:* total variance must not fall with maturity at any $k$; drops
   worth less than 0.1 vol points are treated as fitting noise.

### The synthetic chain
Prices from a power-law SSVI surface (Gatheral and Jacquier, 2014), which is
arbitrage-free, plus quote noise and a spread. Because the answer is known,
the tests check that the pipeline recovers it, and that the checks flag
arbitrage injected on purpose.
""")
