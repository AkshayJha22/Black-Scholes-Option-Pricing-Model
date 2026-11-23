"""Plotly figures for the explorer and the calculator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .bs import bs_greeks, bs_price
from .svi import risk_neutral_density, svi_implied_vol, svi_total_variance

PALETTE = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2", "#be185d", "#65a30d"]


def _layout(fig, height=430, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      title=title, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
    return fig


def fig_smiles(surface, expiries: list) -> go.Figure:
    """Market implied vols (dots) and SVI fits (lines) against log-moneyness."""
    fig = go.Figure()
    for i, exp in enumerate(expiries):
        fit = surface.fits[exp]
        pts = surface.ivs[surface.ivs["expiry"] == exp]
        color = PALETTE[i % len(PALETTE)]
        days = int(round(fit.T * 365))
        fig.add_trace(go.Scatter(x=pts["k"], y=100 * pts["iv"], mode="markers",
                                 marker=dict(size=5, color=color, opacity=0.55),
                                 name=f"{exp} ({days}d) quotes", legendgroup=exp, showlegend=False))
        k = np.linspace(*fit.k_range, 200)
        fig.add_trace(go.Scatter(x=k, y=100 * fit.implied_vol(k), mode="lines",
                                 line=dict(color=color, width=2), name=f"{days}d", legendgroup=exp))
    fig.update_xaxes(title="log-moneyness k = ln(K / F)")
    fig.update_yaxes(title="implied volatility (%)")
    return _layout(fig, 460, "Implied volatility smiles: quotes and SVI fits")


def fig_surface(surface, kind: str = "iv") -> go.Figure:
    """The fitted surface on a grid of standardised moneyness and maturity (3-D)."""
    fits = [f for f in sorted(surface.fits.values(), key=lambda f: f.T) if f.T * 365 >= 7] or \
        sorted(surface.fits.values(), key=lambda f: f.T)
    T = np.array([f.T for f in fits])
    x = np.linspace(-2.0, 1.5, 36)          # k in units of ATM standard deviations
    Z, K = [], []
    for f in fits:
        sd = float(f.implied_vol(0.0)) * np.sqrt(f.T)
        k = x * sd
        z = 100 * f.implied_vol(k) if kind == "iv" else svi_total_variance(k, f.params)
        z = np.where((k >= f.k_range[0]) & (k <= f.k_range[1]), z, np.nan)   # no extrapolation
        Z.append(z)
        K.append(k)
    fig = go.Figure(go.Surface(x=np.array(K), y=np.tile(365 * T[:, None], (1, len(x))), z=np.array(Z),
                               colorscale="Viridis", showscale=True,
                               colorbar=dict(title="IV %" if kind == "iv" else "w")))
    fig.update_layout(scene=dict(xaxis_title="k = ln(K/F)", yaxis_title="days",
                                 zaxis_title="implied vol (%)" if kind == "iv" else "total variance"))
    return _layout(fig, 560, "Fitted volatility surface")


def fig_term_structure(surface) -> go.Figure:
    s = surface.summary
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=s["days"], y=100 * s["atm_iv"], mode="lines+markers", name="ATM implied vol",
                             line=dict(color=PALETTE[0])), secondary_y=False)
    fig.add_trace(go.Bar(x=s["days"], y=s["skew_25d_put_minus_call"], name="25-delta put minus call (vol pts)",
                         marker_color="rgba(217,119,6,0.45)"), secondary_y=True)
    fig.update_xaxes(title="days to expiry")
    fig.update_yaxes(title_text="ATM implied vol (%)", secondary_y=False)
    fig.update_yaxes(title_text="skew (vol points)", secondary_y=True)
    return _layout(fig, 400, "Term structure of ATM volatility and skew")


def fig_density(surface, expiry: str) -> go.Figure:
    fit = surface.fits[expiry]
    lo, hi = fit.k_range
    k = np.linspace(lo, hi, 400)
    dens = risk_neutral_density(k, fit.T, fit.params)
    fig = go.Figure(go.Scatter(x=k, y=dens, mode="lines", fill="tozeroy", line=dict(color=PALETTE[2])))
    fig.update_xaxes(title="log-moneyness k")
    fig.update_yaxes(title="risk-neutral density of k")
    return _layout(fig, 330, f"Risk-neutral density implied by the {expiry} smile")


def fig_price_heatmaps(K, T, r, q, spots, vols) -> go.Figure:
    """Call and put prices over a spot x volatility grid (version 1's heatmaps, vectorised)."""
    S, V = np.meshgrid(spots, vols)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Call price", "Put price"), horizontal_spacing=0.08)
    for col, kind in ((1, "call"), (2, "put")):
        Z = bs_price(S, K, T, r, V, kind, q)
        fig.add_trace(go.Heatmap(x=np.round(spots, 2), y=np.round(100 * vols, 1), z=Z, colorscale="Viridis",
                                 text=np.round(Z, 2), texttemplate="%{text}", textfont=dict(size=10),
                                 showscale=False), row=1, col=col)
    fig.update_xaxes(title_text="spot")
    fig.update_yaxes(title_text="volatility (%)", col=1)
    return _layout(fig, 470)


def fig_greeks_vs_spot(K, T, r, sigma, q, kind) -> go.Figure:
    spots = np.linspace(0.6 * K, 1.4 * K, 161)
    g = bs_greeks(spots, K, T, r, sigma, kind, q)
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Delta", "Gamma", "Vega (per vol pt)", "Theta (per day)"))
    series = [g["delta"], g["gamma"], g["vega"] / 100, g["theta"] / 365]
    for i, y in enumerate(series, start=1):
        fig.add_trace(go.Scatter(x=spots, y=y, mode="lines", line=dict(color=PALETTE[i - 1]), showlegend=False),
                      row=1, col=i)
        fig.add_vline(x=K, line=dict(color="#9ca3af", dash="dot", width=1), row=1, col=i)
    return _layout(fig, 300)


def fig_option_through_time(path: pd.DataFrame) -> go.Figure:
    """Stock price and the option's model value as it ages to expiry."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=path.index, y=path["spot"], name="stock", line=dict(color="#6b7280")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=path.index, y=path["option"], name="option value", line=dict(color=PALETTE[0], width=2)),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=path.index, y=path["intrinsic"], name="intrinsic value",
                             line=dict(color=PALETTE[1], dash="dot")), secondary_y=True)
    fig.update_yaxes(title_text="stock", secondary_y=False)
    fig.update_yaxes(title_text="option", secondary_y=True)
    return _layout(fig, 420, "One option from purchase to expiry")
