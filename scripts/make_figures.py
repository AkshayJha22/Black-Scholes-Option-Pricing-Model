"""Fit the bundled SPY chain and write the README's figures and table.

Writes results/spy_surface_summary.csv, results/spy_smiles.png,
results/spy_term_structure.png and results/spy_surface.png.

Usage:  python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ivsurface.chain import build_surface, load_snapshot  # noqa: E402

OUT = ROOT / "results"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chain, meta = load_snapshot("SPY")
    surface = build_surface(chain)
    s = surface.summary
    OUT.mkdir(exist_ok=True)
    s.to_csv(OUT / "spy_surface_summary.csv", index=False, float_format="%.6f")

    colors = plt.cm.viridis(np.linspace(0, 0.9, 6))
    picks = [s["expiry"].iloc[i] for i in np.unique(np.linspace(0, len(s) - 1, 6).astype(int))]
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, exp in zip(colors, picks):
        fit = surface.fits[exp]
        pts = surface.ivs[surface.ivs["expiry"] == exp]
        days = int(round(fit.T * 365))
        ax.scatter(pts["k"], 100 * pts["iv"], s=8, color=c, alpha=0.45)
        k = np.linspace(*fit.k_range, 200)
        ax.plot(k, 100 * fit.implied_vol(k), color=c, lw=1.8, label=f"{exp} ({days} days)")
    ax.set_xlabel("log-moneyness k = ln(K / F)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(f"SPY implied volatility smiles, {meta.get('as_of_utc', '')} UTC: quotes (dots) and SVI fits")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "spy_smiles.png", dpi=140)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(s["days"], 100 * s["atm_iv"], "o-", color="#2563eb", label="ATM implied vol")
    ax1.set_xlabel("days to expiry")
    ax1.set_ylabel("ATM implied vol (%)", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.bar(s["days"], s["skew_25d_put_minus_call"], width=4, color="#d97706", alpha=0.45,
            label="25-delta put minus call")
    ax2.set_ylabel("25-delta put minus call (vol pts)", color="#b45309")
    ax1.set_title("SPY term structure of ATM volatility and skew")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "spy_term_structure.png", dpi=140)
    plt.close(fig)

    fits = [f for f in sorted(surface.fits.values(), key=lambda f: f.T) if f.T * 365 >= 7]
    x = np.linspace(-2.0, 1.5, 36)      # k in ATM standard deviations, inside every quoted range
    K, D, Z = [], [], []
    for f in fits:
        sd = float(f.implied_vol(0.0)) * np.sqrt(f.T)
        K.append(x * sd)
        D.append(np.full_like(x, 365 * f.T))
        Z.append(100 * f.implied_vol(x * sd))
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(np.array(K), np.array(D), np.array(Z), cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("k = ln(K/F)")
    ax.set_ylabel("days")
    ax.set_zlabel("implied vol (%)")
    ax.view_init(elev=24, azim=-130)
    ax.set_title("Fitted SPY volatility surface")
    fig.tight_layout()
    fig.savefig(OUT / "spy_surface.png", dpi=140)
    plt.close(fig)

    stats = {
        "as_of_utc": meta.get("as_of_utc"), "spot": meta.get("spot"),
        "quotes_in_snapshot": int(len(chain)), "quotes_used": int(s["options"].sum()),
        "expiries_fitted": int(len(s)), "atm_iv_30d": float(np.interp(30, s["days"], s["atm_iv"])),
        "median_rmse_vol_pts": float(s["rmse_vol_pts"].median()),
        "max_rmse_vol_pts": float(s["rmse_vol_pts"].max()),
        "skew_25d_range": [float(s["skew_25d_put_minus_call"].min()), float(s["skew_25d_put_minus_call"].max())],
        "butterfly_ok": int(s["butterfly_ok"].sum()),
        "calendar_flags": [{"days": int(round(c["T2"] * 365)), "drop_vol_pts": round(c["drop_vol_pts"], 3)}
                           for c in surface.calendar if not c["arbitrage_free"]],
        "quoted_butterfly_arbitrages": int(s["quoted_butterfly_arbitrages"].sum()),
        "expiries_with_quoted_arbitrage": [int(d) for d in s.loc[s["quoted_butterfly_arbitrages"] > 0, "days"]],
    }
    (OUT / "spy_surface_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
