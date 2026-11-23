"""Snapshot a real listed option chain on GitHub's runners.

Yahoo Finance is reachable from GitHub Actions, so
.github/workflows/chain-snapshot.yml runs this script and commits the result.
The explorer can then show a real surface offline, and the tests stay offline.

Writes data/<TICKER>_chain_snapshot.csv, one row per listed option:
  expiry, days, type (C/P), strike, bid, ask, last, volume, open_interest,
  yahoo_iv, spot, as_of_utc
and data/<TICKER>_chain_meta.json (spot, 13-week T-bill yield, time of capture).

Usage:  python scripts/snapshot_chain.py [--ticker SPY] [--max-days 400]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import yfinance as yf

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--max-days", type=int, default=400)
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    tk = yf.Ticker(args.ticker)
    spot = float(tk.history(period="5d")["Close"].dropna().iloc[-1])
    irx = yf.Ticker("^IRX").history(period="10d")["Close"].dropna()
    rate = float(irx.iloc[-1]) / 100.0 if len(irx) else None

    rows = []
    for exp in tk.options:
        days = (pd.Timestamp(exp).date() - now.date()).days
        if days < 1 or days > args.max_days:
            continue
        for attempt in range(3):
            try:
                chain = tk.option_chain(exp)
                break
            except Exception as exc:  # transient network errors
                print(f"  {exp}: {exc} (attempt {attempt + 1})")
                time.sleep(3 * (attempt + 1))
        else:
            continue
        for kind, df in (("C", chain.calls), ("P", chain.puts)):
            for _, r in df.iterrows():
                rows.append({
                    "expiry": exp, "days": days, "type": kind,
                    "strike": float(r["strike"]),
                    "bid": float(r.get("bid", float("nan"))),
                    "ask": float(r.get("ask", float("nan"))),
                    "last": float(r.get("lastPrice", float("nan"))),
                    "volume": float(r.get("volume", float("nan"))),
                    "open_interest": float(r.get("openInterest", float("nan"))),
                    "yahoo_iv": float(r.get("impliedVolatility", float("nan"))),
                })
        time.sleep(0.5)

    out = pd.DataFrame(rows)
    out["spot"] = round(spot, 4)
    out["as_of_utc"] = now.strftime("%Y-%m-%d %H:%M")
    path = ROOT / "data" / f"{args.ticker}_chain_snapshot.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    meta = {"ticker": args.ticker, "spot": spot, "tbill_13w": rate,
            "as_of_utc": now.strftime("%Y-%m-%d %H:%M"),
            "expiries": int(out["expiry"].nunique()) if len(out) else 0,
            "options": int(len(out))}
    (path.parent / f"{args.ticker}_chain_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
