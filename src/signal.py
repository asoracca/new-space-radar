"""
signal.py
---------
Daily scan — which new space stock is showing unusual activity today.
Run every morning for a quick portfolio check.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import date

from src.data import (fetch_price_history, fetch_volume_history,
                      compute_returns, compute_beta)
from src.momentum import (compute_volume_zscore, compute_price_zscore,
                           compute_rsi, detect_anomalies)
from src.correlation import compute_rolling_correlation

SPACE_TICKERS = ['RKLB', 'ASTS', 'LUNR']


def _signal_label(vz: float, pz: float, rsi: float) -> str:
    if abs(vz) >= 2 and abs(pz) >= 2:
        direction = "⬆️  SPIKE UP" if pz > 0 else "⬇️  SPIKE DOWN"
        return f"⚠️  ANOMALY — {direction}"
    if rsi > 70:
        return "🔴 Overbought — RSI > 70"
    if rsi < 30:
        return "🟢 Oversold — RSI < 30"
    if abs(pz) >= 1.5:
        return "👀 Elevated move — watch for catalyst"
    return "— Normal"


def run_signal():
    print(f"\n{'='*55}")
    print(f"  NEW SPACE RADAR — {date.today()}")
    print(f"{'='*55}")

    prices  = fetch_price_history(period="1y")
    volume  = fetch_volume_history(period="1y")
    returns = compute_returns(prices)
    betas   = compute_beta(returns)

    vol_z     = compute_volume_zscore(volume)
    price_z   = compute_price_zscore(returns)
    rsi       = compute_rsi(prices)
    anomalies = detect_anomalies(vol_z, price_z)
    corr      = compute_rolling_correlation(returns)

    print(f"\n── Portfolio Snapshot ──────────────────────────────")
    for ticker in SPACE_TICKERS:
        if ticker not in prices.columns:
            continue

        price  = prices[ticker].iloc[-1]
        ret_1d = returns[ticker].iloc[-1]
        ret_1w = prices[ticker].pct_change(5).iloc[-1]
        ret_1m = prices[ticker].pct_change(21).iloc[-1]
        beta   = betas[ticker].iloc[-1] if ticker in betas.columns else np.nan
        vz     = vol_z[ticker].iloc[-1]  if ticker in vol_z.columns  else np.nan
        pz     = price_z[ticker].iloc[-1] if ticker in price_z.columns else np.nan
        r      = rsi[ticker].iloc[-1]    if ticker in rsi.columns     else np.nan
        spy_key = f"{ticker}_vs_SPY"
        spy_corr = corr[spy_key].dropna().iloc[-1] if spy_key in corr else np.nan

        label = _signal_label(vz, pz, r)

        print(f"\n  {ticker} — ${price:.2f}")
        print(f"    Signal:       {label}")
        print(f"    1D / 1W / 1M: {ret_1d:.1%} / {ret_1w:.1%} / {ret_1m:.1%}")
        print(f"    Beta (60d):   {beta:.2f}")
        print(f"    Vol z-score:  {vz:+.2f}")
        print(f"    Price z-score:{pz:+.2f}")
        print(f"    RSI (14d):    {r:.1f}")
        print(f"    SPY corr:     {spy_corr:.2f}")

    print(f"\n── Alerts (last 3 days) ────────────────────────────")
    if not anomalies.empty:
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=3)
        recent = anomalies[pd.to_datetime(anomalies['date']) >= cutoff]
        if not recent.empty:
            for _, row in recent.iterrows():
                print(f"  {row['date']} {row['ticker']}: "
                      f"vol_z={row['vol_zscore']:+.2f} "
                      f"price_z={row['price_zscore']:+.2f} "
                      f"direction={row['direction']}")
        else:
            print("  No anomalies in last 3 days — market quiet")
    else:
        print("  No anomalies in dataset — market quiet")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    run_signal()