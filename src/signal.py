cat > src/signal.py << 'ENDOFFILE'
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
                          detect_anomalies)
from src.correlation import compute_rolling_correlation

SPACE_TICKERS = ['RKLB', 'ASTS', 'LUNR']


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
    anomalies = detect_anomalies(vol_z, price_z)
    corr      = compute_rolling_correlation(returns)

    print(f"\n── Portfolio Snapshot ──────────────────────────────")
    for ticker in SPACE_TICKERS:
        if ticker not in prices.columns:
            continue
        price    = prices[ticker].iloc[-1]
        ret_1d   = returns[ticker].iloc[-1]
        ret_1w   = prices[ticker].pct_change(5).iloc[-1]
        beta     = betas[ticker].iloc[-1] if ticker in betas.columns else np.nan
        vz       = vol_z[ticker].iloc[-1] if ticker in vol_z.columns else np.nan
        pz       = price_z[ticker].iloc[-1] if ticker in price_z.columns else np.nan
        spy_corr = corr.get(f"{ticker}_vs_SPY", pd.Series()).iloc[-1] if f"{ticker}_vs_SPY" in corr else np.nan

        flag = ""
        if abs(vz) >= 2 and abs(pz) >= 2:
            flag = "  ⚠️  ANOMALY DETECTED"
        elif abs(pz) >= 1.5:
            flag = "  👀 Watch"

        print(f"\n  {ticker} — ${price:.2f}{flag}")
        print(f"    1D return:    {ret_1d:.1%}")
        print(f"    1W return:    {ret_1w:.1%}")
        print(f"    Beta (60d):   {beta:.2f}")
        print(f"    Vol z-score:  {vz:.2f}")
        print(f"    Price z-score:{pz:.2f}")
        print(f"    SPY corr:     {spy_corr:.2f}")

    print(f"\n── Today's Alerts ──────────────────────────────────")
    today_anomalies = anomalies[
        pd.to_datetime(anomalies['date']) >=
        pd.Timestamp.today() - pd.Timedelta(days=3)
    ] if not anomalies.empty else pd.DataFrame()

    if today_anomalies.empty:
        print("  No anomalies in last 3 days — market quiet")
    else:
        for _, row in today_anomalies.iterrows():
            print(f"  {row['date']} {row['ticker']}: "
                  f"vol_z={row['vol_zscore']} "
                  f"price_z={row['price_zscore']} "
                  f"direction={row['direction']}")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    run_signal()
ENDOFFILE