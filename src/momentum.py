cat > src/momentum.py << 'ENDOFFILE'
"""
momentum.py
-----------
Detect abnormal volume and price momentum around events.
Z-score based approach — flags moves more than 2 std devs from normal.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SPACE_TICKERS = ['RKLB', 'ASTS', 'LUNR']


def compute_volume_zscore(volume: pd.DataFrame,
                          window: int = 20) -> pd.DataFrame:
    """
    Z-score of volume vs rolling mean.
    Z > 2 = unusually high volume = something is happening.
    """
    zscores = pd.DataFrame(index=volume.index)
    for ticker in SPACE_TICKERS:
        if ticker not in volume.columns:
            continue
        roll_mean = volume[ticker].rolling(window).mean()
        roll_std  = volume[ticker].rolling(window).std()
        zscores[ticker] = (volume[ticker] - roll_mean) / roll_std
    return zscores.dropna()


def compute_price_zscore(returns: pd.DataFrame,
                         window: int = 20) -> pd.DataFrame:
    """
    Z-score of daily return vs rolling distribution.
    Z > 2 = unusually large move = potential catalyst.
    """
    zscores = pd.DataFrame(index=returns.index)
    for ticker in SPACE_TICKERS:
        if ticker not in returns.columns:
            continue
        roll_mean = returns[ticker].rolling(window).mean()
        roll_std  = returns[ticker].rolling(window).std()
        zscores[ticker] = (returns[ticker] - roll_mean) / roll_std
    return zscores.dropna()


def detect_anomalies(vol_zscore: pd.DataFrame,
                     price_zscore: pd.DataFrame,
                     vol_threshold: float = 2.0,
                     price_threshold: float = 2.0) -> pd.DataFrame:
    """
    Flag days where BOTH volume AND price move are anomalous.
    Both firing together = high confidence something real happened.
    """
    anomalies = []
    common_idx = vol_zscore.index.intersection(price_zscore.index)

    for ticker in SPACE_TICKERS:
        if ticker not in vol_zscore.columns:
            continue
        if ticker not in price_zscore.columns:
            continue

        v = vol_zscore.loc[common_idx, ticker]
        p = price_zscore.loc[common_idx, ticker]

        both_flags = (v.abs() >= vol_threshold) & (p.abs() >= price_threshold)

        for date in both_flags[both_flags].index:
            anomalies.append({
                'date':          date.date(),
                'ticker':        ticker,
                'vol_zscore':    round(v.loc[date], 2),
                'price_zscore':  round(p.loc[date], 2),
                'direction':     'UP' if p.loc[date] > 0 else 'DOWN',
                'signal':        'ANOMALY — investigate catalyst'
            })

    df = pd.DataFrame(anomalies)
    if not df.empty:
        df = df.sort_values('date', ascending=False)
    return df


def print_momentum_summary(vol_z: pd.DataFrame,
                           price_z: pd.DataFrame,
                           anomalies: pd.DataFrame):
    print("\n" + "="*55)
    print("  MOMENTUM & ANOMALY DETECTION")
    print("="*55)

    print("\n  Current z-scores (today):")
    for ticker in SPACE_TICKERS:
        if ticker in vol_z.columns and ticker in price_z.columns:
            vz = vol_z[ticker].iloc[-1]
            pz = price_z[ticker].iloc[-1]
            flag = " ⚠️ ANOMALY" if (abs(vz) >= 2 and abs(pz) >= 2) else ""
            print(f"    {ticker}: vol_z={vz:.2f}  price_z={pz:.2f}{flag}")

    print(f"\n  Recent anomalies (last 30 days):")
    if anomalies.empty:
        print("    None detected")
    else:
        recent = anomalies[
            pd.to_datetime(anomalies['date']) >=
            pd.Timestamp.today() - pd.Timedelta(days=30)
        ]
        if recent.empty:
            print("    None in last 30 days")
        else:
            print(recent[['date', 'ticker', 'vol_zscore',
                          'price_zscore', 'direction']].to_string(index=False))
    print("="*55)


def plot_momentum(vol_zscore: pd.DataFrame,
                  price_zscore: pd.DataFrame,
                  anomalies: pd.DataFrame,
                  save=True):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('New Space — Volume & Price Anomaly Detection',
                 fontsize=14, fontweight='bold')

    colors = {'RKLB': '#1D9E75', 'ASTS': '#378ADD', 'LUNR': '#7F77DD'}

    for i, ticker in enumerate(SPACE_TICKERS):
        if ticker not in vol_zscore.columns:
            continue

        # Volume z-score
        axes[i, 0].plot(vol_zscore.index, vol_zscore[ticker],
                        color=colors[ticker], linewidth=1, alpha=0.8)
        axes[i, 0].axhline(2,  color='red', linestyle='--',
                           alpha=0.5, label='±2σ threshold')
        axes[i, 0].axhline(-2, color='red', linestyle='--', alpha=0.5)
        axes[i, 0].axhline(0,  color='black', linestyle='--', alpha=0.3)
        axes[i, 0].fill_between(vol_zscore.index, vol_zscore[ticker],
                                where=(vol_zscore[ticker].abs() >= 2),
                                color='red', alpha=0.2, label='Anomaly')
        axes[i, 0].set_title(f'{ticker} Volume Z-Score')
        axes[i, 0].set_ylabel('Z-Score')
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].grid(True, alpha=0.25)

        # Price z-score
        axes[i, 1].plot(price_zscore.index, price_zscore[ticker],
                        color=colors[ticker], linewidth=1, alpha=0.8)
        axes[i, 1].axhline(2,  color='red', linestyle='--', alpha=0.5)
        axes[i, 1].axhline(-2, color='red', linestyle='--', alpha=0.5)
        axes[i, 1].axhline(0,  color='black', linestyle='--', alpha=0.3)
        axes[i, 1].fill_between(price_zscore.index, price_zscore[ticker],
                                where=(price_zscore[ticker].abs() >= 2),
                                color='orange', alpha=0.2)
        axes[i, 1].set_title(f'{ticker} Price Move Z-Score')
        axes[i, 1].set_ylabel('Z-Score')
        axes[i, 1].grid(True, alpha=0.25)

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/momentum_anomalies.png",
                    dpi=150, bbox_inches='tight')
        print("Saved: data/momentum_anomalies.png")


if __name__ == "__main__":
    from src.data import (fetch_price_history, fetch_volume_history,
                          compute_returns)
    prices  = fetch_price_history()
    volume  = fetch_volume_history()
    returns = compute_returns(prices)
    vol_z   = compute_volume_zscore(volume)
    price_z = compute_price_zscore(returns)
    anomalies = detect_anomalies(vol_z, price_z)
    print_momentum_summary(vol_z, price_z, anomalies)
    plot_momentum(vol_z, price_z, anomalies)
ENDOFFILE