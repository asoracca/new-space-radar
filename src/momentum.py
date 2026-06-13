"""
momentum.py
-----------
Detect abnormal volume and price momentum around events.
Z-score based approach — flags moves more than 2 std devs from normal.
Also computes RSI and rolling Sharpe as secondary signals.
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


def compute_volume_zscore(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    zscores = pd.DataFrame(index=volume.index)
    for ticker in SPACE_TICKERS:
        if ticker not in volume.columns:
            continue
        roll_mean     = volume[ticker].rolling(window).mean()
        roll_std      = volume[ticker].rolling(window).std()
        zscores[ticker] = (volume[ticker] - roll_mean) / roll_std
    return zscores.dropna()


def compute_price_zscore(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    zscores = pd.DataFrame(index=returns.index)
    for ticker in SPACE_TICKERS:
        if ticker not in returns.columns:
            continue
        roll_mean     = returns[ticker].rolling(window).mean()
        roll_std      = returns[ticker].rolling(window).std()
        zscores[ticker] = (returns[ticker] - roll_mean) / roll_std
    return zscores.dropna()


def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """RSI momentum indicator — overbought >70, oversold <30."""
    rsi_df = pd.DataFrame(index=prices.index)
    for ticker in SPACE_TICKERS:
        if ticker not in prices.columns:
            continue
        delta   = prices[ticker].diff()
        gain    = delta.clip(lower=0).rolling(window).mean()
        loss    = (-delta.clip(upper=0)).rolling(window).mean()
        rs      = gain / loss.replace(0, np.nan)
        rsi_df[ticker] = 100 - (100 / (1 + rs))
    return rsi_df.dropna()


def compute_rolling_sharpe(returns: pd.DataFrame,
                            window: int = 60,
                            rf_annual: float = 0.05) -> pd.DataFrame:
    """Rolling annualized Sharpe ratio."""
    rf_daily = rf_annual / 252
    sharpe_df = pd.DataFrame(index=returns.index)
    for ticker in SPACE_TICKERS:
        if ticker not in returns.columns:
            continue
        excess          = returns[ticker] - rf_daily
        sharpe_df[ticker] = (excess.rolling(window).mean() /
                              excess.rolling(window).std()) * np.sqrt(252)
    return sharpe_df.dropna()


def detect_anomalies(vol_zscore: pd.DataFrame,
                     price_zscore: pd.DataFrame,
                     vol_threshold: float = 2.0,
                     price_threshold: float = 2.0) -> pd.DataFrame:
    anomalies  = []
    common_idx = vol_zscore.index.intersection(price_zscore.index)

    for ticker in SPACE_TICKERS:
        if ticker not in vol_zscore.columns or ticker not in price_zscore.columns:
            continue
        v = vol_zscore.loc[common_idx, ticker]
        p = price_zscore.loc[common_idx, ticker]
        both_flags = (v.abs() >= vol_threshold) & (p.abs() >= price_threshold)

        for date in both_flags[both_flags].index:
            anomalies.append({
                'date':         date.date(),
                'ticker':       ticker,
                'vol_zscore':   round(v.loc[date], 2),
                'price_zscore': round(p.loc[date], 2),
                'direction':    'UP' if p.loc[date] > 0 else 'DOWN',
                'signal':       'ANOMALY — investigate catalyst',
            })

    df = pd.DataFrame(anomalies)
    if not df.empty:
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
    return df


def print_momentum_summary(vol_z: pd.DataFrame,
                           price_z: pd.DataFrame,
                           anomalies: pd.DataFrame,
                           rsi: pd.DataFrame = None):
    print("\n" + "="*55)
    print("  MOMENTUM & ANOMALY DETECTION")
    print("="*55)

    print("\n  Current z-scores:")
    for ticker in SPACE_TICKERS:
        if ticker in vol_z.columns and ticker in price_z.columns:
            vz   = vol_z[ticker].iloc[-1]
            pz   = price_z[ticker].iloc[-1]
            flag = "  ⚠️  ANOMALY" if (abs(vz) >= 2 and abs(pz) >= 2) else ""
            rsi_str = ""
            if rsi is not None and ticker in rsi.columns:
                r = rsi[ticker].iloc[-1]
                rsi_str = f"  RSI={r:.0f}"
                if r > 70:
                    rsi_str += " 🔴 overbought"
                elif r < 30:
                    rsi_str += " 🟢 oversold"
            print(f"    {ticker}: vol_z={vz:+.2f}  price_z={pz:+.2f}{rsi_str}{flag}")

    print(f"\n  Recent anomalies (last 30 days):")
    if anomalies.empty:
        print("    None detected")
    else:
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
        recent = anomalies[pd.to_datetime(anomalies['date']) >= cutoff]
        if recent.empty:
            print("    None in last 30 days")
        else:
            print(recent[['date', 'ticker', 'vol_zscore',
                           'price_zscore', 'direction']].to_string(index=False))
    print("="*55)


def plot_momentum(vol_zscore: pd.DataFrame,
                  price_zscore: pd.DataFrame,
                  anomalies: pd.DataFrame,
                  rsi: pd.DataFrame = None,
                  save=True):
    n_rows = 4 if rsi is not None else 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    fig.suptitle('New Space — Volume & Price Anomaly Detection',
                 fontsize=14, fontweight='bold')

    colors = {'RKLB': '#1D9E75', 'ASTS': '#378ADD', 'LUNR': '#7F77DD'}

    for i, ticker in enumerate(SPACE_TICKERS):
        if ticker not in vol_zscore.columns:
            continue

        # Volume z-score
        axes[i, 0].plot(vol_zscore.index, vol_zscore[ticker],
                        color=colors[ticker], linewidth=1, alpha=0.8)
        axes[i, 0].axhline( 2, color='red', linestyle='--', alpha=0.5, label='±2σ')
        axes[i, 0].axhline(-2, color='red', linestyle='--', alpha=0.5)
        axes[i, 0].axhline( 0, color='black', linestyle='--', alpha=0.3)
        axes[i, 0].fill_between(vol_zscore.index, vol_zscore[ticker],
                                where=(vol_zscore[ticker].abs() >= 2),
                                color='red', alpha=0.2, label='Anomaly')
        axes[i, 0].set_title(f'{ticker} Volume Z-Score', fontweight='bold')
        axes[i, 0].set_ylabel('Z-Score')
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].grid(True, alpha=0.25)

        # Price z-score
        axes[i, 1].plot(price_zscore.index, price_zscore[ticker],
                        color=colors[ticker], linewidth=1, alpha=0.8)
        axes[i, 1].axhline( 2, color='red', linestyle='--', alpha=0.5)
        axes[i, 1].axhline(-2, color='red', linestyle='--', alpha=0.5)
        axes[i, 1].axhline( 0, color='black', linestyle='--', alpha=0.3)
        axes[i, 1].fill_between(price_zscore.index, price_zscore[ticker],
                                where=(price_zscore[ticker].abs() >= 2),
                                color='orange', alpha=0.2, label='Anomaly')
        axes[i, 1].set_title(f'{ticker} Price Move Z-Score', fontweight='bold')
        axes[i, 1].set_ylabel('Z-Score')
        axes[i, 1].legend(fontsize=8)
        axes[i, 1].grid(True, alpha=0.25)

    # Optional RSI row
    if rsi is not None and n_rows > 3:
        for i, ticker in enumerate(SPACE_TICKERS):
            if ticker not in rsi.columns:
                continue
            col = i % 2
            row = 3
            if i == 0:
                ax = axes[row, 0]
            elif i == 1:
                ax = axes[row, 1]
            else:
                # merge into a third subplot – just reuse col 0 with all tickers
                break

        ax_rsi = axes[3, 0]
        ax_rsi2 = axes[3, 1]
        for ticker in SPACE_TICKERS:
            if ticker in rsi.columns:
                ax_rsi.plot(rsi.index, rsi[ticker],
                            label=ticker, color=colors[ticker], linewidth=1.5)
        ax_rsi.axhline(70, color='red',   linestyle='--', alpha=0.5, label='Overbought (70)')
        ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax_rsi.set_title('RSI (14-day)', fontweight='bold')
        ax_rsi.set_ylabel('RSI')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend(fontsize=9)
        ax_rsi.grid(True, alpha=0.25)
        ax_rsi2.set_visible(False)

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/momentum_anomalies.png", dpi=150, bbox_inches='tight')
        print("Saved: data/momentum_anomalies.png")
    plt.close()


if __name__ == "__main__":
    from src.data import fetch_price_history, fetch_volume_history, compute_returns
    prices    = fetch_price_history()
    volume    = fetch_volume_history()
    returns   = compute_returns(prices)
    vol_z     = compute_volume_zscore(volume)
    price_z   = compute_price_zscore(returns)
    rsi       = compute_rsi(prices)
    anomalies = detect_anomalies(vol_z, price_z)
    print_momentum_summary(vol_z, price_z, anomalies, rsi)
    plot_momentum(vol_z, price_z, anomalies, rsi)