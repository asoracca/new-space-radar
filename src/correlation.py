"""
correlation.py
--------------
Rolling correlation analysis for RKLB, ASTS, LUNR vs each other and SPY.
When correlation with SPY drops, a stock is moving on its own catalyst.
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
PAIRS = [
    ('RKLB', 'SPY'),
    ('ASTS', 'SPY'),
    ('LUNR', 'SPY'),
    ('RKLB', 'ASTS'),
    ('RKLB', 'LUNR'),
    ('ASTS', 'LUNR'),
]


def compute_rolling_correlation(returns: pd.DataFrame,
                                window: int = 30) -> dict:
    correlations = {}
    for t1, t2 in PAIRS:
        if t1 in returns.columns and t2 in returns.columns:
            key = f"{t1}_vs_{t2}"
            correlations[key] = returns[t1].rolling(window).corr(returns[t2])
    return correlations


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Full correlation matrix for the last 60 trading days."""
    tickers = [t for t in SPACE_TICKERS + ['SPY'] if t in returns.columns]
    return returns[tickers].tail(60).corr()


def detect_correlation_breaks(correlations: dict,
                               threshold: float = 0.3) -> pd.DataFrame:
    breaks = []
    spy_pairs = {k: v for k, v in correlations.items() if 'SPY' in k}

    for pair, corr_series in spy_pairs.items():
        ticker    = pair.replace('_vs_SPY', '')
        low_corr  = corr_series[corr_series < threshold]

        for date, corr_val in low_corr.items():
            breaks.append({
                'date':        date.date(),
                'ticker':      ticker,
                'correlation': round(corr_val, 3),
                'signal':      'LOW CORR — possible catalyst',
            })

    return pd.DataFrame(breaks).drop_duplicates()


def print_correlation_summary(correlations: dict, returns: pd.DataFrame):
    print("\n" + "="*55)
    print("  CORRELATION ANALYSIS")
    print("="*55)

    print("\n  Current 30-day rolling correlations vs SPY:")
    for ticker in SPACE_TICKERS:
        key = f"{ticker}_vs_SPY"
        if key in correlations and len(correlations[key].dropna()) > 0:
            current = correlations[key].dropna().iloc[-1]
            avg     = correlations[key].mean()
            flag    = "  ← LOW (catalyst?)" if current < 0.3 else ""
            print(f"    {ticker}: {current:.2f}  (1Y avg: {avg:.2f}){flag}")

    print("\n  Inter-stock correlations (current 30d):")
    for t1, t2 in [('RKLB', 'ASTS'), ('RKLB', 'LUNR'), ('ASTS', 'LUNR')]:
        key = f"{t1}_vs_{t2}"
        if key in correlations and len(correlations[key].dropna()) > 0:
            current = correlations[key].dropna().iloc[-1]
            print(f"    {t1} vs {t2}: {current:.2f}")

    print("\n  Full correlation matrix (last 60 days):")
    matrix = compute_correlation_matrix(returns)
    print(matrix.round(2).to_string())
    print("="*55)


def plot_correlations(correlations: dict, save=True):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle('New Space Stocks — Rolling 30-Day Correlation',
                 fontsize=14, fontweight='bold')

    colors = {'RKLB': '#1D9E75', 'ASTS': '#378ADD', 'LUNR': '#7F77DD'}

    # vs SPY
    for ticker in SPACE_TICKERS:
        key = f"{ticker}_vs_SPY"
        if key in correlations:
            series = correlations[key].dropna()
            axes[0].plot(series.index, series,
                         label=ticker, color=colors[ticker], linewidth=1.5)

    axes[0].axhline(0.3, color='red', linestyle='--',
                    alpha=0.5, label='Low corr threshold (0.3)')
    axes[0].fill_between(
        correlations.get('RKLB_vs_SPY', pd.Series()).dropna().index,
        0.3, -1,
        alpha=0.04, color='red',
        label='Catalyst zone'
    )
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_title('Correlation vs SPY — below 0.3 = moving independently')
    axes[0].set_ylabel('Correlation')
    axes[0].set_ylim(-1, 1)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    # Inter-stock
    pair_colors = {
        'RKLB_vs_ASTS': '#1D9E75',
        'RKLB_vs_LUNR': '#A32D2D',
        'ASTS_vs_LUNR': '#EF9F27',
    }
    for key, color in pair_colors.items():
        if key in correlations:
            series = correlations[key].dropna()
            axes[1].plot(series.index, series,
                         label=key.replace('_vs_', ' vs '),
                         color=color, linewidth=1.5)

    axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_title('Inter-stock Correlations')
    axes[1].set_ylabel('Correlation')
    axes[1].set_ylim(-1, 1)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/correlations.png", dpi=150, bbox_inches='tight')
        print("Saved: data/correlations.png")
    plt.close()


if __name__ == "__main__":
    from src.data import fetch_price_history, compute_returns
    prices  = fetch_price_history()
    returns = compute_returns(prices)
    corr    = compute_rolling_correlation(returns)
    print_correlation_summary(corr, returns)
    plot_correlations(corr)
    breaks  = detect_correlation_breaks(corr)
    if not breaks.empty:
        print(f"\nCorrelation breaks detected: {len(breaks)}")
        print(breaks.tail(10).to_string(index=False))