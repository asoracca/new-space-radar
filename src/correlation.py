cat > src/correlation.py << 'ENDOFFILE'
"""
correlation.py
--------------
Rolling correlation analysis for RKLB, ASTS, LUNR vs each other and SPY.
When correlation breaks down, a stock is moving on its own catalyst.
That's the signal.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def compute_rolling_correlation(returns: pd.DataFrame,
                                window: int = 30) -> dict:
    """
    Rolling pairwise correlations between space stocks and vs SPY.
    Returns dict of DataFrames keyed by ticker pair.
    """
    pairs = [
        ('RKLB', 'SPY'),
        ('ASTS', 'SPY'),
        ('LUNR', 'SPY'),
        ('RKLB', 'ASTS'),
        ('RKLB', 'LUNR'),
        ('ASTS', 'LUNR'),
    ]

    correlations = {}
    for t1, t2 in pairs:
        if t1 in returns.columns and t2 in returns.columns:
            key = f"{t1}_vs_{t2}"
            correlations[key] = (returns[t1]
                                 .rolling(window)
                                 .corr(returns[t2]))
    return correlations


def detect_correlation_breaks(correlations: dict,
                               threshold: float = 0.3) -> pd.DataFrame:
    """
    Detect when a stock's correlation with SPY drops below threshold.
    Low correlation = stock moving independently = potential catalyst.
    """
    breaks = []
    spy_pairs = {k: v for k, v in correlations.items()
                 if 'SPY' in k}

    for pair, corr_series in spy_pairs.items():
        ticker = pair.replace('_vs_SPY', '')
        low_corr = corr_series[corr_series < threshold]

        for date, corr_val in low_corr.items():
            breaks.append({
                'date':        date.date(),
                'ticker':      ticker,
                'correlation': round(corr_val, 3),
                'signal':      'LOW CORR — possible catalyst'
            })

    return pd.DataFrame(breaks).drop_duplicates()


def print_correlation_summary(correlations: dict, returns: pd.DataFrame):
    print("\n" + "="*55)
    print("  CORRELATION ANALYSIS")
    print("="*55)

    print("\n  Current 30-day correlations vs SPY:")
    for ticker in ['RKLB', 'ASTS', 'LUNR']:
        key = f"{ticker}_vs_SPY"
        if key in correlations:
            current = correlations[key].iloc[-1]
            avg = correlations[key].mean()
            print(f"    {ticker}: {current:.2f} (avg: {avg:.2f})")

    print("\n  Inter-stock correlations (current):")
    pairs = [('RKLB', 'ASTS'), ('RKLB', 'LUNR'), ('ASTS', 'LUNR')]
    for t1, t2 in pairs:
        key = f"{t1}_vs_{t2}"
        if key in correlations:
            current = correlations[key].iloc[-1]
            print(f"    {t1} vs {t2}: {current:.2f}")
    print("="*55)


def plot_correlations(correlations: dict, save=True):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle('New Space Stocks — Rolling 30-Day Correlation',
                 fontsize=14, fontweight='bold')

    # vs SPY
    colors = {'RKLB': '#1D9E75', 'ASTS': '#378ADD', 'LUNR': '#7F77DD'}
    for ticker in ['RKLB', 'ASTS', 'LUNR']:
        key = f"{ticker}_vs_SPY"
        if key in correlations:
            axes[0].plot(correlations[key].index,
                        correlations[key],
                        label=ticker,
                        color=colors[ticker],
                        linewidth=1.5)

    axes[0].axhline(0.3, color='red', linestyle='--',
                    alpha=0.5, label='Low corr threshold (0.3)')
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_title('Correlation vs SPY — below 0.3 = moving independently')
    axes[0].set_ylabel('Correlation')
    axes[0].set_ylim(-1, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    # Inter-stock
    pair_colors = {'RKLB_vs_ASTS': '#1D9E75',
                   'RKLB_vs_LUNR': '#A32D2D',
                   'ASTS_vs_LUNR': '#EF9F27'}
    for key, color in pair_colors.items():
        if key in correlations:
            label = key.replace('_vs_', ' vs ')
            axes[1].plot(correlations[key].index,
                        correlations[key],
                        label=label, color=color, linewidth=1.5)

    axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_title('Inter-stock Correlations')
    axes[1].set_ylabel('Correlation')
    axes[1].set_ylim(-1, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/correlations.png", dpi=150, bbox_inches='tight')
        print("Saved: data/correlations.png")


if __name__ == "__main__":
    from src.data import fetch_price_history, compute_returns
    prices = fetch_price_history()
    returns = compute_returns(prices)
    corr = compute_rolling_correlation(returns)
    print_correlation_summary(corr, returns)
    plot_correlations(corr)
    breaks = detect_correlation_breaks(corr)
    if not breaks.empty:
        print(f"\nCorrelation breaks detected: {len(breaks)}")
        print(breaks.tail(10).to_string(index=False))
ENDOFFILE