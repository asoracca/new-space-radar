"""
cross_portfolio.py
-------------------
The capstone module: a single correlation heatmap and crash-day analysis
spanning EVERY ticker across all three projects — SOXL, QLD, FNGO, USD,
RKLB, ASTS, LUNR, VTI, AMZN.

WHY THIS MODULE EXISTS:
  Each individual project (SOXL vol surface, new-space-radar, leveraged ETF
  risk lab) analyzes ONE slice of the portfolio in isolation. But a real
  portfolio doesn't experience risk in isolated slices — when markets
  stress, correlations don't just stay the same, they tend to CONVERGE
  toward 1. Things that "shouldn't" be related suddenly move together.

  This module answers the question that actually matters for risk
  management: "On the worst days, does ANYTHING in my portfolio actually
  diversify me, or is it all just one big leveraged bet on tech beta?"

THE UNIVERSE:
  SOXL  — 3x leveraged semiconductor ETF (your options income trade)
  QLD   — 2x leveraged Nasdaq-100
  FNGO  — 2x leveraged "FANG+" mega-cap tech
  USD   — 2x leveraged semiconductors (ProShares Ultra Semiconductors)
  RKLB  — Rocket Lab (new space, launch services)
  ASTS  — AST SpaceMobile (new space, satellite connectivity)
  LUNR  — Intuitive Machines (new space, lunar/NASA contracts)
  VTI   — Total US stock market (the "boring" diversifier)
  AMZN  — Single mega-cap tech name (cloud/retail/AI capex exposure)

TWO ANALYSES:

  1. UNCONDITIONAL CORRELATION HEATMAP
     Standard Pearson correlation of daily log returns over the full period.
     This is the "normal day" picture — how things move together on average.

  2. CONDITIONAL ("CRASH DAY") CORRELATION
     Filter to only the worst N% of days for SOXL (the most levered,
     most volatile name — effectively your portfolio's "canary").
     Recompute correlations using ONLY those days.

     This is the test that matters: correlations measured on ALL days
     often understate how connected things become during stress.
     A pair that shows correlation 0.4 normally might show 0.85 on
     SOXL's worst days — meaning your "diversifier" disappears exactly
     when you need it most.

WHY THIS MATTERS FOR YOUR PORTFOLIO SPECIFICALLY:
  SOXL, USD are both leveraged semiconductor bets — if they're not
  near-perfectly correlated, something is wrong with the data, not
  the portfolio (sanity check).

  QLD and FNGO both track mega-cap tech — similarly should be highly
  correlated to each other and to AMZN.

  RKLB/ASTS/LUNR are "new space" — theoretically a different sector,
  but if they spike in correlation with SOXL on down days, it means
  they're trading as "high beta risk-on" rather than on their own
  fundamentals during stress — i.e., they offer ZERO diversification
  exactly when the rest of the portfolio is hurting.

  VTI is included as the benchmark "what if I just owned the market"
  case. If VTI's correlation with everything else is high even on
  calm days, the whole portfolio is essentially a leveraged bet on
  the broad market with extra steps.

HOW TO READ THE OUTPUT:
  - Heatmap 1 (normal days): baseline relationships
  - Heatmap 2 (SOXL crash days): same pairs, recomputed on stress days
  - Heatmap 3 (difference): crash-day corr minus normal-day corr.
    Large POSITIVE values = correlation increases in a crash =
    "fake diversification" that vanishes when you need it.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path

UNIVERSE = ['SOXL', 'QLD', 'FNGO', 'USD', 'RKLB', 'ASTS', 'LUNR', 'VTI', 'AMZN']

# Theme groupings for annotation / sanity-checking
THEMES = {
    'SOXL': 'Leveraged Semis (3x)',
    'USD':  'Leveraged Semis (2x)',
    'QLD':  'Leveraged Mega-Tech (2x)',
    'FNGO': 'Leveraged Mega-Tech (2x)',
    'RKLB': 'New Space',
    'ASTS': 'New Space',
    'LUNR': 'New Space',
    'VTI':  'Broad Market',
    'AMZN': 'Single Mega-Cap Tech',
}


def fetch_universe_returns(period: str = "2y") -> pd.DataFrame:
    """
    Fetch all tickers in the universe and return daily log returns.
    """
    print(f"Fetching {len(UNIVERSE)} tickers: {UNIVERSE}")
    raw = yf.download(UNIVERSE, period=period, interval="1d", progress=False)
    prices = raw['Close'].copy()
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    # Reorder columns to match UNIVERSE for consistent heatmap layout
    cols = [t for t in UNIVERSE if t in prices.columns]
    prices = prices[cols]

    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  Loaded {len(returns)} trading days for {len(cols)} tickers")
    return returns


def compute_full_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Unconditional correlation matrix over the full sample."""
    return returns.corr()


def compute_crash_correlation(returns: pd.DataFrame,
                               crash_ticker: str = 'SOXL',
                               percentile: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Compute correlation matrix using only the worst `percentile`% of days
    for `crash_ticker` (by that ticker's own return).

    Returns (crash_corr, diff_from_full, n_days_used)
    """
    if crash_ticker not in returns.columns:
        raise ValueError(f"{crash_ticker} not in returns columns")

    threshold = np.percentile(returns[crash_ticker], percentile)
    crash_days = returns[returns[crash_ticker] <= threshold]

    crash_corr = crash_days.corr()
    full_corr  = returns.corr()
    diff       = crash_corr - full_corr

    return crash_corr, diff, len(crash_days)


def print_correlation_summary(returns: pd.DataFrame,
                              full_corr: pd.DataFrame,
                              crash_corr: pd.DataFrame,
                              diff: pd.DataFrame,
                              n_crash_days: int,
                              crash_ticker: str = 'SOXL'):
    print("\n" + "="*70)
    print("  CROSS-PORTFOLIO CORRELATION ANALYSIS")
    print("="*70)
    print(f"  Universe: {', '.join(returns.columns)}")
    print(f"  Period:   {returns.index[0].date()} to {returns.index[-1].date()}"
          f"  ({len(returns)} trading days)")
    print(f"  Crash days: worst 10% of {crash_ticker} returns "
          f"(n={n_crash_days} days)")

    print(f"\n  Sanity checks (full-sample correlation):")
    sanity_pairs = [
        ('SOXL', 'USD',  'Both leveraged semis — should be HIGH'),
        ('QLD',  'FNGO', 'Both leveraged mega-tech — should be HIGH'),
        ('QLD',  'AMZN', 'Mega-tech ETF vs AMZN — should be HIGH'),
        ('VTI',  'AMZN', 'Broad market vs single mega-cap — moderate-high'),
    ]
    for t1, t2, note in sanity_pairs:
        if t1 in full_corr.index and t2 in full_corr.columns:
            v = full_corr.loc[t1, t2]
            print(f"    {t1} vs {t2}: {v:.2f}   ({note})")

    print(f"\n  New Space vs SOXL — normal days vs {crash_ticker} crash days:")
    for ticker in ['RKLB', 'ASTS', 'LUNR']:
        if ticker in full_corr.index and crash_ticker in full_corr.columns:
            normal = full_corr.loc[ticker, crash_ticker]
            crash  = crash_corr.loc[ticker, crash_ticker]
            delta  = diff.loc[ticker, crash_ticker]
            flag   = "  ⚠️  diversification VANISHES in crash" if delta > 0.15 else ""
            print(f"    {ticker}: normal={normal:.2f}  "
                  f"crash={crash:.2f}  Δ={delta:+.2f}{flag}")

    print(f"\n  VTI as diversifier — normal vs crash correlation with {crash_ticker}:")
    if 'VTI' in full_corr.index and crash_ticker in full_corr.columns:
        normal = full_corr.loc['VTI', crash_ticker]
        crash  = crash_corr.loc['VTI', crash_ticker]
        print(f"    VTI: normal={normal:.2f}  crash={crash:.2f}  "
              f"Δ={crash-normal:+.2f}")
        if crash > 0.6:
            print(f"    → Even 'broad market' VTI moves WITH {crash_ticker} "
                  f"during stress. There may be no true diversifier here.")

    print(f"\n  Largest correlation INCREASES during {crash_ticker} crash days")
    print(f"  (these are the pairs where 'diversification' disappears most):")
    diff_flat = diff.where(~np.eye(len(diff), dtype=bool)).stack()
    diff_flat = diff_flat[diff_flat.index.get_level_values(0) <
                          diff_flat.index.get_level_values(1)]
    top5 = diff_flat.sort_values(ascending=False).head(5)
    for (t1, t2), val in top5.items():
        print(f"    {t1} vs {t2}: {full_corr.loc[t1,t2]:.2f} → "
              f"{crash_corr.loc[t1,t2]:.2f}  (Δ={val:+.2f})")

    print("="*70)


def plot_correlation_heatmaps(full_corr: pd.DataFrame,
                              crash_corr: pd.DataFrame,
                              diff: pd.DataFrame,
                              n_crash_days: int,
                              crash_ticker: str = 'SOXL',
                              save=True):
    """
    Three side-by-side heatmaps: full-sample, crash-day, and the difference.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f'Cross-Portfolio Correlation — SOXL, QLD, FNGO, USD, RKLB, ASTS, '
        f'LUNR, VTI, AMZN\n'
        f'Left: normal days. Middle: worst 10% of {crash_ticker} days '
        f'(n={n_crash_days}). Right: difference (red = correlation rises in crash).',
        fontsize=11, fontweight='bold'
    )

    matrices = [
        (full_corr,  'All Days (Unconditional)',                  'RdYlGn_r', -1, 1),
        (crash_corr, f'{crash_ticker} Crash Days (worst 10%)',     'RdYlGn_r', -1, 1),
        (diff,       'Difference (Crash − Normal)',                'RdBu_r',  -0.5, 0.5),
    ]

    for ax, (matrix, title, cmap, vmin, vmax) in zip(axes, matrices):
        im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(matrix.index, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Annotate cells
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                val = matrix.values[i, j]
                color = 'white' if abs(val) > (vmax * 0.6) else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=8, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/cross_portfolio_correlation.png", dpi=150, bbox_inches='tight')
        print("Saved: data/cross_portfolio_correlation.png")
    plt.close()


def plot_theme_grouped_heatmap(full_corr: pd.DataFrame, save=True):
    """
    Single heatmap with tickers grouped/colored by theme, for a cleaner
    "at a glance" view of where real diversification exists.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(full_corr.values, cmap='RdYlGn_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(full_corr.columns)))
    ax.set_yticks(range(len(full_corr.index)))

    labels = [f"{t}\n({THEMES.get(t, '')})" for t in full_corr.columns]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(full_corr.index)):
        for j in range(len(full_corr.columns)):
            val = full_corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=8, color=color)

    ax.set_title(
        'Full Portfolio Correlation Matrix (All Days)\n'
        'Theme labels show WHY correlations are high — '
        'same-theme pairs should cluster near 1.0',
        fontsize=11, fontweight='bold'
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/correlation_by_theme.png", dpi=150, bbox_inches='tight')
        print("Saved: data/correlation_by_theme.png")
    plt.close()


if __name__ == "__main__":
    returns = fetch_universe_returns(period="2y")
    full_corr = compute_full_correlation(returns)
    crash_corr, diff, n_crash = compute_crash_correlation(returns, 'SOXL', 10.0)

    print_correlation_summary(returns, full_corr, crash_corr, diff, n_crash, 'SOXL')
    plot_correlation_heatmaps(full_corr, crash_corr, diff, n_crash, 'SOXL')
    plot_theme_grouped_heatmap(full_corr)