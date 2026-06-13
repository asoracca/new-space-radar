"""
events.py
---------
Catalog of real RKLB, ASTS, LUNR events with price impact analysis.
Event study methodology: measure abnormal return around each event.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ── Real event catalog ────────────────────────────────────────────────────────
# Manually curated from public announcements, SEC filings, and news
EVENTS = {
    'RKLB': [
        {'date': '2023-03-24', 'event': 'Electron launch success — NROL-162', 'type': 'launch'},
        {'date': '2023-07-17', 'event': 'Electron launch success — TROPICS-2', 'type': 'launch'},
        {'date': '2023-09-19', 'event': 'NASA ESCAPADE contract award', 'type': 'contract'},
        {'date': '2024-02-18', 'event': 'Electron launch — NROL-123', 'type': 'launch'},
        {'date': '2024-07-24', 'event': 'Neutron rocket development update', 'type': 'milestone'},
        {'date': '2024-11-05', 'event': 'Electron launch — Kinéis 5', 'type': 'launch'},
        {'date': '2025-03-14', 'event': 'Neutron upper stage contract', 'type': 'contract'},
    ],
    'ASTS': [
        {'date': '2023-04-01', 'event': 'BlueWalker 3 first call completed', 'type': 'milestone'},
        {'date': '2023-09-10', 'event': 'AT&T partnership expansion', 'type': 'contract'},
        {'date': '2024-04-04', 'event': 'BlueBird satellite manufacturing begins', 'type': 'milestone'},
        {'date': '2024-09-12', 'event': 'BlueBird 1-5 launch success', 'type': 'launch'},
        {'date': '2024-11-01', 'event': 'First commercial satellite call', 'type': 'milestone'},
        {'date': '2025-01-15', 'event': 'DoD contract for satellite comms', 'type': 'contract'},
    ],
    'LUNR': [
        {'date': '2024-02-22', 'event': 'IM-1 Odysseus moon landing', 'type': 'launch'},
        {'date': '2024-03-01', 'event': 'NASA contract follow-on announced', 'type': 'contract'},
        {'date': '2024-11-20', 'event': 'IM-2 mission preparation update', 'type': 'milestone'},
        {'date': '2025-02-26', 'event': 'IM-2 launch to lunar south pole', 'type': 'launch'},
        {'date': '2025-03-06', 'event': 'IM-2 lunar landing attempt', 'type': 'launch'},
    ]
}


def compute_event_impact(ticker: str, events: list,
                         prices: pd.DataFrame,
                         returns: pd.DataFrame,
                         window: int = 5) -> pd.DataFrame:
    """
    For each event, compute:
    - Return on event day
    - Cumulative abnormal return over [-1, +window] days
    - SPY return over same window (market context)
    - Abnormal return = stock return - SPY return
    """
    results = []

    for ev in events:
        ev_date = pd.Timestamp(ev['date'])

        # Find nearest trading day
        available = prices.index[prices.index >= ev_date]
        if len(available) == 0:
            continue
        actual_date = available[0]

        # Window around event
        idx = prices.index.get_loc(actual_date)
        start_idx = max(0, idx - 1)
        end_idx   = min(len(prices) - 1, idx + window)

        if end_idx <= start_idx:
            continue

        # Cumulative returns over window
        start_price_stock = prices[ticker].iloc[start_idx]
        end_price_stock   = prices[ticker].iloc[end_idx]
        stock_cum_ret     = (end_price_stock / start_price_stock) - 1

        start_price_spy = prices['SPY'].iloc[start_idx]
        end_price_spy   = prices['SPY'].iloc[end_idx]
        spy_cum_ret     = (end_price_spy / start_price_spy) - 1

        abnormal_ret = stock_cum_ret - spy_cum_ret
        event_day_ret = returns[ticker].iloc[idx] if idx < len(returns) else np.nan

        results.append({
            'ticker':        ticker,
            'event_date':    ev_date.date(),
            'event':         ev['event'],
            'type':          ev['type'],
            'event_day_ret': round(event_day_ret, 4),
            'stock_5d_ret':  round(stock_cum_ret, 4),
            'spy_5d_ret':    round(spy_cum_ret, 4),
            'abnormal_ret':  round(abnormal_ret, 4),
        })

    return pd.DataFrame(results)


def analyze_all_events(prices: pd.DataFrame,
                       returns: pd.DataFrame) -> pd.DataFrame:
    """
    Run event impact analysis for all tickers.
    """
    all_results = []
    for ticker, events in EVENTS.items():
        if ticker not in prices.columns:
            continue
        df = compute_event_impact(ticker, events, prices, returns)
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def print_event_summary(event_df: pd.DataFrame):
    if event_df.empty:
        print("No events analyzed.")
        return

    print("\n" + "="*60)
    print("  EVENT IMPACT ANALYSIS")
    print("="*60)

    for ticker in ['RKLB', 'ASTS', 'LUNR']:
        df = event_df[event_df['ticker'] == ticker]
        if df.empty:
            continue

        avg_abnormal = df['abnormal_ret'].mean()
        avg_event_day = df['event_day_ret'].mean()
        launches = df[df['type'] == 'launch']['abnormal_ret'].mean()
        contracts = df[df['type'] == 'contract']['abnormal_ret'].mean()

        print(f"\n  {ticker}")
        print(f"    Events analyzed:      {len(df)}")
        print(f"    Avg abnormal ret:     {avg_abnormal:.1%}")
        print(f"    Avg event-day ret:    {avg_event_day:.1%}")
        if not np.isnan(launches):
            print(f"    Launch avg impact:    {launches:.1%}")
        if not np.isnan(contracts):
            print(f"    Contract avg impact:  {contracts:.1%}")

    print("\n  All events:")
    print(event_df[['ticker', 'event_date', 'type',
                     'event_day_ret', 'abnormal_ret',
                     'event']].to_string(index=False))
    print("="*60)


def plot_event_impact(event_df: pd.DataFrame, save=True):
    if event_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('New Space — Event Impact Analysis',
                 fontsize=14, fontweight='bold')

    colors = {'launch': '#1D9E75', 'contract': '#378ADD',
              'milestone': '#7F77DD'}

    for i, ticker in enumerate(['RKLB', 'ASTS', 'LUNR']):
        df = event_df[event_df['ticker'] == ticker]
        if df.empty:
            axes[i].set_title(f'{ticker} — no data')
            continue

        bar_colors = [colors.get(t, '#888780') for t in df['type']]
        axes[i].bar(range(len(df)), df['abnormal_ret'] * 100,
                    color=bar_colors, alpha=0.8, edgecolor='white')
        axes[i].axhline(0, color='black', linestyle='--', alpha=0.4)
        axes[i].axhline(df['abnormal_ret'].mean() * 100,
                        color='red', linestyle='--', alpha=0.6,
                        label=f"Avg: {df['abnormal_ret'].mean():.1%}")
        axes[i].set_title(f'{ticker} Abnormal Returns', fontweight='bold')
        axes[i].set_ylabel('Abnormal Return (%)')
        axes[i].set_xlabel('Event #')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.25)

    # Legend for event types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.capitalize())
                       for t, c in colors.items()]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/event_impact.png", dpi=150, bbox_inches='tight')
        print("Saved: data/event_impact.png")


if __name__ == "__main__":
    from src.data import fetch_price_history, compute_returns
    prices  = fetch_price_history()
    returns = compute_returns(prices)
    events  = analyze_all_events(prices, returns)
    print_event_summary(events)
    plot_event_impact(events)