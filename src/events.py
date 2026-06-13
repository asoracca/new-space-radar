"""
events.py
---------
Catalog of real RKLB, ASTS, LUNR events with price impact analysis.

Methodology: event study with market-model abnormal returns.
  Expected return = alpha + beta * SPY_return  (estimated on pre-event window)
  Abnormal return = actual return - expected return
  CAR = cumulative abnormal return over event window
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

EVENTS = {
    'RKLB': [
        {'date': '2023-03-24', 'event': 'Electron launch — NROL-162',         'type': 'launch'},
        {'date': '2023-07-17', 'event': 'Electron launch — TROPICS-2',         'type': 'launch'},
        {'date': '2023-09-19', 'event': 'NASA ESCAPADE contract award',         'type': 'contract'},
        {'date': '2024-02-18', 'event': 'Electron launch — NROL-123',           'type': 'launch'},
        {'date': '2024-07-24', 'event': 'Neutron rocket development update',    'type': 'milestone'},
        {'date': '2024-11-05', 'event': 'Electron launch — Kinéis 5',          'type': 'launch'},
        {'date': '2025-03-14', 'event': 'Neutron upper stage contract',         'type': 'contract'},
    ],
    'ASTS': [
        {'date': '2023-04-01', 'event': 'BlueWalker 3 first call completed',   'type': 'milestone'},
        {'date': '2023-09-10', 'event': 'AT&T partnership expansion',           'type': 'contract'},
        {'date': '2024-04-04', 'event': 'BlueBird satellite manufacturing begins','type': 'milestone'},
        {'date': '2024-09-12', 'event': 'BlueBird 1-5 launch success',         'type': 'launch'},
        {'date': '2024-11-01', 'event': 'First commercial satellite call',      'type': 'milestone'},
        {'date': '2025-01-15', 'event': 'DoD contract for satellite comms',     'type': 'contract'},
    ],
    'LUNR': [
        {'date': '2024-02-22', 'event': 'IM-1 Odysseus moon landing',          'type': 'launch'},
        {'date': '2024-03-01', 'event': 'NASA contract follow-on announced',    'type': 'contract'},
        {'date': '2024-11-20', 'event': 'IM-2 mission preparation update',      'type': 'milestone'},
        {'date': '2025-02-26', 'event': 'IM-2 launch to lunar south pole',     'type': 'launch'},
        {'date': '2025-03-06', 'event': 'IM-2 lunar landing attempt',          'type': 'launch'},
    ]
}

EVENT_COLORS = {
    'launch':    '#1D9E75',
    'contract':  '#378ADD',
    'milestone': '#7F77DD',
}


def _estimate_market_model(ticker: str,
                            ev_idx: int,
                            returns: pd.DataFrame,
                            estimation_window: int = 120) -> tuple[float, float]:
    """
    OLS regression of stock returns on SPY returns using pre-event window.
    Returns (alpha, beta).
    """
    start = max(0, ev_idx - estimation_window - 10)
    end   = max(0, ev_idx - 10)           # 10-day buffer before event

    if end <= start + 20:                  # need at least 20 obs
        return 0.0, 1.0

    y = returns[ticker].iloc[start:end].values
    x = returns['SPY'].iloc[start:end].values

    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 20:
        return 0.0, 1.0

    slope, intercept, *_ = stats.linregress(x[mask], y[mask])
    return intercept, slope


def compute_event_impact(ticker: str,
                          events: list,
                          prices: pd.DataFrame,
                          returns: pd.DataFrame,
                          pre_window: int = 2,
                          post_window: int = 5) -> pd.DataFrame:
    """
    For each event compute:
      - Event-day return and abnormal return
      - CAR[-pre, +post] using market-model residuals
      - t-statistic for abnormal return
    """
    results = []

    for ev in events:
        ev_date    = pd.Timestamp(ev['date'])
        available  = prices.index[prices.index >= ev_date]
        if len(available) == 0:
            continue
        actual_date = available[0]
        idx = prices.index.get_loc(actual_date)

        # Market model params from pre-event window
        alpha, beta = _estimate_market_model(ticker, idx, returns)

        # Collect abnormal returns over window
        ar_list = []
        window_indices = range(max(0, idx - pre_window),
                               min(len(returns) - 1, idx + post_window + 1))

        for wi in window_indices:
            r_stock = returns[ticker].iloc[wi]
            r_spy   = returns['SPY'].iloc[wi]
            if np.isnan(r_stock) or np.isnan(r_spy):
                continue
            ar = r_stock - (alpha + beta * r_spy)
            ar_list.append(ar)

        if not ar_list:
            continue

        car = sum(ar_list)

        # Event-day abnormal return
        if idx < len(returns):
            r_ev     = returns[ticker].iloc[idx]
            r_spy_ev = returns['SPY'].iloc[idx]
            ar_day   = r_ev - (alpha + beta * r_spy_ev)
        else:
            ar_day = np.nan

        # Compute t-stat (against null hypothesis CAR = 0)
        # Use estimation-window residual std as benchmark
        est_start = max(0, idx - 130)
        est_end   = max(0, idx - 10)
        est_ars   = []
        for wi in range(est_start, est_end):
            if wi >= len(returns):
                break
            r_s = returns[ticker].iloc[wi]
            r_m = returns['SPY'].iloc[wi]
            if not (np.isnan(r_s) or np.isnan(r_m)):
                est_ars.append(r_s - (alpha + beta * r_m))

        sigma = np.std(est_ars) * np.sqrt(len(ar_list)) if len(est_ars) > 5 else np.nan
        t_stat = car / sigma if sigma and sigma > 0 else np.nan

        results.append({
            'ticker':      ticker,
            'event_date':  ev_date.date(),
            'event':       ev['event'],
            'type':        ev['type'],
            'alpha':       round(alpha, 5),
            'beta':        round(beta, 3),
            'ar_day':      round(ar_day, 4) if not np.isnan(ar_day) else np.nan,
            'CAR':         round(car, 4),
            't_stat':      round(t_stat, 2) if not np.isnan(t_stat) else np.nan,
            'significant': abs(t_stat) > 1.96 if not np.isnan(t_stat) else False,
        })

    return pd.DataFrame(results)


def analyze_all_events(prices: pd.DataFrame,
                        returns: pd.DataFrame) -> pd.DataFrame:
    all_results = []
    for ticker, events in EVENTS.items():
        if ticker not in prices.columns:
            continue
        df = compute_event_impact(ticker, events, prices, returns)
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def print_event_summary(event_df: pd.DataFrame):
    if event_df.empty:
        print("No events analyzed.")
        return

    print("\n" + "="*65)
    print("  EVENT IMPACT ANALYSIS (Market-Model Abnormal Returns)")
    print("="*65)

    for ticker in ['RKLB', 'ASTS', 'LUNR']:
        df = event_df[event_df['ticker'] == ticker]
        if df.empty:
            continue

        sig = df[df['significant']]
        print(f"\n  {ticker}  (β̄ = {df['beta'].mean():.2f})")
        print(f"    Events analyzed:        {len(df)}")
        print(f"    Avg CAR [-2,+5]:        {df['CAR'].mean():.1%}")
        print(f"    Avg event-day AR:       {df['ar_day'].mean():.1%}")
        print(f"    Statistically sig (|t|>1.96): {len(sig)}/{len(df)}")

        for etype in ['launch', 'contract', 'milestone']:
            sub = df[df['type'] == etype]
            if not sub.empty:
                print(f"    {etype.capitalize()} avg CAR: {sub['CAR'].mean():.1%} "
                      f"(n={len(sub)})")

    print(f"\n{'─'*65}")
    print("  All events:")
    cols = ['ticker', 'event_date', 'type', 'beta', 'ar_day', 'CAR', 't_stat', 'significant', 'event']
    print(event_df[cols].to_string(index=False))
    print("="*65)


def plot_event_impact(event_df: pd.DataFrame, save=True):
    if event_df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('New Space — Event Impact Analysis (Market-Model CARs)',
                 fontsize=14, fontweight='bold')

    for i, ticker in enumerate(['RKLB', 'ASTS', 'LUNR']):
        df = event_df[event_df['ticker'] == ticker].copy()
        if df.empty:
            axes[0, i].set_title(f'{ticker} — no data')
            axes[1, i].set_title(f'{ticker} — no data')
            continue

        bar_colors = [EVENT_COLORS.get(t, '#888780') for t in df['type']]

        # Row 0: CAR bar chart
        bars = axes[0, i].bar(range(len(df)), df['CAR'] * 100,
                               color=bar_colors, alpha=0.85, edgecolor='white',
                               linewidth=0.5)

        # Hatch significant bars
        for bar, sig in zip(bars, df['significant']):
            if sig:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

        axes[0, i].axhline(0, color='black', linestyle='--', alpha=0.4)
        axes[0, i].axhline(df['CAR'].mean() * 100, color='red',
                            linestyle='--', alpha=0.6,
                            label=f"Avg: {df['CAR'].mean():.1%}")
        axes[0, i].set_title(f'{ticker}  CAR [-2,+5]', fontweight='bold')
        axes[0, i].set_ylabel('Cumulative Abnormal Return (%)')
        axes[0, i].set_xlabel('Event #')
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(True, alpha=0.25)

        # Row 1: t-stats
        t_colors = ['#1D9E75' if v > 0 else '#A32D2D'
                    for v in df['t_stat'].fillna(0)]
        axes[1, i].bar(range(len(df)), df['t_stat'].fillna(0),
                        color=t_colors, alpha=0.75, edgecolor='white')
        axes[1, i].axhline( 1.96, color='red', linestyle='--',
                             alpha=0.5, label='95% sig (±1.96)')
        axes[1, i].axhline(-1.96, color='red', linestyle='--', alpha=0.5)
        axes[1, i].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[1, i].set_title(f'{ticker}  t-statistics', fontweight='bold')
        axes[1, i].set_ylabel('t-stat')
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.25)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.capitalize())
                       for t, c in EVENT_COLORS.items()]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/event_impact.png", dpi=150, bbox_inches='tight')
        print("Saved: data/event_impact.png")
    plt.close()


if __name__ == "__main__":
    from src.data import fetch_price_history, compute_returns
    prices  = fetch_price_history()
    returns = compute_returns(prices)
    events  = analyze_all_events(prices, returns)
    print_event_summary(events)
    plot_event_impact(events)