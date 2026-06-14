"""
car_curve.py
------------
The standard event-study output: average Cumulative Abnormal Return (CAR)
across a [-5, +10] trading-day window around events, per ticker and event type.

WHY THIS IS THE OUTPUT QUANT RESEARCHERS LOOK FOR:
  events.py (the original module) computes a single CAR number per event —
  "did this event move the stock, net of market, over a 7-day window?"

  This module instead asks: "WHAT SHAPE does the reaction take over time?"

  Academic event studies (Fama, MacKinlay 1997 — the standard reference)
  always show this as a line chart:
    x-axis: trading days relative to event (day 0 = event day)
    y-axis: average cumulative abnormal return (%)

  The SHAPE of this curve tells a story:
    - Flat before day 0, jump at day 0, flat after
        → Market reacts instantly and efficiently. No drift to exploit.
    - Flat before day 0, jump at day 0, CONTINUES RISING after
        → "Post-event drift" — the market underreacts initially.
          This is one of the most studied anomalies in finance (Bernard
          & Thomas 1989 for earnings). If you see this, it suggests
          buying AFTER the event (not before) could capture extra return.
    - Rising BEFORE day 0
        → Information leakage / anticipation. Someone knew before the
          public announcement (insider activity, rumor, or the market
          correctly anticipating scheduled launches).
    - Jump at day 0 then REVERSAL
        → Overreaction. The initial move is too large and partially
          unwinds. Classic mean-reversion setup.

METHODOLOGY:
  For each event, compute daily abnormal returns AR_t for t in [-5, +10]
  using the same market-model (alpha, beta from pre-event window) as events.py.

  Average AR_t across all events of the same type (or same ticker) at each
  relative day t. This gives the "average reaction path".

  Cumulate: CAR_t = sum(AR_0..t) for t >= 0, and CAR_t = -sum(AR_t..-1) for t < 0
  (so CAR_0 = AR_0, and the curve reads naturally left to right).

  Standard error bands: SE_t = std(AR_t across events) / sqrt(n_events)
  Shaded band = ±1.96 * SE (95% confidence interval on the average).
  If zero is OUTSIDE the band, the average reaction at that day is
  statistically significant.

HOW TO READ THE OUTPUT CHART:
  - Solid line = average CAR across all events for that ticker
  - Shaded band = 95% confidence interval
  - Vertical dashed line at day 0 = the event date
  - If the band excludes zero AFTER day 0 → the event has a lasting,
    statistically distinguishable effect
  - If the band is huge and always includes zero → too few events /
    too noisy to draw conclusions (be honest about this in your writeup)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.events import EVENTS, _estimate_market_model, EVENT_COLORS

PRE_DAYS  = 5    # days before event to include
POST_DAYS = 10   # days after event to include


def compute_ar_paths(ticker: str, events: list,
                     prices: pd.DataFrame,
                     returns: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, compute AR_t for t in [-PRE_DAYS, +POST_DAYS].
    Returns a DataFrame: rows = events, columns = relative day offsets.
    """
    rel_days = list(range(-PRE_DAYS, POST_DAYS + 1))
    rows = []

    for ev in events:
        ev_date   = pd.Timestamp(ev['date'])
        available = prices.index[prices.index >= ev_date]
        if len(available) == 0:
            continue
        actual_date = available[0]
        idx = prices.index.get_loc(actual_date)

        # Need enough room on both sides
        if idx - PRE_DAYS < 0 or idx + POST_DAYS >= len(returns):
            continue

        alpha, beta = _estimate_market_model(ticker, idx, returns)

        ar_row = {'type': ev['type'], 'event': ev['event'],
                  'event_date': ev_date.date()}
        for d in rel_days:
            wi = idx + d
            r_s = returns[ticker].iloc[wi]
            r_m = returns['SPY'].iloc[wi]
            if np.isnan(r_s) or np.isnan(r_m):
                ar_row[d] = np.nan
            else:
                ar_row[d] = r_s - (alpha + beta * r_m)
        rows.append(ar_row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_car_curve(ar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-event AR paths into an average CAR curve with SE bands.
    Returns DataFrame indexed by relative day with columns:
      mean_ar, mean_car, se, ci_lower, ci_upper, n
    """
    rel_days = list(range(-PRE_DAYS, POST_DAYS + 1))
    if ar_df.empty:
        return pd.DataFrame()

    records = []
    for d in rel_days:
        col = ar_df[d].dropna()
        n   = len(col)
        mean_ar = col.mean() if n > 0 else np.nan
        se      = col.std() / np.sqrt(n) if n > 1 else np.nan
        records.append({'rel_day': d, 'mean_ar': mean_ar, 'se': se, 'n': n})

    curve = pd.DataFrame(records).set_index('rel_day')

    # Cumulate from day -PRE_DAYS forward
    curve['mean_car'] = curve['mean_ar'].cumsum()

    # Re-center so CAR = 0 at the event day's start (just before AR_0 added)
    # i.e. CAR_t for t < 0 should be the "pre-drift", CAR_0 includes AR_0
    # (cumsum already does this correctly given ordering -5..+10)

    # CI band on the CUMULATIVE series: variance of a sum of independent
    # ARs accumulates -> se_car_t = sqrt(sum of se_i^2 for i=-5..t)
    se_sq_cum = (curve['se'] ** 2).cumsum()
    curve['car_se'] = np.sqrt(se_sq_cum)
    curve['ci_lower'] = curve['mean_car'] - 1.96 * curve['car_se']
    curve['ci_upper'] = curve['mean_car'] + 1.96 * curve['car_se']

    return curve


def plot_car_curves(prices: pd.DataFrame, returns: pd.DataFrame, save=True):
    """
    One panel per ticker: average CAR curve from -5 to +10 days,
    with 95% confidence band, plus a second row split by event type.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        'Event Study — Average Cumulative Abnormal Return (CAR)\n'
        'Day 0 = event date. Shaded band = 95% CI. '
        'Band excluding zero = statistically distinguishable reaction.',
        fontsize=12, fontweight='bold'
    )

    colors = {'RKLB': '#1D9E75', 'ASTS': '#378ADD', 'LUNR': '#7F77DD'}

    for i, ticker in enumerate(['RKLB', 'ASTS', 'LUNR']):
        events = EVENTS.get(ticker, [])
        ar_df  = compute_ar_paths(ticker, events, prices, returns)
        curve  = compute_car_curve(ar_df)

        # ── Row 0: all events combined ──────────────────────────────────
        ax = axes[0, i]
        if curve.empty:
            ax.set_title(f'{ticker} — insufficient data')
        else:
            x = curve.index
            ax.plot(x, curve['mean_car'] * 100, color=colors[ticker],
                    linewidth=2.2, marker='o', markersize=4,
                    label=f'Avg CAR (n={int(curve["n"].max())} events)')
            ax.fill_between(x, curve['ci_lower'] * 100, curve['ci_upper'] * 100,
                            alpha=0.18, color=colors[ticker], label='95% CI')
            ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Event day')
            ax.axhline(0, color='black', linestyle=':', alpha=0.3)
            ax.set_title(f'{ticker} — All Events', fontweight='bold')
            ax.set_xlabel('Trading days relative to event')
            ax.set_ylabel('Avg CAR (%)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.22)

        # ── Row 1: split by event type (launch vs contract vs milestone) ──
        ax2 = axes[1, i]
        any_plotted = False
        for etype, color in EVENT_COLORS.items():
            sub_events = [e for e in events if e['type'] == etype]
            if len(sub_events) < 2:
                continue
            ar_sub   = compute_ar_paths(ticker, sub_events, prices, returns)
            curve_sub = compute_car_curve(ar_sub)
            if curve_sub.empty:
                continue
            x = curve_sub.index
            ax2.plot(x, curve_sub['mean_car'] * 100, color=color,
                     linewidth=1.8, marker='o', markersize=3,
                     label=f'{etype.capitalize()} (n={int(curve_sub["n"].max())})')
            any_plotted = True

        ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(0, color='black', linestyle=':', alpha=0.3)
        ax2.set_title(f'{ticker} — By Event Type', fontweight='bold')
        ax2.set_xlabel('Trading days relative to event')
        ax2.set_ylabel('Avg CAR (%)')
        if any_plotted:
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'Not enough events\nper type to split',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=9)
        ax2.grid(True, alpha=0.22)

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/car_curves.png", dpi=150, bbox_inches='tight')
        print("Saved: data/car_curves.png")
    plt.close()


def print_car_summary(prices: pd.DataFrame, returns: pd.DataFrame):
    print("\n" + "="*65)
    print("  CAR EVENT-WINDOW CURVES  (Day 0 = event, window [-5, +10])")
    print("="*65)

    for ticker in ['RKLB', 'ASTS', 'LUNR']:
        events = EVENTS.get(ticker, [])
        ar_df  = compute_ar_paths(ticker, events, prices, returns)
        curve  = compute_car_curve(ar_df)
        if curve.empty:
            print(f"\n  {ticker}: insufficient data")
            continue

        n = int(curve['n'].max())
        car_at_0   = curve.loc[0, 'mean_car'] * 100
        car_at_end = curve.loc[POST_DAYS, 'mean_car'] * 100
        car_pre    = curve.loc[-PRE_DAYS, 'mean_car'] * 100  # essentially AR at -5

        # Is day +10 CI excluding zero?
        sig_end = (curve.loc[POST_DAYS, 'ci_lower'] > 0 or
                   curve.loc[POST_DAYS, 'ci_upper'] < 0)

        print(f"\n  {ticker}  (n={n} events)")
        print(f"    CAR at day  0:  {car_at_0:+.2f}%")
        print(f"    CAR at day +10: {car_at_end:+.2f}%  "
              f"{'(significant)' if sig_end else '(not significant — wide CI)'}")

        drift = car_at_end - car_at_0
        if abs(drift) > 1 and sig_end:
            direction = "continues rising" if drift > 0 else "reverses"
            print(f"    → Post-event drift detected: CAR {direction} "
                  f"by {drift:+.2f}% after the event day.")
        else:
            print(f"    → No clear post-event drift; reaction appears "
                  f"to settle near day 0.")

    print("\n  CAVEAT: with only 5-7 events per ticker, confidence intervals")
    print("  are wide. Treat these as exploratory patterns, not proven edges.")
    print("="*65)


if __name__ == "__main__":
    from src.data import fetch_price_history, compute_returns
    prices  = fetch_price_history()
    returns = compute_returns(prices)
    print_car_summary(prices, returns)
    plot_car_curves(prices, returns)
