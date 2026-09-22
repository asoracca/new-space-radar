"""
backtest.py
-----------
Backtest a simple event-driven follow-through strategy for RKLB, ASTS, and LUNR.

Strategy:
  Buy a stock when:
    1. Volume z-score >= 2
    2. Price z-score >= 2
    3. 30-day correlation vs SPY < 0.30

  Hold for 5 trading days.

This tests whether unusual, stock-specific moves in new-space names tend to
continue after the initial catalyst.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import SPACE_TICKERS, compute_returns
from src.momentum import compute_volume_zscore, compute_price_zscore
from src.correlation import compute_rolling_correlation


def evaluate_signals(prices, signals, hold_days=5, transaction_cost_bps=20.0, schedule=None):
    """Signals are known AFTER the indicated close; fill at next session close.

    Close-only input cannot justify a next-open fill. Missing fills are skipped,
    never delayed to a favorable available bar. One active trade per symbol.
    These are trade observations, not a funded portfolio equity curve.
    """
    from src.events import load_schedule
    if hold_days < 1 or transaction_cost_bps < 0:
        raise ValueError('Invalid holding period or costs')
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Prices must have unique sorted session dates')
    schedule = load_schedule() if schedule is None else schedule
    bars = prices.reindex(schedule.index)
    trades, busy_until = [], {}
    for signal in sorted(signals, key=lambda x: (x['signal_date'], x['ticker'])):
        ticker = signal['ticker']
        day = pd.Timestamp(signal['signal_date'])
        if day not in schedule.index or ticker not in bars or 'SPY' not in bars:
            continue
        pos = schedule.index.get_loc(day)
        entry, end = pos+1, pos+1+hold_days
        if end >= len(bars) or entry <= busy_until.get(ticker, -1):
            continue
        values = bars.iloc[[entry,end]][[ticker,'SPY']].to_numpy()
        if not np.isfinite(values).all() or (values <= 0).any():
            continue
        net = values[1,0]/values[0,0]-1-transaction_cost_bps/10000
        baseline = values[1,1]/values[0,1]-1
        trades.append({**signal, 'signal_timestamp': schedule.iloc[pos]['close'].isoformat(),
                       'entry_timestamp': schedule.iloc[entry]['close'].isoformat(),
                       'entry_date': bars.index[entry].date(), 'exit_date': bars.index[end].date(),
                       'entry_price': float(values[0,0]), 'exit_price': float(values[1,0]),
                       'hold_days': hold_days, 'return': float(net), 'spy_return': float(baseline),
                       'alpha_vs_spy': float(net-baseline),
                       'execution': 'next_session_close', 'cost_bps_round_trip': transaction_cost_bps})
        busy_until[ticker] = end
    return pd.DataFrame(trades)


def run_anomaly_backtest(prices, volume, hold_days=5, vol_threshold=2.0,
                         price_threshold=2.0, corr_threshold=.30,
                         transaction_cost_bps=20.0):
    from src.events import load_schedule
    schedule = load_schedule()
    # Preserve exchange sessions before computing returns, even when a bar is absent.
    dates = schedule.index[(schedule.index >= prices.index.min()) & (schedule.index <= prices.index.max())]
    prices, volume = prices.reindex(dates), volume.reindex(dates)
    returns = compute_returns(prices)
    vol_z = compute_volume_zscore(volume)
    price_z = compute_price_zscore(returns)
    corr = compute_rolling_correlation(returns)
    signals = []
    for ticker in SPACE_TICKERS:
        key = f'{ticker}_vs_SPY'
        if ticker not in vol_z or ticker not in price_z or key not in corr:
            continue
        for day in vol_z.index.intersection(price_z.index).intersection(corr[key].index):
            vz, pz, c = vol_z.loc[day,ticker], price_z.loc[day,ticker], corr[key].loc[day]
            if vz >= vol_threshold and pz >= price_threshold and c < corr_threshold:
                signals.append(dict(ticker=ticker,signal_date=day,vol_z=float(vz),
                                    price_z=float(pz),spy_corr=float(c)))
    return evaluate_signals(prices, signals, hold_days, transaction_cost_bps, schedule)


def print_backtest_summary(trades):
    if trades.empty:
        print('No executable trades.')
        return
    print(trades.groupby('ticker').agg(n=('return','count'),mean_return=('return','mean'),
                                      mean_excess_vs_spy=('alpha_vs_spy','mean')))
    print('Exploratory trade observations. Same-stock overlap suppressed. Cross-stock overlap, '
          'capital allocation and selection bias prevent a portfolio-performance claim.')


def plot_backtest(trades, save=True):
    if trades.empty:
        return
    fig, ax = plt.subplots()
    ax.scatter(trades['spy_return']*100, trades['return']*100)
    ax.set(xlabel='Same-period SPY return (%)',ylabel='Net trade return (%)',
           title='Exploratory next-session-close fills; not portfolio equity')
    if save:
        Path('data').mkdir(exist_ok=True)
        fig.savefig('data/anomaly_backtest.png',bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    from src.data import fetch_price_history, fetch_volume_history

    prices = fetch_price_history(period="2y")
    volume = fetch_volume_history(period="2y")
    trades = run_anomaly_backtest(prices, volume)
    print_backtest_summary(trades)
    plot_backtest(trades)
