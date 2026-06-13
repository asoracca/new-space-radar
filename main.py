"""
main.py
-------
Full new-space-radar pipeline. Run this to generate all outputs.

Usage:
    python main.py              # full pipeline
    python main.py --signal     # today's signal only (fast)
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

from src.data import (fetch_price_history, fetch_volume_history,
                      compute_returns, compute_beta, print_summary)
from src.events import analyze_all_events, print_event_summary, plot_event_impact
from src.correlation import (compute_rolling_correlation,
                              print_correlation_summary,
                              plot_correlations,
                              detect_correlation_breaks)
from src.momentum import (compute_volume_zscore, compute_price_zscore,
                           compute_rsi, detect_anomalies,
                           print_momentum_summary, plot_momentum)
from src.charts import (plot_normalized_prices, plot_rolling_beta,
                        plot_drawdown, plot_return_distribution)
from src.signal import run_signal


def main(signal_only: bool = False):
    Path("data").mkdir(exist_ok=True)

    if signal_only:
        run_signal()
        return

    print("\n── 1. Portfolio Data ───────────────────────────────")
    prices  = fetch_price_history(period="2y")
    volume  = fetch_volume_history(period="2y")
    returns = compute_returns(prices)
    betas   = compute_beta(returns)
    print_summary(prices, returns)

    print("\n── 2. Portfolio Charts ─────────────────────────────")
    plot_normalized_prices(prices)
    plot_rolling_beta(betas)
    plot_drawdown(prices)
    plot_return_distribution(returns)

    print("\n── 3. Event Impact Analysis ────────────────────────")
    events = analyze_all_events(prices, returns)
    print_event_summary(events)
    plot_event_impact(events)

    print("\n── 4. Correlation Analysis ─────────────────────────")
    corr = compute_rolling_correlation(returns)
    print_correlation_summary(corr, returns)
    plot_correlations(corr)
    breaks = detect_correlation_breaks(corr)
    if not breaks.empty:
        print(f"\n  Correlation breaks found: {len(breaks)}")
        print(breaks.tail(5).to_string(index=False))

    print("\n── 5. Momentum & Anomaly Detection ─────────────────")
    vol_z     = compute_volume_zscore(volume)
    price_z   = compute_price_zscore(returns)
    rsi       = compute_rsi(prices)
    anomalies = detect_anomalies(vol_z, price_z)
    print_momentum_summary(vol_z, price_z, anomalies, rsi)
    plot_momentum(vol_z, price_z, anomalies, rsi)

    print("\n── 6. Today's Signal ───────────────────────────────")
    run_signal()

    print("\n✅  All outputs saved to data/")


if __name__ == "__main__":
    signal_only = "--signal" in sys.argv
    main(signal_only=signal_only)