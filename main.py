cat > main.py << 'ENDOFFILE'
"""
main.py
-------
Full new-space-radar pipeline.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from src.data import (fetch_price_history, fetch_volume_history,
                      compute_returns, print_summary)
from src.events import analyze_all_events, print_event_summary, plot_event_impact
from src.correlation import (compute_rolling_correlation,
                              print_correlation_summary,
                              plot_correlations,
                              detect_correlation_breaks)
from src.momentum import (compute_volume_zscore, compute_price_zscore,
                          detect_anomalies, print_momentum_summary,
                          plot_momentum)
from src.signal import run_signal


def main():
    Path("data").mkdir(exist_ok=True)

    print("\n── 1. Portfolio Data ───────────────────────────────")
    prices  = fetch_price_history(period="2y")
    volume  = fetch_volume_history(period="2y")
    returns = compute_returns(prices)
    print_summary(prices, returns)

    print("\n── 2. Event Impact Analysis ────────────────────────")
    events = analyze_all_events(prices, returns)
    print_event_summary(events)
    plot_event_impact(events)

    print("\n── 3. Correlation Analysis ─────────────────────────")
    corr = compute_rolling_correlation(returns)
    print_correlation_summary(corr, returns)
    plot_correlations(corr)
    breaks = detect_correlation_breaks(corr)
    if not breaks.empty:
        print(f"\n  Correlation breaks found: {len(breaks)}")
        print(breaks.tail(5).to_string(index=False))

    print("\n── 4. Momentum & Anomaly Detection ─────────────────")
    vol_z     = compute_volume_zscore(volume)
    price_z   = compute_price_zscore(returns)
    anomalies = detect_anomalies(vol_z, price_z)
    print_momentum_summary(vol_z, price_z, anomalies)
    plot_momentum(vol_z, price_z, anomalies)

    print("\n── 5. Today's Signal ───────────────────────────────")
    run_signal()


if __name__ == "__main__":
    main()
ENDOFFILE