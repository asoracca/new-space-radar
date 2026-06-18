"""
event_filter.py  —  add to new-space-radar/src/
------------------------------------------------
Statistical significance filter for CAR (Cumulative Abnormal Return) event studies.

What this does:
  Your event study already computes CARs and t-statistics for each event.
  This module adds a filter: only keep events where the result is REAL,
  not just noise.

The rule: |t-stat| > 1.96  →  95% confidence the abnormal return is real
           |t-stat| > 2.58  →  99% confidence (stricter)

Why this matters:
  If you have 20 events and report all of them, some will look significant by
  pure chance (false positives). Filtering on |t| > 1.96 means you only
  report the ones that are statistically meaningful.

  This is exactly what academic papers do. If you put this on your GitHub
  with proper t-stat filtering, it looks like proper research.

Also adds:
  - Cohen's d effect size (how BIG is the abnormal return, not just significant)
  - Cross-event aggregation (average CAR across all events for a ticker)
  - Publication-style summary table

Run standalone:
    python src/event_filter.py

Or import into your existing event study:
    from src.event_filter import filter_significant_events, summarise_by_ticker
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Core filter ──────────────────────────────────────────────────────────────

def filter_significant_events(events_df, t_col="t_stat", confidence=0.95):
    """
    Filter events DataFrame to only statistically significant results.

    events_df must have a column with t-statistics (default: 't_stat').
    confidence: 0.95 = 95% (t > 1.96), 0.99 = 99% (t > 2.58)

    Returns (significant_df, rejected_df, threshold)
    """
    thresholds = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    threshold = thresholds.get(confidence, 1.960)

    if t_col not in events_df.columns:
        raise ValueError(f"Column '{t_col}' not found. Available: {events_df.columns.tolist()}")

    t_vals = events_df[t_col].abs()
    significant = events_df[t_vals >= threshold].copy()
    rejected    = events_df[t_vals <  threshold].copy()

    sig_rate = len(significant) / len(events_df) if len(events_df) > 0 else 0

    print(f"\n── Significance Filter ({confidence*100:.0f}% confidence, |t| ≥ {threshold}) ──")
    print(f"  Total events:       {len(events_df)}")
    print(f"  Significant:        {len(significant)}  ({sig_rate:.0%})")
    print(f"  Rejected (noise):   {len(rejected)}")
    print(f"  Expected false pos: ~{len(events_df) * (1 - confidence):.1f} "
          f"(events that look sig by chance)")

    return significant, rejected, threshold


def cohens_d(car_values, benchmark_mean=0.0):
    """
    Cohen's d = (mean CAR - benchmark) / std(CAR)

    Measures effect SIZE — how large is the abnormal return relative to its variation?
    d < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large

    benchmark_mean: what we'd expect under H0 (usually 0 — no abnormal return)
    """
    cars = np.array(car_values)
    if len(cars) < 2 or cars.std() == 0:
        return np.nan
    return (cars.mean() - benchmark_mean) / cars.std()


# ── Aggregation ──────────────────────────────────────────────────────────────

def summarise_by_ticker(events_df, ticker_col="ticker", car_col="car",
                         t_col="t_stat", date_col="event_date"):
    """
    Aggregate significant events by ticker.

    For each ticker, computes:
      - n_events: how many significant events
      - mean_car: average cumulative abnormal return
      - cohens_d: effect size
      - pct_positive: what % of events were positive (bullish)
      - best/worst event
    """
    rows = []

    for ticker, group in events_df.groupby(ticker_col):
        cars  = group[car_col].values
        t_stats = group[t_col].values if t_col in group.columns else [np.nan]*len(group)

        row = {
            "ticker":        ticker,
            "n_events":      len(group),
            "mean_car":      cars.mean(),
            "median_car":    np.median(cars),
            "std_car":       cars.std(),
            "cohens_d":      cohens_d(cars),
            "pct_positive":  (cars > 0).mean(),
            "best_car":      cars.max(),
            "worst_car":     cars.min(),
            "mean_t_stat":   np.nanmean(t_stats),
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("mean_car", ascending=False)
    return summary


def print_summary_table(summary_df):
    """Print a clean publication-style summary table."""
    print("\n" + "="*72)
    print("  EVENT STUDY SUMMARY — Significant Events Only")
    print("  (Cumulative Abnormal Returns, t-filtered at 95% confidence)")
    print("="*72)
    print(f"  {'Ticker':8}  {'N':>4}  {'Mean CAR':>10}  {'Cohen d':>9}  "
          f"{'% Positive':>11}  {'Mean |t|':>9}")
    print(f"  {'─'*66}")

    for _, row in summary_df.iterrows():
        d_label = ""
        if abs(row["cohens_d"]) > 0.8:
            d_label = "★ large"
        elif abs(row["cohens_d"]) > 0.5:
            d_label = "◆ medium"
        elif abs(row["cohens_d"]) > 0.2:
            d_label = "· small"

        print(f"  {row['ticker']:8}  {row['n_events']:>4}  "
              f"{row['mean_car']:>+9.2%}  {row['cohens_d']:>7.2f} {d_label:<8}  "
              f"{row['pct_positive']:>10.0%}  {row['mean_t_stat']:>8.2f}")

    print(f"  {'─'*66}")
    print(f"\n  Cohen's d: < 0.2 negligible | 0.2–0.5 small | 0.5–0.8 medium | > 0.8 large")
    print(f"  This tells you HOW BIG the effect is, not just whether it's real.")
    print("="*72)


def plot_car_significance(events_df, car_col="car", t_col="t_stat",
                           ticker_col="ticker", threshold=1.96):
    """
    Scatter plot: t-stat vs CAR for all events.
    Green = significant, grey = noise.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("CAR Event Study — Significance Analysis", fontsize=13, fontweight="bold")

    t_vals = events_df[t_col].abs()
    sig    = events_df[t_vals >= threshold]
    noise  = events_df[t_vals <  threshold]

    # Panel 1: Scatter t-stat vs CAR
    ax = axes[0]
    if len(noise) > 0:
        ax.scatter(noise[t_col], noise[car_col] * 100, color="grey", alpha=0.5,
                   s=40, label=f"Not significant (n={len(noise)})")
    if len(sig) > 0:
        ax.scatter(sig[t_col], sig[car_col] * 100, color="#1D9E75", alpha=0.8,
                   s=60, label=f"Significant (n={len(sig)})", zorder=5)
        if ticker_col in events_df.columns:
            for _, row in sig.iterrows():
                ax.annotate(row[ticker_col], (row[t_col], row[car_col]*100),
                            fontsize=7, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")

    ax.axvline( threshold, color="orange", linestyle="--", alpha=0.7, label=f"|t|={threshold}")
    ax.axvline(-threshold, color="orange", linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("CAR (%)")
    ax.set_title("All Events: t-stat vs CAR")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: CAR distribution — significant vs noise
    ax2 = axes[1]
    if len(noise) > 0:
        ax2.hist(noise[car_col] * 100, bins=15, alpha=0.5, color="grey",
                 label=f"Noise (n={len(noise)})", density=True)
    if len(sig) > 0:
        ax2.hist(sig[car_col] * 100, bins=15, alpha=0.7, color="#1D9E75",
                 label=f"Significant (n={len(sig)})", density=True)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("CAR (%)")
    ax2.set_ylabel("Density")
    ax2.set_title("CAR Distribution: Significant vs Noise")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    Path("data").mkdir(exist_ok=True)
    plt.savefig("data/car_significance.png", dpi=150, bbox_inches="tight")
    print("Saved: data/car_significance.png")


# ── Demo (runs if you don't have real events yet) ────────────────────────────

def demo_with_synthetic_events():
    """
    Demo using synthetic event data — replace with your real event study output.
    Shows exactly what the output looks like.
    """
    np.random.seed(42)
    tickers = ["RKLB", "ASTS", "LUNR", "RDW", "MNTS", "SPCE", "ASTR"]
    events  = []

    for ticker in tickers:
        n = np.random.randint(4, 10)
        for _ in range(n):
            # Mix of real signal + noise
            car = np.random.normal(0.03, 0.08)   # mean 3% abnormal return
            noise_std = np.random.uniform(0.01, 0.05)
            t_stat = car / noise_std + np.random.normal(0, 0.3)
            events.append({
                "ticker":     ticker,
                "event_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 500)),
                "car":        car,
                "t_stat":     t_stat,
            })

    df = pd.DataFrame(events)

    print("\n⚠️  Running with SYNTHETIC demo data.")
    print("    Replace this with output from your real event_study.py\n")

    sig, noise, threshold = filter_significant_events(df, confidence=0.95)

    if len(sig) > 0:
        summary = summarise_by_ticker(sig)
        print_summary_table(summary)
        plot_car_significance(df, threshold=threshold)
    else:
        print("  No significant events in demo data — try lowering confidence to 0.90")

    return df, sig


if __name__ == "__main__":
    # Try to load real event study output first
    real_path = Path("data/event_results.csv")
    if real_path.exists():
        print(f"Loading real event data from {real_path}")
        df = pd.read_csv(real_path)
        sig, noise, threshold = filter_significant_events(df, confidence=0.95)
        if len(sig) > 0:
            summary = summarise_by_ticker(sig)
            print_summary_table(summary)
            plot_car_significance(df, threshold=threshold)
    else:
        print(f"No data/event_results.csv found — running demo.")
        print(f"To use with your real data: save your event study results to data/event_results.csv")
        print(f"Required columns: ticker, car (float), t_stat (float), event_date")
        demo_with_synthetic_events()
