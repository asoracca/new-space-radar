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


def run_anomaly_backtest(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    hold_days: int = 5,
    vol_threshold: float = 2.0,
    price_threshold: float = 2.0,
    corr_threshold: float = 0.30,
    transaction_cost_bps: float = 20.0,
) -> pd.DataFrame:
    returns = compute_returns(prices)
    vol_z = compute_volume_zscore(volume)
    price_z = compute_price_zscore(returns)
    corr = compute_rolling_correlation(returns)

    trades = []
    cost = transaction_cost_bps / 10000

    for ticker in SPACE_TICKERS:
        if ticker not in prices.columns:
            continue

        spy_corr_key = f"{ticker}_vs_SPY"
        if spy_corr_key not in corr:
            continue

        common_dates = (
            prices.index
            .intersection(vol_z.index)
            .intersection(price_z.index)
            .intersection(corr[spy_corr_key].dropna().index)
        )

        for entry_date in common_dates:
            vz = float(vol_z.loc[entry_date, ticker])
            pz = float(price_z.loc[entry_date, ticker])
            spy_corr = float(corr[spy_corr_key].loc[entry_date])

            signal = (
                vz >= vol_threshold
                and pz >= price_threshold
                and spy_corr < corr_threshold
            )

            if not signal:
                continue

            entry_idx = prices.index.get_loc(entry_date)
            exit_idx = entry_idx + hold_days

            if exit_idx >= len(prices):
                continue

            exit_date = prices.index[exit_idx]
            entry_price = float(prices.loc[entry_date, ticker])
            exit_price = float(prices.loc[exit_date, ticker])

            if entry_price <= 0:
                continue

            gross_return = exit_price / entry_price - 1
            net_return = gross_return - cost

            spy_entry = float(prices.loc[entry_date, "SPY"])
            spy_exit = float(prices.loc[exit_date, "SPY"])
            spy_return = spy_exit / spy_entry - 1

            trades.append({
                "ticker": ticker,
                "entry_date": entry_date.date(),
                "exit_date": exit_date.date(),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "hold_days": hold_days,
                "vol_z": round(vz, 2),
                "price_z": round(pz, 2),
                "spy_corr": round(spy_corr, 2),
                "return": round(net_return, 4),
                "spy_return": round(spy_return, 4),
                "alpha_vs_spy": round(net_return - spy_return, 4),
                "outcome": "WIN" if net_return > 0 else "LOSS",
            })

    df = pd.DataFrame(trades)

    if not df.empty:
        df = df.sort_values("entry_date").reset_index(drop=True)
        Path("data").mkdir(exist_ok=True)
        df.to_csv("data/anomaly_backtest_trades.csv", index=False)

    return df


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def print_backtest_summary(trades: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  ANOMALY FOLLOW-THROUGH BACKTEST")
    print("=" * 65)

    if trades.empty:
        print("  No trades generated.")
        print("  Try lowering thresholds or using a longer data period.")
        print("=" * 65)
        return

    rets = trades["return"]
    alpha = trades["alpha_vs_spy"]

    equity = (1 + rets).cumprod()
    total_return = equity.iloc[-1] - 1
    win_rate = (rets > 0).mean()
    avg_return = rets.mean()
    median_return = rets.median()
    avg_alpha = alpha.mean()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252 / trades["hold_days"].iloc[0]) if rets.std() > 0 else np.nan
    max_dd = _max_drawdown(equity)

    print(f"  Total trades:        {len(trades)}")
    print(f"  Win rate:            {win_rate:.1%}")
    print(f"  Avg trade return:    {avg_return:.2%}")
    print(f"  Median trade return: {median_return:.2%}")
    print(f"  Total compounded:    {total_return:.2%}")
    print(f"  Avg alpha vs SPY:    {avg_alpha:.2%}")
    print(f"  Trade Sharpe:        {sharpe:.2f}")
    print(f"  Max drawdown:        {max_dd:.2%}")

    print("\n  By ticker:")
    grouped = trades.groupby("ticker").agg(
        trades=("return", "count"),
        win_rate=("return", lambda x: (x > 0).mean()),
        avg_return=("return", "mean"),
        avg_alpha=("alpha_vs_spy", "mean"),
    )
    print(grouped.to_string(float_format=lambda x: f"{x:.2%}"))

    print("\n  Recent trades:")
    print(trades.tail(10).to_string(index=False))
    print("=" * 65)


def plot_backtest(trades: pd.DataFrame, save: bool = True):
    if trades.empty:
        return

    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["equity"] = (1 + trades["return"]).cumprod()
    trades["spy_equity"] = (1 + trades["spy_return"]).cumprod()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("New Space Radar - Anomaly Follow-Through Backtest", fontweight="bold")

    axes[0, 0].plot(trades["entry_date"], trades["equity"], marker="o", label="Strategy")
    axes[0, 0].plot(trades["entry_date"], trades["spy_equity"], marker="o", label="SPY during trades", alpha=0.75)
    axes[0, 0].set_title("Compounded Trade Equity")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.25)

    colors = ["#1D9E75" if r > 0 else "#A32D2D" for r in trades["return"]]
    axes[0, 1].bar(range(len(trades)), trades["return"] * 100, color=colors, alpha=0.85)
    axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.4)
    axes[0, 1].set_title("Per-Trade Return")
    axes[0, 1].set_ylabel("Return (%)")
    axes[0, 1].grid(True, alpha=0.25)

    trades.boxplot(column="return", by="ticker", ax=axes[1, 0])
    axes[1, 0].set_title("Return Distribution by Ticker")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Return")
    fig.suptitle("New Space Radar - Anomaly Follow-Through Backtest", fontweight="bold")

    axes[1, 1].scatter(trades["spy_corr"], trades["return"] * 100, alpha=0.8)
    axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.4)
    axes[1, 1].set_title("Return vs SPY Correlation at Entry")
    axes[1, 1].set_xlabel("30-day corr vs SPY")
    axes[1, 1].set_ylabel("Trade return (%)")
    axes[1, 1].grid(True, alpha=0.25)

    plt.tight_layout()

    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/anomaly_backtest.png", dpi=150, bbox_inches="tight")
        print("Saved: data/anomaly_backtest.png")

    plt.close()


if __name__ == "__main__":
    from src.data import fetch_price_history, fetch_volume_history

    prices = fetch_price_history(period="2y")
    volume = fetch_volume_history(period="2y")
    trades = run_anomaly_backtest(prices, volume)
    print_backtest_summary(trades)
    plot_backtest(trades)
