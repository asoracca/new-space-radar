"""
charts.py
---------
Portfolio-level charts for new-space-radar.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPACE_TICKERS = ["RKLB", "ASTS", "LUNR"]


def _save(name):
    Path("data").mkdir(exist_ok=True)
    path = f"data/{name}"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_normalized_prices(prices):
    norm = prices / prices.iloc[0] * 100
    plt.figure(figsize=(12, 6))
    for col in norm.columns:
        lw = 2.2 if col in SPACE_TICKERS else 1.6
        alpha = 0.9 if col in SPACE_TICKERS else 0.55
        plt.plot(norm.index, norm[col], label=col, linewidth=lw, alpha=alpha)
    plt.axhline(100, color="black", linestyle="--", alpha=0.25)
    plt.title("Normalized Performance: New Space vs SPY")
    plt.ylabel("Growth of $100")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save("normalized_prices.png")


def plot_rolling_beta(betas):
    plt.figure(figsize=(12, 6))
    for col in betas.columns:
        plt.plot(betas.index, betas[col], label=col, linewidth=1.8)
    plt.axhline(1, color="black", linestyle="--", alpha=0.35)
    plt.title("60-Day Rolling Beta vs SPY")
    plt.ylabel("Beta")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save("rolling_beta.png")


def plot_drawdown(prices):
    plt.figure(figsize=(12, 6))
    for ticker in [t for t in SPACE_TICKERS + ["SPY"] if t in prices.columns]:
        dd = prices[ticker] / prices[ticker].cummax() - 1
        plt.plot(dd.index, dd, label=ticker, linewidth=1.7)
    plt.title("Drawdown From Prior Peak")
    plt.ylabel("Drawdown")
    plt.gca().yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save("drawdown.png")


def plot_return_distribution(returns):
    tickers = [t for t in SPACE_TICKERS if t in returns.columns]
    fig, axes = plt.subplots(1, len(tickers), figsize=(5 * len(tickers), 4))
    if len(tickers) == 1:
        axes = [axes]
    for ax, ticker in zip(axes, tickers):
        r = returns[ticker].dropna()
        ax.hist(r, bins=40, alpha=0.75, edgecolor="white")
        ax.axvline(r.mean(), color="red", linestyle="--", label="Mean")
        ax.set_title(f"{ticker} Daily Return Distribution")
        ax.set_xlabel("Log return")
        ax.grid(True, alpha=0.25)
        ax.legend()
    _save("return_distributions.png")
