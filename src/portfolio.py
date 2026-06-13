"""
portfolio.py
------------
Portfolio risk analysis for RKLB, ASTS, and LUNR.

This module answers:
  If I own these space stocks, how concentrated is my risk?
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import SPACE_TICKERS, compute_returns


DEFAULT_WEIGHTS = {
    "RKLB": 0.40,
    "ASTS": 0.35,
    "LUNR": 0.25,
}


def compute_portfolio_returns(returns, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    available = [t for t in weights if t in returns.columns]
    w = pd.Series({t: weights[t] for t in available})
    w = w / w.sum()
    return returns[available].mul(w, axis=1).sum(axis=1)


def compute_portfolio_beta(returns, portfolio_returns):
    cov = portfolio_returns.cov(returns["SPY"])
    var = returns["SPY"].var()
    return cov / var


def compute_risk_contribution(returns, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    tickers = [t for t in weights if t in returns.columns]
    w = pd.Series({t: weights[t] for t in tickers})
    w = w / w.sum()

    cov = returns[tickers].cov() * 252
    portfolio_vol = np.sqrt(w.T @ cov @ w)
    marginal = cov @ w / portfolio_vol
    contribution = w * marginal / portfolio_vol

    return contribution.sort_values(ascending=False)


def compute_portfolio_summary(prices, weights=None):
    returns = compute_returns(prices)
    port = compute_portfolio_returns(returns, weights)

    equity = (1 + port).cumprod()
    drawdown = equity / equity.cummax() - 1

    beta = compute_portfolio_beta(returns, port)
    annual_return = port.mean() * 252
    annual_vol = port.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    summary = {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "beta_vs_spy": beta,
        "total_return": equity.iloc[-1] - 1,
    }

    risk_contrib = compute_risk_contribution(returns, weights)
    return summary, risk_contrib, port


def print_portfolio_summary(prices, weights=None):
    summary, risk_contrib, _ = compute_portfolio_summary(prices, weights)

    print("\n" + "=" * 60)
    print("  NEW SPACE PORTFOLIO RISK MODEL")
    print("=" * 60)
    print(f"  Total return:       {summary['total_return']:.1%}")
    print(f"  Annual return:      {summary['annual_return']:.1%}")
    print(f"  Annual volatility:  {summary['annual_vol']:.1%}")
    print(f"  Sharpe ratio:       {summary['sharpe']:.2f}")
    print(f"  Max drawdown:       {summary['max_drawdown']:.1%}")
    print(f"  Beta vs SPY:        {summary['beta_vs_spy']:.2f}")

    print("\n  Risk contribution:")
    for ticker, contribution in risk_contrib.items():
        print(f"    {ticker}: {contribution:.1%}")

    print("=" * 60)


def plot_portfolio_risk(prices, weights=None, save=True):
    summary, risk_contrib, port = compute_portfolio_summary(prices, weights)
    equity = (1 + port).cumprod()
    drawdown = equity / equity.cummax() - 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("New Space Portfolio Risk Model", fontweight="bold")

    axes[0].plot(equity.index, equity, linewidth=2)
    axes[0].set_title("Portfolio Equity")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(drawdown.index, drawdown, color="#A32D2D", linewidth=2)
    axes[1].set_title("Portfolio Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].grid(True, alpha=0.25)

    axes[2].bar(risk_contrib.index, risk_contrib.values)
    axes[2].set_title("Contribution to Risk")
    axes[2].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()

    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/portfolio_risk.png", dpi=150, bbox_inches="tight")
        print("Saved: data/portfolio_risk.png")

    plt.close()


if __name__ == "__main__":
    from src.data import fetch_price_history

    prices = fetch_price_history(period="2y")
    print_portfolio_summary(prices)
    plot_portfolio_risk(prices)
