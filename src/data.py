"""
data.py
-------
Price, volume, and risk metrics for RKLB, ASTS, LUNR vs SPY benchmark.

WHY THESE STOCKS:
  RKLB (Rocket Lab): Only US public company launching rockets regularly
    besides SpaceX. Revenue-generating, growing backlog, Neutron rocket
    in development for larger payloads. The "picks and shovels" play.

  ASTS (AST SpaceMobile): Building a satellite constellation to provide
    broadband directly to standard smartphones — no special hardware.
    Partnerships with AT&T, Verizon, Vodafone. Pre-revenue but BlueBird
    satellites now in orbit. Enormous TAM if it works.

  LUNR (Intuitive Machines): NASA's primary commercial lunar lander
    contractor. IM-1 was the first US soft lunar landing in 50 years.
    Entirely dependent on government contracts — binary risk profile.

WHY SPY AS BENCHMARK:
  Any return you earn in RKLB/ASTS/LUNR needs to be compared to what
  you could have earned just holding the S&P 500 with zero effort and
  much less risk. If RKLB returns 40% but SPY returns 35%, you only
  earned 5% of *true* alpha — and took on massive extra volatility to get it.
  This is the core insight of the Sharpe ratio.

METRICS COMPUTED:
  - Sharpe ratio: return per unit of risk (higher = better risk-adjusted return)
  - Max drawdown: worst peak-to-trough loss (the "how bad could it get" number)
  - Rolling beta: how much market exposure you have at any given time
  - Excess return: return above SPY (the only return that justifies the extra risk)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

TICKERS       = ['RKLB', 'ASTS', 'LUNR', 'SPY']
SPACE_TICKERS = ['RKLB', 'ASTS', 'LUNR']


def fetch_price_history(period="2y"):
    print(f"Fetching price history for {TICKERS}...")
    raw    = yf.download(TICKERS, period=period, interval="1d", progress=False)
    prices = raw['Close'].copy()
    prices.index = pd.to_datetime(prices.index)
    # Flatten MultiIndex columns if present
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    print(f"  Loaded {len(prices)} trading days")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def fetch_volume_history(period="2y"):
    raw = yf.download(TICKERS, period=period, interval="1d", progress=False)
    vol = raw['Volume'].copy()
    if isinstance(vol.columns, pd.MultiIndex):
        vol.columns = vol.columns.get_level_values(0)
    return vol


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns: r_t = ln(P_t / P_{t-1})

    We use log returns (not simple returns) because:
    1. They're time-additive: weekly return = sum of daily log returns
    2. They're more normally distributed (easier to model statistically)
    3. Standard in quantitative finance
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_excess_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Excess return = stock return - SPY return (same day).
    This strips out the market move and isolates stock-specific return.
    If SPY drops 2% and RKLB drops 2%, RKLB's excess return is 0 —
    it didn't do anything special, it just moved with the market.
    """
    excess = pd.DataFrame(index=returns.index)
    for ticker in SPACE_TICKERS:
        if ticker in returns.columns:
            excess[ticker] = returns[ticker] - returns['SPY']
    return excess.dropna()


def compute_beta(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling beta: Cov(stock, SPY) / Var(SPY) over 60 trading days.

    Beta interpretation:
      β = 1.0  → stock moves 1:1 with market (no amplification)
      β = 2.0  → stock moves 2x the market (high leverage to broad moves)
      β = 0.0  → stock is uncorrelated with market (pure alpha or noise)
      β < 0    → stock moves opposite to market (rare, happens around LUNR events)

    60-day window chosen because:
      - Short enough to capture regime changes (post-launch vs pre-launch)
      - Long enough to have statistical significance (≥30 obs recommended)
    """
    betas = pd.DataFrame(index=returns.index)
    for ticker in SPACE_TICKERS:
        if ticker in returns.columns:
            cov           = returns[ticker].rolling(window).cov(returns['SPY'])
            var           = returns['SPY'].rolling(window).var()
            betas[ticker] = cov / var
    return betas.dropna()


def compute_sharpe(returns: pd.DataFrame,
                   risk_free_annual: float = 0.05) -> dict:
    """
    Annualized Sharpe ratio = (mean excess return / std) * sqrt(252)

    Risk-free rate set to 5% (approx US T-bill yield as of 2024-2025).
    Sharpe > 1.0 is considered good. > 2.0 is exceptional.
    Most new space stocks will have negative Sharpe over most periods —
    the volatility is too high relative to the return.

    This is why event-driven trading (not buy-and-hold) makes more sense here.
    """
    rf_daily = risk_free_annual / 252
    sharpes  = {}
    for ticker in SPACE_TICKERS:
        if ticker not in returns.columns:
            continue
        excess         = returns[ticker] - rf_daily
        sharpes[ticker] = (excess.mean() / excess.std()) * np.sqrt(252)
    return sharpes


def compute_max_drawdown(prices: pd.DataFrame) -> dict:
    """
    Max drawdown = largest peak-to-trough decline in the period.
    If a stock went from $50 → $10, max drawdown = -80%.

    This is the most important risk metric for position sizing:
      Kelly criterion and risk-of-ruin calculations both depend on it.
      Rule of thumb: never risk more than (max_drawdown / 2) of portfolio in one name.
    """
    drawdowns = {}
    for ticker in SPACE_TICKERS:
        if ticker not in prices.columns:
            continue
        roll_max          = prices[ticker].cummax()
        dd                = (prices[ticker] - roll_max) / roll_max
        drawdowns[ticker] = dd.min()
    return drawdowns


def print_summary(prices: pd.DataFrame, returns: pd.DataFrame):
    sharpes   = compute_sharpe(returns)
    drawdowns = compute_max_drawdown(prices)

    print("\n" + "="*60)
    print("  NEW SPACE PORTFOLIO — RISK & RETURN SUMMARY")
    print("="*60)
    print(f"  {'Metric':<20} {'RKLB':>10} {'ASTS':>10} {'LUNR':>10} {'SPY':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Price
    row = "  Price ($)"
    for t in SPACE_TICKERS + ['SPY']:
        row += f" {prices[t].iloc[-1]:>10.2f}" if t in prices.columns else f" {'N/A':>10}"
    print(row)

    # 1M return
    row = "  1M return"
    for t in SPACE_TICKERS + ['SPY']:
        v = prices[t].pct_change(21).iloc[-1] if t in prices.columns else np.nan
        row += f" {v:>10.1%}" if not np.isnan(v) else f" {'N/A':>10}"
    print(row)

    # 1Y return
    row = "  1Y return"
    for t in SPACE_TICKERS + ['SPY']:
        v = prices[t].pct_change(252).iloc[-1] if t in prices.columns else np.nan
        row += f" {v:>10.1%}" if not np.isnan(v) else f" {'N/A':>10}"
    print(row)

    # Annual vol
    row = "  Annual vol"
    for t in SPACE_TICKERS + ['SPY']:
        v = returns[t].std() * np.sqrt(252) if t in returns.columns else np.nan
        row += f" {v:>10.1%}" if not np.isnan(v) else f" {'N/A':>10}"
    print(row)

    # Sharpe
    spy_sharpe = ((returns['SPY'] - 0.05/252).mean() /
                  (returns['SPY'] - 0.05/252).std()) * np.sqrt(252)
    row = "  Sharpe (1Y)"
    for t in SPACE_TICKERS:
        v = sharpes.get(t, np.nan)
        row += f" {v:>10.2f}" if not np.isnan(v) else f" {'N/A':>10}"
    row += f" {spy_sharpe:>10.2f}"
    print(row)

    # Max drawdown
    row = "  Max drawdown"
    for t in SPACE_TICKERS:
        v = drawdowns.get(t, np.nan)
        row += f" {v:>10.1%}" if not np.isnan(v) else f" {'N/A':>10}"
    spy_dd = ((prices['SPY'] - prices['SPY'].cummax()) / prices['SPY'].cummax()).min()
    row += f" {spy_dd:>10.1%}"
    print(row)

    print("="*60)
    print("\n  Interpretation:")
    print("  Sharpe < 0 = not worth the risk vs T-bills")
    print("  Sharpe 0-1 = some compensation for risk")
    print("  Sharpe > 1 = good risk-adjusted return")
    print("  Max drawdown tells you how much you'd have lost buying at the peak.")
    print("="*60)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    prices  = fetch_price_history()
    volume  = fetch_volume_history()
    returns = compute_returns(prices)
    print_summary(prices, returns)
