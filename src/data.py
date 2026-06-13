"""
data.py
-------
Pulls price and volume history for RKLB, ASTS, LUNR and SPY benchmark.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

TICKERS = ['RKLB', 'ASTS', 'LUNR', 'SPY']
SPACE_TICKERS = ['RKLB', 'ASTS', 'LUNR']


def fetch_price_history(period="2y"):
    """
    Fetch adjusted close prices for all tickers.
    Returns a DataFrame with tickers as columns.
    """
    print(f"Fetching price history for {TICKERS}...")
    raw = yf.download(TICKERS, period=period, 
                      interval="1d", progress=False)
    prices = raw['Close'].copy()
    prices.index = pd.to_datetime(prices.index)
    print(f"  Loaded {len(prices)} trading days")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def fetch_volume_history(period="2y"):
    """
    Fetch volume data for all tickers.
    """
    raw = yf.download(TICKERS, period=period,
                      interval="1d", progress=False)
    return raw['Volume'].copy()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Daily log returns for each ticker.
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_excess_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Excess return = stock return - SPY return (market-adjusted).
    Removes broad market moves to isolate stock-specific action.
    """
    excess = pd.DataFrame(index=returns.index)
    for ticker in SPACE_TICKERS:
        if ticker in returns.columns:
            excess[ticker] = returns[ticker] - returns['SPY']
    return excess.dropna()


def compute_beta(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling 60-day beta for each space stock vs SPY.
    Beta > 1 = amplifies market moves.
    Beta < 0 = moves opposite to market (rare but happens with LUNR).
    """
    betas = pd.DataFrame(index=returns.index)
    for ticker in SPACE_TICKERS:
        if ticker in returns.columns:
            cov = (returns[ticker].rolling(window)
                   .cov(returns['SPY']))
            var = returns['SPY'].rolling(window).var()
            betas[ticker] = cov / var
    return betas.dropna()


def print_summary(prices: pd.DataFrame, returns: pd.DataFrame):
    print("\n" + "="*55)
    print("  NEW SPACE PORTFOLIO SUMMARY")
    print("="*55)
    for ticker in SPACE_TICKERS:
        if ticker not in prices.columns:
            continue
        current = prices[ticker].iloc[-1]
        ret_1m  = prices[ticker].pct_change(21).iloc[-1]
        ret_3m  = prices[ticker].pct_change(63).iloc[-1]
        vol_ann = returns[ticker].std() * np.sqrt(252)
        print(f"\n  {ticker}")
        print(f"    Price:       ${current:.2f}")
        print(f"    1M return:   {ret_1m:.1%}")
        print(f"    3M return:   {ret_3m:.1%}")
        print(f"    Annual vol:  {vol_ann:.1%}")
    print("="*55)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    prices  = fetch_price_history()
    volume  = fetch_volume_history()
    returns = compute_returns(prices)
    print_summary(prices, returns)