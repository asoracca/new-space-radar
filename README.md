# new-space-radar 🛰️

A quantitative event-driven analysis toolkit for the emerging space economy.
Tracks **RKLB** (Rocket Lab), **ASTS** (AST SpaceMobile), and **LUNR** (Intuitive Machines) —
three pure-play new space stocks with high volatility and catalyst-driven price action.

---


## Research Question

Do unusual volume spikes combined with stock-specific price movement predict short-term follow-through in RKLB, ASTS, and LUNR?

I built this because I personally hold new-space stocks and wanted a disciplined way to separate real catalyst-driven moves from broad market noise. These companies move on launches, NASA contracts, FCC approvals, satellite milestones, and retail momentum. The goal is not to blindly trade every alert, but to measure whether these events create abnormal returns that are actually worth paying attention to.

## How I Use This

I use this as a daily risk and catalyst scanner, not as an automatic trading system.

When the model flags an anomaly, I check:

- whether the move was stock-specific or market-wide
- whether there was a launch, contract, FCC/NASA update, earnings event, or partnership announcement
- whether the stock historically shows follow-through after similar moves
- whether my current position size still makes sense given volatility, drawdown, and beta
- whether the signal is telling me to add, trim, hedge, or simply investigate

The project is personal: I am interested in the space economy and have put real money into these names, so this is my attempt to turn interest and conviction into a more disciplined research process.

## The idea

New space stocks don't move like normal equities. They react to:
- Rocket launches (success or failure)
- Satellite deployment milestones
- Government contracts (NASA, DoD, Space Force)
- FCC licensing approvals
- Partnership announcements

This project builds a quantitative framework to detect when these events create
**abnormal returns** — price moves that can't be explained by broad market movement alone.

---

## Modules

| Module | What it does |
|---|---|
| `src/data.py` | Price + volume history for RKLB, ASTS, LUNR vs SPY. Sharpe, beta, drawdown stats. |
| `src/charts.py` | Normalized returns, rolling beta, drawdown, return distribution plots |
| `src/events.py` | Catalog of real launch/contract/milestone events + market-model CAR analysis |
| `src/correlation.py` | Rolling 30-day correlations between stocks and vs SPY |
| `src/momentum.py` | Volume + price z-scores, RSI, anomaly detection |
| `src/signal.py` | Daily morning scan — which stock is showing unusual activity right now |
| `src/backtest.py` | Tests whether anomaly signals lead to 5-day follow-through |
| `src/portfolio.py` | Measures portfolio beta, drawdown, and risk contribution |
---

## Methodology

### Event study (market-model CARs)

For each catalogued event:
1. Estimate alpha and beta from a 120-day pre-event estimation window using OLS
2. Compute daily abnormal returns: `AR = r_stock − (α + β × r_SPY)`
3. Accumulate over `[-2, +5]` trading days: `CAR = Σ AR`
4. Compute t-statistic to assess statistical significance

This isolates stock-specific reactions from broad market noise.

### Anomaly detection

Volume and price z-scores computed on a rolling 20-day window:
- `Z_vol = (volume − μ_vol) / σ_vol`
- `Z_price = (return − μ_ret) / σ_ret`

When both exceed 2.0 on the same day → flag as anomaly → investigate for catalyst.

---

## Outputs (saved to `data/`)

| File | Description |
|---|---|
| `normalized_prices.png` | Relative performance vs SPY, base = 100 |
| `rolling_beta.png` | 60-day rolling beta for each stock |
| `drawdown.png` | Drawdown from peak |
| `return_distributions.png` | Daily return histogram + normal fit |
| `event_impact.png` | CARs and t-statistics per event |
| `correlations.png` | Rolling correlations vs SPY and inter-stock |
| `momentum_anomalies.png` | Volume + price z-scores, RSI |

---

## Quickstart

```bash
git clone https://github.com/asoracca/new-space-radar.git
cd new-space-radar
pip install -r requirements.txt

# Full pipeline (~2 min, generates all charts)
python main.py

# Fast daily signal only
python main.py --signal
```

---

## Signal interpretation

```
⚠️  ANOMALY     vol_z > ±2  AND  price_z > ±2  →  investigate catalyst
🔴  Overbought  RSI > 70                         →  momentum may reverse
🟢  Oversold    RSI < 30                         →  potential entry
👀  Watch       |price_z| > 1.5                  →  elevated move, monitor
```

Low SPY correlation (< 0.3) alongside an anomaly = high probability of stock-specific catalyst.

---

## Statistical caveats

- Event catalog is manually curated — selection bias is possible
- Small sample sizes (5–7 events per ticker) limit statistical power
- Market model uses realized vol as a proxy; true IV requires options history
- Past event impact does not predict future event impact

---

## Personal context

I hold positions in RKLB, ASTS, and LUNR. This toolkit was built to move from
intuition-based to data-driven position management — systematically measuring
whether major catalysts actually create lasting abnormal returns, or just noise.

Part of a broader quant portfolio project series:
- **Project 1**: [SOXL Vol Surface](https://github.com/asoracca/soxl-vol-surface) — options IV rank and put-selling signal
- **Project 2**: New Space Radar — event-driven equity analysis

---

## Tech stack

`yfinance` · `pandas` · `numpy` · `scipy` · `matplotlib`

---

*Built as a quant portfolio project. Not financial advice.*
