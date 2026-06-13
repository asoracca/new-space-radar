# new-space-radar 🛰️

A quantitative event-driven analysis toolkit for the emerging space economy.
Tracks RKLB (Rocket Lab), ASTS (AST SpaceMobile), and LUNR (Intuitive Machines)
— three pure-play new space stocks with high volatility and catalyst-driven price action.

---

## The idea

New space stocks don't move like normal equities. They react to:
- Rocket launches (success or failure)
- Satellite deployment milestones
- Government contracts (NASA, DoD, Space Force)
- FCC licensing approvals
- Partnership announcements

This project builds a quantitative framework to detect when these events
create **abnormal returns** — price moves that can't be explained by
broad market movement alone.

---

## What it builds

| Module | What it does |
|---|---|
| `data.py` | Pull price + volume history for RKLB, ASTS, LUNR vs SPY benchmark |
| `events.py` | Catalog of real launch/contract/milestone events + their price impact |
| `correlation.py` | Rolling correlation between the three stocks and vs SPY |
| `momentum.py` | Detect abnormal volume and price momentum around events |
| `signal.py` | Daily scan — which stock is showing unusual activity today |

---

## Methodology

Uses **event study methodology** — a standard technique in quantitative
equity research to measure abnormal returns around known events.

For each event:
1. Estimate expected return using market model (beta × SPY return)
2. Compute abnormal return = actual return − expected return
3. Aggregate abnormal returns across events to measure average impact
4. Test statistical significance using t-test

---

## Personal context

I hold positions in RKLB, ASTS, and LUNR. This toolkit was built to
systematically analyze the catalysts that drive these stocks — moving
from intuition-based to data-driven position management.

Part of a broader quant portfolio project series:
- Project 1: [SOXL Vol Surface](https://github.com/asoracca/soxl-vol-surface) — options IV analysis
- Project 2: New Space Radar — event-driven equity analysis

---

## Quickstart

```bash
git clone https://github.com/asoracca/new-space-radar.git
cd new-space-radar
pip install -r requirements.txt
python main.py
```

---

## Statistical disclaimer

- Abnormal returns are estimated using a simplified market model.
  True alpha requires controlling for additional risk factors (size, momentum, value).
- Event catalog is manually curated — selection bias is possible.
- Small sample sizes (few launches per ticker) limit statistical power.
- Past event impact does not predict future event impact.

*Built as a quant portfolio project. Not financial advice.*