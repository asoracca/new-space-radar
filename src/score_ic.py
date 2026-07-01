"""
score_ic.py — new-space-radar

Turns your journal into a measured SKILL number: the Information Coefficient (IC).

THE IDEA
--------
IC = the correlation between what you PREDICTED and what ACTUALLY happened.
It only counts if you write the prediction down BEFORE the move (no hindsight).
This is THE skill metric in quant (Grinold & Kahn). A small positive IC that is
stable and statistically significant is a real, hireable edge.

We score predictions against the MARKET (default benchmark SPY), not raw direction —
because a high-beta stock going up when everything goes up is beta, not skill.

WORKFLOW
--------
  # 1. Log a forecast today (prediction: +1 = beat market, -1 = lag market; size optional)
  python score_ic.py add RKLB +1
  python score_ic.py add LUNR -1 --horizon 5 --note "post-launch fade"

  # 2. Days later, score everything that has matured:
  python score_ic.py
"""

from __future__ import annotations
import os
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = "data"
FORECASTS_CSV = os.path.join(DATA_DIR, "forecasts.csv")
SCORED_CSV    = os.path.join(DATA_DIR, "forecast_scores.csv")
BENCHMARK = "SPY"          # forecasts are scored as ASSET return minus this


# ----------------------------------------------------------------------
# 1. Logging a forecast
# ----------------------------------------------------------------------
def log_forecast(asset: str, prediction: float, horizon: int = 5,
                 conviction: float | None = None, note: str = "") -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "asset": asset.upper(),
        "horizon_days": horizon,
        "prediction": prediction,          # +1 beat market, -1 lag; magnitude optional
        "conviction": conviction if conviction is not None else "",
        "note": note,
    }
    if os.path.exists(FORECASTS_CSV):
        df = pd.read_csv(FORECASTS_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(FORECASTS_CSV, index=False)
    print(f"Logged: {row['date']}  {row['asset']}  pred={prediction:+g}  "
          f"horizon={horizon}d  {note}")


# ----------------------------------------------------------------------
# 2. Scoring
# ----------------------------------------------------------------------
def _closes(ticker: str, start: str) -> pd.Series:
    s = yf.Ticker(ticker).history(start=start)["Close"].dropna()
    s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
    return s


def _fwd_return(closes: pd.Series, date: str, horizon: int) -> float | None:
    """Return from first bar on/after `date` to `horizon` trading days later."""
    pos = closes.index.searchsorted(pd.Timestamp(date))
    end = pos + horizon
    if pos >= len(closes) or end >= len(closes):
        return None                      # forecast not matured yet
    return float(closes.iloc[end] / closes.iloc[pos] - 1.0)


def score_forecasts(benchmark: str = BENCHMARK) -> pd.DataFrame:
    if not os.path.exists(FORECASTS_CSV):
        print(f"No forecasts yet — log one with:  python score_ic.py add RKLB +1")
        return pd.DataFrame()

    fc = pd.read_csv(FORECASTS_CSV)
    start = (pd.Timestamp(fc["date"].min()) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    bench = _closes(benchmark, start)

    price_cache: dict[str, pd.Series] = {}
    rows = []
    for _, r in fc.iterrows():
        tkr = str(r["asset"]).upper()
        if tkr not in price_cache:
            price_cache[tkr] = _closes(tkr, start)
        h = int(r["horizon_days"])
        a_ret = _fwd_return(price_cache[tkr], r["date"], h)
        b_ret = _fwd_return(bench, r["date"], h)
        if a_ret is None or b_ret is None:
            continue                     # still pending
        rows.append({
            "date": r["date"], "asset": tkr, "horizon_days": h,
            "prediction": float(r["prediction"]),
            "asset_return": a_ret, "bench_return": b_ret,
            "abnormal_return": a_ret - b_ret,     # what we score against
            "correct": int(np.sign(r["prediction"]) == np.sign(a_ret - b_ret)),
            "note": r.get("note", ""),
        })

    scored = pd.DataFrame(rows)
    n_pending = len(fc) - len(scored)
    if not scored.empty:
        os.makedirs(DATA_DIR, exist_ok=True)
        scored.to_csv(SCORED_CSV, index=False)
    scored.attrs["pending"] = n_pending
    return scored


def ic_stats(scored: pd.DataFrame) -> dict:
    n = len(scored)
    if n < 2:
        return {"n": n}
    pred, real = scored["prediction"], scored["abnormal_return"]
    ic = float(pred.corr(real))                       # Pearson IC
    rank_ic = float(pred.corr(real, method="spearman"))
    hit = float(scored["correct"].mean())
    # t-stat of the IC: ic * sqrt((n-2)/(1-ic^2))
    t = (ic * np.sqrt((n - 2) / (1 - ic**2))) if abs(ic) < 1 else np.nan
    return {"n": n, "ic": ic, "rank_ic": rank_ic, "hit_rate": hit, "t_stat": t}


# ----------------------------------------------------------------------
# 3. Report
# ----------------------------------------------------------------------
def report_ic(benchmark: str = BENCHMARK) -> None:
    scored = score_forecasts(benchmark)
    if scored.empty:
        if os.path.exists(FORECASTS_CSV):
            print("Forecasts logged but none matured yet — check back after their horizon.")
        return

    s = ic_stats(scored)
    print("=" * 56)
    print(f"  INFORMATION COEFFICIENT — scored vs {benchmark}")
    print("=" * 56)
    print(f"  Forecasts scored : {s['n']}   (pending: {scored.attrs.get('pending', 0)})")
    if s["n"] >= 2:
        print(f"  Hit rate         : {s['hit_rate']*100:5.1f}%   (>50% = better than a coin)")
        print(f"  IC (Pearson)     : {s['ic']:+.3f}")
        print(f"  Rank IC (Spearman): {s['rank_ic']:+.3f}")
        print(f"  t-stat of IC     : {s['t_stat']:+.2f}   (|t|>2 ~ significant)")
        print("-" * 56)
        if s["n"] < 30:
            print(f"  NOTE: only {s['n']} forecasts — way too few to claim skill.")
            print(f"        Keep logging; aim for 30-50+ before trusting the IC.")
        elif abs(s["t_stat"]) < 2:
            print("  Read: IC not yet distinguishable from luck. Keep going.")
        else:
            print("  Read: statistically significant IC — a real, reportable edge.")
    print("=" * 56)
    print(f"  Scored rows saved -> {SCORED_CSV}")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Log and score directional forecasts (IC).")
    sub = p.add_subparsers(dest="cmd")
    a = sub.add_parser("add", help="log a new forecast")
    a.add_argument("asset")
    a.add_argument("prediction", type=float, help="+1 beat market, -1 lag market")
    a.add_argument("--horizon", type=int, default=5, help="trading days (default 5)")
    a.add_argument("--conviction", type=float, default=None)
    a.add_argument("--note", default="")
    args = p.parse_args()

    if args.cmd == "add":
        log_forecast(args.asset, args.prediction, args.horizon, args.conviction, args.note)
    else:
        report_ic()


if __name__ == "__main__":
    main()