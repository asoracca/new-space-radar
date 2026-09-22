"""Regenerate a bounded regular-session fixture; no provider or network needed."""
from pathlib import Path
import exchange_calendars as xc
ROOT = Path(__file__).resolve().parents[1]
cal = xc.get_calendar('XNYS', start='2022-01-01', end='2025-12-31')
cal.schedule[['open', 'close']].rename_axis('session').to_csv(ROOT / 'fixtures/xnys_sessions.csv')
