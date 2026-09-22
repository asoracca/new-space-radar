"""Deterministic synthetic validation; never presented as historical stock returns."""
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
from src.events import Event, StudyConfig, load_schedule, study

SEED = 1729

def synthetic_case():
    schedule = load_schedule()
    rng = np.random.default_rng(SEED)
    market = rng.normal(0, .01, len(schedule))
    returns = pd.DataFrame({'SPY': market, 'SHOCK': .0002+1.3*market,
                            'CONTROL': .0002+1.3*market}, index=schedule.index)
    i = schedule.index.get_loc(pd.Timestamp('2024-02-23'))
    returns.iloc[i, returns.columns.get_loc('SHOCK')] += .05
    events = [Event(id=f'synthetic-{s.lower()}', symbol=s, title=f'Synthetic {s.lower()} validation',
                    original_date='2024-02-22', timestamp='2024-02-22T18:23:00-05:00',
                    timezone='America/New_York', source_url='https://github.com/asoracca/new-space-radar/blob/fix/event-evidence-browser/src/demo.py',
                    event_type='validation', announcement_status='confirmed', input_kind='synthetic',
                    source_note='Generated fixture; not a real catalyst or security.') for s in ['SHOCK','CONTROL']]
    return schedule, returns, events


def selection_experiment():
    rng = np.random.default_rng(SEED)
    # Independent precommitted dates vs the ten largest returns chosen in hindsight.
    null = rng.normal(0, .01, 1000)
    return {'seed': SEED, 'candidate_count': len(null), 'selected_count': 10,
            'precommitted_mean': float(null[np.arange(0,1000,100)].mean()),
            'hindsight_top10_mean': float(np.sort(null)[-10:].mean()),
            'all_dates_mean': float(null.mean()),
            'interpretation': 'Selecting large realized returns manufactures an apparent effect under a no-edge null.'}


def main():
    schedule, returns, events = synthetic_case()
    rows = study(events, returns, schedule)
    print(json.dumps({'seed': SEED, 'config': asdict(StudyConfig()), 'events': rows,
                      'selection_bias': selection_experiment()}, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
