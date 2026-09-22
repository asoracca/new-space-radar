"""CAR curves reuse the provenance-aware event engine; no separate alignment/model."""
import numpy as np
import pandas as pd
from pathlib import Path
from src.events import EVENTS, Event, StudyConfig, load_schedule, study, LIMITATIONS

PRE_DAYS, POST_DAYS = 5, 10


def compute_ar_paths(ticker, events, prices, returns):
    rows = study([e if isinstance(e,Event) else Event(**e) for e in events], returns,
                 load_schedule(), StudyConfig(pre=PRE_DAYS,post=POST_DAYS))
    return pd.DataFrame([{'id': r['id'], 'type': r['event_type'],
                          **{p['offset']:p['ar'] for p in r['path']}}
                         for r in rows if r['status']=='included'])


def compute_car_curve(ar_df):
    if ar_df.empty:
        return pd.DataFrame()
    paths=ar_df[list(range(-PRE_DAYS,POST_DAYS+1))].dropna().cumsum(axis=1)
    if paths.empty:
        return pd.DataFrame()
    # Observed spread across whole CAR paths preserves within-event covariance.
    # No aggregate inferential band: cross-event dependence is not identified.
    return pd.DataFrame({'mean_car': paths.mean(), 'min_car': paths.min(),
                         'max_car': paths.max(), 'n': len(paths)})


def plot_car_curves(prices, returns, save=True):
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(14,4))
    for ax,ticker in zip(axes,EVENTS):
        curve=compute_car_curve(compute_ar_paths(ticker,EVENTS[ticker],prices,returns))
        ax.set(title=ticker,xlabel='Session offset',ylabel='CAR from day -5 (%)')
        if curve.empty:
            ax.text(.1,.5,'No eligible sourced events',transform=ax.transAxes)
        else:
            ax.plot(curve.index,curve.mean_car*100)
            ax.fill_between(curve.index,curve.min_car*100,curve.max_car*100,alpha=.2,
                            label='Observed range, not confidence interval')
            ax.legend()
    fig.suptitle('Descriptive event curves — exploratory')
    if save:
        Path('data').mkdir(exist_ok=True)
        fig.savefig('data/car_curves.png',bbox_inches='tight')
    plt.close(fig)


def print_car_summary(prices,returns):
    for ticker in EVENTS:
        curve=compute_car_curve(compute_ar_paths(ticker,EVENTS[ticker],prices,returns))
        print(ticker, 'No eligible events' if curve.empty else curve.to_string())
    print(LIMITATIONS)
