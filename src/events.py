"""Provenance-aware, calendar-aligned descriptive market-model event studies."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'first_session_closing_after_timestamp'
LIMITATIONS = ('Exploratory descriptive study, not a trading-edge or catalyst-probability claim. '
               'Curated selection, cross-stock dependence and multiple comparisons preclude '
               'confirmatory inference. Intervals assume IID constant-variance model errors; '
               'they are pointwise model diagnostics, not simultaneous confidence bands.')
EVENT_COLORS = {'launch': '#1D9E75', 'contract': '#378ADD', 'milestone': '#7F77DD'}

@dataclass(frozen=True)
class Event:
    id: str
    symbol: str
    title: str
    original_date: str
    timestamp: str | None
    timezone: str
    source_url: str | None
    event_type: str
    announcement_status: str
    alignment: str = POLICY
    input_kind: str = 'historical'
    timestamp_basis: str = 'announcement'
    source_note: str = ''
    original_record: dict | None = None

    def __post_init__(self):
        if not self.id or not self.symbol or not self.event_type:
            raise ValueError('ID, symbol and event type are required')
        if self.alignment != POLICY or self.input_kind not in {'historical', 'synthetic', 'live'}:
            raise ValueError('Unsupported alignment or input kind')
        if self.announcement_status not in {'confirmed', 'scheduled', 'unverified', 'duplicate'}:
            raise ValueError('Invalid announcement status')
        if self.timestamp_basis not in {'announcement', 'occurrence', 'unknown'}:
            raise ValueError('Invalid timestamp basis')
        zone = ZoneInfo(self.timezone)
        if self.timestamp is not None:
            ts = pd.Timestamp(self.timestamp)
            if ts.tzinfo is None or pd.isna(ts):
                raise ValueError('Timestamp must have a UTC offset')
            if ts.utcoffset() != ts.tz_convert(zone).utcoffset():
                raise ValueError('Timestamp offset disagrees with timezone')
        if self.source_url and not self.source_url.startswith(('https://', 'http://')):
            raise ValueError('Source URL must be HTTP(S)')


def load_catalog(path=ROOT / 'fixtures/catalog.json'):
    return [Event(**row) for row in json.loads(Path(path).read_text())]

# Compatibility for existing pipeline callers; one canonical catalog.
EVENTS = {s: [asdict(e) for e in load_catalog() if e.symbol == s] for s in ['RKLB','ASTS','LUNR']}


def load_schedule(path=ROOT / 'fixtures/xnys_sessions.csv'):
    frame = pd.read_csv(path, index_col='session', parse_dates=['session'])
    for col in ['open', 'close']:
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def align_event(event, schedule):
    if event.timestamp is None:
        raise ValueError('unknown_timestamp')
    ts = pd.Timestamp(event.timestamp).tz_convert('UTC')
    # Do not extrapolate outside the frozen schedule, including its first midnight.
    if ts < schedule.index[0].tz_localize(event.timezone).tz_convert('UTC'):
        raise ValueError('calendar_out_of_range')
    candidates = schedule.index[schedule['close'] > ts]
    if len(candidates) == 0:
        raise ValueError('calendar_out_of_range')
    return candidates[0]


@dataclass(frozen=True)
class StudyConfig:
    estimation: int = 120
    gap: int = 5
    pre: int = 2
    post: int = 5

    def __post_init__(self):
        if self.estimation < 3 or min(self.gap, self.pre, self.post) < 0:
            raise ValueError('Invalid study windows')


def study(events, returns, schedule, config=StudyConfig()):
    """Fixed complete windows. Missing rows remain missing; never skip to a later price.

    Input returns are close-to-close log returns indexed by naive session date.
    Both overlapping same-stock event windows and other known events in the
    estimation window are conservatively excluded. Duplicate canonical keys keep
    the first catalog entry; every submission gets an outcome.
    """
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError('Return index must be unique and sorted')
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.tz is not None:
        raise ValueError('Returns require naive session-date DatetimeIndex')
    if not returns.index.equals(returns.index.normalize()):
        raise ValueError('Returns require midnight session labels')
    if len({e.id for e in events}) != len(events):
        raise ValueError('Duplicate stable event ID')
    aligned, reasons, seen = {}, {}, set()
    for e in events:
        if e.announcement_status != 'confirmed':
            reasons[e.id] = 'announcement_' + e.announcement_status
            continue
        if not e.source_url:
            reasons[e.id] = 'missing_source'
            continue
        try:
            session = align_event(e, schedule)
        except ValueError as ex:
            reasons[e.id] = str(ex)
            continue
        key = (e.symbol, pd.Timestamp(e.timestamp).tz_convert('UTC').isoformat(), e.event_type)
        if key in seen:
            reasons[e.id] = 'duplicate_announcement'
        else:
            seen.add(key)
            aligned[e.id] = schedule.index.get_loc(session)
    rows = []
    for e in events:
        row = {**asdict(e), 'aligned_session': None, 'reason': reasons.get(e.id),
               'n_estimation': 0, 'n_event': 0, 'path': [], 'car': None,
               'limitations': LIMITATIONS}
        if e.id not in aligned:
            rows.append(row)
            continue
        i = aligned[e.id]
        a, b = i-config.pre, i+config.post+1
        end = a-config.gap
        start = end-config.estimation
        row['aligned_session'] = str(schedule.index[i].date())
        others = [j for key,j in aligned.items() if key != e.id and
                  next(x.symbol for x in events if x.id == key) == e.symbol]
        if any(a <= j+config.post and j-config.pre < b for j in others):
            row['reason'] = 'overlapping_event_window'
        elif any(start <= j+config.post and j-config.pre < end for j in others):
            row['reason'] = 'contaminated_estimation_window'
        elif start < 0:
            row['reason'] = 'insufficient_history'
        elif b > len(schedule):
            row['reason'] = 'incomplete_event_window'
        elif e.symbol not in returns or 'SPY' not in returns:
            row['reason'] = 'missing_stock_or_benchmark'
        else:
            est_dates, event_dates = schedule.index[start:end], schedule.index[a:b]
            assert not len(est_dates.intersection(event_dates))
            est = returns.reindex(est_dates)[['SPY', e.symbol]]
            win = returns.reindex(event_dates)[['SPY', e.symbol]]
            row.update(estimation_start=str(est_dates[0].date()), estimation_end=str(est_dates[-1].date()),
                       event_start=str(event_dates[0].date()), event_end=str(event_dates[-1].date()),
                       n_estimation=int(np.isfinite(est).all(axis=1).sum()),
                       n_event=int(np.isfinite(win).all(axis=1).sum()))
            if not np.isfinite(est.to_numpy()).all():
                row['reason'] = 'missing_estimation_observation'
            elif not np.isfinite(win.to_numpy()).all():
                row['reason'] = 'missing_event_observation'
            else:
                x = np.column_stack([np.ones(len(est)), est['SPY']])
                y = est[e.symbol].to_numpy()
                coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
                if rank != 2:
                    row['reason'] = 'singular_market_model'
                else:
                    z = np.column_stack([np.ones(len(win)), win['SPY']])
                    ar = win[e.symbol].to_numpy() - z @ coef
                    residual = y - x @ coef
                    variance = float(residual @ residual / (len(est)-2))
                    inv = np.linalg.inv(x.T @ x)
                    path = []
                    for k, car in enumerate(np.cumsum(ar)):
                        v = z[:k+1].sum(axis=0)
                        se = float(np.sqrt(max(0, variance*((k+1) + v @ inv @ v))))
                        half = float(t.ppf(.975, len(est)-2) * se)
                        path.append(dict(offset=k-config.pre, session=str(event_dates[k].date()),
                                         ar=float(ar[k]), car=float(car), se=se,
                                         lower=float(car-half), upper=float(car+half)))
                    row.update(alpha=float(coef[0]), beta=float(coef[1]), path=path,
                               car=path[-1]['car'], ar_day=float(ar[config.pre]),
                               null_car=0.0, car_se=path[-1]['se'])
        row['status'] = 'included' if row['reason'] is None else 'excluded'
        rows.append(row)
    for r in rows:
        r['status'] = 'included' if r['reason'] is None else 'excluded'
    return rows


def compute_event_impact(ticker, events, prices, returns, pre_window=2, post_window=5):
    objects = [e if isinstance(e, Event) else Event(**e) for e in events]
    rows = study(objects, returns, load_schedule(), StudyConfig(pre=pre_window, post=post_window))
    return pd.DataFrame(rows)


def analyze_all_events(prices, returns):
    return pd.DataFrame(study(load_catalog(), returns, load_schedule()))


def print_event_summary(event_df):
    if not event_df.empty:
        print(event_df[['id','symbol','aligned_session','status','reason','car']].to_string(index=False))
    print(LIMITATIONS)


def plot_event_impact(event_df, save=True):
    import matplotlib.pyplot as plt
    valid = event_df[event_df['status'] == 'included'] if not event_df.empty else event_df
    if valid.empty:
        return
    fig, ax = plt.subplots()
    ax.bar(valid['id'], valid['car'] * 100)
    ax.set(title='Descriptive event CAR (exploratory)', ylabel='Sum of log abnormal returns (%)')
    ax.tick_params(axis='x', rotation=45)
    if save:
        Path('data').mkdir(exist_ok=True)
        fig.savefig('data/event_impact.png', bbox_inches='tight')
    plt.close(fig)
