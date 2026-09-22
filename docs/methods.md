# Methods and evidence

## Time, source and sample contract

A record has a stable ID, symbol, original date/text, offset-aware timestamp,
IANA timezone, source URL, event type, announcement status, timestamp basis,
input kind and alignment policy. Unknown source/time values are explicit nulls.
Original date-only entries came from `src/events.py` at commit `4f12b8b` and are
preserved in `fixtures/original_catalog.json`; descriptions have not been
silently corrected or independently endorsed. `catalog.json` retains each raw
record alongside annotations. IDs identify records, not validated factual claims.

The sole sourced annotation is [NASA's retrospective account of the IM-1
landing](https://www.nasa.gov/news-release/nasa-intuitive-machines-to-discuss-historic-moon-mission-today/),
published February 23, 2024, reporting the landing at 18:23 EST on February 22.
It supports descriptive occurrence alignment to February 23. It does **not**
establish when an executable announcement signal became public. Scheduled,
unverified and explicitly duplicate announcements are excluded. No live inputs
are part of the demo. Do not use occurrence timestamps as announcement trades.

XNYS regular-session dates/open/close times are frozen from
[exchange_calendars 4.11.1](https://pypi.org/project/exchange_calendars/4.11.1/)
([upstream documentation](https://github.com/gerrymanoim/exchange_calendars)).
This calendar is used as the shared US-equity regular-session convention,
including for Nasdaq-listed stocks; it does not model security-specific halts,
extended-hours fills or exceptional listing conditions. Events outside the
2022–2025 fixture are rejected until the fixture is deliberately extended.

## Estimation and uncertainty

Input rows are close-to-close **log returns**, with a unique sorted naive
session-date index. Returns calculated from prices must preserve missing sessions
before taking consecutive differences. This prevents a multi-day return from
being mistaken for a one-day return. The study does not fill missing data.

For day-0 session index `i`, event dates are `[i-pre, i+post]`. Estimation ends
exclusively at `i-pre-gap`, starts `estimation` sessions earlier, and is required
to contain exactly 120 complete stock/SPY observations by default. Event windows
also require all 8 observations. No fallback alpha/beta is substituted.

OLS estimates `r_stock = alpha + beta*r_SPY + error`. Abnormal returns subtract
that prediction. CAR is their arithmetic sum from day −2 onward; it is neither
a buy-and-hold return nor an executable strategy return.

For a prefix of `m` event sessions, with design-row sum `z`, the model variance is
`s² * (m + z' (X'X)^−1 z)`, where `s² = SSE/(n−2)`. This includes parameter
estimation error. Bounds use the 97.5th percentile of Student t with `n−2` degrees
of freedom. They assume a correct linear model and IID homoskedastic errors.
A noisy synthetic test checks an independent closed-form variance calculation.
The exact-linear shock/control demo has approximately zero residual variance:
its narrow intervals reflect fixture construction, not empirical certainty.

Inference remains **exploratory**. The small, manually curated sample was not
preregistered; source coverage is incomplete. Cross-stock/common-date dependence,
serial correlation, event-induced volatility and multiple symbol/type/window
comparisons invalidate a blanket 95% significance interpretation. No p-value
screen or edge claim is issued. Cross-event plots show observed ranges of whole
CAR paths, not bands formed by adding independent daily standard errors.

Both same-stock events with overlapping event windows are excluded; known event
windows intersecting another estimation sample contaminate it. These rules
cannot detect uncatalogued catalysts. Duplicate canonical announcements keep the
first input occurrence for analysis and label later occurrences as exclusions;
database natural-key constraints reject duplicate catalog imports. Imports must
resolve duplicate records before saving a run. Date-only unverified records
cannot establish precise overlap or causal attribution.

## Signal evaluation

A full-bar price/volume signal is only known after that close. With close-only
inputs, execution uses the next exchange session's close, never the signal bar.
Missing entry/exit prices invalidate a trade rather than postponing its fill.
Tests mutate the signal-bar price and future bars to check execution and feature
causality. Costs default to 20 basis points round trip. A symbol can have one
active trade at a time. Cross-symbol capital allocation, slippage beyond that
assumption, borrow, halts and provider price adjustments are not modeled.
Reported observations and SPY comparisons therefore do not establish portfolio
Sharpe, drawdown or deployable alpha. Threshold choice remains exploratory.

## Persistence and reproduction

SQLite enables foreign keys on every connection. Sequential migrations store
checksums and reject changes to already applied scripts. Catalog records are
immutable by ID; corrections require a new versioned record ID with a note.
Run IDs hash command/configuration, runtime versions, fixture and generated-input
hashes, and source-module hashes. Repeat runs check outputs rather than overwrite
evidence. Environment versions may produce a different run ID; numerical tests
use tolerances. A transaction rolls back on invalid provenance or constraints.

`query_events` binds symbol, type and inclusive date bounds. Date filtering uses
the aligned session, falling back to the original date only for unaligned records.
`reconcile` checks SQL counts, mean CAR and sums of per-session AR against Python.
Exports include exclusions rather than silently restricting the denominator.

The Python environment is pinned in `requirements-offline.lock`; the browser
compiler is pinned in `explorer/pnpm-lock.yaml`. The lock installs third-party
packages under their original licenses; no third-party implementation is copied.
The generated calendar fixture is attributed above. Existing project files,
archived figures, journals and any upstream licensing are preserved.

Implementation references consulted: [NumPy 2.2 least squares](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.lstsq.html),
[pandas 2.2 reindex](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.DataFrame.reindex.html),
[SciPy 1.15.3 Student t](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.t.html),
[SQLite foreign keys](https://www.sqlite.org/foreignkeys.html), and
[TypeScript 5.8](https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-8.html).

The legacy journal IC module (`src/score_ic.py`) is outside the validated execution
path: it stores date-only forecasts and must not be used as evidence of executable
announcement-time performance. Its old strong-edge wording has been removed.
