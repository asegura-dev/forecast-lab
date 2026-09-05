# Chapter 2 - The features

This chapter answers what the nineteen columns of the feature matrix are, how each is computed,
and why the set stops where it does. Three of those answers are decisions rather than
definitions: no column may carry a price level, indicators are computed before the symbols are
aligned onto a common clock, and the library's warm-up values are overwritten because one of
them is a zero pretending to be a measurement. Every number below comes from
`docs/status/features-policy.json`, `docs/status/eda-summary.json` or
`docs/status/notebook-baseline.json`; every formula is implemented in
`src/forecast_lab/research/features/technical.py`.

## 1. The nineteen columns

One function builds all of them:
`src/forecast_lab/research/features/technical.py::indicators`. It takes one symbol's bars on
that symbol's own index, returns a frame with the same index, and prefixes nothing -
namespacing by symbol happens later, in
`src/forecast_lab/research/align.py::align_to_target`, which is why the columns in
`docs/status/features-policy.json` read `XAUUSD_return` rather than `return`.

Notation: `C_t`, `H_t`, `L_t` are the close, high and low of bar `t`; `w` is a window in
bars; `SMA_w` and `EMA_w` are the simple and exponential moving averages of the close, the
latter with `alpha = 2/(w+1)` and no adjustment (`ta.utils::_ema` uses
`ewm(span=w, adjust=False)`, a plain recursion `E_t = alpha*C_t + (1-alpha)*E_{t-1}`). The
"window" column gives the indicator's own window; where `technical.py::_warm` is passed a
different number - MACD and ADX - it is named, and `_warm` blanks the first `n - 1` rows of it.

| # | Column | Group | Window | Formula | Built by |
|---:|---|---|---:|---|---|
| 1 | `return` | returns | 1 | `C_t / C_{t-1} - 1` | `technical.py::indicators` |
| 2 | `log_return` | returns | 1 | `ln(C_t / C_{t-1})` | `technical.py::_log_return` |
| 3 | `range_pct` | returns | 1 | `(H_t - L_t) / C_t` | `technical.py::indicators` |
| 4 | `realised_vol_24` | volatility | 24 | `sd(l_{t-23..t}) * sqrt(24*365)`, `l` = log return, `sd` sample (`ddof=1`) | `technical.py::indicators` |
| 5 | `realised_vol_168` | volatility | 168 | `sd(l_{t-167..t}) * sqrt(24*365)` | `technical.py::indicators` |
| 6 | `atr_pct` | volatility | 14 | `ATR_14(t) / C_t`, with `TR_t = max(H_t-L_t, abs(H_t-C_{t-1}), abs(L_t-C_{t-1}))` and `ATR_t = ((w-1)*ATR_{t-1} + TR_t)/w` | `technical.py::indicators`, `::_warm` |
| 7 | `bb_wband` | volatility | 20 | `100 * (U_t - Lo_t) / M_t`, where `M = SMA_20` and `U/Lo = M +/- 2*sigma_20` (`sigma` population, `ddof=0`) | `technical.py::indicators` |
| 8 | `dist_sma_20` | trend | 20 | `C_t / SMA_20(t) - 1` | `technical.py::indicators` |
| 9 | `dist_sma_50` | trend | 50 | `C_t / SMA_50(t) - 1` | `technical.py::indicators` |
| 10 | `dist_sma_200` | trend | 200 | `C_t / SMA_200(t) - 1` | `technical.py::indicators` |
| 11 | `dist_ema_12` | trend | 12 | `C_t / EMA_12(t) - 1` | `technical.py::indicators` |
| 12 | `dist_ema_26` | trend | 26 | `C_t / EMA_26(t) - 1` | `technical.py::indicators` |
| 13 | `macd` | trend | 12/26, blanked to 35 | `(EMA_12(t) - EMA_26(t)) / C_t` | `technical.py::indicators`, `::_warm` |
| 14 | `macd_signal` | trend | 12/26/9, blanked to 35 | `EMA_9(EMA_12 - EMA_26)(t) / C_t` | `technical.py::indicators`, `::_warm` |
| 15 | `macd_diff` | trend | 12/26/9, blanked to 35 | `(macd_line(t) - signal(t)) / C_t` | `technical.py::indicators`, `::_warm` |
| 16 | `rsi_14` | momentum | 14 | `100 - 100/(1 + RS_t)`, `RS = E(gain)/E(loss)` with Wilder `alpha = 1/14` | `technical.py::indicators` |
| 17 | `roc_12` | momentum | 12 | `100 * (C_t - C_{t-12}) / C_{t-12}` | `technical.py::indicators` |
| 18 | `adx_14` | momentum | 14, blanked to 28 | Wilder `DI+`/`DI-`, then `DX = 100 * abs(DI+ - DI-) / (DI+ + DI-)` and `ADX` = Wilder mean of `DX` over 14 | `technical.py::indicators`, `::_warm` |
| 19 | `bb_pband` | position in range | 20 | `(C_t - Lo_t) / (U_t - Lo_t)` | `technical.py::indicators` |

Three details in that table are easy to misread later.

**`roc_12` is in percent and everything else in the return family is a fraction.** `ta`'s
`ROCIndicator` multiplies by 100; `technical.py::indicators` does not undo it, and the comment
there says so deliberately - "already a percentage, so it needs no normalisation". The
inconsistency is harmless for tree models, invariant as they are to a monotone rescaling of a
single column, and would not be for a model that regularises coefficients. Nothing enforces it
either way.

**The three MACD columns are divided by the close, and the raw form never exists.** A MACD is
a difference of two moving averages, so it inherits the price's units and grows with it.
`technical.py::indicators` blanks the warm-up and then divides -
`_warm(series, MACD_WINDOW) / close` - with `MACD_WINDOW = 26 + 9 = 35`, because the signal
line smooths an already-smoothed quantity.

**`realised_vol_*` uses `min_periods` equal to the window, and rolls rather than expands.** A
partial window would report a confident annualised number computed from three bars; an
expanding window would give the column a different meaning at bar `t` than at bar `t+1`.
Annualisation is `sqrt(24*365)`, set by `BARS_PER_YEAR`: this venue trades around the clock on
weekdays, so the series has weekend gaps rather than weekend zeros, and a 252-day equity
convention would be wrong. The 7-day window was chosen by measurement - the module records
correlations against VIX's level of +0.645 at 24h, +0.744 at 5d, +0.754 at 7d and +0.755 at
21d, flat past a week, so seven days buys the tracking at the shortest warm-up.

The windows themselves are the original project's, unchanged. `technical.py` states the reason
at the constants: altering them would count as a fresh trial in the deflated-Sharpe
accounting, and the object of this work is to isolate the effect of the corrections, not to
run a hyperparameter sweep.

## 2. The scale-freedom policy

Gold runs from **1,616.66 to 4,378.00** across this sample (`docs/status/eda-summary.json`,
`price.minimum` and `price.maximum`). A column carrying that level puts the test block outside
the support of the training block, so a model fitted on it is extrapolating rather than
predicting. That is the rule: no column may be a price level.

What makes the rule work is *how* it is checked.
`src/forecast_lab/research/features/stationarity.py::probe_scale` multiplies `open`, `high`,
`low` and `close` by `SCALE_PROBE = 10.0`, rebuilds the whole matrix through the same
callable, and compares the two frames row by row. The deviation is relative -
`abs(a - b) / max(abs(a))` - because an absolute difference means nothing across columns whose
magnitudes differ by orders of magnitude. Multiplying every price by a constant is an
economically meaningless change, the same instrument quoted in a different unit, so anything
that reacts to it is measuring the unit rather than the market.

A name list is the obvious alternative and is strictly weaker: a name list is a claim about
what a function does, a rescaling an observation of what it did. `bollinger_wband` is the case
that settled it. It was listed as scale-free on the assumption that it normalises internally,
and it does - `ta.volatility::BollingerBands.bollinger_wband` computes
`(hband - lband) / mavg * 100` - but the assumption was carrying the classification until
something checked, and this project exists because an unchecked assumption became a published
number.

**Both halves of the probe's calibration were wrong, and real data corrected them in that
order.** `SCALE_TOLERANCE` began at `1e-9`, on the reasoning that a real violation moves nine
orders of magnitude, so anything above numerical noise must be genuine. It flagged `bb_pband`,
which is dimensionless by construction but divides by a Bollinger band's width - small in
quiet stretches, and therefore an amplifier of floating-point error. `bb_pband`'s worst single
row in `docs/status/features-policy.json` is `1.3392e-09`. The threshold moved to `1e-6`.

Then ADX broke it again. `adx_14`'s `worst_change` is **7.1270e-03**
(`docs/status/features-policy.json`), four orders above the new threshold, on a quantity that
is also dimensionless by construction. The diagnosis was in the shape of the deviation rather
than its size: **134 rows out of 23,181 (0.58%), at consecutive positions** beginning at 4586
(`docs/status/STATUS-2026-08-features.md` sec. 3). Consecutive rows are the signature of a
recursion. `ta.trend::ADXIndicator._run` builds directional movement from the comparison
`(diff_up > diff_down) & (diff_up > 0)`; where the two sides are equal to within
floating-point error, rescaling breaks the tie the other way, and the Wilder recursion carries
that one flipped comparison forward until it decays.

So the verdict became `SCALE_PERCENTILE = 99.0`, the 99th percentile of the per-row deviation,
with the maximum still reported beside it as `worst_change`. The percentile cannot hide a
genuine price level, because a level moves on *every* row; the maximum was measuring float64
rather than a property of the feature. Dropping the maximum would have been the dishonest half
of the decision: a large `worst` beside a tiny `change` is a real numerical instability worth
knowing about, and simply not a price level. On `adx_14` the two differ by seven orders of
magnitude - `relative_change` `6.9876e-10` against `worst_change` `7.1270e-03` - and the JSON
prints both.

The result: `policy_passes: true`, `violations: []`, all nineteen columns `scale-free`, and
the matrix's largest `relative_change` anywhere is that same `6.9876e-10`, three orders below
the threshold. A threshold chosen by reasoning about magnitudes was wrong twice, for two
unrelated reasons, and both times the data said so within seconds of being asked. The statistic
mattered more than the number.

## 3. Order of operations

`src/forecast_lab/research/features/build.py::build_features` does two steps, and the order is
the entire content of the module. Step 1 computes indicators on each symbol's native grid.
Step 2 hands the resulting frames to `align.py::align_to_target`, which reindexes them onto
the target's timeline. The function takes the raw per-symbol mapping and performs the
alignment itself, precisely so that it cannot be handed an already-aligned panel.

The other order looks identical in a diff and manufactures data. After reindexing, a row on
which an auxiliary symbol is stale holds a copy of that symbol's previous price. A return
computed there is **exactly zero** - not a small number, an artefact; an RSI drifts toward 50;
an ATR contracts toward nothing. What makes it dangerous rather than merely noisy is that the
artefact is not scattered at random: a symbol is stale precisely when its venue is shut, so
the pattern tracks the hour of the day almost perfectly. VIX is carried on 8.0% of rows and
DXY on 7.1% (`docs/adr/ADR-006-features-and-stationarity.md` sec. 1, citing ADR-003 sec. 2). A
tree fed those columns learns a session clock, and the report calls it a macro signal;
`tests/unit/test_build.py::test_a_stale_auxiliary_does_not_get_a_fabricated_zero_return` pins
it.

Carrying a *finished* indicator forward is a different act and is legitimate: at 03:00 the
last thing known about the S&P really was its 21:00 RSI. Computing an indicator *from* carried
prices is not, because those inputs were never observed at those instants.

**The accepted cost, written down so it is not mistaken for a bug later:** `ta` windows are
positional, not temporal, and there is no option to change that. The target series has
**1,013 gaps** (`docs/status/notebook-baseline.json`, `labels.gapped`), so an `RSI_14` on a
Monday bar uses thirteen bars from Friday and `SMA_200` can span nine calendar days. The
obvious fix, reindexing onto a complete grid first, reintroduces exactly the defect above, so
the positional window is the lesser evil and becomes an invariant of the study.

Warm-up is then dropped **by position**: `LONGEST_WINDOW - 1 = 199` rows, where
`LONGEST_WINDOW` is the maximum over every window in `technical.py` (200, from `SMA_200`).
`docs/status/features-policy.json` records `warmup_dropped: 199` and `rows: 22982`. `dropna()`
was rejected because it would additionally delete rows a *stale auxiliary* left empty - real
target bars carrying real target data, which belong in the matrix marked rather than removed.
`tests/unit/test_build.py::test_the_warmup_is_dropped_by_position_not_by_dropna` records it.

FOCUS yields 19 columns from 1 symbol; WHOLE yields 199 from 10
(`docs/status/STATUS-2026-08-features.md` sec. 1), the arithmetic being `19 + 9 * (19 + 1)` -
the target contributes nineteen and no staleness column, since it defines the timeline and is
never filled, while each auxiliary contributes nineteen plus one
(`align.py::align_to_target`). Both modes span **22,982** rows and share an index exactly
(`tests/unit/test_build.py::test_whole_and_focus_share_an_index`), where the original ran its
two comparisons over 24,232 and 22,441 rows.

## 4. The warm-up problem

`technical.py::_warm` blanks the first `window - 1` rows of a series, whatever the library put
there. It is idempotent where `ta` already returns NaN - SMA, EMA, RSI, MACD and Bollinger all
honour `fillna=False` - and load-bearing where it does not.

`ta.volatility::AverageTrueRange._run` allocates `atr = np.zeros(len(close))` and starts
filling at index `window - 1`, so the warm-up rows are literally the zeros of that allocation
and `fillna=False` never touches them; `ta.trend::ADXIndicator._run` does the same with three
more `np.zeros` arrays. Measured: a 14-bar ATR returns exactly zero on rows 0 through 12, with
no NaN anywhere in the column, and ADX returns 27 zeros
(`docs/status/STATUS-2026-08-features.md` sec. 6). It is the Wilder-recursion family, not an
isolated bug.

A zero is worse than a NaN in every respect that matters here. It is not "no data", it is nil
volatility or no trend at all - readings the market can plausibly produce. `dropna()` will not
remove it, any normalisation dividing by it explodes, and a tree will learn it as a genuine
low-volatility regime that happens to coincide with the start of the sample. A NaN is loud and
removable; a zero is quiet and wrong.

`indicators` therefore wraps ADX in `_warm(..., 2 * ADX_WINDOW)` rather than `ADX_WINDOW`,
because ADX smooths a smoothed quantity and its warm-up is roughly twice its window.
`tests/unit/test_technical.py::test_the_wilder_indicators_emit_zero_not_nan_during_warmup`
asserts **both halves** - that `ta` really does this, and that the wrapper undoes it - so a
future release fixing the library makes the first assertion fail and tells us the workaround
can go. That is the reason to assert a dependency's misbehaviour rather than silently work
around it. `ta` is quarantined in this one module for the same family of reasons, enforced by
`tests/test_layering.py`.

## 5. ADF and KPSS

Both run on every column in
`src/forecast_lab/research/features/stationarity.py::evaluate_stationarity`, and neither can
fail the command.

ADF's null hypothesis is that the series has a unit root, so a **small** p-value argues for
stationarity. KPSS's null is the opposite - that the series is stationary - so a **small**
p-value argues against it. The nulls point in opposite directions, which is why both are
reported: a single test can only ever fail to reject, and "failed to reject" is not evidence.
The agreement rule is therefore `adf < 0.05 and kpss > 0.05` - ADF rejects the unit root while
KPSS declines to reject stationarity, the same conclusion from opposite starting points.
`ColumnReport.adf_rejects_unit_root` and `ColumnReport.kpss_rejects_stationarity` encode the
two thresholds, and `PolicyReport.disagreements` returns the columns where **both** reject,
which is the contradiction.

On the reference run (`docs/status/features-policy.json`, `n_finite: 5000` for every column):

- **14 columns agree on stationary**: `return`, `log_return`, the five `dist_*`, `rsi_14`,
  `roc_12`, `adx_14`, the three `macd*`, and `bb_pband`.
- **4 disagree**, both tests rejecting: `range_pct` (ADF 9.75e-07, KPSS 0.010),
  `realised_vol_24` (1.81e-04, 0.0120), `atr_pct` (0.0030, 0.010) and `bb_wband` (1.35e-06,
  0.0158).
- **1 agrees on non-stationary**: `realised_vol_168`, the only column ADF does not reject
  (p = 0.1264), with KPSS at 0.010.

Many p-values are exactly `0.1` or exactly `0.01` because KPSS clips at the edges of its
tabulated range; `_tests` silences `InterpolationWarning` for that reason and the module
docstring states the limitation instead, rather than burying the report under one warning per
column. Four ADF p-values come back as exactly `0.0`.

Neither test is a gate, and `ColumnReport.allowed` says so in one line: it returns
`self.scale is not Scale.DEPENDENT`, consulting layer 1 and nothing else. A gate here would be
a false guarantee, which is worse than none because it stops people looking. The tests read
only the last 5,000 rows of each column (`max_rows`), and at that length ADF rejects a unit
root on almost anything; both are also invalid under heteroskedasticity and regime change,
which is the entire character of this data. And `realised_vol_168` makes the case better than
any argument could: a gate would have discarded the feature this project built to replace VIX,
for being persistent - which is what seven-day volatility should look like - on a test that
does not apply to this data anyway.
`tests/unit/test_stationarity.py::test_the_tests_are_reported_but_do_not_decide` pins it.

"Carries no price level" is necessary, not sufficient. `dist_sma_200` shifts its mean from
0.0024 to 0.0068 between train and test, and `atr_pct` rises 37% in test
(`docs/status/STATUS-2026-08-features.md` sec. 5). Both are scale-free; both clear layers 1
and 2. What catches them is distribution shift measured between blocks - PSI, KS, adversarial
validation - which needs the splits and therefore belongs with the validation work.

## 6. What the original computed and this does not

The original `indicator_creator` produced **23 columns per symbol**, and every one is accounted
for in `docs/status/STATUS-2026-08-features.md` sec. 2. Twelve were dropped, all for the same
reason - they carry the price level: three raw SMAs (`SMA_200/50/20`), two raw EMAs
(`EMA_12/26`), three Bollinger levels (`BB_HIGH`, `BB_MID`, `BB_LOW`), three undivided MACD
columns, and a raw `ATR`.

The instructive part is that **the original was not ignorant of the correction**. It computed
`Dist_SMA200`, `Dist_SMA50`, `Dist_SMA20`, `Dist_EMA12` and `Dist_EMA26` itself - exactly the
normalisation used here. What it did not do was remove what the correction replaced: the
distances and the levels they were derived from sat side by side in the same matrix. Nobody
forgot the correction; nobody deleted the thing it corrected. That is a far commoner failure
than ignorance, and no test in that pipeline could have caught it, because none existed.

Two columns here are genuinely new - `range_pct`, and `bb_pband`, which replaces three raw band
levels with a dimensionless position inside them. `realised_vol_168` is new; `realised_vol_24`
is not, since a 24-bar version already existed in the original's other notebook, a claim
corrected on 2026-09-02 in that same STATUS section.

Removing the levels did **not** fix the matrix's redundancy, and the ADR records the correction
to its own first guess. The original's PCA at 90% variance collapsed 63 features into 6
components, which looked like the levels' doing; with every level gone, PCA still retains 6
from 19. `docs/adr/ADR-009-exploratory-analysis.md` sec. 5 measured why: ten of 171 pairs
correlate above 0.9, `return` and `log_return` at exactly 1.0000, and the first principal
component alone carries 44.3% of the variance. Indicators derived from one price series are
intrinsically redundant; the levels made it worse rather than caused it. Nothing was dropped in
response - removing `log_return` would be defensible and would also be a change to the feature
set, with its own before-and-after to run.

## 7. Where the documents and the code disagree

Three contradictions, none of which changes a decision, all of which cost an hour if met cold.

- `docs/adr/ADR-006-features-and-stationarity.md` sec. 4 says **three** columns have ADF and
  KPSS pointing opposite ways. `docs/status/features-policy.json` gives **four** -
  `range_pct`, `realised_vol_24`, `atr_pct`, `bb_wband` - as does
  `docs/status/STATUS-2026-08-features.md` sec. 5. The ADR sentence is stale; the JSON is the
  artefact of the run.
- The module docstring of `stationarity.py` says PCA "still retains 6 from **17**". The matrix
  has **19** columns (`features-policy.json`, `columns: 19`), which is what ADR-006 sec. 2
  says. The docstring predates ADX and ROC joining the set.
- The `SCALE_TOLERANCE` comment in `stationarity.py` says a raw price level moves by
  **8.5e+00**; ADR-006 and `STATUS-2026-08-features.md` sec. 3 both say **9.0e+00**. Neither
  appears in `features-policy.json`, because no level survives to be measured there - both come
  from the probe's own fixtures, recorded on different runs.
