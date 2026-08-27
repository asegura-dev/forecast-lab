# ADR-009 - What exploratory analysis is for, and the three ways it goes wrong quietly

- **Status:** Accepted - **built** (`research/eda.py`, seven figures in `research/plots.py`, the `explore` command, and 22 tests).
- **Date:** 2026-08-27
- **Follows:** [ADR-006](ADR-006-features-and-stationarity.md), which established that scale-freedom is checked by measurement rather than by trusting a name. This applies the same standard to the descriptive work that comes before any feature exists.
- **Context:** Exploratory analysis is where a project decides what it believes before there is a model to check it against, so a mistake made here survives every correction applied downstream. The original project's EDA is not sloppy - it runs the right procedures, computes them correctly, and reports them clearly. It goes wrong in a way no reviewer catches by reading the code: **each test is pointed at a question that did not need asking**, and the answer is reported as though it had.

## Decision

### 1. Normality is tested on returns, and on the price only for contrast

Both are computed and printed side by side.

*Why:* the original runs D'Agostino-Pearson on `XAUUSD_Close` and reports "the data do NOT follow a normal distribution". That is true, uninformative, and unavoidable - **any series with a drift fails it**, and a price is not what a model at this horizon consumes. Nothing downstream changes whether it rejects or not.

The variable that matters is the return, and there the rejection has a consequence. Measured on the reference series, the Q-Q panel shows *why* each one fails, which a p-value cannot:

| | Lower tail | Upper tail |
|---|---:|---:|
| Normal, theoretical | -3.66 | +3.66 |
| Price | -1.14 | +3.12 |
| **Returns** | **-9.57** | **+10.94** |

The price's tails are **shorter** than a normal's, because a price is bounded below and was trending - it fails for a reason with no consequence. Returns reach roughly ten standard deviations where a normal reaches 3.7, and *that* is why a Sharpe ratio's textbook confidence interval is wrong on this data and why the evaluation layer will need a bootstrap rather than a t-statistic.

**Trade-off:** reporting both invites the reader to conclude "neither is normal, so it did not matter". The command prints the distinction rather than leaving it to be inferred, which is the only defence available in a table.

### 2. A group comparison may not use the variable that defined the groups

`compare_by_direction()` takes the columns to compare as an explicit argument, and its docstring carries the warning.

*Why:* the original defines `Direction = (Price_Change > 0)` and then runs a t-test of `Price_Change` between the resulting groups, reporting p < 0.001 and "Significant: Yes". **By construction the UP group cannot contain a negative value.** The test is a certainty dressed as a discovery, and it feeds the report's interpretation section.

No guard can catch this in general - whether a column leaks the outcome is a fact about how the column was built, not about its values - so the API makes the caller name the columns rather than defaulting to "all of them". A test reproduces the tautology (p < 1e-100 on the leaking column, p > 0.01 on an honest one beside it) so the warning has evidence rather than only a claim.

Welch rather than Student, incidentally: the two groups have no reason to share a variance, and assuming they do inflates significance when they do not.

### 3. Correlations are reported on levels and on returns, together

`correlation_pairs()` returns both columns and the gap between them.

*Why:* the original's EDA reports gold against the S&P at **+0.923** and reads it as an economic relationship between the two assets. Two series that both trend upward correlate on levels regardless of any relationship. Measured on the same data:

| | On levels | On returns | Inflated by |
|---|---:|---:|---:|
| SPX | **+0.919** | **+0.139** | +0.780 |
| NDX | +0.908 | +0.128 | +0.779 |
| BTCUSD | +0.896 | +0.095 | **+0.801** |
| WTIUSD | **-0.688** | **+0.173** | +0.514 |
| XAGUSD | +0.968 | +0.750 | +0.218 |
| DXY | -0.444 | -0.439 | **+0.005** |

Eight tenths of the gold-S&P correlation is shared trend. **WTIUSD changes sign** - on levels oil appears to move against gold, on returns it moves with it, so a reading based on the level would have been backwards. And **DXY barely moves**: the dollar is the one genuine relationship in the list, and it is the one the original did not highlight. Silver survives at +0.750, which is what a real relationship looks like.

This is [ADR-006](ADR-006-features-and-stationarity.md)'s feature-scale argument applied to the exploratory stage: the same reason a price level cannot be a feature is the reason a level correlation cannot be a finding.

### 4. The yearly breakdown is reported because the class balance drifts

`by_year()` returns bars, mean price, annualised volatility, price range and the share of rising bars, per year.

*Why:* the original plotted these four panels and read the volatility one as a regime story. The panel that matters is a different one. Measured on the reference series, the share of rising bars runs **49.6%, 50.2%, 52.5%, 52.5%** across 2022-2025 - a spread of **2.95 points**, which is larger than any edge this project is trying to detect.

That is the `oracle_gap` of [ADR-004](ADR-004-labels-splits-and-baselines.md) sec. 5 seen from the data's side rather than the model's: **which years land in the test block decides a meaningful part of any accuracy measured on it.** The same fact explains why a single chronological split is fragile here and why walk-forward is not a stylistic preference.

### 5. Two measurements about the matrix itself, reported before any model

`feature_correlations()` ranks every column by its correlation with the direction; `multicollinear_pairs()` lists the columns that are near-duplicates of each other.

*Why the first:* the original ranked these and printed a top twenty, which is the lesser half of the information. The number that decides what follows is the **largest** one. Measured on the reference matrix, the strongest single feature correlates with the direction at **0.0192** - and if the best individual predictor barely leaves zero, no combination of nineteen such columns produces a large edge. That is the same verdict the models reach, available before any model is fitted.

*Why the second:* it is the measurement behind a claim this repository had been making without evidence - that PCA collapses the matrix because indicators derived from one price series are variations of one another. Measured:

| Pair | Correlation |
|---|---:|
| `return` / `log_return` | **+1.0000** |
| `dist_sma_50` / `macd` | +0.9657 |
| `dist_sma_20` / `dist_ema_26` | +0.9639 |
| `macd` / `macd_signal` | +0.9546 |
| `dist_ema_26` / `rsi_14` | +0.9284 |

**Ten of 171 pairs exceed 0.9.** `return` and `log_return` correlate at exactly 1.0000, because log(1+r) is r for moves this small - the column is literally redundant. MACD turns out to be nearly the same thing as the distance to a 50-bar mean, which it is: both are differences of moving averages. And a bounded oscillator (`rsi_14`) tracks a distance-to-mean at 0.93, which is the least obvious pair on the list and the most instructive.

The PCA curve says the same from the other side: **the first component alone carries 44.3%** of the variance of nineteen features, and the first three carry 73%. A matrix of near-duplicates has one dominant direction and a tail of small corrections.

**Trade-off:** none of this removes a column. Dropping `log_return` because it duplicates `return` would be defensible, but it would also be a change to the feature set, and this project has just measured how much a two-column change moves the winner ([STATUS 2026-08-24](../status/STATUS-2026-08-models.md) sec. 5). The redundancy is measured and reported; acting on it is a decision for the modelling work, with its own before-and-after.

## Consequences

**The statistics live in `research`, the drawing in `plots`, and the writing in the CLI.** `qq_points()` computes the quantiles so the plotting module needs no statistics library, exactly as `roc_points()` was moved out of it in [ADR-008](ADR-008-figures-are-built-in-memory.md). Four figures were added and the layering guard's `savefig` rule covers them unchanged.

**`correlation_comparison` finally has a caller.** It was written, tested and dead for several days - ADR-008 recorded that honestly at the time. The `explore` command now draws it, and the figure is the clearest single image the project has produced: two bars per symbol, and for WTIUSD and USDJPY they point in opposite directions.

**`explore` offers `--json` and `--figures`**, like the other analysis commands, so the dashboard can render the exploratory panels without reimplementing any of this ([ADR-005](ADR-005-the-dashboard-runs-the-cli.md)).

**What is deliberately not built.** No hypothesis tests beyond these three, no autocorrelation or stationarity panels - those belong with the evaluation layer, where the bootstrap that depends on them lives. No seasonality analysis by hour of day, which is tempting given that staleness tracks the session clock, but which would be a new family of trials rather than a description of what is there.

**Replicated on the canonical dataset, and the replication is the strongest part.** Running `explore` on both - 23,181 bars from one venue over four years, 51,147 from another over nine - separates what is a property of the pair from what is a property of the window:

| | reference, levels | reference, returns | canonical, levels | canonical, returns |
|---|---:|---:|---:|---:|
| SPX | 0.919 | **0.139** | 0.919 | **0.131** |
| NDX | 0.908 | 0.128 | 0.918 | 0.139 |
| XAGUSD | 0.968 | **0.750** | 0.952 | **0.767** |
| **DXY** | **-0.444** | -0.439 | **+0.195** | -0.414 |
| **WTIUSD** | **-0.688** | 0.173 | **+0.195** | 0.040 |

**Every returns correlation is stable across venues and windows.** Silver holds at 0.75-0.77, the S&P at 0.13, the dollar at -0.41 to -0.44. Those are properties of the instruments.

**Two level correlations change sign.** DXY runs from -0.444 to **+0.195** and WTIUSD from -0.688 to **+0.195**, purely from extending the window back to 2018. Nothing about gold, oil or the dollar changed; the shape of their shared trend did. A level correlation is a fact about the stretch of history you looked at, and the original's EDA read one as an economic relationship.

The yearly panel replicates too, over nine years rather than four: the share of rising bars runs from **49.6% to 52.6%**, a spread of three points. The argument for walk-forward gets stronger with more data rather than weaker.

**What is still measured on one dataset:** the Q-Q tail figures quoted in sec. 1. The canonical price is more skewed (1.53 against 1.16) and considerably more peaked (kurtosis 1.50 against 0.37), so the price panel would look different; the returns panel is the one carrying the consequence and its conclusion - tails far past a normal's - is not in doubt.
