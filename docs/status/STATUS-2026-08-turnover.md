# STATUS 2026-08-28 - Two assumed parameters, measured, in opposite directions

- **Question:** the verdict rests on two numbers this project never computed - how often each model actually trades, and how much serial dependence inflates its standard errors. What are they?
- **Verdict:** **The conclusion survives and its margin collapses.** Turnover was assumed at 0.5 flips per bar; measured it runs **17.6% to 38.4%**, so every published break-even was quoted at a frequency no model has. The best case goes from *"short by 2.26 points"* to **short by 0.46 - 1.86 standard errors**. Serial dependence, meanwhile, was asserted to be inflating the standard error by an unmeasured factor; measured it is **0.97**, so the naive figure was already honest. One assumption flattered the conclusion, the other undermined it, and neither was caught by reading carefully.
- **Command:** `forecast-lab validate --target XAUUSD --timeframe 1H`
- **Machine-readable output:** [walk-forward-canonical.json](walk-forward-canonical.json)
- **Inputs:** `raw/*.csv` (Dukascopy, 2018-01 to 2026-08), hashed in [data-manifest.json](data-manifest.json). 50,948 rows x 19 columns, focus mode, 40,587 bars scored.

## 1. The headline

| Model | Accuracy | Baseline | Edge | Flip rate | Bars held | Break-even | Short by |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Naive Bayes** | 50.77% | 50.26% | +0.51% | **17.6%** | **5.7** | **51.23%** | **-0.46%** |
| Logistic Regression | 50.67% | 50.26% | +0.41% | 26.9% | 3.7 | 51.88% | -1.21% |
| HistGradientBoosting | 51.23% | 50.26% | +0.97% | 38.4% | 2.6 | 52.68% | -1.45% |
| Random Forest | 51.12% | 50.26% | +0.86% | 37.1% | 2.7 | 52.59% | -1.47% |
| LightGBM | 51.11% | 50.26% | +0.85% | 37.3% | 2.7 | 52.61% | -1.50% |
| XGBoost | 51.07% | 50.26% | +0.82% | 37.4% | 2.7 | 52.61% | -1.54% |

**The ordering inverts.** By accuracy, HistGradientBoosting wins and Naive Bayes is fifth. By the question that decides anything - does it pay for itself - Naive Bayes is closest by a full point, because it trades less than half as often. Accuracy alone was ranking the models by the wrong criterion the entire time.

## 2. The flip rate, and why 0.5 was wrong

[ADR-010](../adr/ADR-010-costs-are-measured-not-assumed.md) sec. 2 wrote: *"At `f = 0.5` - which is what a model with no persistence produces, and every model measured here qualifies"*. The clause reads like a measurement. It was an assumption, and it is false:

| Model | Lag-1 autocorrelation of predictions |
|---|---:|
| HistGradientBoosting | +0.2159 |
| XGBoost | +0.2376 |
| Random Forest | +0.2384 |
| LightGBM | +0.2390 |
| Logistic Regression | +0.3932 |
| **Naive Bayes** | **+0.6118** |

Computed **inside folds**, like the flip rate. A first pass concatenated the five folds and
measured across the joins, which inflated every figure by two to four hundredths - a small
error, in a table whose whole point is that the small figure everyone assumed was zero is
not. It is corrected here and the command computes it the same way turnover is counted.

These models are persistent. They hold positions for 2.6 to 5.7 bars, pay the spread proportionally less often, and face thresholds up to 2.26 points below the one they were all being judged against.

Turnover is counted **inside folds only**. Charging a round trip between the last bar of fold 2 and the first of fold 3 would invent a trade separated by eighteen months.

## 3. The dependence correction, which turned out not to be needed

[ADR-011](../adr/ADR-011-power-before-verdict.md) attached this caveat to every power figure it published: overlapping feature windows make bars dependent, the effective sample is smaller than the row count, *"100% power must not be read literally"*. Measured:

| Series | Lag-1 autocorrelation |
|---|---:|
| RSI-14 | **+0.9293** |
| Realised volatility, 24h | **+0.9888** |
| Log return | -0.0173 |
| The label (direction) | -0.0246 |
| **Model correctness** | **-0.0199** |

The premise was right and the conclusion did not follow. The features are as dependent as feared; **the quantity being averaged is not a feature.** It is whether the model was right, and that series is indistinguishable from independent draws - because the thing it tracks, the direction itself, is too.

| | Value |
|---|---:|
| Bars | 40,587 |
| Naive standard error, `0.5/sqrt(n)` | 0.2482% |
| Stationary bootstrap standard error | **0.2413%** |
| **Inflation** | **0.97x** |
| Effective sample | 42,937 |
| Block lengths across folds | 1.5 - 3.7 |

The correction runs *toward* the conclusion, not away: the closest model's shortfall becomes **1.91 standard errors** rather than 1.86.

**Why this null is credible.** Pointed at a synthetic AR(1) correctness series, the same code returns an inflation of **3.64x** and an effective sample of 1,513 out of 20,000. That case is pinned by a test, so "we measured 0.97" cannot be confused with "we measured nothing".

## 4. Is the remaining gap real?

Naive Bayes at 50.77% against a 51.23% break-even. Everything now hangs on 0.46 points.

| Spread assumption | Break-even | Short by | Standard errors |
|---|---:|---:|---:|
| **Median, 1.86 bps** | 51.23% | 0.46% | **1.86** |
| Mean, 2.13 bps | 51.40% | 0.63% | 2.56 |
| 95th percentile, 3.60 bps | 52.37% | 1.60% | 6.46 |

At the friendliest assumption the rejection is thin - a one-sided 5% test needs 1.645 and this is 1.86, so it holds, barely. At any harsher and realistic assumption it is comfortable. **A strategy cannot choose to trade only in median conditions**, which is what makes the second and third rows the operative ones and the first the honest worst case for the argument.

It also fails the other test independently: its edge over the constant predictor is **+0.51%**, and a constant predictor is not a strategy anyone would run.

## 5. Widened to eighteen configurations

`validate --pca` (2026-08-30) scores the same six estimators on raw features and on PCA at 95% and 90% of training variance - the eighteen configurations the original project compared, now across folds and against per-model thresholds.

**None clears its own break-even, and every PCA row lands below its raw counterpart.** The closest three:

| Configuration | Accuracy | Flip rate | Break-even | Short by |
|---|---:|---:|---:|---:|
| **Naive Bayes** (raw) | 50.77% | 17.6% | 51.23% | **-0.46%** |
| Naive Bayes (pca-95) | 50.72% | 20.7% | 51.44% | -0.72% |
| Naive Bayes (pca-90) | 50.96% | 25.0% | 51.75% | -0.79% |

The worst is LightGBM (pca-95) at 2.43 points short. PCA is off by default in `validate` because it triples an already slow command for a representation that lost every single-split run; that reason was an expectation until this run, and is now a measurement.

Note what PCA does to *turnover*: it raises the flip rate in every case (Naive Bayes from 17.6% to 20.7% and 25.0%), so a reduced representation both scores slightly worse and trades slightly more, and is penalised twice.

## 6. What this run does not settle

- **The swap is still unmodelled**, and it is the one that matters. It raises every threshold above, and it is charged in exactly the hours the gap finding lives in.
- **Slippage and the rollover surcharge** are likewise absent. Both raise the bar.
- **PCA representations are still not scored across folds**, carried from ADR-011.
- **The remaining escape route is narrower than ADR-010 suggested.** It offered "a trend follower holding twenty bars pays a fifth as much". These models already hold up to 5.7 bars and already take part of that discount. The room left is between 5.7 bars and twenty, not between one and twenty.

## 7. What this cost to find

Nothing but running it. Both defects sat in prose that reads as careful - one clause phrased as a measurement, one caveat phrased as rigour - and both survived being written, reviewed and published. Neither survived being computed. That is the same argument this repository makes about the analysis it corrects, with the repository itself as the worked example.
