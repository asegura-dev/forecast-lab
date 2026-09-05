# ADR-012 - The last two assumed parameters, measured

- **Status:** Accepted - **built** (`research/dependence.py`, `flip_rate()` in `research/costs.py`, per-model break-evens in `validate`, 12 tests for the bootstrap and 8 more for turnover).
- **Date:** 2026-08-28
- **Follows:** [ADR-010](ADR-010-costs-are-measured-not-assumed.md), which measured the spread and then assumed the turnover it multiplies; and [ADR-011](ADR-011-power-before-verdict.md), which asserted a correction it never computed.
- **Context:** This repository exists to correct an analysis that assumed things. Two of its own numbers were still assumed, in the two modules built to stop exactly that. One was assumed **in the project's favour** and one **against it**, and both are corrected here. Neither was caught by review; both were caught by measuring.

## Decision

### 1. Each model's break-even is computed from its own measured turnover

`flip_rate()` counts how often a prediction series changes, inside folds only. `validate` gives every model the threshold that rate implies instead of one shared figure.

*Why:* ADR-010 sec. 2 stated the arithmetic honestly - `p = 0.5 + f·c/(2·E|r|)` - and then said of `f`: *"At `f = 0.5` - which is what a model with no persistence produces, and every model measured here qualifies"*. **That clause was never measured, and it is false.** Predictions autocorrelate from **+0.22** (the trees) to **+0.61** (Naive Bayes) - measured inside folds, so no join between 2020 and 2022 contributes - and no model flips anywhere near half the time:

| Model | Accuracy | Flip rate | Bars held | Break-even at `f=0.5` | Break-even measured | Short by |
|---|---:|---:|---:|---:|---:|---:|
| **Naive Bayes** | 50.77% | **17.6%** | 5.7 | 53.49% | **51.23%** | **0.46%** |
| Logistic Regression | 50.67% | 26.9% | 3.7 | 53.49% | 51.88% | 1.21% |
| Random Forest | 51.12% | 37.1% | 2.7 | 53.49% | 52.59% | 1.47% |
| LightGBM | 51.11% | 37.3% | 2.7 | 53.49% | 52.61% | 1.50% |
| XGBoost | 51.07% | 37.4% | 2.7 | 53.49% | 52.61% | 1.54% |
| HistGradientBoosting | 51.23% | 38.4% | 2.6 | 53.49% | 52.68% | 1.45% |

A persistent model pays the spread less often and faces a lower bar. **The published margin was therefore too generous by more than a point**, and by two and a quarter for the model it mattered most to: the best case was reported as *"short by 2.26 points"* and is really **short by 0.46** - a near miss rather than a rout.

**Trade-off:** measuring `f` from a model's own predictions makes the threshold depend on the model, which is less quotable than one number and invites the objection that a model could game it by trading less. That objection is correct and is the point: a model that holds longer *is* cheaper to run, and the arithmetic should say so rather than charging every strategy for a turnover it does not have. The constant predictor is the limiting case - flip rate zero, break-even exactly 50% - which is why it is scored against the *baseline* as well, and fails there.

### 2. The dependence correction is computed, and it is approximately one

`measure_dependence()` runs a stationary bootstrap per fold and reports how much the standard error inflates.

*Why:* ADR-011 attached a caveat to every power figure it published - overlapping feature windows make bars dependent, so the effective sample is smaller than the row count and *"100% power must not be read literally"*. Reasonable, prominent, repeated, and **wrong**:

| | Lag-1 autocorrelation |
|---|---:|
| Realised volatility (168h) | **+0.9994** |
| MACD signal | **+0.9970** |
| The label (direction) | **-0.0246** |
| **Model correctness** | **-0.0199** |

The features are strongly dependent. The thing being averaged is not a feature - it is whether the model was right - and **that series is indistinguishable from independent draws**. A near-coin-flip outcome inherits almost none of the dependence of its inputs. Measured over 40,587 bars the bootstrap standard error is **0.2413%** against **0.2482%** for independent draws: an inflation of **0.97**, an effective sample of 42,937, and blocks of 1.5 to 3.7 bars.

So `0.5/√n` was already honest, if anything conservative, and the corrected figure makes the closest model's shortfall **1.91 standard errors rather than 1.86** - a caveat that dissolves in the direction of the conclusion it was hedging.

**Trade-off, and the reason this is not self-serving:** a negative result about one's own caveat is worthless if the instrument could not have found the opposite. Pointed at a synthetic AR(1) correctness series the same code returns an inflation of **3.64x** and an effective sample of 1,513 out of 20,000. A test pins the property rather than these numbers - on 6,000 synthetic rows it asserts an inflation above 2.0 and an effective sample below half of n - so the null finding stays falsifiable, and the two figures above are an illustration from a one-off probe rather than a fixed expectation.

### 3. `arch` is used and deliberately not quarantined

*Why:* the three existing quarantines (`ta`, the estimator stack, matplotlib) all exist for one reason - those libraries ship no type information, and `mypy --strict` is only worth having while the untyped surface stays auditable. **`arch` ships `py.typed`**, so that reason does not apply and adding a fourth guard would be a rule without its motive. One annotation is wrong rather than missing - `StationaryBootstrap` is typed as taking an `int` block size when a *stationary* bootstrap's block length is real by construction (Politis-White returns 2.1 and 101.3) - and that is overridden at the single call site with the reason written beside it. Rounding to satisfy the stub would have changed a published number to work around someone else's error.

## Consequences

**The verdict survives and its margin does not.** No model clears the accuracy that would pay for its own turnover. But the closest, Naive Bayes holding a position 5.7 bars, misses by **0.46 points - 1.86 standard errors**, where the published figure implied 9.1. At a 5% one-sided test that is still a rejection, and it is a thin one. The honest phrasing is no longer *"nothing comes close"* but **"nothing clears it, and the closest is within two standard errors at the median spread"**.

**The spread distribution is what makes it firm.** ADR-010 chose the median and reported the rest of the distribution beside it. That choice now carries weight it did not before:

| Spread used | Naive Bayes break-even | Short by |
|---|---:|---:|
| Median, 1.86 bps | 51.23% | 0.46% (1.86σ) |
| Mean, 2.13 bps | 51.40% | 0.63% (2.56σ) |
| 95th percentile, 3.60 bps | 52.37% | 1.60% (6.46σ) |

A strategy cannot choose to trade only in median conditions. The verdict is thin against the friendliest assumption and comfortable against a realistic one, and both are published.

**Two assumptions, opposite directions, one lesson.** The turnover assumption flattered the conclusion; the dependence assumption undermined it. Neither was noticed by writing carefully - ADR-010's clause reads as a measurement and ADR-011's caveat reads as rigour. **Both were only caught by computing them**, which is the same argument this project makes about the original analysis, now with the project itself as the example.

**What is still assumed**, and each raises the bar rather than lowering it: slippage beyond the quoted spread, the rollover surcharge on bars next to gaps, and the overnight swap. The swap is the one that matters, because it is charged in exactly the hours the gap finding lives in.

~~**PCA representations still are not scored across folds**~~ - **discharged 2026-08-30** as `validate --pca`, off by default because it triples the runtime. Measured across folds, **18 configurations and none clears its own break-even**; every PCA row lands below its raw counterpart, which is what the default assumed and had not verified.

**The escape route ADR-010 named is narrower than it looked.** It offered *"a trend follower holding for twenty bars pays a fifth as much"* as the way out. Measured, these models already hold 2.6 to 5.7 bars and already take part of that discount - and still fall short. The remaining room is between 5.7 bars and twenty, not between one and twenty.
