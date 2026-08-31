# STATUS 2026-08-27 - Scored on the whole history, with the power to say what that means

- **Question:** across nine years rather than one block, does any model clear the accuracy that pays for its own costs? And could this design have seen such an edge if it existed?
- **Verdict:** **No, and yes.** The best of six models scores **51.23%** pooled over 40,587 bars against a **53.49%** break-even - short by **2.26 points** *(at an assumed turnover; the measured shortfall is 0.46 for the closest model - see the correction note below)*. The design resolves **0.62%** at 80% power and would detect a profitable edge essentially every time, so the negative result is a finding rather than a failure to look. A nominally significant edge of **+0.97%** over the constant predictor does appear, survives a Holm correction, and is shown in section 4 to be mostly an artefact of the baseline moving.
- **Command:** `forecast-lab validate --target XAUUSD --timeframe 1H`
- **Machine-readable output:** [walk-forward-canonical.json](walk-forward-canonical.json) - *regenerated 2026-08-28 with per-model break-evens, so it carries the corrected figures rather than the ones below.*
- **Corrected by [STATUS 2026-08-28](STATUS-2026-08-turnover.md).** Every break-even quoted here assumes a turnover of 0.5 flips per bar. Measured, no model trades that often, and the shortfall of the closest is **0.46 points rather than 2.26**. The verdict is unchanged; the margins below are all too generous. Kept as published.
- **Inputs:** `raw/*.csv` (Dukascopy, 2018-01 to 2026-08), hashed in [data-manifest.json](data-manifest.json). Figures are pinned to the **2026-08-26 snapshot**; `fetch` extends the series forward, so a later download changes every hash and `verify` will say so. 50,948 rows x 19 columns, focus mode.

## 1. The headline

| Model | n | Accuracy | Baseline | Edge | Fold range | vs break-even |
|---|---:|---:|---:|---:|---:|---:|
| **HistGradientBoosting** | 40,587 | **51.23%** | 50.26% | **+0.97%** | 50.44%-51.87% | **-2.26%** |
| Random Forest | 40,587 | 51.12% | 50.26% | +0.86% | 50.70%-51.42% | -2.37% |
| LightGBM | 40,587 | 51.11% | 50.26% | +0.85% | 50.91%-51.60% | -2.39% |
| XGBoost | 40,587 | 51.07% | 50.26% | +0.82% | 50.03%-52.00% | -2.42% |
| Naive Bayes | 40,587 | 50.77% | 50.26% | +0.51% | 49.32%-51.23% | -2.73% |
| Logistic Regression | 40,587 | 50.67% | 50.26% | +0.41% | 49.93%-51.66% | -2.82% |

Six models, all positive against the baseline, none within two points of paying for itself.

## 2. The folds

Five expanding windows, each training on everything before its test block. 5 bars purged - one per boundary, for the label that reaches one bar forward.

| Fold | Train | Test | Tests from | To |
|---:|---:|---:|---|---|
| 1 | 8,490 | 8,491 | 2019-06 | 2020-11 |
| 2 | 16,981 | 8,491 | 2020-11 | 2022-05 |
| 3 | 25,472 | 8,491 | 2022-05 | 2023-10 |
| 4 | 33,963 | 8,491 | 2023-10 | 2025-03 |
| 5 | 42,454 | 8,491 | 2025-03 | 2026-08 |

42,455 bars scored, of which **40,587 carry a label** - the remainder are the bars whose horizon spans one of the venue's pauses, dropped rather than guessed.

Fold 1 tests through the March 2020 crash and fold 5 through the most recent year. The range that matters is not the average but the spread: the best model swings **50.44% to 51.87%** across regimes, which is 1.4 points of movement on a question where 3.49 would be needed.

## 3. Could this design have seen an edge worth having?

This is the section the original project never had, and without it none of the above means anything.

| Design | Out-of-sample bars | MDE at 80% power | Power for +3.49 pp |
|---|---:|---:|---:|
| Reference, single split | 3,294 | 2.17% | 99.1% |
| Canonical, single split | 7,306 | 1.45% | 100.0% |
| **Canonical, walk-forward** | **40,587** | **0.62%** | **100.0%** |

**A profitable edge needs 1,268 bars to detect. There are 40,587.**

For a week this project argued the opposite - that a single split "cannot resolve the effect it exists to test", true against the *assumed* 1 bp round trip that implies a 1.92-point effect against a 2.17-point MDE. [ADR-010](../adr/ADR-010-costs-are-measured-not-assumed.md) measured the venue's real spread at 1.86 bps, which moves the effect worth finding to **3.49 points**, and every design here sees that comfortably. The verdict turns from *"we could not see"* into **"we looked with power to spare and there was nothing"**.

## 4. The trap: why every edge turned positive

On a single split the selected model scored **-0.21%**. Here all six are positive. That looks like walk-forward rescuing the models. It is not.

| | Model accuracy | Baseline | Edge |
|---|---:|---:|---:|
| Single split, HistGradientBoosting | 51.44% | 50.93% | +0.51% |
| Walk-forward, HistGradientBoosting | **51.23%** | **50.26%** | **+0.97%** |

**The accuracy falls by 0.21 points. The baseline falls by 0.67.** The single split lands on one rising stretch where always-UP scores 50.93%; averaged across five stretches the majority class sits nearer a half, and the same model looks better against a weaker opponent.

The edge moved because **the thing it is measured against moved** - which is the error this whole repository is a correction of, arriving from a new direction. It is invisible unless both terms are printed, so `validate` prints both on every row and says so in prose underneath.

## 5. Is the +0.97% real?

Nominally, yes. It should be reported before it is discounted.

| | Value |
|---|---:|
| Bars pooled | 40,587 |
| Standard error | 0.2482% |
| Best edge | **+0.97% = 3.91 sigma** |
| Holm threshold for the best of 6 | z >= 2.39 |
| Survives multiplicity correction | **Yes** |
| Models with a positive edge | **6 of 6** |

And it is still not a strategy, for three reasons stated in order of weight:

1. **It is a quarter of what costs demand.** 0.97 against 3.49 points. No amount of statistical significance closes a factor of 3.6.
2. **Section 4 shows most of it is the baseline moving**, not the models predicting.
3. ~~**The standard error is optimistic by an unmeasured amount.**~~ **Measured 2026-08-28: it is not.** The stationary bootstrap puts the inflation at **0.97** - the correctness series autocorrelates at -0.020 even though RSI does at +0.93 - so `0.5/sqrt(n)` was already honest and 100% power is literal. The reasoning here was sound and its conclusion did not follow; [STATUS 2026-08-28](STATUS-2026-08-turnover.md) sec. 3 has the measurement.

Point 1 is arithmetic and survives every caveat attached to points 2 and 3.

## 6. What this run does not settle

- **PCA representations were not scored across folds.** Six models on raw features, not eighteen configurations. PCA lost every single-split run and fitting it per fold per model triples the runtime; recorded as a debt in [ADR-011](../adr/ADR-011-power-before-verdict.md), not an oversight.
- ~~The dependence correction is owed~~ - **discharged**, per point 3 above.
- **The gap benchmark is still owed.** Price rises through the venue's pauses 56-59% of the time, which clears 53.49% on its face - but those are precisely the hours that pay overnight financing. Until the swap is modelled, that stays a measurement rather than a strategy.
- **Nothing here rules out a lower frequency.** The break-even scales with how often the position turns over; a strategy holding twenty bars faces roughly a fifth of this bar. This project tested the hourly one.

## 7. Defects found by running this

**`rich` was silently eating the model representation.** `ModelScore.key` renders as `Random Forest [raw]`, and `[raw]` is valid console markup, so `train` has been printing `Selected on validation AUC: Random Forest` since the models commit - dropping the half of the identifier that says whether raw features or PCA won. Found because the same bug appeared in `validate`'s new output. Fixed with `rich.markup.escape` at all three call sites; the table in `_score_table` was never affected because it carries model and representation in separate columns.
