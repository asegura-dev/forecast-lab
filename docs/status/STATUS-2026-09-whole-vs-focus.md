# STATUS 2026-09-02 - WHOLE against FOCUS, the question the original asked and could not answer

- **Question:** `Proyecto_ASM` set out to determine *"si WHOLE (multi-mercado) o FOCUS (solo oro) es más efectivo"*. It never could: its two modes ran over 24,232 and 22,441 rows, so every difference between them mixed a change of feature set with a change of sample. This repository built both modes on a shared index in August and then **never ran a model in WHOLE** - the machinery existed and no committed artefact used it.
- **Verdict:** **FOCUS wins, and it is not close.** Matched on folds, bars and design, adding nine more markets turns four positive edges into one. The best edge falls from **+0.67% to +0.16%**, and Random Forest - the strongest configuration in FOCUS - drops from +0.67% to **-0.14%**. Nine times the features, less signal.
- **Commands:**
  `forecast-lab validate --target XAUUSD --timeframe 1H --mode focus --folds 3 --json`
  `forecast-lab validate --target XAUUSD --timeframe 1H --mode whole --folds 3 --json`
- **Inputs:** `raw/*.csv`, hashed in [data-manifest.json](data-manifest.json). Both runs score **36,567 bars** over **38,250 rows** from the same three expanding folds, and both resolve **0.65%** at 80% power. The only thing that differs is the feature set: 19 columns against 179.

## 1. The comparison

| Configuration | FOCUS accuracy | FOCUS edge | WHOLE accuracy | WHOLE edge | Change |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | 51.54% | **+0.67%** | 50.72% | **-0.14%** | **-0.81** |
| XGBoost | 51.20% | +0.34% | 50.73% | -0.14% | -0.48 |
| Naive Bayes | 50.79% | -0.07% | 49.83% | -1.03% | -0.96 |
| Logistic Regression | 50.87% | +0.00% | 50.44% | -0.42% | -0.42 |
| HistGradientBoosting | 50.97% | +0.11% | 50.83% | -0.03% | -0.14 |
| LightGBM | 51.01% | +0.14% | 51.03% | **+0.16%** | +0.02 |

| | FOCUS | WHOLE |
|---|---:|---:|
| Feature columns | **19** | **179** |
| Configurations with a positive edge | **4 of 6** | **1 of 6** |
| Best edge | **+0.67%** | +0.16% |
| Minimum detectable effect | 0.65% | 0.65% |

**Five of six estimators do worse with more markets.** The exception, LightGBM, improves by two hundredths of a point - well inside the noise this design can resolve.

## 2. Why this comparison is valid and the original's was not

The original's WHOLE and FOCUS ran over **24,232** and **22,441** rows. The cause is not what this repository said for a fortnight (see [the parity audit](STATUS-2026-09-parity.md) sec. 2.1): the notebook's indicator function ends with `dropna()` and runs once per symbol in a loop, so a 199-row warm-up is paid once per contributor. Ten contributors, ten warm-ups.

A 1,791-row difference between the two samples is larger than any effect either mode could produce. Whatever the notebook concluded from that comparison, it could not have been about the feature set.

Here a test asserts the two modes share an index exactly, and this run confirms it in the field: 38,250 rows and 36,567 scored bars, identical on both sides. **The only thing that varies is the thing being compared.**

## 3. What it does not change

**Nothing pays for itself in either mode.** Every configuration falls short of the accuracy its own turnover demands - by 0.59 to 1.81 points in FOCUS, and 1.09 to 1.81 in WHOLE. The verdict in [FINDINGS](../../FINDINGS.md) is untouched; this answers a different question, about which feature set is less bad.

**And the winning edge is barely resolvable.** Random Forest's +0.67% sits just above this design's 0.65% detection floor at three folds. It is a real ordering, not a large one, and the five-fold run in [STATUS 2026-08-27](STATUS-2026-08-walk-forward.md) is the sharper instrument.

## 4. What it suggests

The obvious reading is that nine extra markets contribute noise rather than signal, and the feature-correlation work supports it: the strongest single correlation with the target across the FOCUS matrix is **0.0192**, and adding 160 more columns of comparable strength adds variance to estimate without adding information to find.

There is a second reading this run cannot separate from the first, and it should be named rather than assumed away: WHOLE's auxiliaries are **carried forward** across their own venues' pauses, so a large share of their rows repeat a stale value. [ADR-003](../adr/ADR-003-target-anchored-alignment.md) records the staleness per symbol and refuses to fabricate a bar, but a stale column is still a column whose value did not move for the reason the model will attribute it to. Distinguishing "these markets carry no signal" from "these markets carry signal that staleness destroys" needs a run that drops stale rows rather than carrying them, and `align --max-staleness` already exists to do it. Not run here.
