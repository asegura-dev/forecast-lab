# Chapter 4 - Costs, power and validation

This chapter answers two questions the original analysis never asked, which between them decide whether any accuracy in this repository means anything. **What does a prediction have to beat?** - the accuracy at which a directional strategy stops losing money, computed from the venue's own quoted spread and from how often each model actually trades. And **could the experiment have seen it?** - the smallest edge the design resolves, and whether the folds it was measured over are what they appear to be. Without the first, "51.23%" is a number without a threshold. Without the second, "no model beat its baseline" is a shrug rather than a finding. Every figure below comes from a committed artefact named where it is used, under `docs/status/`.

---

## 1. The break-even arithmetic

A directional strategy holds a position of +1 or -1 into every bar. With accuracy `p` and a mean absolute move `E|r|` per bar, the expected gross return per bar is

    E[gross] = p·E|r| - (1 - p)·E|r| = (2p - 1)·E|r|

It pays the round-trip cost `c` on the share `f` of bars where the position changes, so the expected cost per bar is `f·c`. That one flip costs exactly one round trip and not two is the convention enforced in `net_of_costs()` in `src/forecast_lab/research/costs.py`: turnover is `|Δposition|`, which is 2 on a full reversal, and the charge is `turnover / 2 × round_trip`.

Setting expected profit to zero:

    (2p - 1)·E|r| = f·c
    p = 0.5 + f·c / (2·E|r|)

Implemented as the `accuracy` property of `BreakEven` in `costs.py`, one line, exactly as written; `edge_required` on the same dataclass returns `accuracy - 0.5`, which is the number an edge is compared against. At `f = 0.5` - a predictor with no persistence - it collapses to `0.5 + c / (4·E|r|)`.

**The terms.** `c` is the round trip in basis points, measured by `summarise_spread()`: it reads the `spread` column `fetch` writes on every bar, divides by `close`, multiplies by 10,000, and reports median, mean, 95th percentile and max. `E|r|` is `mean_absolute_return_bps` on the same `SpreadSummary`, the mean of `|Δclose / close|` - what a correct prediction earns before costs. `f` is turnover, and section 2 is about the fact that it was assumed rather than measured.

**Measured over 51,147 hourly bars of gold** (the row count is in `docs/status/data-manifest.json`), ADR-010 sec. 1 publishes:

| | Round trip | Break-even at `f = 0.5` |
|---|---:|---:|
| Assumed (planning) | 1.00 bps | 51.87% |
| Probe, short window | 1.60 bps | 53.00% |
| **Measured, median** | **1.86 bps** | **53.49%** |
| Measured, mean | 2.13 bps | 53.99% |
| Measured, 95th percentile | 3.60 bps | 56.74% |

`verdict-canonical.json` carries the median as `costs.round_trip_bps = 1.8648852404334182`; `walk-forward-canonical.json` carries the threshold as `break_even.at_assumed_flip_rate = 0.5349268747697784` and names its inputs in `break_even.source`: *"from a measured median spread of 1.86 bps against a 13.35 bps average move"*.

**The ratio is the whole problem.** 1.86 / 13.35 is **14%** - the median round trip is a seventh of the average move a correct prediction earns, and `SpreadSummary.cost_to_move_ratio` computes exactly this and nothing else. The required edge above chance is proportional to `c/E|r|` and to nothing else about the market: 14% of the move, halved by the `2`, is 3.49 points at `f = 0.5`. No modelling improvement changes that ratio. The only two levers are trading less often (section 2) and trading a horizon where the move is larger against a fixed round trip - which is why every ADR from 010 onward names the same escape route and refuses to close it.

Two behaviours to remember. `break_even()` *raises* a `CostError` rather than returning an accuracy at or above 1.0, because "104%" is arithmetic saying the strategy cannot exist at that frequency and a percentage invites a reader to treat it as merely demanding. And where no spread column exists, `_break_even_for()` in `src/forecast_lab/interfaces/cli.py` returns the literal constant `FALLBACK_BREAK_EVEN = 0.5192`; it does **not** recompute `0.5 + c/(4·E|r|)` from that series' own average move. That is why ADR-010's table shows 51.87% for a 1 bp round trip while the figure quoted throughout the project's history is 51.92% - the first is the assumption evaluated against the canonical `E|r|`, the second a hard-coded number inherited from a different series. The command prints which one it used.

---

## 2. Turnover: the parameter assumed inside the module built to stop assuming

`DEFAULT_FLIP_RATE = 0.5` in `costs.py`. The default itself is defensible - a break-even computed without a prediction series has to assume something, and the expensive case is the safe direction to be wrong in. The defect was ADR-010 sec. 2 writing *"At `f = 0.5` - which is what a model with no persistence produces, and every model measured here qualifies"*: a clause that reads as a measurement, was never computed, and is false. ADR-010 now carries a correction block at the point of the error.

`flip_rate()` in `costs.py` measures it. Given `segments`, a sequence of contiguous prediction stretches, it counts `(array[1:] != array[:-1]).sum()` per segment and divides by the total *within-segment* pairs. The segment structure is load-bearing: counting a change between the last bar of fold 2 and the first of fold 3 would invent a trade separated by eighteen months. `PooledScore.prediction_segments` in `src/forecast_lab/research/models/validation.py` feeds it, and `FoldScore` keeps `predictions` and `correct` as tuples rather than reducing them to a mean for exactly this reason - a flip rate cannot be recovered from an accuracy. `bars_held()` returns `1/f`, the readable form.

Measured, from `walk-forward-canonical.json` (`models[]`):

| Model | Accuracy | Lag-1 of predictions | Flip rate | Bars held | Break-even | Short by |
|---|---:|---:|---:|---:|---:|---:|
| **Naive Bayes** | 50.77% | **+0.6118** | **17.6%** | **5.68** | **51.23%** | **0.46** |
| Logistic Regression | 50.67% | +0.3932 | 26.9% | 3.72 | 51.88% | 1.21 |
| Random Forest | 51.12% | +0.2384 | 37.1% | 2.69 | 52.59% | 1.47 |
| LightGBM | 51.11% | +0.2390 | 37.3% | 2.68 | 52.61% | 1.50 |
| XGBoost | 51.07% | +0.2376 | 37.4% | 2.67 | 52.61% | 1.54 |
| HistGradientBoosting | 51.23% | +0.2159 | 38.4% | 2.61 | 52.68% | 1.45 |

The predictions autocorrelate at +0.22 to +0.61 (`PooledScore.persistence`, via `lag_one_autocorrelation()` in `dependence.py`, also inside folds only). No model flips anywhere near half the time. Because `p = 0.5 + f·c/(2·E|r|)` is linear in `f`, each threshold is 53.49% rescaled by `f/0.5`; `_break_even_at_assumed()` in `cli.py` inverts that same rescaling to keep the old shared figure in the payload beside the new per-model ones, and `test_break_even_is_linear_in_the_flip_rate` pins the linearity.

Two consequences. First, the published margin was too generous by more than a point: the best case was reported as "short by 2.26 points" and is short by **0.46**, or 1.86 naive standard errors - a thin rejection at a one-sided 5% test, not a rout. Second, **the ordering inverts**. By accuracy, HistGradientBoosting wins and Naive Bayes is fifth; by the question that decides anything, Naive Bayes is closest by a full point, because it trades less than half as often. `cli.py` therefore sorts `pooled` by `accuracy - thresholds[key]`, so `pooled[0]` is the model a reader has to argue with.

*Checking figures against each other:* the module docstring of `costs.py` quotes Naive Bayes at "1.84 standard errors" and HistGradientBoosting as short by "1.44", where ADR-012, `STATUS-2026-08-turnover.md` and the JSON all give 1.86 and 1.45. The JSON is right; the docstring rounds stale.

---

## 3. Power: whether the experiment could have seen it

An experiment that finds nothing has two explanations - there was nothing, or it could not have found it - and only a power calculation separates them. `src/forecast_lab/research/power.py` computes it:

    SE  = 0.5 / sqrt(n)                                Design.standard_error
    MDE = (z_(1-α) + z_(power)) · SE                   Design.minimum_detectable_effect
    n   = ((z_(1-α) + z_(power)) · 0.5 / effect)² + 1  required_sample()

**Why 0.5 in the standard error.** The variance of a proportion is `p(1-p)`, maximised at `p = 0.5`. Using 0.5 rather than the observed accuracy gives the widest interval and the most conservative claim, and every accuracy here sits within two points of a half, so the conservatism costs nothing and removes an argument.

**Why one-sided.** `design()` takes `one_sided: bool = True`. The question is "is this better than chance", not "is this different from chance". A two-sided test demands a larger effect for the same power; using one while asking a directional question is quiet conservatism that would make a negative result harder to falsify - the opposite of what the module is for. `power_for()` uses the same critical value: `norm.cdf(effect / SE - z_(1-α))`.

From ADR-011 sec. 1, at α = 0.05 and 80% power:

| Design | Out-of-sample bars | MDE | Power for +3.49 pp |
|---|---:|---:|---:|
| Reference, single split | 3,294 | 2.17% | 99.1% |
| Canonical, single split | 7,306 | 1.45% | 100.0% |
| **Canonical, walk-forward** | **40,587** | **0.62%** | **100.0%** |

`walk-forward-canonical.json` carries the bottom two rows exactly: `power.walk_forward_mde = 0.006171071825814692`, `power.single_split_mde = 0.014545017330852384`, `power.single_split_rows = 7306`. Detecting the 3.49-point effect needs **1,268 bars**; there are 40,587. It is 1,268 rather than 1,269 because the break-even is 53.4927% before rounding and `required_sample()` truncates with `int()` before adding one.

**Read the JSON's own power figure carefully.** `power.power_for_break_even = 0.99954` and `power.bars_required = 10216` are *not* the 100.0% and 1,268 of the table above: `cli.py` computes them against `gap = thresholds[best.key] - 0.5`, the **per-model** threshold from section 2 - 1.23 points for Naive Bayes, not 3.49. Both readings clear: 10,216 bars required against 40,587 scored.

**Computing this reversed the project's own argument.** For a week the claim was that a single 70/15/15 split *"cannot resolve the effect it exists to test"*: MDE 2.17 points against a break-even 1.92 points above chance. True of the *assumed* cost. Once ADR-010 measured the real one, the effect worth detecting became 3.49 points and every design sees it. The verdict turns from "we could not see" into **"we looked with power to spare and there was nothing"**. A reader is entitled to suspect a calculation that flatters its author; ADR-011's defence is that the same module, run a week earlier, produced the opposite reading and it was published anyway. The input changed, not the arithmetic.

---

## 4. Walk-forward: expanding folds, pooled, baseline refitted

`src/forecast_lab/research/walkforward.py` cuts a `DatetimeIndex` into train/test pairs. With `folds = 5` it divides the index into `folds + 1 = 6` blocks of `n // 6` rows - the first fold needs a training block before its test block, so there is always one extra. Fold `k` trains on `index[0 : block·k - horizon]` and tests on `index[block·k : block·(k+1)]`. Expanding is the default because it trains on all history to date, which is what a deployment would have; a fixed-length rolling window asks whether recent history predicts better than distant, a hypothesis about regime change rather than a validation design.

**Purging happens at every boundary, not once for the scheme.** A label at the last training bar reaches `horizon` bars into the test block, so `horizon` rows come off the end of every training window - five folds at horizon 1 purge five bars, which is `scheme.purged = 5`. No embargo, for ADR-004's reason: nothing after a test block trains in the *same* fold. A **later** fold does train on it, which is correct - by then it is history - and is precisely what makes this walk-forward rather than k-fold on shuffled data.

The committed scheme:

| Fold | Train | Test | Tests from | To |
|---:|---:|---:|---|---|
| 1 | 8,490 | 8,491 | 2019-06-21 | 2020-11-26 |
| 2 | 16,981 | 8,491 | 2020-11-26 | 2022-05-06 |
| 3 | 25,472 | 8,491 | 2022-05-06 | 2023-10-11 |
| 4 | 33,963 | 8,491 | 2023-10-11 | 2025-03-19 |
| 5 | 42,454 | 8,491 | 2025-03-19 | 2026-08-26 |

`scheme.scored = 42455`; `models[].n = 40587` carry a label. The difference is the bars whose horizon spans one of the venue's pauses, dropped rather than guessed. And because 50,948 rows floor-divide into blocks of 8,491, the final two rows fall outside every fold - integer division, not a bug, but the reason `test_rows` and the row count never match.

**Folds are pooled by row count and never averaged.** `PooledScore.accuracy` in `validation.py` is `sum(f.accuracy * f.n) / sum(f.n)`. Averaging the five fold accuracies and quoting a t-statistic over them is the obvious alternative and it is wrong: **the folds share training data** - fold 5 trains on everything fold 1 trained on - so they are not five independent experiments and their spread is not a sampling distribution. Pooling makes one accuracy over one row count, which is what `SE = 0.5/√n` describes. The spread is still reported, via `worst_fold` and `best_fold`, as a description of how much the answer moves across regimes and never as an interval: HistGradientBoosting runs 50.44% to 51.87%, 1.4 points of movement on a question where 3.49 would be needed.

**The baseline is refitted inside every fold.** In `score_walk_forward()` the constant predictor's class is `1 if float((train_labels == 1.0).mean()) > 0.5 else 0`, computed from `labels.reindex(fold.train)` - that fold's own training block and nothing after it. Carrying one global majority across all five would score the baseline with information the model was denied, which is this project's own leak pointed the other way.

**And here is the trap.** Under walk-forward every model's edge over the baseline turns positive, after being negative or negligible on a single split. It reads like walk-forward rescuing the models. It is not:

| | Model accuracy | Baseline | Edge |
|---|---:|---:|---:|
| Single split, HistGradientBoosting | 51.44% | 50.93% | +0.51% |
| Walk-forward, HistGradientBoosting | **51.23%** | **50.26%** | **+0.97%** |

The first row is `model-comparison-canonical.json` (HistGradientBoosting, `raw`, block `test`: accuracy 0.514372, baseline 0.509307, n = 7,306); the second is `walk-forward-canonical.json`. **The accuracy falls by 0.21 points. The baseline falls by 0.67.** The single split lands on one rising stretch where always-UP scores 50.93%; averaged over five stretches the majority class sits nearer a half, and the same model looks better against a weaker opponent. **The edge moved because the thing it is measured against moved** - the identical error this repository was built to correct, from a new direction. It is invisible unless both terms are printed, so `validate` carries both on every row.

---

## 5. Serial dependence: the caveat that dissolved

`power.py` treats the `n` outcomes as independent draws. ADR-011 flagged that as broken everywhere the figure appeared: overlapping feature windows make neighbouring rows dependent, the effective sample must be smaller than the row count, *"100% power must not be read literally"*. Prominent, repeated, and wrong.

`src/forecast_lab/research/dependence.py` measures it. `measure_dependence(segments)` takes one contiguous stretch per fold. Per segment it selects a block length with `optimal_block()` - Politis and White's rule, delegated to `arch.bootstrap.optimal_block_length(...)["stationary"]` and floored at `MINIMUM_BLOCK = 1.0`, since a stationary bootstrap draws block lengths from a geometric distribution with mean `b` and a selected value below one honestly means "resample independently". It then runs a `StationaryBootstrap` at that length, seeded (`BOOTSTRAP_SEED = 42`, `BOOTSTRAP_REPLICATIONS = 1_000`) so the figure is byte-reproducible, and takes the standard deviation of the resampled means.

**Segments are bootstrapped separately and their variances combined**, because a block straddling a fold boundary would treat 2020 as adjacent to 2022:

    Var(pooled mean) = Σ (n_i · SE_i)² / n²
    SE_bootstrap     = sqrt( Σ (n_i · SE_i)² ) / n

From which `inflation = SE_bootstrap / SE_naive` and `effective_sample = (0.5 / SE_bootstrap)²`, the number of independent draws the series is worth. `material` is `|inflation - 1| >= 0.10`, a stated judgement: below that the MDE moves by less than the rounding accuracies are reported at.

Measured on the closest model's `correct_segments` (`dependence` in `walk-forward-canonical.json`):

| | Value |
|---|---:|
| Bars | 40,587 |
| Lag-one autocorrelation of correctness | **-0.019874** |
| Naive standard error, `0.5/√n` | **0.2482%** |
| Stationary bootstrap standard error | **0.2413%** |
| **Inflation** | **0.9722** |
| Effective sample | 42,937 |
| Block lengths across the five folds | 1.500, 1.932, 3.108, 3.241, 3.673 |
| `material` | `false` |

**The contrast is the finding.** The features *are* as dependent as feared: the same JSON's `serial_dependence_contrast` records realised volatility (168h) at **+0.9994**, the MACD signal at +0.9970, ATR% at +0.9956 and ADX-14 at +0.9947, against log return at -0.0173 and the label itself at **-0.0246**. But **the quantity being averaged is not a feature** - it is whether the model got the direction right, and that series is indistinguishable from independent draws, because the thing it tracks is too. A near-coin-flip outcome inherits almost none of the dependence of its inputs. So `0.5/√n` was already honest, if anything conservative, and the correction moves the closest model's shortfall to 1.91 standard errors rather than 1.86 - a caveat that dissolves in the direction of the conclusion it was hedging.

**Two discrepancies to know about.** ADR-012 and `STATUS-2026-08-turnover.md` quote the contrast as "RSI-14 +0.9293, realised volatility (24h) +0.9888". Those figures are not in the committed JSON, whose `serial_dependence_contrast` names a different five columns as the most autocorrelated - and the code producing it (the `contrast` dict in `cli.py`) computes lag-one for *every* column, so the JSON's top five are what the run actually found. Second: the claim, in both `dependence.py` and ADR-012, that the AR(1) falsification case "returns an inflation of 3.64x and an effective sample of 1,513 out of 20,000" and is "pinned by a test" is half true. The test that pins it, `test_the_instrument_finds_dependence_that_is_really_there` in `tests/unit/test_dependence.py`, runs on **6,000** rows and asserts `inflation > 2.0`, `effective_sample < n/2`, `lag_one > 0.5`, `max(blocks) > 10.0`. That the instrument can find dependence which is really there is genuinely pinned; the specific 3.64 / 1,513 / 20,000 figures are not.

---

## What the two halves license together

`verdict-canonical.json` records **0 of 18** configurations profitable under either position framing, against an always-long benchmark returning +100.9% over the scored period (`profit.benchmark_cumulative = 1.0090873...`), while **9 of 18** carry a directional signal surviving Holm (`skill.surviving_holm`). Section 2's accuracy-and-turnover framework and the returns framework of `directional_returns()` are different computations over different quantities and they agree - which is the check that matters; a disagreement would have meant a defect in one of them.

Still assumed, each raising the bar rather than lowering it: slippage beyond the quoted spread, the rollover surcharge, and the overnight swap on long gold. A finding of profitability would have to survive all three; a finding of unprofitability is safe from them by construction. One stale line to fix when next in ADR-010: its Consequences still say *"`net_of_costs()` exists but has no caller yet"*. It has one - `directional_returns()` in the same module, reached from `cli.py` in the `verdict` command ADR-013 built.

And the escape route, stated again because section 1's arithmetic is what keeps it open: the break-even scales linearly with `f`. These models already hold 2.6 to 5.7 bars, already take part of that discount, and still fall short. The room left is between 5.7 bars and twenty, not between one and twenty. What this machinery rules out is the hourly directional trade - which is what the original analysis claimed, and what this project set out to check.
