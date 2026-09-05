# Chapter 3 - Labels, splits and models

This chapter answers three questions that each look like one line of code and are not: what a bar is asked to predict, where the boundaries between train, validation and test fall, and what gets fitted between them. It covers `research/labeling.py`, `research/splitting.py`, `research/baselines.py` and the `research/models/` package, the two leaks the design refuses, and the price that was measured for the second of them. Every figure quoted below names the committed file it came from - `docs/status/notebook-baseline.json`, `docs/status/model-comparison.json`, or `docs/status/model-comparison-canonical.json` - and the last section lists the places where this repository's own prose disagrees with those files.

## 1. The label

A direction label is the sign of `close(t + horizon) - close(t)`. Everything interesting is in what "`t + horizon`" means.

`label_direction()` in `src/forecast_lab/research/labeling.py` takes a close series, a `Timeframe`, a `horizon` in bars and an optional `dead_band`, and returns a frame plus a `LabelReport`. It never shifts by position. It builds `wanted = index + Timedelta(seconds=timeframe.seconds * horizon)`, locates each wanted instant with `index.searchsorted(wanted, side="left")`, and keeps two separate booleans: `found` (a later bar exists at all) and `landed` (`index[position] == wanted`, the bar sits exactly where the horizon asks). Only `landed` rows get a label; the rest are set to `NaN` and flagged by the `label_valid` column.

This matters because gold's venue pauses. `docs/status/notebook-baseline.json` records the reference run over **23,181 rows**: **22,167** labelled, **1,013 gapped**. The two counts leave one row unaccounted for, and that is the last bar, for which no later bar exists at all - `found` is false, so it is neither labelled nor gapped. `tests/unit/test_labeling.py::test_the_last_bar_has_no_answer` pins it.

The gapped rows are measured, not discarded. `label_direction()` also computes `next_change` - the move to whatever bar actually comes next, at whatever distance, which is exactly what `close.shift(-1)` would have produced - and `_report()` splits it into `gapped_up` and `gapped_down`. In the same file: **600 up, 411 down**, a `gapped_up_rate` of **59.35%** against an `up_rate` of **50.87%** on the rows that landed. (600 + 411 = 1,011, so two gapped bars closed exactly flat across the pause.) Those rows are a different prediction problem with a visibly different class balance, and answering them with a bar at the wrong distance would not recover information, it would forge it. `seconds_ahead` is computed as `(index[safe] - index).total_seconds()` rather than by dividing an int64 by a hard-coded `1e9`, because pandas stores whatever time unit the source implied - the docstring records that the shortcut once produced a staleness figure a thousand times too small.

**A tie is FLAT.** `Direction` is an `IntEnum` with `DOWN = 0`, `UP = 1`, `FLAT = 2`. The label array is initialised to `FLAT` and only strict comparisons overwrite it: `np.where(change > 0, UP, ...)` then `np.where(change < 0, DOWN, ...)`. A bar closing exactly where it opened therefore stays FLAT and is excluded from the direction rather than assigned to one. On clean data this is pedantic - `notebook-baseline.json` reports **36** flat bars in 23,181. It is decisive on dirty data: ADR-004 sec. 2 records that every row a forward fill fabricates is an exact tie, so a strictly-greater comparison deposits the whole alignment defect onto DOWN, where it reads as signal instead of as a count.

The optional dead band is a per-row threshold on `|change|` below which the move is called FLAT. `rolling_dead_band()` computes it as `close.diff().abs().rolling(window).mean().shift(1) * multiple`; the `.shift(1)` is load-bearing, because a threshold that can see the move it is judging selects the test set on its own answer. It is off by default, and the `train` command in `src/forecast_lab/interfaces/cli.py` calls `label_direction(close, interval, horizon=horizon)` with no band - so every number in the committed sidecars is a no-dead-band run.

## 2. The split

`temporal_split(index, train=0.70, validation=0.15, horizon=1)` in `src/forecast_lab/research/splitting.py` cuts the index in time order:

```
train_end      = int(n * train)
validation_end = int(n * (train + validation))
train      = index[: train_end - horizon]
validation = index[train_end : validation_end - horizon]
test       = index[validation_end :]
```

The test share is never computed - it is whatever the other two leave - so blocks plus `purged` always sum to `n` and a rounding error cannot silently drop rows (`TemporalSplit.rows`). The function refuses an unsorted index, fractions that leave no room for test, and any index shorter than `3 * (horizon + 1)`; it also refuses a split that produced an empty block.

Never shuffled: a shuffled time series lets a model train on Thursday to predict Wednesday. The module states this and then spends its docstring on the part that is actually easy to get wrong.

**Purging.** The last bar of train carries an answer that lives inside validation, so leaving it in puts the thing being predicted on both sides of the line. The purge width is exactly the label's reach - `horizon` bars, no more - dropped at each *internal* boundary. On the reference run that is **2 bars in total** (`notebook-baseline.json`, `"purged": 2`), and the effect is visible in the block boundaries of that same file: train ends `2024-09-27T09:00`, validation starts `2024-09-27T11:00`; validation ends `2025-05-02T01:00`, test starts `2025-05-02T03:00`.

**The embargo is zero, deliberately.** An embargo discards a stretch *after* the test block, and it exists because k-fold cross-validation places training data on both sides of a fold. A single chronological split has nothing after test, so an embargo would delete real observations to prevent a leak that cannot occur. ADR-004 sec. 4 also records a correction worth keeping: an earlier draft justified purging as protection against rolling feature windows crossing the boundary, and that was wrong - a causal window looking backward from `t` never reaches forward. What overlapping windows actually do is make neighbouring rows dependent, which shrinks the effective sample; that is the evaluation layer's problem, and `docs/status/STATUS-2026-08-turnover.md` reports it measured at **0.97x**, i.e. essentially absent here.

Block sizes on the reference run (`notebook-baseline.json`): train **16,225**, validation **3,476**, test **3,478**. Note that a block's row count is not its scored count - the baselines score only decided labels, so the 3,478-row test block yields **3,323** scored bars.

## 3. The three baselines

`evaluate_baselines(train_labels, block_labels, block, seed=DEFAULT_SEED)` in `src/forecast_lab/research/baselines.py` fits on train and scores one block. `_decided()` strips everything that is not UP or DOWN first, because a FLAT bar has no direction to be right or wrong about.

- **Majority class**, from train. `majority = UP if train_up_rate > 0.5 else DOWN`, applied blind. Not the scored block's own majority - that is an oracle, and it flatters every model that inherits the same drift. `tests/unit/test_baselines.py::test_the_majority_comes_from_train_not_from_the_block_scored` pins the distinction.
- **Persistence**: this bar repeats the previous one. `_persistence()` shifts the decided sequence by one and gives the first bar `DOWN`, which is arbitrary and stated rather than hidden. Read the code carefully: the shift is over the *decided* subsequence, so "the previous bar" skips FLAT and unlabelled rows. This is the specific rule any momentum claim has to clear.
- **Random**, seeded: `rng.choice([DOWN, UP], p=[1 - train_up_rate, train_up_rate])` with `DEFAULT_SEED = 20260101`, so the number is a fact about the data rather than about the day it ran.

`Score` carries four confusion counts and derives accuracy, precision (`TP/(TP+FP)`), recall (`TP/(TP+FN)`) and specificity (`TN/(TN+FP)`). Accuracy alone cannot tell a model from a constant - always-UP has perfect recall and **zero** specificity - which is the whole reason those three travel together.

`BaselineReport.oracle_gap` is `abs(up_rate - train_up_rate)`: how much a rule gains purely by knowing the scored block's own class balance. On the reference run (`notebook-baseline.json`) it is **2.008 points on validation** and **1.059 points on test** - both larger than the effect anyone is trying to detect. That number says which stretch of history was used as test, and nothing at all about any model.

The three rules on the reference test block, from `notebook-baseline.json` (3,323 scored bars):

| Rule | Accuracy | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|
| majority-class (UP, from train) | **51.46%** | 51.46% | 100.00% | **0.00%** |
| persistence | 49.32% | 50.76% | 50.76% | 47.80% |
| random (train frequencies) | **51.70%** | 53.25% | 50.29% | 53.19% |

A seeded coin flip beats the honest majority rule. Nothing has been discovered - a different seed moves it - and that is the point: at this sample size the noise is the size of everything being argued about.

## 4. The estimators

Six specs in `CATALOGUE`, in `src/forecast_lab/research/models/catalogue.py`, in fixed order because a table that reorders itself between runs cannot be compared with a previous one. `RANDOM_STATE = 42` everywhere it is accepted. Each `ModelSpec` carries a deferred `build` callable, so importing the module never imports an estimator that might be blocked.

| Model | Constructor and hyperparameters | `needs_scaling` | How it works, honestly |
|---|---|:--:|---|
| Logistic Regression | `LogisticRegression(max_iter=1000, random_state=42)` | yes | A linear combination of the features squashed through a logistic function, fitted by maximising likelihood. It can only draw a hyperplane, so it finds a monotone tilt or nothing. `max_iter=1000` raises scikit-learn's default of 100 so the solver converges rather than warns. |
| Naive Bayes | `GaussianNB()` | yes | Assumes every feature is Gaussian and conditionally independent given the class, then applies Bayes' rule. The independence assumption is flatly false on indicators computed from one price series, which makes its probabilities badly calibrated even where its ranking is not - visible as the worst Brier scores in the comparison. No hyperparameters. |
| Random Forest | `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)` | no | 100 decision trees, each on a bootstrap sample and a random feature subset, voting. Depth 10 is the original's cap; `Proyecto_Final_Completo` left it unconstrained, which is why its forest reports a training accuracy of exactly 1.0000. |
| XGBoost | `XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss")` | no | Gradient boosting: trees fitted in sequence, each one on the previous ensemble's residual, shrunk by the learning rate. `eval_metric` is set explicitly to pin a version-dependent default. |
| LightGBM | `LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)` | no | The same idea with histogram-binned features and leaf-wise growth - faster, and more prone to overfit at equal depth. This is the estimator the original analysis selected, at 51.53% accuracy. |
| HistGradientBoosting | `HistGradientBoostingClassifier(max_depth=5, random_state=42)` | no | scikit-learn's own histogram booster. Marked `added_here=True`: it is *not* a stand-in for `Proyecto_Final_Completo`'s `GradientBoostingClassifier`, which this catalogue does not carry. It exists because it loads no native DLL and therefore runs where the other two boosters may not. |

**Why the forest is single-threaded.** `n_jobs=1` is the one setting deliberately changed from the original's `-1`, and it is not a hyperparameter: `random_state` fixes every tree, so the forest is identical either way. What changes is the order in which 100 tree votes are summed, and floating-point addition is not associative. Measured, per the module docstring and ADR-007: probabilities move by up to **3.3e-16** between runs with `-1`, and the command's JSON is not byte-reproducible; with `1` it is exact. The cost is **0.29s against 3.04s per fit**, taking the whole command from about six seconds to twelve. `tests/unit/test_models.py::test_fitting_twice_gives_bit_identical_probabilities` asserts it with `array_equal` rather than `allclose`, on the grounds that approximate equality is what let the defect go unnoticed.

**Why availability is discovered.** `availability()` *builds* each estimator and records what happened, catching `ImportError` (not installed) and `OSError` (what a blocked `ctypes.CDLL` raises) and nothing broader, so a genuine bug in a constructor still surfaces as a bug. Checking for the module would not do: LightGBM and XGBoost import fine while the shared library they wrap is refused by Windows Smart App Control, which judges a binary's reputation - and a freshly published wheel has none. `_reason()` matches Windows error **4551** numerically, because the message text is localised. The `train` command prints every model it could not run; a comparison that quietly omits two models describes a different experiment from the one its table claims. Both committed sidecars report `"models_unavailable": []`, so all six built on the machines that produced them. ADR-007 records the actual fix: version pins (LightGBM 4.6.0, XGBoost 3.0.5), because a future `uv lock --upgrade` would silently remove the two most interesting models from the comparison.

## 5. The pipeline

`build_pipeline(spec, variance)` in `src/forecast_lab/research/models/training.py` assembles an `sklearn.Pipeline`: a `StandardScaler` when `spec.needs_scaling or variance is not None`, then `PCA(n_components=variance, random_state=42)` when a variance is given, then the estimator. `PCA_VARIANCE = (0.95, 0.90)`, so each model is fitted in three representations - `raw`, `pca-95`, `pca-90` - and six models give **18 configurations**.

One `fit` call on train propagates to every step, and no step can be fitted twice by accident. That is the entire defence against the commonest leak in tabular work: a scaler fitted on the whole matrix leaks the test block's mean and variance backwards, nothing crashes, and the number that comes out is merely optimistic. Two tests pin it - `test_the_scaler_is_fitted_on_train_only` and `test_pca_components_come_from_train_only`.

**The scaler is included even for trees, whenever PCA is requested.** Trees are invariant to monotone rescaling and do not need it; PCA does. PCA maximises variance, so a column measured in larger units would dominate the components purely by being larger. On 19 feature columns the canonical run retains **8 components at 95%** variance and **6 at 90%** (`model-comparison-canonical.json`).

`fit_and_predict()` prepares each block through `_usable()`, which keeps only rows carrying both a label and a complete feature vector, per block rather than globally, so a block's score is honest about its own denominator. Two different reasons a row is dropped, both meaningful: a missing label means the horizon spanned a gap or the move was an exact tie; a missing feature means an auxiliary symbol had not started trading yet. It refuses a missing train block, an empty one, and one with a single class; it asserts that every block's columns equal train's; and `_predict_up()` takes `predict_proba(...)[:, 1]` - column 0 would give a perfectly inverted model that still scores near 50%, which looks like noise rather than like a bug. The LightGBM feature-name warning is suppressed by category *and* message, narrowly, so a different validation warning still gets through.

FLAT never reaches the estimator: `cli.py` masks it out (`labels["label"].where(labels["label"] != FLAT)`) before splitting, so `_usable()` only ever sees 0 and 1. `hard()` thresholds probabilities at 0.5, and its docstring labels that a placeholder - the threshold that matters comes from transaction costs.

## 6. The metrics

`score_model()` in `src/forecast_lab/research/models/metrics.py` builds four confusion counts from `predicted = (p >= threshold)` and `y == 1`, then:

| Metric | Formula | Implementation note |
|---|---|---|
| accuracy | `mean(predicted == y)` | The number that cannot distinguish a model from a constant. |
| precision | `TP / (TP + FP)` | Of the bars called UP, how many rose. |
| recall | `TP / (TP + FN)` | Of the bars that rose, how many were called. |
| specificity | `TN / (TN + FP)` | Of the bars that fell, how many were called. The column that exposes always-UP. |
| negative predictive value | `TN / (TN + FN)` | Precision's mirror on DOWN. Carried because the original's `classification_report` printed it. |
| F1 | `2PR / (P + R)` via `_harmonic()`; `0.0` when `P + R == 0` | Harmonic mean, so one good half cannot rescue it. |
| AUC | `roc_auc_score(y, p)` on **probabilities**; `0.5` when the block has one class | `_auc()` abstains rather than crashing the report it belongs to. |
| Brier | `mean((p - y) ** 2)` | Mean squared error of the probability itself, so confident-and-wrong is penalised where accuracy would not notice. |
| edge | `accuracy - baseline_accuracy` | A `@property` on `ModelScore`. The first number anyone should read. |
| predicted-UP rate | `mean(predicted == 1)` | Where "a constant with extra steps" becomes obvious. |

All four ratios go through `_ratio()`, which returns `0.0` on a zero denominator - an abstention, not a value.

**What F1 is worth here: nothing.** A predictor that always says UP has recall exactly 1 and precision exactly the class balance `p`, so its F1 is `2p / (1 + p)` - the identity `_score_table()` in `cli.py` prints on its `always-UP (from train)` row, and that `tests/unit/test_models.py::test_a_constant_up_predictor_scores_the_f1_the_original_published` pins. On the canonical test block `p = 0.50931` (`model-comparison-canonical.json`), so the constant scores **F1 = 0.6749**. The committed sidecars carry no `f1` field - `_train_payload()` emits one today, but both files predate it - so F1 has to be recomputed from the `precision` and `recall` they do carry, with the formula above. Doing that: the best of the eighteen configurations on canonical test is Logistic Regression [pca-95] at **F1 = 0.5911**, and **0 of 18 reach 0.6749**. On the reference run (`model-comparison.json`) the constant scores **0.6782** and the best model reaches **0.6208**; again **0 of 18**. The same holds on validation in both files. The metric the original published as a headline is one that every model fitted here loses to, and it loses because the parameter-free rule maximising it is exactly the degenerate one.

## 7. The two leaks this design refuses, and what the second is worth

**Transforms fitted on the whole matrix.** Answered structurally by the `Pipeline` in `build_pipeline()`, as above. This one has no measured cost in this repository, because it was never committed - the correct version was the easy one to write.

**Selection on the test block.** The original ranked its results by `Test_Accuracy` in one notebook and picked the winner with `results_df['Test_AUC'].idxmax()` in the other. Here `cli.py` chooses `max(validation, key=lambda s: s.auc)` and only then looks up that key's test row; every model's test score is computed and printed, and none of them participates in the choice.

The cost was measured on both datasets, and it is in the sidecars:

| | reference (`model-comparison.json`) | canonical (`model-comparison-canonical.json`) |
|---|---|---|
| Selected on validation AUC | XGBoost [raw] | Random Forest [raw] |
| Its edge on test | **-0.21%** | **+0.15%** |
| Best edge on test (what selecting on test gives) | **+1.85%** (Naive Bayes [pca-90]) | **+0.55%** (Logistic Regression [pca-95]) |
| Difference between the two procedures | **2.06 points** | **0.40 points** |
| Configurations with a positive test edge | 5 of 18 | 12 of 18 |

Two points, honestly. First, 2.06 points is larger than any effect this design could resolve - ADR-004 sec. 4 puts the reference split's minimum detectable effect at **2.11 points** at 80% power - and the entire difference is produced by *when* the test set was consulted, not by any change to the data or the models. Second, the same comparison on twice the data gives 0.40 points, so 2.06 is itself a noisy measurement of a real mechanism rather than a constant. The selected model's edge changes sign between the two datasets (-0.21% against +0.15%), which `docs/status/STATUS-2026-08-models.md` calls the cleanest demonstration the project has produced that the sign of a result this size carries no information.

## 8. Where the prose disagrees with the files

Read these before trusting a sentence over a number.

- **The tie count.** `labeling.py`'s module docstring says *"38 genuine ties out of 23,180"*. ADR-004 sec. 2 says 36 in 23,181, and `notebook-baseline.json` reports `"flat": 36` over `"rows": 23181`. Trust the JSON; the docstring is stale by two.
- **The gap breakdown.** The same docstring says *"780 two-hour gaps ... 183 weekend gaps of about fifty hours, and a handful longer still"*. No committed sidecar carries that breakdown - `notebook-baseline.json` reports only the total, 1,013. The 4.37% it quotes does check out (1,013 / 23,181).
- **Two different blocks, both called "always-UP".** `metrics.py`'s docstring quotes 51.46% and ADR-007 sec. 4 quotes 51.31%. Both are right and they are not the same block: the `baseline` command scores the label index (test 3,478 rows, 3,323 decided, `notebook-baseline.json`), while `train` scores the feature matrix, which is 199 rows shorter for the 200-period moving average's warm-up (22,982 rows against 23,181; test 3,448 rows, 3,294 scored, `model-comparison.json`).
- **`f1` and `negative_predictive_value` are computed but absent from the committed sidecars.** `ModelScore` carries both and `_train_payload()` serialises both today; neither field appears in either committed JSON. Every F1 in this chapter was recomputed from the `precision` and `recall` that are there.
