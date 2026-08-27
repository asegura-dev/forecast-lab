# ADR-007 - Fitting on train, selecting on validation, scoring test once

- **Status:** Accepted - **built** (`research/models/`, the `train` command, 17 tests, and a STATUS log that reproduces the original's conclusion and then dismantles it).
- **Date:** 2026-08-24
- **Follows:** [ADR-004](ADR-004-labels-splits-and-baselines.md), which fixes the blocks and the baselines this decision is measured against, and [ADR-006](ADR-006-features-and-stationarity.md), which fixes the matrix being fitted.
- **Context:** Everything before this ADR was preparation. This is where a model finally gets fitted, and where the original analysis made the three mistakes that turned an inconclusive experiment into a published claim. None of the three produces an error, a warning or a crash. All three make the final number larger.

## Decision

### 1. Every transform is fitted on the training block alone

Scaler and PCA live inside an `sklearn.Pipeline` that is fitted exactly once, on train.

*Why:* fitting a scaler on the full matrix before splitting is the commonest leak in tabular work and the hardest to see, because nothing breaks - the number that comes out is merely optimistic. The `Pipeline` makes the correct version the easy one: one `fit` call propagates to every step, and no step can be fitted twice by accident.

The scaler is included even for tree models that do not need it, whenever a PCA follows. PCA maximises variance, so a column measured in larger units would dominate the components purely by being larger.

### 2. Selection happens on validation. Test is scored once, afterwards

The best validation AUC picks the model. Every model's test score is computed and printed, but no test figure participates in the choice.

*Why:* the original did the opposite, in both notebooks and in two different ways. `Proyecto_ASM` ranks its results table by `Test_Accuracy`. `Proyecto_Final_Completo` - the one the technical report documents - selects with `results_df['Test_AUC'].idxmax()`. Choosing a model by the number that is supposed to be measuring it means the reported figure is not an out-of-sample result at all; it is the **maximum of thirty draws**, and the maximum of thirty draws is larger than the average of thirty draws even when every draw is noise.

**This is not an abstract objection. The size of it was measured on this data** ([STATUS 2026-08-24](../status/STATUS-2026-08-models.md)): selecting honestly on validation gives an edge of **-0.21%** on test; selecting by looking at test gives **+1.85%**. The gap between the two procedures is **2.06 points** - larger than any effect this project could plausibly detect, produced by nothing but the choice of where to look.

### 3. AUC comes from probabilities, never from hard labels

`predict_proba`, column 1.

*Why:* `Proyecto_Final_Completo` passes hard predictions to `roc_auc_score`. With binary input the ROC curve has a single interior point, so the statistic degenerates into a rescaled balanced accuracy - which is why its Naive Bayes reports an AUC of exactly **0.5000** and why nothing in that column carries the information it appears to. `Proyecto_ASM` used `predict_proba` correctly; the report documents the notebook that did not. A test pins the distinction by scoring a perfectly-ranked model whose probabilities all sit below the threshold: AUC 1.0, accuracy 0.5.

### 4. No score is printed without its baseline on the same line

Every row carries `edge` = accuracy - baseline, coloured, immediately after accuracy.

*Why:* an accuracy of 51.5% is not a result, it is half of one. The other half is that predicting UP every time scores 51.31% on the same block. The original had **both numbers in its own output, two rows apart**, and no place where their difference was computed. Putting them on one line is a two-character change to a table and it is the difference between a finding and a mistake.

The table also carries a literal `always-UP (from train)` row, so the comparison is visible even to someone skimming.

### 5. Which models exist is discovered by building them, not assumed

`availability()` constructs each estimator and records what happened. The command prints every model it could not run, and why.

*Why:* two of the six load a native DLL through `ctypes`, and an application control policy can refuse it while the Python package imports perfectly well - so checking for the module would report available and then crash. More importantly, a comparison that quietly omits two models is a comparison of a different experiment than the one the table claims to describe. Silence here would be the same class of error as everything else this ADR is about.

**Trade-off:** the fix that made both models available on this machine was not code but a version pin. LightGBM 4.7.0 and XGBoost 3.4.1 are refused; 4.6.0 and 3.0.5 load first try. The policy judges a binary's reputation, and a freshly published wheel has none. That is recorded in `pyproject.toml` and in the RUNBOOK, because a future `uv lock --upgrade` would silently remove the two most interesting models from the comparison.

## Consequences

**The original's conclusion is reproduced, and it dies in the same table that reproduces it.** The selected model scores 51.09% against a 51.31% baseline: **-0.21%**. The original reported 51.53% against 51.86%: **-0.33%**. Different data handling throughout, the same sign.

**And the selection turned out to be unstable, which is worth more than the reproduction.** Adding two realised-volatility columns moved the winner from LightGBM to XGBoost without changing any conclusion ([STATUS 2026-08-24](../status/STATUS-2026-08-models.md) sec. 5). Every feature added from here therefore counts as a trial in the multiplicity accounting, and its effect on the selection is measured before and after.

**The multiplicity problem is now measurable rather than argued.** Eighteen configurations are scored on test (six models x three representations). Under a true null, the maximum of eighteen draws is expected at **1.82 sigma**; the best observed is **+1.85%, or 2.13 sigma**. It is above the expectation but does not survive the Holm threshold for the most significant of eighteen tests (z >= 2.77, p = 0.0168 against 0.00278). And only **5 of 18** configurations have a positive edge at all, where a coin would give about 9. The full multiplicity battery - Romano-Wolf, Hansen SPA - arrives in Phase 3; this is the arithmetic that shows why it is needed.

**PCA on clean features still collapses 17 columns into 6.** [ADR-006](ADR-006-features-and-stationarity.md) sec. 2 offered price levels as "the plainest explanation" for the original's 63-into-6 collapse. That was too quick, and this run corrects it: with every price level already removed, PCA at 90% variance still retains only **6 components** from 17 features, and 7 at 95%. Technical indicators computed from one price series are intrinsically redundant, regardless of whether they carry the level. The levels made it worse; they were not the cause.

**One setting is deliberately changed from the original, and only one.** `RandomForestClassifier` runs with `n_jobs=1` where the original used `-1`. It is not a hyperparameter - `random_state` fixes every tree, so the forest is identical - but a parallel reduction sums 100 tree votes in whatever order the workers finish, and floating-point addition is not associative. Measured: probabilities move by up to 3.3e-16 between runs and the command's JSON is not byte-reproducible. Fixing it takes the command from about six seconds to twelve, and a test pins it with `array_equal` rather than `allclose` - approximate equality being what allowed the problem to go unnoticed.

**What is deliberately not built.** No hyperparameter search - the original's settings are kept unchanged, because tuning them would add trials to an accounting that is already the problem. No walk-forward, no cost model, no economic benchmarks, no significance battery. The 0.5 decision threshold is a placeholder and is labelled as one in the code: the threshold that matters comes from transaction costs.
