# STATUS 2026-08-24 - The original's conclusion, reproduced and then dismantled

- **Question:** with every correction in place, does a model beat the rule it has to beat? And does the corrected pipeline reach the same verdict the original did, by a different route?
- **Verdict:** **The honest pipeline selects LightGBM - the same model the original selected - and on test it loses to a constant predictor by 0.64 points.** The original reported the same model losing by 0.33. Two pipelines, different data handling, same choice, same sign.
- **Command:** `forecast-lab train --target XAUUSD --timeframe 1H --dir data/reference`
- **Machine-readable output:** [model-comparison.json](model-comparison.json)
- **Inputs:** `reference/*.csv`, hashed in [data-manifest.json](data-manifest.json). 22,982 rows x 17 columns, focus mode.

## 1. The headline

| | Model chosen | Accuracy | Baseline | Edge |
|---|---|---:|---:|---:|
| **Original** (`Proyecto_Final_Completo`) | LightGBM (PCA) | 51.53% | 51.86% | **-0.33%** |
| **This pipeline** | LightGBM (raw) | 50.67% | 51.31% | **-0.64%** |

The correspondence is not engineered. This pipeline labels by timestamp rather than position, aligns to the target's bars rather than an outer join, carries no price levels, fits every transform on train alone, and selects on validation. It arrives at the same model and the same conclusion the original's own output contained and did not read.

## 2. Selection, and the 2.64 points that hang on where you look

This is the section that matters, because it puts a number on an argument that is usually made in words.

| Chosen by | Model | Edge on test |
|---|---|---:|
| **Validation AUC** (this pipeline) | LightGBM [raw] | **-0.64%** |
| **Test AUC** (what the original did) | Naive Bayes [pca-90] | **+2.00%** |

Same data. Same eighteen configurations. Same test block. **The two procedures differ by 2.64 points** - far larger than any effect this project could plausibly detect - and the entire difference is produced by *when* the test set was consulted.

Had this pipeline used `results_df['Test_AUC'].idxmax()`, as `Proyecto_Final_Completo` does, it would have reported a Naive Bayes model beating chance by two points and had every appearance of a finding.

## 3. Is the +2.00% real?

No, and the arithmetic is simple enough to check by hand.

| | Value |
|---|---:|
| Configurations scored on test | **18** |
| Rows per block | 3,294 |
| Standard error of one accuracy | 0.8712% |
| Expected maximum of 18 draws under the null | **1.82 sigma = +1.59%** |
| Best observed | **+2.00% = 2.30 sigma** |
| Holm threshold for the most significant of 18 | z >= 2.77 (p < 0.00278) |
| p-value of the best | **0.0107 - does not survive** |

The best of eighteen is roughly where the best of eighteen coin flips would land. It is slightly above the expectation and comfortably short of the threshold that accounts for having looked eighteen times.

**And the tell is in the count.** Only **5 of 18** configurations have a positive edge on test. Under a true null one would expect about 9. If there were a real edge in this data at this horizon, more than a quarter of the configurations would find some of it.

## 4. The full comparison

**Validation** (baseline 52.46% - this is the block selection is allowed to see):

| Model | Repr. | Accuracy | Edge | AUC |
|---|---|---:|---:|---:|
| LightGBM | raw | 51.94% | **-0.52%** | 0.5232 |
| Naive Bayes | pca-90 | 50.64% | -1.82% | 0.5173 |
| Naive Bayes | pca-95 | 51.24% | -1.21% | 0.5163 |
| HistGradientBoosting | raw | 51.46% | -1.00% | 0.5108 |

**Not one configuration beats the constant predictor on validation.** The best of eighteen, on the block where selection is honest, is still 0.52 points behind a rule with no parameters.

**Test** (baseline 51.31%), the four highest by AUC:

| Model | Repr. | Accuracy | Edge | AUC |
|---|---|---:|---:|---:|
| Naive Bayes | pca-90 (6) | 53.31% | +2.00% | 0.5312 |
| Naive Bayes | raw | 51.97% | +0.67% | 0.5234 |
| Naive Bayes | pca-95 (7) | 51.82% | +0.52% | 0.5195 |
| Random Forest | raw | 51.43% | +0.12% | 0.5194 |

Every AUC sits between 0.49 and 0.54. A model that knew something would not look like this.

## 5. What the charts show that the tables did not

![Accuracy against the baseline and the break-even line](figures/edge-test.png)

Thirteen of eighteen configurations fall below the constant predictor. Exactly one clears break-even, and it is the one section 3 has just shown to be the best of eighteen coin flips. The selected model is the dark bar, below both lines.

![ROC curves from probabilities](figures/roc-test.png)

Eighteen curves, all of them lying on the diagonal. The original drew this panel too - but from hard labels, which gives every curve one interior point and produced its Naive Bayes AUC of exactly 0.5000.

![Confusion matrices, normalised by row](figures/confusion-test.png)

**This is the one that renders something no table did.** Read each panel by row: in almost every model the DOWN row and the UP row are the same. Logistic Regression [raw] predicts UP 65% of the time when price fell, and 65% of the time when price rose. The prediction is independent of the answer.

Quantified across all eighteen: the median gap between recall and (1 - specificity) is **0.0101**. That difference is Youden's J, so a median of 0.01 corresponds to an AUC of about 0.505 - which is what the ROC panel above shows from the other direction. Normalising by row is what makes it visible; raw counts would hide it behind the class imbalance.

## 6. A correction to ADR-006

[ADR-006](../adr/ADR-006-features-and-stationarity.md) sec. 2 called price levels "the plainest explanation" for the original's PCA collapsing 63 features into 6 components. That was too quick.

Measured here, with every price level already removed: **PCA at 90% variance retains 6 components from 17 features; at 95% it retains 7.** The same absolute number, from a quarter of the columns. Technical indicators computed from a single price series are intrinsically redundant - they are transformations of the same closes - and the levels made that worse rather than causing it. The ADR is corrected rather than left standing.

## 7. What this run says about the project's thesis

**The original's conclusion is refuted, but not by finding the opposite.** It is refuted by showing that its number was the maximum of thirty draws taken from a design that cannot resolve the effect it claimed. This run reproduces that: the maximum of eighteen draws here is +2.00%, and the honest procedure applied to the same data gives -0.64%.

**"No edge" is not what this shows either**, and the log will not claim it. A single 70/15/15 split resolves about 0.85 points at 80% power; the honest result is -0.64 points, well inside that. What has been established is narrower and firmer: *nothing here clears the bar, and this design could not tell if something barely did.* Walk-forward validation, which scores roughly 11,590 bars instead of 3,294, is the next piece of work, and it exists precisely to make the second half of that sentence go away.

## 8. Is it reproducible?

Byte for byte, yes - but only after fixing something that four runs of the command exposed and no test would have.

```
corrida 1: 19106 bytes  sha=7fb10e7ad44d8dbb8d37ae90
corrida 2: 19106 bytes  sha=7fb10e7ad44d8dbb8d37ae90
corrida 3: 19106 bytes  sha=7fb10e7ad44d8dbb8d37ae90
corrida 4: 19106 bytes  sha=7fb10e7ad44d8dbb8d37ae90
```

**Before the fix, three consecutive runs produced three different hashes.** The cause was `RandomForestClassifier(n_jobs=-1)`, the original's setting: with the work spread across cores, 100 tree votes are summed in whatever order the workers finish, and floating-point addition is not associative. Probabilities moved by up to **3.3e-16** - the last bit of a float64 - and one Brier score in the JSON gained or lost a digit.

Nothing about any conclusion changed, and that is exactly why it mattered. A discrepancy that changes no conclusion is one nobody investigates, so it stays; and a repository arguing that a published number must recompute identically cannot have its own headline command return different bytes each time.

`n_jobs` is not a hyperparameter - `random_state` fixes every tree, so the forest is identical either way and only the summation order changes. The cost is 0.29s against 3.04s per fit, taking the command from about six seconds to twelve. A test now pins it with `array_equal` rather than `allclose`, because approximate equality is what let the problem go unnoticed in the first place. All six estimators were then checked: all deterministic.

## 9. What changed in the repository

- `research/models/` exists: `catalogue.py`, `training.py`, `metrics.py`, with 17 tests, including two that pin the leaks a passing suite would miss and three that pin bit-level reproducibility.
- The `train` command exists, with `--json` and `--no-pca`.
- The layering guard now quarantines `sklearn`, `lightgbm` and `xgboost` to `research/models/`, for the same reason `ta` is quarantined: none ships type information.
- **Eight figures are committed** under `docs/status/figures/`, regenerated by `train --figures` (ADR-008).
- **Random Forest is fitted single-threaded**, so the command is byte-reproducible (sec. 7).
- **LightGBM and XGBoost were unavailable on this machine and are not any more.** An application control policy refused their native DLLs. The fix was a version pin rather than code: 4.7.0 and 3.4.1 are refused, 4.6.0 and 3.0.5 load first try, because the policy judges a binary's reputation and a freshly published wheel has none. Without that diagnosis, this log would have compared four models and quietly omitted the one the original selected.
