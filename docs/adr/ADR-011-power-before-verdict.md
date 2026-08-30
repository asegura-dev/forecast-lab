# ADR-011 - A negative result is only a finding if the design could have seen the effect

- **Status:** Accepted - **built** (`research/power.py`, `research/walkforward.py`, `research/models/validation.py`, the `validate` command, and 35 tests).
- **Date:** 2026-08-27
- **Follows:** [ADR-004](ADR-004-labels-splits-and-baselines.md), which built the single split and the constant predictor; [ADR-007](ADR-007-fitting-models-without-leaking.md), whose scores this re-measures across the whole history; and [ADR-010](ADR-010-costs-are-measured-not-assumed.md), which supplies the effect size that everything below is measured against.
- **Context:** This project has now said "no model beats its baseline" four times, on two datasets, across 36 configurations. Every one of those statements is worthless without the number this ADR computes. An experiment that finds nothing has two possible explanations - **there was nothing to find**, or **it could not have found it** - and only a power calculation separates them. The original project ran none. Neither, until now, did this one, which meant its correction was open to exactly the criticism it was making.

## Decision

### 1. The design's resolution is computed and printed beside every verdict

`research/power.py` turns an out-of-sample row count into a minimum detectable effect: `MDE = (z_alpha + z_beta) * SE`, with `SE = 0.5 / sqrt(n)`.

*Why 0.5:* `p(1-p)` is maximised at one half, so it is the widest and therefore the most conservative standard error. Every accuracy in this project sits within two points of a half, so the conservatism costs nothing.

*Why one-sided:* the question is "is this better than chance", not "is this different from chance". Using a two-sided test while asking a directional question demands a larger effect for the same power - quiet conservatism that would make a negative result harder to falsify, which is the opposite of what this ADR is for.

**And computing it reversed this project's own argument.** For a week the claim was that a single 70/15/15 split "cannot resolve the effect it exists to test": its MDE is 2.17 points against a break-even 1.92 points above chance. That was true of the **assumed** break-even. ADR-010 measured the real one at 53.49%, so the effect worth detecting is **3.49 points**, not 1.92:

| Design | Out-of-sample bars | MDE at 80% power | Power for +3.49 pp |
|---|---:|---:|---:|
| Reference, single split | 3,294 | 2.17% | 99.1% |
| Canonical, single split | 7,306 | 1.45% | 100.0% |
| **Canonical, walk-forward** | **40,587** | **0.62%** | **100.0%** |

Detecting a profitable edge needs **1,268 bars**. There are 40,587. The verdict therefore changes from *"we could not see"* to **"we looked with power to spare and there was nothing"** - which is a result rather than a shrug. It does not rule out an edge of half a point; it rules out one worth having, which is the only kind the question was ever about.

**Trade-off:** this makes the negative result *stronger*, and a reader is entitled to suspect a calculation that flatters its author's conclusion. The defence is that the same module, run a week earlier against the assumed cost, produced the opposite reading and it was published anyway. The input that changed is ADR-010's measurement, not this arithmetic.

### 2. Models are scored across walk-forward folds, not one block

`research/walkforward.py` cuts the index into expanding train/test pairs; `research/models/validation.py` fits and scores on them.

*Why:* a single split scores one block, and that block has a character. The reference test set is a rally where gold rose 30%; its share of rising bars differs from the training block's by more than the effect anyone is trying to detect. An accuracy measured there is partly a statement about which months it landed on. Five folds cover 40,587 bars instead of 7,306 and take the MDE from 1.45% to 0.62%.

*Why expanding rather than rolling:* an expanding window trains on all history to date, which is what someone deploying this would have. A fixed-length rolling window answers a different question - whether recent history predicts better than distant - and that is a hypothesis about regime change, not a validation design. Both are available; `--expanding` is the default because it matches the deployment.

*Why purged per fold and with no embargo:* a label at the last training bar reaches `horizon` bars into the test block, so `horizon` bars come off the end of every training window - once per boundary, five in total here, not once for the whole scheme. No embargo, for ADR-004's reason: nothing after a test block trains in the *same* fold. A **later** fold does train on it, which is correct - by then it is history - and is precisely what makes this walk-forward rather than k-fold on shuffled data.

### 3. Folds are pooled by row count, never averaged

*Why:* averaging the five fold accuracies and quoting a t-statistic over them is the obvious move and it is wrong. The folds **share training data** - fold 5 trains on everything fold 1 trained on - so they are not five independent experiments and their spread is not a sampling distribution. Pooling makes one accuracy over one row count, which is what the standard error above describes. The spread across folds is still reported, as a description of how much the answer moves between regimes, never as an interval.

### 4. The constant predictor is refitted inside every fold

*Why:* the baseline picks the majority class of **its own** training block, exactly as the model sees only its own training block. Carrying one global majority across all five folds would score the baseline with information the model was denied - the same leak this project exists to refuse, pointed the other way.

**And this is what produced the finding of this ADR.** Under walk-forward, **all six models turn a positive edge**, after being negative or negligible on a single split. It reads like the models improving. They do not:

| | model accuracy | baseline | edge |
|---|---:|---:|---:|
| Single split, HistGradientBoosting | 51.44% | 50.93% | +0.51% |
| Walk-forward, HistGradientBoosting | 51.23% | **50.26%** | **+0.97%** |

The accuracy **falls** by 0.21 points. The baseline falls by 0.67. The single split lands on one rising stretch where always-UP scores 50.93%; averaged over five stretches the majority class sits nearer a half, and the same model looks better against a weaker opponent. **The edge moved because the thing it is measured against moved.**

That is the identical error this repository was built to correct, arriving from a new direction - and it would have been invisible if the edge were reported without the two terms behind it. Both are carried on every row of `validate`, and a paragraph of the command's own output says so in plain language.

## Consequences

**The headline result is unchanged, and now it is falsifiable.** The best model pooled over 40,587 bars scores **51.23%** against a **53.49%** break-even, short by **2.26 points**. *(That threshold assumes a turnover none of these models has; [ADR-012](ADR-012-turnover-and-dependence-are-measured.md) measures it and the real shortfall is 0.46 points for the closest model. The conclusion holds; the margin does not.)* No model clears it; the closest of the six is off by a factor of 3.6 on the edge. Because the design resolves 0.62% and would detect a profitable edge essentially every time, *that* is a finding, not a failure to look.

**A nominally significant edge is reported and then discounted, in that order.** The best pooled edge of +0.97% is 3.91 standard errors, which survives a Holm correction across the six models (z >= 2.39). It would be dishonest to omit it. It is also not a strategy: it is a quarter of what costs demand, and section 4 shows it is mostly a statement about the baseline. The number is published with both facts attached rather than dropped for being inconvenient in either direction.

> **Resolved 2026-08-28 ([ADR-012](ADR-012-turnover-and-dependence-are-measured.md)).** The paragraph below was wrong. Measured with a stationary bootstrap, the inflation is **0.97** - the correctness series autocorrelates at -0.020 even though its features autocorrelate at +0.93 and +0.99, because a near-coin-flip outcome inherits almost none of the dependence of its inputs. `0.5/sqrt(n)` was already honest. The reasoning is kept below as published; only its verdict changed.

**The standard error is optimistic and the amount is unmeasured.** `SE = 0.5/sqrt(n)` assumes independent bars. Overlapping feature windows make neighbouring rows dependent, so the effective sample is smaller than 40,587 and every power figure here is generous by a factor nobody has yet computed. The correction is a stationary bootstrap and it belongs with the significance battery. Until it exists, **100% power must not be read literally**. The verdict does not rest on it: 0.97 against 3.49 points is arithmetic, and no variance correction moves a factor of 3.6.

~~**PCA representations are not scored across folds yet.**~~ **Discharged 2026-08-30** as `validate --pca`, off by default. Measured: **18 configurations, none clears its own break-even**, and every PCA row scores below its raw counterpart - so the reason for the default was right, and it is now a measurement rather than an expectation. The closest configuration is still Naive Bayes on raw features.

**`validate` is the first command whose output argues with itself.** It prints an edge, then a paragraph explaining why that edge is smaller than it looks. That is deliberate and it is the house style this project is arguing for: a number without the thing it is measured against is not a result. It also makes the command's output long, which is a real cost for a reader in a hurry, and `--json` exists for them.

**The gap benchmark is still owed.** ADR-010 left the 56-59% rise-through-a-pause finding as a measurement rather than a strategy, pending the overnight swap that is charged in exactly those hours. This ADR supplies the validation machinery that benchmark will be scored on and does not discharge the debt.
