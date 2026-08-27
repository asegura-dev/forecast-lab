# ADR-004 - What a bar's answer is, where the boundaries fall, and what a model must beat

- **Status:** Accepted - **built** (`research/labeling.py`, `research/splitting.py`, `research/baselines.py`, the `baseline` command, and the STATUS log that reproduces the figures).
- **Date:** 2026-08-23
- **Follows:** [ADR-003](ADR-003-target-anchored-alignment.md) - which fixes *which rows exist*. This one fixes what each row is asked to predict, how the rows are divided, and what number the prediction is judged against.
- **Context:** With the timeline settled, three decisions stand between the panel and a result, and all three look like one line of code each. The original analysis wrote those three lines - `close.shift(-1) > close`, a 70/15/15 slice, and a comparison against 50% - and each one hid a defect large enough to reverse its conclusion. They are decided here together, because they interact: the labelling policy changes the class balance, the class balance is what the baseline predicts, and the baseline is the number the whole project's verdict is measured from.

## Decision

### 1. The horizon is a timestamp, not a position

A bar's answer is the close at `t + horizon * interval`. If no bar exists at exactly that instant, the row is left unlabelled and marked, rather than answered by whatever bar happens to come next.

*Why:* `close.shift(-1)` assumes the next row is one interval later. On the target it frequently is not - the venue pauses daily, closes at weekends, and observes holidays. Measured on the reference series, 1,013 of 23,181 bars (**4.37%**) have no bar at the stated horizon: for those, `shift(-1)` reaches across two hours, or across fifty. Those rows are not a rounding error in a metric; they are a materially different prediction problem with several times the variance, folded into the same average without a word.

**The two distributions, which are reported together and never averaged:**

| Subset | Decided bars | UP |
|---|---:|---:|
| Answer lands at the stated horizon | 22,131 | **50.87%** |
| Answer lands across a gap | 1,011 | **59.35%** |

The second row is the more interesting number in this dataset, and no notebook in the original project computed it. **Gold rises through the pause about 59% of the time.** Whatever edge this data contains, the most visible candidate is not hourly direction - it is being long while the venue is shut. Two consequences follow, and both are inconvenient: that is precisely the position which pays overnight financing, so the cost model decides whether it survives; and it is precisely the set of rows this labelling policy declines to label, so it cannot be studied through the main metric. It gets its own treatment in Phase 2 rather than being smuggled into this one.

**Trade-off:** 4.37% of the sample carries no label, and that is real information given up. The alternative - answering those rows with a bar at the wrong distance - does not recover the information, it forges it. The rows stay in the frame with `seconds_ahead`, `next_change` and `label_valid`, so a later stage can model the regime deliberately instead of inheriting it silently.

### 2. An exact tie is FLAT, not DOWN

`>` is strictly greater, so a bar closing exactly where it opened is swept into DOWN. Every tie is named `FLAT` and excluded from the direction rather than assigned to one.

*Why:* this is pedantic on clean data - 36 genuine ties in 23,181 bars - and decisive on dirty data. Every row the forward fill fabricated was an exact tie ([ADR-003](ADR-003-target-anchored-alignment.md) sec. 1), so a strictly-greater comparison deposited the entire defect onto one class and flipped the majority. The class that a corrupted pipeline inflates should be visible as its own count, not absorbed into a direction where it reads as signal.

### 3. A threshold on the move may not see the move it judges

`rolling_dead_band()` computes its threshold from bars strictly before the one being judged - the `.shift(1)` is the load-bearing line - and the labeller refuses a band it was not handed.

*Why:* a threshold derived from the move it is thresholding selects the test set on its own answer. It is the same error as scaling before splitting, hidden inside a preprocessing step that reads as harmless. The band is optional and off by default; when it is used the reason is economic - a move smaller than the round trip is not an opportunity, and a model rewarded for predicting one is being trained on noise - so its size comes from the cost model rather than from the data's own dispersion.

### 4. The split is chronological, purged by `horizon` bars, and carries no embargo

Train, validation and test in time order, with `horizon` bars dropped at each internal boundary. Nothing is removed after test.

*Why the purge:* the last bar of train carries an answer that lives inside validation. Left in place, the training set literally contains the thing being predicted on the other side of the line. The correct width is exactly the label's reach - `horizon` bars, no more - and on the reference run that is **2 bars in total**, which is the honest size of this particular problem rather than a dramatic one.

*Why no embargo:* the standard companion to purging discards a stretch *after* the test block, because k-fold cross-validation places training data on both sides of it. A single chronological split has nothing after test, so an embargo would delete real observations to prevent a leak that cannot occur. It arrives with combinatorial cross-validation in Phase 2, or not at all.

An earlier draft of this reasoning justified purging as protection against rolling feature windows leaking across the boundary. That was wrong, and it is recorded as wrong rather than quietly dropped: a causal window looking backward from `t` uses only data at or before `t`, and no amount of it crosses a future boundary. What overlapping windows actually do is make neighbouring rows dependent, which shrinks the *effective* sample and therefore narrows every confidence interval computed downstream. That is the evaluation layer's problem, and it is answered with a stationary bootstrap, not with a purge.

**Trade-off, and it is the central one in this project:** a single 70/15/15 split leaves 3,478 out-of-sample bars. One standard error on an accuracy over that many bars is 0.85 points, and the **minimum detectable effect** at 80% power is 2.49 standard errors: **2.11 points**. The accuracy that makes an hourly strategy break even against a one basis point round trip is 51.92% - 1.92 points from chance - so **this design cannot resolve the very effect it exists to test**, let alone a realistic break-even. That is not a statement about gold; it is arithmetic about the sample size, available before any model is fitted. Walk-forward validation recovers the resolution by scoring roughly 11,590 bars. **The split is kept here because reproducing the original result requires reproducing the original design.** It is replaced in Phase 2, and the fact that a replacement is necessary is a finding about experimental design rather than about gold.

### 5. The majority-class baseline is fitted on train and applied blind

The rule predicts whichever direction was more common **in train**, and is scored on validation and test without refitting. The distance to each block's own majority is computed and printed as `oracle_gap`.

*Why:* this is the trap the original analysis fell into, and it is worth stating precisely, because it does not look like a mistake. Its selected model - LightGBM on PCA components - reported **51.53%** accuracy on its test set. Its own classification report printed the class support two lines below: 1,547 of 2,983 test bars rose, so predicting UP every single time would have scored **51.86%**. The reported winner lost to the most trivial rule available, by 0.33 points, and the pipeline contained no place where that comparison could occur. The report concluded from it that machine learning can beat chance on hourly gold.

A baseline that reads the block it is scored on is not a baseline, it is an oracle, and the gap is not small enough to wave away. On the reference run it is **2.01 points on validation** and **1.06 points on test** - both larger than the entire effect anyone is trying to detect. That gap measures which stretch of history was used as the test set. It says nothing whatever about any model.

### 6. Accuracy is never reported alone

Every rule reports precision, recall and specificity beside its accuracy.

*Why:* the original report's winning models show **100% recall** on UP. That is not a model detecting rises; it is a model that has learned to say UP, and one column makes it unmistakable - its specificity is **0.00%**. A single accuracy figure cannot distinguish a model from a constant, and this project's entire premise is that a constant was mistaken for a model.

### 7. Three baselines, not one

Majority class from train, persistence (this bar repeats the last one), and a seeded random draw from train's class frequencies.

*Why three:* each answers a different "well, obviously" objection. Majority class asks whether the model beat the class balance. Persistence asks whether it beat the cheapest possible use of the data, and is the specific rule any momentum claim has to clear. Random-from-train-frequencies establishes what the metric looks like when nothing is known, which is not 50% once the classes are unbalanced.

**The seed is fixed** (`DEFAULT_SEED = 20260101`) so that a random baseline is a property of the data rather than of the day it ran. The reference run makes the case on its own: on test the seeded random rule scores **51.70%**, *above* the honest majority-class rule at 51.46%. Nothing has been discovered - that is one draw of a coin, and a different seed moves it - but it is the plainest available demonstration that at this sample size the noise is the same size as everything being argued about.

## Consequences

**A number in this repository is now produced by the repository.** Before this ADR, the README asserted 51.53% and 51.86% - figures read out of a notebook, which the code here could not compute. `forecast-lab baseline` computes the corrected pipeline's equivalents from the reference exports, and [STATUS 2026-08-23](../status/STATUS-2026-08-notebook-baseline.md) records the output. A repository that argues from unverifiable numbers is not entitled to criticise one.

**The verdict has a floor to clear, and nothing is near it.** On the corrected test block, majority class scores 51.46%, persistence 49.32% and seeded random 51.70%; break-even against the most optimistic cost assumption is 51.92%. Every rule available without a model sits below the line where trading would pay for itself. That is not evidence that no edge exists - it is the neighbourhood any claimed edge must be measured against, established before a model exists rather than after one is chosen.

**Every analysis command now offers `--json`.** The dashboard is going to run these commands rather than reimplement them ([ADR-005](ADR-005-the-dashboard-runs-the-cli.md)), which only works if a result can leave the process without being scraped off a rendered table.

**What is deliberately not built here.** No walk-forward or combinatorial cross-validation, no economic baselines (buy and hold, always-long, always-flat, random at equal turnover), no cost model, no significance testing. Those are Phase 2 and Phase 3, and the split built here is the one being replaced. Building the replacement now would mean owning two split implementations while only one has a caller.

**The gapped bars are a debt, recorded openly.** 4.37% of the sample is measured, reported, and then set aside. It is the subpopulation with the largest visible directional skew in the whole dataset, and it stays unmodelled until a cost model exists that can say whether holding through a closed venue survives its financing.
