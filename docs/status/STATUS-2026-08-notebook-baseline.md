# STATUS 2026-08-23 - The baseline the original result had to beat, computed here

- **Question:** the README claims a published model lost to the most trivial rule available. That claim came out of a notebook. **Can this repository compute the numbers behind it, from data, with one command?**
- **Verdict:** **Yes for the corrected pipeline, and the claim survives contact with it.** Every rule available without a model sits below the accuracy at which trading would pay for itself, and the honest baseline beats two of the three alternatives to it. The notebook's own figures remain quoted rather than recomputed, and the reason is stated in section 4 rather than glossed.
- **Command:** `forecast-lab baseline --target XAUUSD --timeframe 1H --dir data/reference`
- **Machine-readable output:** [notebook-baseline.json](notebook-baseline.json)
- **Inputs:** `reference/XAUUSD_1H.csv`, hashed in [data-manifest.json](data-manifest.json). 23,181 hourly bars, 2022-01-02 to 2025-12-01.

## 1. What the labelling found

| | Bars | Share |
|---|---:|---:|
| Total | 23,181 | |
| Labelled at the stated horizon | 22,167 | 95.63% |
| Unlabelled - the horizon spans a gap | 1,013 | **4.37%** |
| Exact ties, held out of the direction | 36 | 0.16% |

The 1,013 are the rows where `close.shift(-1)` would have reached across two hours, or across fifty, while reporting the result as an hourly move. They are marked, not dropped: `seconds_ahead` records how far the next bar actually sits, and `next_change` records what price did over that distance.

**And what it did over that distance is the most interesting number in the dataset:**

| Subset | Decided bars | UP |
|---|---:|---:|
| Answer lands at the stated horizon | 22,131 | **50.87%** |
| Answer lands across a gap | 1,011 | **59.35%** |

Gold rose through the pause 600 times out of 1,011. That is an 8.5 point difference from the hourly rate, in the subpopulation a positional shift folds silently into the headline average - and no notebook in the original project separated the two. It is not a strategy yet: those are exactly the hours that pay overnight financing, and the cost model in Phase 2 decides whether anything survives it. It is recorded here because measuring it and setting it aside is a different act from never looking.

*A note on the two "UP rate" denominators, so they are not mistaken for a discrepancy.* 11,257 UP out of 22,167 labelled bars is 50.78%; out of 22,131 **decided** bars (labelled minus the 36 ties) it is 50.87%. This repository reports the second, because a tie has no direction to be right or wrong about and including it in the denominator of a directional rate is the same error the strictly-greater comparison makes.

## 2. The blocks

| Block | Index rows | Scored | Period | UP rate |
|---|---:|---:|---|---:|
| train | 16,225 | 15,484 | 2022-01-02 to 2024-09-27 | 50.40% |
| validation | 3,476 | 3,322 | 2024-09-27 to 2025-05-02 | **52.41%** |
| test | 3,478 | 3,323 | 2025-05-02 to 2025-12-01 | **51.46%** |

Purged at the boundaries: **2 bars**, one at each internal edge, which is exactly the label's reach at horizon 1. "Scored" is smaller than "index rows" because the unlabelled and tied bars in each block carry no direction to score.

The two right-hand columns are the finding of this section. **The class balance is not stable across the timeline.** Train rises 50.40% of the time; validation rises 52.41%; test rises 51.46%. A rule that read each block's own majority instead of train's would collect **2.01 points on validation** and **1.06 points on test** for free - and both are larger than the entire effect any model in this project is trying to detect. Any accuracy figure reported without saying which stretch of history it was measured on is partly a statement about that drift.

## 3. What the baselines scored

**Test block** (3,323 scored bars):

| Rule | Accuracy | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|
| majority-class (UP, from train) | 51.46% | 51.46% | **100.00%** | **0.00%** |
| persistence | 49.32% | 50.76% | 50.76% | 47.80% |
| random (train frequencies, seeded) | **51.70%** | 53.25% | 50.29% | 53.19% |

**Validation block** (3,322 scored bars):

| Rule | Accuracy | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|
| majority-class (UP, from train) | 52.41% | 52.41% | **100.00%** | **0.00%** |
| persistence | 49.22% | 51.55% | 51.52% | 46.68% |
| random (train frequencies, seeded) | 50.09% | 52.57% | 48.77% | 51.55% |

Three readings, in order of how much they matter.

**The 100.00% recall with 0.00% specificity is the signature to remember.** It is what a constant looks like when only accuracy is reported. The original analysis's selected models show the same pair, and it was read as a model that detects rises.

**On test, the seeded random rule beats the honest baseline** - 51.70% against 51.46%. Nothing has been discovered; that is one draw of a coin, and a different seed moves it by more than the gap. It is included because it is the plainest available demonstration that at this sample size the noise is the same size as everything being argued about. A result that would be celebrated if a model produced it was produced here by a random number generator.

**Persistence loses to both, on both blocks** - 49.32% and 49.22%, below chance. Hourly gold does not repeat its last direction; if anything it mildly alternates. Any momentum claim at this frequency starts from behind.

## 4. What this does and does not reproduce

**Reproduced from data by this repository:** the corrected pipeline's class balance, block boundaries, and the three baselines above. The alignment identity behind the original defect is pinned separately by a regression test - the forward fill fabricates 1,250 rows and adds exactly 1,250 ties ([ADR-003](../adr/ADR-003-target-anchored-alignment.md) sec. 1).

**Quoted, not reproduced:** the original notebook's own figures - 51.53% for its selected model against a 51.86% base rate on its 2,983-row test set. Those come from a run whose alignment this repository deliberately does not implement, so recomputing them would mean building the defect on purpose. What is checkable here is the shape of the mistake rather than its digits: the corrected pipeline shows the same structure - a base rate above 51%, a constant predictor achieving it with zero specificity, and no rule reaching the cost floor.

**Not measured here:** anything requiring a model, a cost model, or a validation design with more resolution than a single split. The break-even accuracy quoted below is carried from the planning analysis and gets its own module and STATUS log in Phase 2.

## 5. The floor, and where everything sits relative to it

Break-even accuracy against the most optimistic cost assumption - a one basis point round trip against a mean absolute hourly move of 13.04 basis points - is **51.92%**.

| | Accuracy |
|---|---:|
| Break-even, optimistic (1 bp round trip) | **51.92%** |
| Random (train frequencies), test | 51.70% |
| Majority class from train, test | 51.46% |
| Persistence, test | 49.32% |

Everything available without a model is below the line. That is not evidence that no edge exists, and this log does not claim it is - it is the neighbourhood a claimed edge has to be measured against, fixed before any model exists rather than after one has been chosen. The point of establishing it now is that it can no longer be moved.

And the resolution problem is visible in the same table: the gap between the top two rows is 0.22 points, while a 3,478-bar test block resolves about 0.85 points at 80% power. **This design cannot tell those two rows apart.** That is the argument for walk-forward validation in Phase 2, stated with the numbers rather than as a preference.

## 6. What changed in the repository because of this

- `research/labeling.py`, `research/splitting.py` and `research/baselines.py` exist, with 31 tests between them.
- The `baseline` command exists, with `--json`.
- The README's claims about 51.53% and 51.86% now carry the pointer to this log, which says which of them the repository computes and which it quotes.
- A unit bug was caught by a test written for a different purpose: `seconds_ahead` was computed by casting to `int64` and dividing by a hard-coded `1e9`, which is only correct when pandas is storing nanoseconds. It was storing microseconds, and the column came out a thousand times too small. The same assumption produced a wrong staleness figure during the alignment work. Both are now computed by subtracting timestamps and asking the result for seconds.
