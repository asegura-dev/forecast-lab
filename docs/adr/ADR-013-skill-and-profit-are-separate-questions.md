# ADR-013 - Skill and profit are separate questions, and they get separate answers

- **Status:** Accepted - **built** (`research/significance.py`, the `verdict` command, and 19 tests).
- **Date:** 2026-08-30
- **Follows:** [ADR-011](ADR-011-power-before-verdict.md), which established that the design could see an effect worth having; and [ADR-012](ADR-012-turnover-and-dependence-are-measured.md), which left one positive result standing and one dead.
- **Context:** After ADR-012 the project had two live claims pointing opposite ways. **Profitability was dead**: nothing clears the accuracy that pays for its own turnover, the closest by 0.46 points. **Skill was alive**: pooled over 40,587 bars every model beats the constant predictor, the best by +0.97% at 3.91 standard errors, and the dependence measurement had just removed the last reason to distrust that standard error. Asserting either "there is a small real edge" or "there is nothing" without testing would have been the project's own accusation turned inward.

## Decision

### 1. The two questions are asked separately and answered separately

`forecast-lab verdict` runs four tests, reports them in order, and states a verdict that names both halves.

*Why:* they can disagree, and on this data they do. Collapsing them into one number - which is what "51.23% accuracy" or "-0.21% edge" does - throws away the more interesting half of the answer. A finding of *"a real signal too small to trade"* is worth more than either *"nothing here"* or *"we found an edge"*, and it is only available if the questions stay apart.

### 2. Skill is tested against independence, not against a coin flip

`pesaran_timmermann()` compares accuracy against `P·Q + (1-P)(1-Q)`, the accuracy the two marginals produce with no information passing between them.

*Why:* a predictor that always says UP on a series that rises 52% of the time scores 52%, beats a coin flip by two points, passes a naive binomial test, and knows **nothing**. This project has spent twelve ADRs insisting that a score without its baseline is not a result; testing accuracy against 0.5 would be that error committed one last time. The independence benchmark is what the marginals alone imply, and against it the constant predictor scores exactly zero - which a test pins.

Measured, the independence benchmark sits at **50.03%-50.19%** across the six configurations tabled below and **50.03%-50.33%** across all eighteen, not the 50.26% of the constant predictor. The two nulls are different and neither substitutes for the other.

### 3. Multiplicity is corrected everywhere, including on this project's own results

Holm for the eighteen Pesaran-Timmermann tests; Hansen's SPA and Romano-Wolf's StepM for the economic comparison, which handle it internally.

*Why:* this project made the multiplicity argument by hand against the original analysis - eighteen configurations, an expected maximum of +1.59% under the null, a best observed that did not survive Holm. Quoting its own smallest p-value out of eighteen would be a double standard visible to any reader who got that far.

*Why Holm rather than Bonferroni:* uniformly more powerful and no less valid, so the weaker correction would concede detections for nothing. A test pins a case where they disagree.

### 4. Profit is measured against holding the asset, never against zero

*Why:* gold rose through most of this sample. A strategy measured against zero collects that drift and reports it as skill - the same error as an accuracy without its baseline, moved into return space. The benchmark is always-long, which is the constant predictor expressed as a position.

*Why both position framings:* long-or-short is the natural reading of a directional prediction and pays a full round trip per flip. Long-or-flat is the friendlier framing - half the turnover, and a wrong call merely forgoes a move instead of taking it backwards. Both are available and `--long-only` runs the friendlier one, so a negative result cannot be blamed on the harsher.

## Consequences

**The verdict, in one line: there is a real directional edge and it is worth less than nothing.**

| Question | Test | Answer |
|---|---|---|
| Is there skill? | Pesaran-Timmermann + Holm | **Yes. 9 of 18** survive across 18 tests; the best is z = 4.61 |
| Does anything make money? | Net of the venue's spread | **No. 0 of 18**, under either position framing |
| Does anything beat holding gold? | Hansen SPA | **No.** p = 0.761 |
| Does the best survive being the best? | Romano-Wolf StepM, DSR | **No.** StepM rejects nothing; DSR = 0.0000 |

The gap is the whole finding. The best configuration returns **-86.9%** over the scored period while holding the asset returns **+100.9%**. Nine configurations carry a statistically robust directional signal of about one point, and one point of directional accuracy is worth less than the spread costs to collect it.

**This is a stronger result than "we found nothing", and it is also the one the data supports.** A reader who suspects the project of motivated reasoning should note that the same battery, on the same data, could have returned "no skill either" - and that the tests are shown a case where each effect is real, so a battery that only ever says "not significant" would fail its own suite.

**The two frameworks agree, which is the check that matters.** ADR-012 concluded from accuracy and turnover that no model pays for itself; this concludes from returns and a benchmark that none makes money. Those are different computations over different quantities, and a disagreement would have meant a defect in one of them.

**A defect this found in itself.** `arch` returns column *names* from `StepM.superior_models` and positional *indices* from `SPA.better_models`, given the same DataFrame. The first version passed both through `str()`, so a superior model at position 4 was reported as `'4'` - a valid-looking label belonging to no model. Nothing but a test asserting a known-superior model by name would have caught it: the count was right and only the identity was wrong.

**What a positive result would have needed, and this negative one does not.** Slippage beyond the quoted spread, the rollover surcharge, and the overnight swap are all unmodelled, and every one of them raises the bar. A finding of profitability would have to survive them first; a finding of unprofitability is safe from them by construction.

**The escape route that remains open, stated for the third time because it is real.** The break-even scales with turnover, these models hold 2.6 to 5.7 bars, and nothing here tests a strategy holding twenty. What this rules out is the hourly directional trade - which is what the original project claimed, and what this one set out to check.
