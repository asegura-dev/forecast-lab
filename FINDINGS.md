# Findings

**The answer, in one place.** Everything below is produced by `forecast-lab verdict` and
`forecast-lab validate` from data anyone can download; the reasoning behind each decision
is in the [ADRs](docs/adr/) and the measurements in the [STATUS logs](docs/status/). This
document exists so a reader does not have to assemble the answer from thirteen of the one
and eight of the other.

---

## The verdict

**There is a real directional edge in hourly gold, and it is worth less than nothing.**

Nine of eighteen model configurations beat statistical independence between prediction and
outcome, after correcting for having tried eighteen. The best does so at **z = 4.61**. That
signal is not an artefact of searching, and it is not what the original analysis claimed to
find - it is smaller, and it is real.

**None of the eighteen makes money.** Traded at the venue's own quoted spread, they turn
buy-and-hold's **+69.8%** into **-196.7%** over the same 40,587 bars. Hansen's test puts the
probability that any of them beat holding gold at **p = 0.763**.

About one point of directional accuracy costs more to collect than it is worth: the
closest configuration clears chance by 0.77 points and needs 1.23 to pay for its own
trading. That is what an efficient market with frictions is supposed to look like.

---

## The two questions, kept apart

They had to be separated, because they disagree.

| | Test | Answer |
|---|---|---|
| **Is there skill?** | Pesaran-Timmermann against independence, corrected with Holm | **Yes.** 16 of 18 significant at 5%; **9 survive** the correction |
| **Does anything make money?** | Strategy returns net of the measured spread | **No.** **0 of 18**, under either position framing |
| **Does anything beat holding gold?** | Hansen SPA | **No.** p = 0.763 |
| **Does the best survive being the best?** | Romano-Wolf StepM, Deflated Sharpe | **No.** StepM rejects nothing; DSR = 0.0000 |

Collapsing these into a single number - which "51.23% accuracy" does - throws away the more
interesting half. *"There is no signal"* would have been wrong. *"We found an edge"* would
have been true and dangerously incomplete.

### The numbers behind each

**Skill.** The best configuration scores **51.23%** where independence between its own
predictions and the outcome would give **50.09%**. The null is independence, not a coin
flip: a predictor that always says UP on a series rising 52% of the time scores 52% and
knows nothing, and against the independence benchmark it scores exactly zero.

**Profit.** Long-or-short at the median spread of 1.86 bps, the least-bad configuration
loses **0.485 bps per bar**. Under the friendlier long-or-flat framing - half the turnover,
and a wrong call merely forgoes a move instead of taking it backwards - it loses 0.156 bps
per bar and the count is still 0 of 18. The result cannot be blamed on the harsher framing.

**The best strategy's Sharpe is -0.021 per bar** (about -1.99 annualised), with skew -0.75
and kurtosis 28.3. A negative Sharpe cannot survive deflation, and the fat left tail is
carried into the calculation rather than assumed away.

---

## What was fixed before the answer was read

A verdict is only worth as much as the thresholds it was measured against, and the git
history is what makes that checkable rather than asserted:

| Fixed | Where | Commit | Date |
|---|---|---|---|
| The break-even accuracy, from the venue's own spread | [ADR-010](docs/adr/ADR-010-costs-are-measured-not-assumed.md) | `85c0d89` | 2026-08-28 |
| The minimum detectable effect, per validation design | [ADR-011](docs/adr/ADR-011-power-before-verdict.md) | `85c0d89` | 2026-08-28 |
| **The verdict** | [ADR-013](docs/adr/ADR-013-skill-and-profit-are-separate-questions.md) | `9cb292c` | 2026-08-31 |

The thresholds were published three days and two commits before the tests that were scored
against them. Selection happens on validation and the test block is scored once
([ADR-007](docs/adr/ADR-007-fitting-models-without-leaking.md)), which is the difference
between -0.21% and +1.85% on identical data.

**What was not pre-registered**, stated because omitting it would be the same failure this
project is a correction of: the *taxonomy of possible outcomes* was never written down in
advance. The thresholds were; the list of verdicts they might produce was not. Reading the
finding below as a pre-registered result would overstate it.

---

## What this establishes

- **The design could have seen a profitable edge.** Over 40,587 out-of-sample bars the
  minimum detectable effect is **0.62%** at 80% power. The closest model would need
  **1.23 points** over chance to pay for its own turnover, which this design detects from
  **10,216 bars** - a quarter of what it has. The negative half is a finding, not a failure
  to look ([ADR-011](docs/adr/ADR-011-power-before-verdict.md)).
- **The standard errors are honest.** Overlapping feature windows make the *features*
  strongly dependent - RSI-14 autocorrelates at +0.93 - but model correctness does not
  (-0.020), so `0.5/√n` was already right. Measured inflation: **0.97**
  ([ADR-012](docs/adr/ADR-012-turnover-and-dependence-are-measured.md)).
- **The two frameworks agree.** Accuracy against per-model break-even says nothing pays for
  itself; returns against a benchmark say nothing makes money. Different computations over
  different quantities, same answer.
- **It survives a data refresh.** On 51,200 bars instead of 51,147, the skill count rises
  from 9 to 13 and the economic verdict is unchanged at 0 of 18.

## What this does **not** establish

- **Not "there is no edge."** There is one. It is about one point of directional accuracy,
  and it is statistically robust.
- **Not that a lower-frequency strategy fails.** Break-even scales with turnover. These
  models hold 2.6 to 5.7 bars; nothing here tests one holding twenty. **This rules out the
  hourly directional trade**, which is what the original project claimed.
- **Not a full cost model.** Slippage, the rollover surcharge and the overnight swap are
  unmodelled. Each *raises* the bar, so they cannot rescue a negative result - but a
  positive one would have needed them first.
- **Not other assets or horizons.** One instrument, one interval, one target.

## What would change the answer

1. **A cheaper venue or a wider move.** Break-even is `0.5 + f·c/(2·E|r|)`. The median round
   trip is 1.86 bps against a 13.35 bps average move - **14% of what a correct prediction
   earns**. Halve the spread and the threshold halves.
2. **Lower turnover.** The same arithmetic. A strategy holding twenty bars pays a fifth as
   much, and these models already take part of that discount by holding up to 5.7.
3. **A larger edge.** The gap is small and it is a gap. The closest configuration holds
   **0.77 points** over chance and needs **1.23** to pay for its own turnover - so roughly
   **half again** the edge it has would do it, at the turnover it already runs. The design
   would detect that from a quarter of the bars available. Nothing here suggests where the
   extra half point would come from.

---

## What happened to the original analysis

This project re-engineers a postgraduate notebook that reported **51.53% accuracy** as
evidence that "it is possible to build ML models that beat random" on hourly gold.

| Original claim | What the rebuild found |
|---|---|
| 51.53% accuracy beats random | Its own always-UP baseline scored **51.86%** on the same block. The model **lost by 0.33 points** ([STATUS](docs/status/STATUS-2026-08-models.md)) |
| The winner chosen by `Test_AUC.idxmax()` | Selecting on test rather than validation is worth **2.06 points** on identical data - larger than any effect being hunted |
| Gold and the S&P correlate at **+0.923** | **+0.13** on returns. Eight tenths of it was shared trend ([STATUS](docs/status/STATUS-2026-08-exploratory.md)) |
| Symbols aligned by outer join and forward fill | Invented **1,250 gold bars** that never traded and flipped which class was the majority ([ADR-003](docs/adr/ADR-003-target-anchored-alignment.md)) |
| No transaction costs | Break-even is **51.2%-52.7%** depending on the model's own turnover ([ADR-012](docs/adr/ADR-012-turnover-and-dependence-are-measured.md)) |
| No power analysis | Without one, a null result cannot be told from a blind one |

**The original's conclusion was wrong, and the corrected pipeline finds something the
original had no way of seeing.** Not "nothing" - a real signal, and a measurement of exactly
how far short of useful it falls.

---

## Reproducing this

```bash
uv sync
uv run forecast-lab fetch                                  # public CDN, no API key
uv run forecast-lab verify                                 # do the bytes match the manifest?
uv run forecast-lab verdict  --target XAUUSD --timeframe 1H
uv run forecast-lab validate --target XAUUSD --timeframe 1H
```

Every figure here comes from those two commands; add `--json` to either for the machine-readable
payload. Sidecars: [verdict-canonical.json](docs/status/verdict-canonical.json) and
[walk-forward-canonical.json](docs/status/walk-forward-canonical.json).

**Figures are pinned to the 2026-08-26 snapshot** that the committed
[manifest](docs/status/data-manifest.json) describes. `fetch` extends the series forward, so
a later download changes every hash and `verify` will say so - that is the mechanism working,
not a fault. To move onto newer data, regenerate the numbers *and* the manifest in one
commit, never the manifest alone.

## Errors this project made and corrected

Kept here because a findings document that lists only its successes is not a findings
document. Each was caught by **measuring**, never by re-reading:

- **The break-even was assumed at a turnover none of these models has.** ADR-010 asserted
  every model flips position half the time. Measured: 17.6% to 38.4%. The verdict's margin
  was overstated by more than a point.
- **The dependence caveat was wrong in the other direction.** ADR-011 warned that every
  power figure was optimistic by an unmeasured factor. Measured: 0.97.
- **`rich` was silently eating the model representation.** `[raw]` and `[pca-95]` are valid
  console markup, so `train` printed the winner without saying which representation won.
- **`arch` is not symmetric.** `StepM` returns column names and `SPA` returns positional
  indices, so a superior model at position 4 was reported as the label `'4'`.
