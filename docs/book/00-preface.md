# The method - a reference for this project

Six chapters covering every formula, decision and guard in this repository, written for one reader: its own author, two years from now, who has forgotten all of it. The ADRs record *why* each decision was made at the moment it was made; the STATUS logs record *what a run measured*. This book is the third thing neither of those is - a single continuous explanation of **how the whole apparatus works**, from a CSV of hourly bars to a sentence about whether the market can be beaten.

## The rule this book was written under

**Every formula cites the module and function that implements it. Every number cites the committed file it came from.** Nothing here is quoted from memory or inferred from a document; where a document and the code disagreed, the disagreement is reported in the chapter rather than smoothed over.

That rule is not decoration. It found things:

- The project's **headline number was wrong** - returns were summed instead of compounded, and summed in log space, so the most-quoted sentence read *"the best loses 196.7%"*, a loss larger than the capital available to lose it. Corrected: **-86.87%**, against buy-and-hold's **+100.91%**. [STATUS 2026-09-04](../status/STATUS-2026-09-return-accounting.md).
- A **live `TypeError`** on exactly the machines the RUNBOOK is written for.
- A **break-even quoting gold's threshold** for any series that arrived without a spread column.
- A dozen figures in ADRs, docstrings and guides that no longer matched the sidecars they were drawn from.

None of it changed the verdict. All of it changed what the documents say, and one of them was the sentence the whole project exists to state.

## The chapters

| | Chapter | What it answers |
|---|---|---|
| 1 | [Data, alignment and provenance](01-data-and-alignment.md) | Where the numbers come from, and what had to be true before any of them meant anything. Target-anchored alignment, staleness, the hash manifest. |
| 2 | [The features](02-features.md) | All nineteen columns with their formulas, the scale-freedom policy and why it is enforced by rescaling rather than by a list of names. |
| 3 | [Labels, splits and models](03-labels-splits-models.md) | What is being predicted, how the data is divided, and what is fitted to it. Six estimators, every metric, and the two leaks the design refuses. |
| 4 | [Costs, power and validation](04-costs-power-validation.md) | What a prediction has to beat, and whether the experiment could have seen it. Break-even, turnover, minimum detectable effect, walk-forward, serial dependence. |
| 5 | [Significance, multiplicity and the verdict](05-significance.md) | Is any of this real? Pesaran-Timmermann, Holm, Hansen's SPA, Romano-Wolf, the Deflated Sharpe - and how a result can be statistically real and economically worthless at once. |
| 6 | [The architecture](06-architecture.md) | How the code is arranged and what the arrangement prevents. The four layers, the AST guard, the `--json` contract, the three gates. |

## Reading orders

**To understand the finding**, read [FINDINGS](../../FINDINGS.md) first, then chapters 4 and 5. That is the argument: what a prediction must beat, and whether anything beat it.

**To trust the finding**, read chapters 1 and 6. Provenance and structure are what make the rest checkable, and they are the two chapters a reviewer should read before the statistics.

**To reproduce it**, the [RUNBOOK](../guides/RUNBOOK-getting-started.md) is the operational path; this book explains what each command is doing while you wait for it.

**To extend it to another instrument**, read chapters 2 and 3 - the feature policy and the label definition are where the assumptions specific to gold at one hour actually live.

## What this book is not

It is not a tutorial on machine learning, and it explains a standard concept only far enough to say what *this* code does with it. It is not a defence of the project's conclusions - where the evidence is thin, the chapters say so, and each one ends by naming what it could not verify. And it is not a substitute for the ADRs: a decision's reasoning at the time it was made is a different document from an explanation of the machine as it now stands, and collapsing the two would lose the record of what was believed before it was measured.
