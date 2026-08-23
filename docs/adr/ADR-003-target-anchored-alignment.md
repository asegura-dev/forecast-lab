# ADR-003 - The target's bars are the timeline

- **Status:** Accepted - **built** (`research/align.py`, the `align` command, and a regression test that pins the identity behind the defect).
- **Date:** 2026-08-22
- **Follows:** [ADR-002](ADR-002-data-source-and-symbol-set.md) - which establishes that a timestamp is the bar's open, the invariant this decision rests on.
- **Context:** Several symbols have to end up on one timeline before anything can be modelled, and they do not trade at the same moments. Gold, the currencies and the US indices keep different hours; crypto never stops. Deciding *whose* clock the panel runs on looks like plumbing, and it is where the original analysis went wrong in a way that invalidated every number downstream.

## Decision

### 1. The target's own bars define the timeline; nothing else may extend it

Auxiliary symbols are read onto that timeline by carrying their last known value forward. The target is never filled.

*Why:* the original pipeline joined every symbol with an outer join and forward-filled the result, so a row appeared at every timestamp *any* symbol traded. On the hours when the currencies were open and gold was not, gold's close became a copy of the previous bar - and a label built from `close[t+1] > close[t]` then compares a price with itself.

Measured on the real exports, with gold as the target and the nine auxiliaries the original project used:

| Strategy | Rows | Exact ties | UP | DOWN |
|---|---:|---:|---:|---:|
| outer join + forward fill | 24,430 | 1,288 | 48.53% | **51.47%** |
| inner join (the other notebook) | 19,932 | 31 | **51.15%** | 48.85% |
| **target-anchored** | **23,180** | 38 | **51.15%** | 48.85% |

Three readings, in order of how much they matter. The forward fill **never creates an UP** - 11,857 either way - so every fabricated row lands on the same side. It therefore **flips which class is the majority**: the corrupted data says gold falls more often than it rises, and that was the baseline every model was compared against. And the inner join, the other notebook's approach, gets the balance right by **discarding 3,248 real gold bars** - 14% of the good data - to buy it.

The identity behind the damage is exact and is fixed by a regression test: **each fabricated row adds exactly one tie**. On the real data that reads 24,430 - 23,180 = 1,250 and 1,288 - 38 = 1,250.

**Trade-off:** the panel is only as long as the target's history, so a symbol with a longer record contributes nothing before the target begins. That is the correct answer rather than a limitation - there is no row to attach it to - but it does mean changing the target changes the sample size, and every comparison across targets has to say so.

### 2. Carrying a value forward is legitimate; pretending it is fresh is not

Each auxiliary gets a `staleness_s` column recording, per row, how old the value it carries is.

*Why:* forward-filling is not a trick. At 03:00 the last thing known about the S&P really was its 21:00 close, and a model deciding at 03:00 would have had exactly that. What is dishonest is losing track of *how* old it is, because "the S&P is at 4,500" and "the S&P was at 4,500, sixteen hours ago" are different statements and only one of them is true.

**Per symbol, not one column** - measured on the real panel, the spread is wide enough that a single number would describe nobody:

| Symbol | Rows carried | Oldest |
|---|---:|---:|
| VIX | 8.0% | 74h |
| DXY | 7.1% | 75h |
| USDJPY | 0.1% | 33h |
| EURUSD, XAGUSD | 0.0% | 51h |

**Trade-off:** one extra column per auxiliary, and a feature the model can see. That is deliberate: staleness is real information about the state of the world, and hiding it would not make it stop affecting the answer.

### 3. Beyond a configurable age, the value goes missing instead of stale

`max_staleness_seconds` drops a carried value rather than letting it stand in.

*Why:* the honest reading of a forward fill decays. Over a weekend, Friday's close would otherwise masquerade as a Monday price for 65 consecutive hours, and a model would happily learn the resulting flat stretch as signal. There is no universally right threshold, which is why there is no default: the caller picks one, and the panel reports what it cost.

**Trade-off:** raising the limit keeps rows and lowers their quality; lowering it does the reverse. The choice belongs to the experiment, so it is a parameter rather than a constant.

### 4. An auxiliary whose bars outlast the target's is refused

*Why:* this is the judgement [ADR-002 sec. 6](ADR-002-data-source-and-symbol-set.md) deliberately declined to make at the reading boundary, because only here are the target and its interval known.

A daily bar carried onto an hourly row **has not closed yet** when that hour's decision is taken. Its high, low and close describe hours that have not happened. No amount of grid cleanliness makes that safe, and the symptom - a model that predicts slightly too well - is precisely the one nobody investigates. So mixing intervals fails loudly with an explanation, instead of producing a panel that looks fine.

**Trade-off:** the daily and 4-hour series, which reach back years further than the hourly ones, cannot currently contribute to an hourly panel at all. Making them contribute means shifting them by one of their own intervals so only closed bars are read, which is a real decision with its own trade-offs and deserves its own ADR rather than a quiet special case here.

### 5. Column order is sorted, not insertion order

*Why:* the original notebook built its symbol list with `list(set(...))`, and the iteration order of a set of strings varies between processes. Its column order therefore varied between runs, and with it the principal components computed from those columns. A pipeline whose output depends on hash seeding is not reproducible, however carefully everything else is pinned.

## Consequences

- The panel has 23,181 rows for gold at hourly bars - every real bar, no invented ones - against 24,430 for the outer join and 19,932 for the inner one.
- The class balance is now a property of the market rather than of the merge, which is the precondition for the majority-class baseline to mean anything.
- Staleness becomes visible and therefore arguable. The measurements above are the first honest look at how much of the panel is carried rather than observed.
- Mixing timeframes is now impossible by accident, and possible only by an explicit later decision.
- `research` gained its first module, so the guard forbidding it to read data stopped skipping and became a real gate.

## Implementation status

**Built:** `research/align.py`, the `align` command, 16 unit tests and 3 regression tests. Verified against the real reference panel, where it reproduces every figure measured during the audit: 23,181 rows, VIX carried on 8.0% of them, DXY on 7.1%, crypto absent from 25.5%.

**Not built:** shifting a coarser auxiliary so it can contribute safely (sec. 4). Deferred until something needs it.
