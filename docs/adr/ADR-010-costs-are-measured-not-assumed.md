# ADR-010 - The cost of trading is measured from the venue, not assumed

- **Status:** Accepted - **built** (`research/costs.py`, the break-even computed inside `train`, and 16 tests).
- **Date:** 2026-08-27
- **Follows:** [ADR-002](ADR-002-data-source-and-symbol-set.md), which chose a venue that quotes both sides of the book *specifically so this number could be measured*, and [ADR-007](ADR-007-fitting-models-without-leaking.md), whose every accuracy was compared against the assumed figure.
- **Context:** Every accuracy this project has produced has been held against a break-even of **51.92%**, carried from a planning document that assumed a one basis point round trip. ADR-002 sec. 2 justified downloading both offer sides on the grounds that "the spread IS the transaction cost, and the break-even accuracy that decides whether any edge is worth having is computed from it". The column has been written into every fetched bar since the first day. It was never read.

## Decision

### 1. The break-even is computed from the venue's own quoted spread

`summarise_spread()` reads the `spread` column that `fetch` writes; `break_even()` turns it into the accuracy that pays for itself.

*Why:* the assumption and the measurement are not close. Over 51,147 hourly bars of gold:

| | Round trip | Break-even |
|---|---:|---:|
| Assumed (planning) | 1.00 bps | 51.87% |
| Probe, short window | 1.60 bps | 53.00% |
| **Measured, median** | **1.86 bps** | **53.49%** |
| Measured, mean | 2.13 bps | 53.99% |
| Measured, 95th percentile | 3.60 bps | 56.74% |

The mean absolute hourly move is **13.35 bps**, so the median round trip is **14% of the average move a correct prediction earns**. That ratio is the whole problem with trading this frequency, stated in one number.

**And it changes the verdict rather than decorating it.** The best accuracy any configuration reached, across both datasets and 36 configurations, was 53.31% - and that one was already shown to be the best of eighteen draws from noise. Against the measured cost, **nothing tested comes within two points of paying for itself**, and the honestly-selected model falls short by **2.41 points**. Under the assumed cost the same run fell short by 0.83, which reads as a near miss. It was not one.

**Trade-off:** the median is a choice. Spread is not constant - it runs from about 1.3 bps mid-session to 57.8 bps at the worst moment measured, and the 95th percentile is 3.60. Using the median describes a strategy that trades in normal conditions and ignores that a real one cannot pick its moments. The full distribution is reported beside it so the reader can pick a harsher assumption; none of them help the conclusion.

### 2. The arithmetic is stated, not buried

    p = 0.5 + f * c / (2 * E|r|)

with `c` the round trip, `E|r|` the mean absolute return per bar, and `f` the share of bars where the position changes.

*Why:* a break-even quoted without its flip rate is unfalsifiable. At `f = 0.5` - which is what a model with no persistence produces, and every model measured here qualifies - it reduces to `0.5 + c / (4 * E|r|)`. A trend follower holding for twenty bars pays a fifth as much and faces a much lower bar. That is a real escape route from this verdict and it is named rather than hidden: **nothing here rules out a lower-frequency strategy**, only the hourly one that was tested.

### 3. Impossible thresholds raise rather than report

`break_even()` refuses to return an accuracy at or above 1.0.

*Why:* at a sufficiently wide spread the formula yields "104%", and a percentage invites the reader to treat it as merely demanding. It is not demanding, it is arithmetic saying the strategy cannot exist at that frequency. An error says so; a number does not.

### 4. Where no spread exists, the assumption is used and labelled

The reference exports carry no spread column - their venue never published one - so runs on that data fall back to the 1 bp figure, and the command prints which one it used and why.

*Why:* silently switching between a measured and an assumed threshold would make two runs incomparable without saying so. The label costs one line of output and makes the difference visible in the log rather than in someone's memory.

## Consequences

**A number quoted sixteen times across eleven files changes meaning.** 51.92% is not deleted - it is still the correct break-even *under a 1 bp assumption*, and it is what applies to the reference dataset. What changes is that it stops being **the** threshold and becomes the optimistic end of a range whose measured middle is 53.49%. Every place that quoted it as fact now quotes both.

**The gap finding gets harder, not easier.** Price rises through the venue's pauses 56-59% of the time, which clears 53.49% on its face. But those are precisely the hours that pay overnight financing, and the swap on long gold is negative and triples on Wednesdays. This module does not model it yet, and until it does, that finding stays a measurement rather than a strategy.

**What is deliberately not modelled**, each of which raises the bar: slippage beyond the quoted spread, the rollover surcharge on the bars adjacent to gaps, the overnight swap, and the fact that a market order at the quoted spread is optimistic for size. The conclusion does not depend on any of them, which is the only reason it is safe to defer them.

**`net_of_costs()` exists but has no caller yet.** It charges turnover against a return series, which is what the economic benchmarks of Phase 2 will need - buy and hold, always-long, always-flat, random-at-equal-turnover. Recorded here as a debt, in the same way ADR-008 recorded `correlation_comparison` before the EDA gave it a caller.
