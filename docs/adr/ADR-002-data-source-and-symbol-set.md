# ADR-002 - The data source and the symbol set: a fetchable venue, and no VIX

- **Status:** Accepted - **built**. Probe done, verdict measured ([STATUS-2026-08-dukascopy-probe](../status/STATUS-2026-08-dukascopy-probe.md)). See *Implementation status* for what has actually been executed since.
- **Date:** 2026-08-18
- **Follows:** [ADR-001](ADR-001-hexagonal-architecture.md) - this decides what flows in through the boundary that ADR-001 defines.
- **Context:** A research repository whose entire claim is that its numbers are trustworthy needs a data source with three properties: **a stranger can fetch it** (otherwise no published figure is verifiable), **it extends forward** (otherwise the sample is frozen and no genuinely out-of-sample window can ever exist), and **its provenance can be hashed** (otherwise a result cannot be tied to the bytes that produced it). This ADR picks that source, and - because history depth is not uniform across instruments - it also settles the symbol set, since the symbol set is what silently decides the modelling window.

## Decision

### 1. Dukascopy is the source

*Why:* it publishes to a public CDN with **no API key and no registration**, so the dataset is regenerable by anyone who clones the repository: `forecast-lab fetch`, and every published number can be checked. It also reaches back far enough that the window is chosen by the research question rather than by the vendor, and it keeps advancing, which is what makes a genuinely unseen holdout possible at all.

A second dataset has a narrow, separate role. The academic project this work re-analyses used hourly exports from a different venue (Capital.com, via TradingView), and its published results are the baseline this repository sets out to reproduce and correct. Those files therefore live in `data/reference/`, are read to regenerate that baseline for the STATUS logs, and do not feed the engine. *That separation was documented on 2026-08-18 and only executed on 2026-08-26; see Implementation status.*

| | `data/raw/` | `data/reference/` |
|---|---|---|
| Content | Dukascopy downloads | the prior project's hourly exports |
| Purpose | the modelling dataset | reproduce the published baseline, once |
| Obtained by | `forecast-lab fetch` | `forecast-lab ingest --from <path>` |

**Trade-off:** two datasets means two ingestion paths and a manifest that has to record which venue each file came from. The alternative, modelling on the reference exports, was rejected because they cannot be refreshed, cannot be fetched by a reader, and their final segment has already been read by everyone who worked on the original analysis.

*Why not yfinance:* verified against the live endpoint. Hourly history is capped at **730 days** (a request for January 2022 returns HTTP 422), and `XAUUSD=X` and `XAGUSD=X` return **HTTP 404**: spot gold and silver do not exist there at all. `^GSPC` yields only 7 bars per session day.

*Why not MetaTrader 5:* it requires Windows with the terminal running and a broker account, and its history depth depends on a GUI setting. A repository whose thesis is reproducibility cannot depend on a desktop application being open.

### 2. The instrument identity was verified, not assumed

A venue's "gold" is not automatically the same instrument as another's. Before adopting it, XAU/USD hourly over 2025-11 was compared against the reference exports: **457 of 457** timestamps matched, and the correlation of hourly returns is **0.9996**. The reference close falls between Dukascopy's bid and ask, implying a spread of about **0.70 USD, or 1.6 bps**.

*Why it is worth the check:* the baseline this repository reproduces was computed on the reference series. If the two venues quoted materially different instruments, every before-and-after comparison in the project would be measuring the venue change rather than the correction being tested.

*Why it matters beyond the go/no-go:* that spread is now a **measured** input to the cost model instead of a guess. The break-even accuracy, the point where a directional edge stops paying for its own transaction costs, was going to be bracketed between an assumed 1 bp and 3 bps. It can now be computed from an observed number.

**Trade-off:** verified on one month of one symbol. Decisive for instrument identity; coverage over the full window still has to be validated when the bulk fetch runs.

### 3. VIX is dropped, and the window becomes 2018-01 to today

Hourly history does not begin at the same date for every symbol, so **the symbol set silently chooses the window, and the window chooses the statistical resolution**:

| Configuration | Bars | Test at 15% | MDE, single split | MDE, walk-forward |
|---|---:|---:|---:|---:|
| All 12 (VIX binds, 2022-10) | 22,937 | 3,440 | 52.12% | 51.16% |
| **Without VIX** (DXY binds, 2018-01) | **51,073** | 7,660 | **51.42%** | **50.78%** |

Against a break-even accuracy of 51.92%, the twelve-symbol configuration cannot resolve a barely-profitable edge on a single split. Dropping VIX resolves it, and more than doubles the sample.

The contribution of VIX was measured rather than argued. Over 21,156 hourly bars, its correlation with gold's next-hour direction is **-0.011**; the realised-volatility proxy's is **+0.001**. Both are zero.

*Why:* keeping VIX costs **28,136 bars, 55% of the available sample**, to buy a feature whose measured association with the target is 0.011. The decision rests on that asymmetry: the cost is certain, the benefit is not.

**Trade-off:** a linear correlation against a binary target is a weak instrument. A tree model could exploit a non-linear interaction it cannot see. This ADR does **not** claim VIX is useless; it claims the price of keeping it is too high for the evidence available. If a later chapter wants to revisit it, the short-window configuration is one flag away, because the engine is symbol-agnostic by construction.

### 4. Realised volatility replaces it, computed causally

Rolling realised volatility of the target and of the S&P 500, from log returns, with explicit window and `min_periods`, never an expanding window over the whole sample. Built as `realised_vol_24` and `realised_vol_168` in `research/features/technical.py`, at the windows this decision's own probe measured: the correlation with VIX's level is +0.645 at 24h and flat past a week at +0.754, so seven days buys the tracking at the shortest warm-up.

*Why:* it tracks the level of the fear regime at **+0.75** against VIX, is free, exists across the entire window, and depends on no external symbol, so it can never be the thing that shortens the sample again.

**Trade-off:** it substitutes as a *regime* indicator, not as a *shock* detector. The correlation of its **changes** with VIX's changes is about 0.00, which is expected: VIX is forward-looking implied volatility and this is its backward-looking twin. Anything that depends on anticipating a spike is not captured.

### 5. Each (symbol, timeframe) is fetched independently; nothing is resampled

*Why:* a venue's higher-timeframe bars are anchored to **its own trading day**, not to UTC midnight, and that anchor **moves with daylight saving time**. Measured on the reference exports: 4-hour bars sit at offsets of 7200 *and* 10800 seconds, and daily bars at 79200 and 82800, a broker day closing at 22:00 or 23:00 UTC depending on the season. A UTC-anchored resample would produce a series that silently disagrees with the venue's own, cutting sessions in half and creating partial buckets around the weekly gap.

Higher timeframes also tend to reach further back than hourly does, so deriving them from the hourly series would discard real history rather than merely re-bucket it.

**Trade-off:** one series per (symbol, timeframe) instead of one derived family, and `SeriesMeta` must carry an anchor **set** rather than a single value. Dukascopy's hourly grid is clean - `timestamp % 3600 == 0` holds for all twelve instruments - but **its 4-hour and daily anchors have not been measured**, and the rejection rule in sec. 6 must not assume they are zero.

### 6. Canonicalisation at the boundary; grid shape is described, not judged

Every timestamp is converted to timezone-aware UTC. The reader rejects only what cannot be interpreted at all - a missing price column, an unparseable timestamp, a duplicated one - and **records** the shape of the grid rather than ruling on it: the set of offsets bars sit at, and the count of consecutive bars closer together than one nominal interval.

*Why the invariant still matters:* the timestamp is the bar **open**. That is what makes forward-fill alignment safe (ADR-003): an auxiliary bar opening at or before `t` closes at or before `t + one interval`, exactly when the decision at `t` is taken. A series on a genuinely irregular grid would smuggle look-ahead into every row, silently.

*Why the check moved anyway:* this ADR first specified rejecting any series with more than two grid offsets. Running that rule against the reference exports rejected six files, and the reason was not corruption. **SPX and NDX daily and 4-hour bars sit at three offsets, and DXY at four**, because a US instrument quoted on a European trading calendar crosses two daylight-saving regimes that do not switch on the same weekend. **SPX daily also has 1,305 gaps shorter than 24 hours** - which is not an overlap but a short session, since a "day" is a session and not a duration. The threshold was measuring the calendar rather than a defect.

The deeper reason it belonged elsewhere: whether a grid property is harmful depends on the target series and the prediction horizon, and the reader knows neither. Aligning a daily auxiliary onto an hourly target is unsafe no matter how clean both grids are, because the daily bar is still open when the hourly decision is taken. That judgement is alignment's, so the reader hands it the facts and stays out of it.

**Trade-off:** the boundary is now more permissive than originally designed, and a genuinely broken grid will be caught one step later instead of at import. In exchange, the rule that remains is one the data supports. The import command prints every unusual grid it records, so nothing is silent.

### 7. Provenance is a versioned manifest and a command, not a test

`docs/status/data-manifest.json` records sha256, row count and time range per file, and is **committed**. The `data/` directory stays ignored. `forecast-lab verify` re-checks it.

*Why:* a manifest that lives inside an ignored directory ties no published result to any data. And a test that reads `data/` would skip itself in a clean clone and in CI - a gate that skips is decoration, not a guarantee. The tests stay hermetic on a synthetic fixture; provenance is a command the operator runs.

**A file that grew and a file that was rewritten are reported apart** (2026-09-06). Both make a whole-file hash disagree, and only one is serious. The series extends forward in time, so *every* operator who fetches after the manifest was cut has longer files than it records - and for a fortnight the command answered that with `CHANGED ... any result computed from this data is no longer reproducible from it`. That sentence was false for the common case and true for the rare one, printed identically for both.

The cost of collapsing them is not a cosmetic one. **A warning that fires on every routine action trains the operator to clear it without reading**, and the way you clear this one is by re-cutting the manifest - which is precisely how a genuine revision gets absorbed with nobody noticing. The alarm being too loud is what would eventually make it useless.

So `verify` truncates each file to the row count the entry recorded, re-hashes that prefix, and reports **EXTENDED** when it matches - exiting 0, because the promise the manifest makes is that a published number traces to bytes, and that promise is intact. **REVISED** and **MISSING** keep exit 1 and the original language. The row count is what makes this decidable at all; a manifest holding only a hash could not tell the two apart, which is a reason to record more than the minimum.

*Checked on the real case:* the working data was two days ahead of the committed snapshot across all eleven files, and all eleven verified as extensions - the recorded prefix hashed identically for every one. That is the evidence the distinction is doing real work rather than excusing drift, and it is the check `tests/unit/test_manifest.py` pins in both directions, including a file that grew *and* rewrote its past.

### 8. The canonical dataset is what every analysis command reads by default

Two commands are exempt and both for the same reason - the prior project's exports are their
subject, not their input. `ingest` writes them; `baseline` reproduces the original's
published numbers on the original's data, which cannot be done on any other data. Everything
else - `align`, `explore`, `features`, `train`, `validate`, `verdict` - defaults to
`data/raw`, the dataset `fetch` downloads.

*Why this is an ADR and not a tidy-up:* it was not true until 2026-09-05, and the way it was
untrue is the interesting part. `features`, `explore` and `train` were built during Phase 1,
when the reference exports were the only data in existence, and nobody revisited their
defaults once `fetch` arrived and `validate` and `verdict` were written against `data/raw`.
**The result was a split nothing announced.** A reader following the RUNBOOK got a break-even
of **51.92%** from `train` and **53.49%** from `verdict` - two different datasets, two
different answers, no line of output saying they were not comparable.

Nothing was inconsistent. Each command was correct about its own dataset, each figure was
reproducible, and all three gates were green throughout. The defect lived in the space
between two commands that nobody had asked to agree.

*The trade-off:* `train` and `explore` now report different numbers than they did, and the
committed sidecars generated from reference (`model-comparison.json`, `eda-summary.json`,
`features-policy.json`) are no longer what the bare command produces. That is accepted, and
handled the way this repository already handled it once - by keeping both and naming them:
each has a `-canonical` sibling, and [features-policy-canonical.json](../status/features-policy-canonical.json)
was generated to complete the set. A figure whose dataset is ambiguous is worse than two
figures whose datasets are labelled.

*What stops it recurring:* `tests/unit/test_cli.py::test_every_command_reads_the_dataset_this_list_says_it_does`
reads every command's default out of the source and compares it against a list with a reason
per entry. A default that drifts now fails a gate instead of quietly answering a different
question.

## Consequences

- The repository becomes **runnable by a stranger**: clone, fetch, and every published number can be reproduced. This is the single largest gain of this ADR, and it was a side effect rather than the goal.
- The modelling window grows to **51,073 bars**, which lowers the minimum detectable effect from 52.12% to 51.42% on a single split - the quantity that governs whether this project's verdict can mean anything.
- Dukascopy returns a **volume** column the reference exports lack, re-opening volume-based indicators (OBV, MFI, CMF, VWAP). Not used yet; recorded because it changes what is possible.
- The engine now depends on a network fetch. The three quality gates stay hermetic because tests construct synthetic bars in code, and network tests are opt-in behind `pytest -m network`.
- Dukascopy's data licensing terms are **not published**. The repository redistributes no raw data and publishes only derived statistics, which is standard practice in quantitative research, but the terms remain unread because they are unavailable to read. Recorded as a known exposure.
- Its indices and commodities are CFDs from its own liquidity pool rather than official exchange prints - the same class of instrument the reference exports were, which is why the two agree so closely.

## Implementation status

The probe that produced the verdict ran outside the repository in an ephemeral environment and wrote nothing into it.

**Built (2026-08-19):** `fetch`, `ingest --from` and `verify`, plus the reader and the manifest behind them. All 34 reference exports import and verify (**462,674 bars**, 12 symbols); `fetch` downloads both offer sides and writes mid prices with the per-bar spread. Fifteen opt-in network tests hold the venue to its side of the contract, including that all twelve mapped instruments still exist.

**Corrected while building**, each caught by running rather than by testing:

- The grid rule of sec. 6 rejected six legitimate files on first contact with real data. Rewritten above with the measurements that refuted it.
- Manifest paths were recorded relative to each command's destination, so a reference import and a fetch both produced the key `XAUUSD_1H.csv` and would have overwritten one another. Paths are now relative to `data/`, and the same collision reappeared once more in the row-count lookup before it was closed there too.
- The spread proved the design: measured at the weekly open it is **9.5 USD against a mid-session 0.75** on gold. A constant cost assumption would have hidden an order of magnitude.

**Built (2026-08-26), and a week late:** the canonical dataset this ADR is about. `fetch` had only ever been run as a 240-bar smoke test, so **every published number in this repository was computed on the reference exports** - the dataset sec. 1 says must *never feed the engine*. The decision was written, agreed, and then not executed, and nothing in the pipeline could notice: both directories read identically, and every command was simply pointed at the one that had data in it.

The canonical set now exists: **11 symbols, 606,250 hourly bars, 2018-01 to date**, verified against a manifest of 45 entries. Gold alone goes from 23,181 bars to **51,147**.

Running the pipeline on both is what makes the two roles worth having, and it produced a corroboration the single dataset could not:

| | reference (Capital.com) | canonical (Dukascopy) |
|---|---:|---:|
| Hourly bars of gold | 23,181 | **51,147** |
| Bars whose horizon spans a gap | 4.37% | **4.37%** |
| UP across those gaps | 59.35% | **56.40%** |
| UP across the stated horizon | 50.87% | 50.76% |
| Exact ties | 36 | 15 |

The gap fraction is **identical to two decimal places across two venues and twice the sample** - it is a property of gold's trading calendar, not of a provider. And the gap skew reproduces: price rises through the pause far more often than through an ordinary hour, at both venues. A finding measured once is an observation; measured twice on independent data it is a finding.

**And it changes the project's central argument.** A single 70/15/15 split of the reference data leaves 3,478 test bars and a minimum detectable effect of 2.11 points - which cannot resolve the 1.92-point break-even it exists to test. The same split of the canonical data leaves **7,334 bars and an MDE of 1.45 points**, which can. What was "this design needs walk-forward" becomes "this design needed walk-forward *or* more data", with both routes now available and measured.

**Also built (2026-08-27):** realised volatility (sec. 4), which this ADR had described in the present tense since 2026-08-18 and which did not exist.

> **Corrected 2026-09-02.** Presented here as the substitute this decision promised, and it is - but not as an invention. `Proyecto_Final_Completo` already computed `RealVol_24h` by the same definition, in a feature function this repository never analysed. It arrived here independently, which is not the same as arriving first ([STATUS](../status/STATUS-2026-09-parity.md) sec. 2.5). Dropping VIX cost 55% of the sample on the promise of a substitute that was never written. It is now `realised_vol_24` and `realised_vol_168`, and adding it changed which model the pipeline selects - see [STATUS 2026-08-24](../status/STATUS-2026-08-models.md) sec. 5.

**The lesson is the one worth keeping.** Three of this ADR's decisions were documented as done and were not: the canonical dataset, the realised volatility, and the fixture that the gitignore described. A Status line saying "built" is a claim like any other, and this repository's own rule - anchor every claim in the code - applies to it. An audit found all three; nothing in the gates could have.

Remaining: measure the venue's 4-hour and daily anchors (sec. 5), still unknown - the anchors measured so far are the reference venue's.
