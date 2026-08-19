# ADR-002 - The data source and the symbol set: a fetchable venue, and no VIX

- **Status:** Accepted - **probe done, verdict measured** ([STATUS-2026-08-dukascopy-probe](../status/STATUS-2026-08-dukascopy-probe.md)). Nothing built yet; the `fetch` and `ingest` commands are slice 2 of Phase 1.
- **Date:** 2026-08-18
- **Follows:** [ADR-001](ADR-001-hexagonal-architecture.md) - this decides what flows in through the boundary that ADR-001 defines.
- **Context:** A research repository whose entire claim is that its numbers are trustworthy needs a data source with three properties: **a stranger can fetch it** (otherwise no published figure is verifiable), **it extends forward** (otherwise the sample is frozen and no genuinely out-of-sample window can ever exist), and **its provenance can be hashed** (otherwise a result cannot be tied to the bytes that produced it). This ADR picks that source, and - because history depth is not uniform across instruments - it also settles the symbol set, since the symbol set is what silently decides the modelling window.

## Decision

### 1. Dukascopy is the source

*Why:* it publishes to a public CDN with **no API key and no registration**, so the dataset is regenerable by anyone who clones the repository: `forecast-lab fetch`, and every published number can be checked. It also reaches back far enough that the window is chosen by the research question rather than by the vendor, and it keeps advancing, which is what makes a genuinely unseen holdout possible at all.

A second dataset has a narrow, separate role. The academic project this work re-analyses used hourly exports from a different venue (Capital.com, via TradingView), and its published results are the baseline this repository sets out to reproduce and correct. Those files therefore live in `data/reference/`, are read exactly once to regenerate that baseline for the STATUS log, and never feed the engine.

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

Rolling realised volatility of the target and of the S&P 500, from log returns, with explicit window and `min_periods`, never an expanding window over the whole sample.

*Why:* it tracks the level of the fear regime at **+0.75** against VIX, is free, exists across the entire window, and depends on no external symbol, so it can never be the thing that shortens the sample again.

**Trade-off:** it substitutes as a *regime* indicator, not as a *shock* detector. The correlation of its **changes** with VIX's changes is about 0.00, which is expected: VIX is forward-looking implied volatility and this is its backward-looking twin. Anything that depends on anticipating a spike is not captured.

### 5. Each (symbol, timeframe) is fetched independently; nothing is resampled

*Why:* a venue's higher-timeframe bars are anchored to **its own trading day**, not to UTC midnight, and that anchor **moves with daylight saving time**. Measured on the reference exports: 4-hour bars sit at offsets of 7200 *and* 10800 seconds, and daily bars at 79200 and 82800, a broker day closing at 22:00 or 23:00 UTC depending on the season. A UTC-anchored resample would produce a series that silently disagrees with the venue's own, cutting sessions in half and creating partial buckets around the weekly gap.

Higher timeframes also tend to reach further back than hourly does, so deriving them from the hourly series would discard real history rather than merely re-bucket it.

**Trade-off:** one series per (symbol, timeframe) instead of one derived family, and `SeriesMeta` must carry an anchor **set** rather than a single value. Dukascopy's hourly grid is clean - `timestamp % 3600 == 0` holds for all twelve instruments - but **its 4-hour and daily anchors have not been measured**, and the rejection rule in sec. 6 must not assume they are zero.

### 6. Canonicalisation at the boundary, and the invariant that makes forward-fill safe

Every timestamp is converted to timezone-aware UTC, and a file is rejected unless `(ts - anchor) % seconds(timeframe) == 0` for a single observed anchor set.

*Why:* the timestamp is the bar **open**. That is the load-bearing invariant of the whole alignment design (ADR-003): an auxiliary bar opening at or before `t` closes at or before `t + one interval`, which is exactly the instant the target's bar at `t` closes and the decision is taken. A symbol delivered on a grid shifted by 30 minutes would smuggle half an hour of look-ahead into every single row, silently. Alignment must therefore **fail loudly** on a grid that does not fit, never align anyway.

**Trade-off:** the naive form of this check, `ts % seconds(tf) == 0`, would reject 100% of the 4-hour and daily reference files, whose anchors are non-zero. The anchor has to be measured from the file and stored, which is more machinery than a modulo.

### 7. Provenance is a versioned manifest and a command, not a test

`docs/status/data-manifest.json` records sha256, row count and time range per file, and is **committed**. The `data/` directory stays ignored. `forecast-lab verify` re-checks it.

*Why:* a manifest that lives inside an ignored directory ties no published result to any data. And a test that reads `data/` would skip itself in a clean clone and in CI - a gate that skips is decoration, not a guarantee. The tests stay hermetic on a synthetic fixture; provenance is a command the operator runs.

## Consequences

- The repository becomes **runnable by a stranger**: clone, fetch, and every published number can be reproduced. This is the single largest gain of this ADR, and it was a side effect rather than the goal.
- The modelling window grows to **51,073 bars**, which lowers the minimum detectable effect from 52.12% to 51.42% on a single split - the quantity that governs whether this project's verdict can mean anything.
- Dukascopy returns a **volume** column the reference exports lack, re-opening volume-based indicators (OBV, MFI, CMF, VWAP). Not used yet; recorded because it changes what is possible.
- The engine now depends on a network fetch. The three quality gates stay hermetic because tests run on a committed synthetic fixture, and network tests are opt-in behind `pytest -m network`.
- Dukascopy's data licensing terms are **not published**. The repository redistributes no raw data and publishes only derived statistics, which is standard practice in quantitative research, but the terms remain unread because they are unavailable to read. Recorded as a known exposure.
- Its indices and commodities are CFDs from its own liquidity pool rather than official exchange prints - the same class of instrument the reference exports were, which is why the two agree so closely.

## Implementation status

Nothing built. The probe that produced the verdict ran outside the repository in an ephemeral environment and wrote nothing into it.

Remaining, each a follow-up executed against this ADR:

1. `fetch` - bulk download into `data/raw/`, with the coverage validation sec. 2 defers.
2. `ingest --from` - copy and validate the reference exports into `data/reference/`.
3. `manifest` and `verify` - the versioned provenance of sec. 7.
4. Measure Dukascopy's 4-hour and daily anchors (sec. 5), which are currently unknown.
