# Documentation index - what is in each file

The reference for finding the right document **without opening it**. One line per file: what it contains. Grouped as the research book is - decisions (ADRs), experiment results (STATUS logs), and guides. For the reading order and the current state, see [README](../README.md).

Only files that exist today are listed. The book grows one line at a time, as each piece lands.

## Decisions - ADRs (`docs/adr/`)

| ADR | File | The decision |
|-----|------|--------------|
| 001 | [hexagonal-architecture](adr/ADR-001-hexagonal-architecture.md) | Four layers (`contracts`, `ingest`, `research`, `interfaces`) with dependencies pointing inward, and the load-bearing rule that **`research` may never reach for data** - a layer that can quietly re-read the world produces results that cannot be falsified. Records what is deliberately **not** built: no ports package and no storage layer, because an abstract interface earns its place when a second implementation exists and today there is one source feeding one caller. Also fixes two holes in the layering guard (relative imports escaping it, and modules at the package root sitting outside it). |
| 002 | [data-source-and-symbol-set](adr/ADR-002-data-source-and-symbol-set.md) | Dukascopy as the engine's source - fetchable with no API key, extends forward, hashable provenance - with the prior project's exports kept only to reproduce its published baseline once. Verifies instrument identity before adopting it (0.9996 return correlation), drops **VIX** because its short history costs 55% of the sample for a feature correlating -0.011 with the target, and settles the window at 2018-01 onward. Each `(symbol, timeframe)` is fetched independently rather than resampled, because venue anchors move with daylight saving time. |

## Experiments - STATUS logs (`docs/status/`)

| File | What it measured |
|------|------------------|
| [2026-08-18 - Dukascopy probe](status/STATUS-2026-08-dukascopy-probe.md) | The five questions answered before adopting a data source: instrument existence, coverage, history depth, grid anchor, and the go/no-go against the reference exports. **Verdict: GO** - same instrument, clean UTC grid, and volume the reference lacks. Also the finding that reshaped the symbol set: VIX hourly history starts only in 2022-10, so keeping it costs 28,136 bars to buy a feature whose correlation with the target is zero. Realised volatility substitutes for its *level* (+0.75) but not for its *changes* (0.00). |

## Data provenance (`docs/status/`)

| File | What's in it |
|------|--------------|
| [data-manifest.json](status/data-manifest.json) | SHA-256, byte size, row count and time span for every series file under `data/`. Committed while the data is not, because the data is regenerable but "regenerable" is not "identical": a venue can revise a bar and a file can be edited, and without a hash a result and its inputs drift apart silently. Checked with `forecast-lab verify`. |

## Guides (`docs/guides/`)

| File | What's in it |
|------|--------------|
| [engineering-conventions](guides/engineering-conventions.md) | The three gates, the four hard architectural rules, the data rules that exist because breaking them once produced a wrong number that looked right, where each kind of document lives, and the epistemic rules the project holds itself to. |

## Top-level

| File | What's in it |
|------|--------------|
| [README](../README.md) | What the project is and is not, the finding that started it (the reported winner loses to the majority-class baseline), why sample size decides the whole question, the architecture at a glance, the quickstart, and the current build status. |
| [CHANGELOG](../CHANGELOG.md) | Notable changes, dated, newest first. Complements the ADRs and STATUS logs rather than replacing them. |
