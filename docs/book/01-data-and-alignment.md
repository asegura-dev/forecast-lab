# Chapter 1 - Data, alignment and provenance

This chapter answers the question everything else depends on: **where do the numbers come from, and what had to be true before any of them meant anything?** Four things. The source had to be fetchable by a stranger, or no published figure is checkable. The symbol set had to be chosen deliberately, because it silently decides the modelling window and therefore the statistical resolution. Several symbols had to be put on one timeline without inventing a single bar. And the bytes had to stay tied to the results computed from them, by a hash rather than by trust.

## 1. The source, and why Dukascopy

[ADR-002](../adr/ADR-002-data-source-and-symbol-set.md) sets three requirements: a stranger can fetch it, it extends forward, its provenance can be hashed. Dukascopy meets all three - a public CDN, no API key, no registration - and the repository became runnable from a clean clone, which the ADR records as "a side effect rather than the goal".

The alternatives were rejected on measurement, not preference: yfinance caps hourly history at **730 days** (January 2022 returns HTTP 422), returns **HTTP 404** for `XAUUSD=X` and `XAGUSD=X`, and yields 7 bars per session day on `^GSPC` (ADR-002 sec. 1, [STATUS-2026-08-dukascopy-probe](../status/STATUS-2026-08-dukascopy-probe.md)).

Instrument identity was verified rather than assumed. XAU/USD hourly over 2025-11 against the reference exports: **457 of 457** timestamps matched, hourly-return correlation **0.999589** (bid) and **0.999681** (ask), with the reference close falling between the two sides at an implied spread of about **0.70 USD, 1.6 bps** (probe sec. 4) - a go/no-go on instrument identity, and a *measured* transaction cost where the cost model had been going to assume one.

`src/forecast_lab/ingest/dukascopy.py::fetch_series` downloads **both sides of the book** and writes mid prices plus a per-bar `spread = ask.close - bid.close`, because the spread is the transaction cost and a constant would hide it - ADR-002's implementation status records the vindication, a gold spread of **9.5 USD at the weekly open against a mid-session 0.75**.

Two datasets exist and are not interchangeable. `data/raw/` is the canonical modelling set from `fetch`. `data/reference/` holds the prior project's Capital.com exports, imported by `src/forecast_lab/ingest/importer.py::import_directory`, read once to regenerate the baseline this repository sets out to correct, and never fed to the engine. `import_directory` validates every file *before* copying it, and anchors manifest paths at `data/` through its `data_root` argument - because relative to the destination, a reference import and a fetch both produce the key `XAUUSD_1H.csv`.

## 2. The symbol set chooses the window

Hourly history does not begin at the same date for every instrument, so the symbol set decides the sample size, and the sample size decides what can be resolved at all:

| Configuration | Bars (1H) | MDE, single split | MDE, walk-forward |
|---|---:|---:|---:|
| All 12 (VIX binds, from 2022-10) | 22,937 | 52.12% | 51.16% |
| Without VIX (DXY binds, from 2018-01) | **51,073** | **51.42%** | **50.78%** |

Against a break-even accuracy of 51.92%, the twelve-symbol configuration **cannot resolve a barely-profitable edge on a single split**. VIX was dropped. What it bought was measured: over 21,156 hourly bars its correlation with gold's next-hour direction is **-0.011**, and the realised-volatility proxy's is **+0.001**. The decision rests on an asymmetry rather than a proof - the cost (28,136 bars, 55% of the sample) is certain, the benefit is not - and ADR-002 sec. 3 says so, naming the weakness of a linear correlation against a binary target. VIX stays mapped in `dukascopy.py::INSTRUMENTS` and out of `DEFAULT_SYMBOLS`. Its substitute, `realised_vol_24` and `realised_vol_168` in `research/features/technical.py`, tracks VIX's *level* at **+0.754** (7 d) but its *changes* at about 0.00 - a regime indicator, not a shock detector.

**Crypto is a different exclusion, at a different layer.** BTCUSD and ETHUSD are fetched, are aligned, and can be targets; what they cannot do is contribute *feature columns* in WHOLE mode, because `research/features/build.py::_contributors` filters them out through `LATE_STARTERS = frozenset({"BTCUSD", "ETHUSD"})` - their history starts a year late and including them would cost 25.5% of the rows. Worth recording: ADR-006 sec. 5 originally blamed crypto for the original project's mismatched row counts (24,232 against 22,441) and was **corrected on 2026-09-02**. The original excluded BTCUSD explicitly; the real cause was a `dropna()` ending an indicator function called once per symbol in a loop, paying a 199-row warm-up ten times - `24,431 - 10x199 = 22,441` and `24,431 - 199 = 24,232`, both exact. The decision survived the correction; the explanation did not.

## 3. The boundary, and the invariant that makes alignment safe

`src/forecast_lab/ingest/csv_reader.py::read_series` establishes timezone-aware UTC, strict ordering and uniqueness rather than hoping for them, and rejects only what cannot be interpreted at all. It deliberately does **not** judge the grid - `_anchors` and `_short_gaps` describe its shape and neither can fail an import, after an earlier rule rejecting more than two offsets threw out six legitimate files whose only defect was being a US instrument quoted on a European trading calendar. What it does establish is the invariant everything downstream rests on: **a timestamp is the bar's open**, so an auxiliary bar opening at or before `t` closes at or before `t + one interval` - exactly when the decision at `t` is taken. The probe confirmed it for all twelve instruments (`timestamp % 3600 == 0`).

## 4. Target-anchored alignment

A forward fill carries the last known value into a row where nothing new was observed. That is legitimate for an *auxiliary* column and catastrophic for a *target*, and the difference is the whole chapter.

The original pipeline joined every symbol with an outer join and forward-filled the result, so a row appeared at every timestamp *any* symbol traded. On the hours when the currencies were open and gold was not, gold's close became a copy of the previous bar - and a label built from `close[t+1] > close[t]` compares a price with itself, returns False, and manufactures a DOWN out of nothing. Measured on the real exports, with gold as the target and the nine auxiliaries the original used ([ADR-003](../adr/ADR-003-target-anchored-alignment.md) sec. 1):

| Strategy | Labels | Exact ties | UP | DOWN |
|---|---:|---:|---:|---:|
| outer join + forward fill | 24,430 | 1,288 | 48.53% | **51.47%** |
| inner join (the other notebook) | 19,932 | 31 | **51.15%** | 48.85% |
| **target-anchored** | **23,180** | 38 | **51.15%** | 48.85% |

Three readings, in the order they matter. The fill **never creates an UP** - 11,857 either way - so every fabricated row lands on the same side. It therefore **flips which class is the majority**: the corrupted data says gold falls more often than it rises, and that was the baseline every model was compared against. And the inner join buys the right balance by discarding **3,248 real gold bars**, 14% of the good data. The *Labels* column counts labels rather than bars, which is why target-anchored reads 23,180 against the panel's 23,181 rows: the last bar has no successor to label it against.

The identity behind the damage is exact: **each fabricated row adds exactly one tie**. On the real data, `24,430 - 23,180 = 1,250` and `1,288 - 38 = 1,250`. It is pinned by `tests/regression/test_ffill_fabricates_labels.py::test_each_fabricated_row_adds_exactly_one_tie`, alongside `::test_the_forward_fill_never_creates_an_up` and `::test_the_target_timeline_is_never_extended` - all three on synthetic bars, reconstructing the naive merge inside the test file so the package never ships a function whose only purpose is to be wrong.

The fix is `src/forecast_lab/research/align.py::align_to_target`. The target's columns go on unchanged and are never filled; every other symbol is read onto that index by `align.py::_carry_forward`, whose entire mechanism is one line - `aux_index.searchsorted(target_index, side="right") - 1`. The `- 1` is what makes it "the last bar at or before `t`" rather than "at or after", the difference between reading the past and reading the future. Rows before an auxiliary's history begins become missing and are never back-filled.

Two guards sit around it. `align.py::_checked_index` refuses a timezone-naive, unsorted or duplicated index. `align.py::_reject_coarser` refuses an auxiliary whose bars outlast the target's, because a daily bar carried onto an hourly row **has not closed yet** when that hour's decision is taken - the judgement ADR-002 sec. 6 declined to make at the reader, since only here are the target and its interval known. Symbols are also iterated through `sorted(...)`: the original used `list(set(...))`, whose order varies between processes, so its column layout - and any PCA fitted on it - was irreproducible.

## 5. Staleness: carried is honest, fresh-looking is not

Carrying a value forward is legitimate. At 03:00 the last thing known about the S&P really *was* its 21:00 close, and a model deciding at 03:00 would have had exactly that. What is dishonest is losing track of how old it is, because "the S&P is at 4,500" and "the S&P was at 4,500, sixteen hours ago" are different statements and only one is true. So every auxiliary gets a `{symbol}_staleness_s` column (`align.py::STALENESS_SUFFIX`) recording the age in seconds of the value that row carries, and `align.py::SymbolAlignment` reports per symbol - never pooled, because the spread is wide enough that one number would describe nobody (ADR-003 sec. 2, on the reference panel):

| Symbol | Rows carried | Oldest |
|---|---:|---:|
| VIX | 8.0% | 74h |
| DXY | 7.1% | 75h |
| USDJPY | 0.1% | 33h |
| EURUSD, XAGUSD | 0.0% | 51h |

Two accounting details are easy to misread. `stale` counts every row whose age is above zero - `int((staleness.fillna(0) > 0).sum())` - because a threshold there would only hide the commonest case. And `missing` pools two things: rows before the symbol's history begins, *and* rows dropped for exceeding `max_staleness_seconds`, which `_carry_forward` also blanks in the staleness column via `age.mask(missing)`.

**Why a stale column is dangerous in a way a missing one is not.** A missing value is visibly absent; a model either drops the row or must be told what to do with it. A stale value is a plausible number that is simply not new, and it distorts *systematically*: a return computed on it is exactly zero, an RSI drifts toward 50, an ATR contracts toward nothing. Worse, the pattern is not scattered at random - a symbol is stale precisely when its venue is shut, so staleness tracks the hour of the day almost perfectly, and a tree fed those columns learns a session clock while the report calls it a macro signal. That is why `research/features/build.py::build_features` computes indicators on each symbol's **native grid first** and aligns second, and why the warm-up is dropped by position rather than by `dropna()`, which would also delete real target bars that a stale auxiliary left empty. It is also a live confound in the results: [STATUS-2026-09-whole-vs-focus](../status/STATUS-2026-09-whole-vs-focus.md) finds FOCUS (19 columns) beating WHOLE (179), best edge **+0.67%** against **+0.16%**, and cannot say whether nine extra markets carry no signal or carry signal that staleness destroys.

## 6. Provenance: a committed manifest over ignored data

`data/` is gitignored; `docs/status/data-manifest.json` is committed. The reasoning in `src/forecast_lab/ingest/manifest.py`: the series is regenerable, so committing it would be committing a cache - but "regenerable" is not "identical", and without a record a result and the data behind it drift apart with no symptom.

`manifest.py::entry_for` records, per file, a SHA-256 of the bytes (`manifest.py::sha256_of`), the size, the row count, the first and last timestamps, and a `source` label. `manifest.py::write` sorts entries by path and writes **no generation timestamp**, so identical data produces a byte-identical file - a manifest that always shows a diff is one nobody reads, and this one has to be read to be worth committing.

`manifest.py::verify`, behind `interfaces/cli.py::verify_command`, re-hashes what is on disk, comparing size first and only then the digest - which lets it name the dangerous case precisely: one edited digit leaves the file exactly as long, and the report says `same size, different content`. Every discrepancy is collected rather than raised at the first. A *changed* or *missing* file fails; an *untracked* one is reported and does not, because a data directory may legitimately hold something new.

It is a **command, not a test**: a pytest reading `data/` would skip itself in a clean clone and in CI, and a gate that skips is decoration - green while guaranteeing nothing (ADR-002 sec. 7).

What the manifest records:

| | files | rows | source label |
|---|---:|---:|---|
| `raw/` (canonical) | 11 | 606,250 | `dukascopy` |
| `reference/` | 34 | 462,674 | `reference` |

with `raw/XAUUSD_1H.csv` at **51,147** bars spanning 2018-01-01T23:00Z to 2026-08-26T14:00Z, against `reference/XAUUSD_1H.csv` at **23,181** bars from 2022-01-02. Two data roots share one manifest without erasing each other, through `interfaces/cli.py::_merge_entries`.

## 7. Where the documents and the code disagree

Three gaps, all real, all worth knowing before trusting a figure:

1. **The RUNBOOK's alignment expectations belong to a dataset the default command does not read.** [RUNBOOK sec. 3](../guides/RUNBOOK-getting-started.md) runs `align --target XAUUSD --timeframe 1H` with no `--dir` - which resolves to `DEFAULT_DATA_DIR = data/raw` - then says to expect "VIX carried on 8.0% of rows". VIX is excluded from `dukascopy.py::DEFAULT_SYMBOLS` and there is **no `raw/VIX_1H.csv` in the committed manifest**. Those percentages, and the 25.5% crypto figure repeated in ADR-003's implementation status, are the *reference* panel's. No committed sidecar reports staleness on `raw/` at all.
2. **There is no default staleness limit, and the ADR's phrasing invites a misreading.** ADR-003 sec. 3 says "there is no default: the caller picks one". In code, `align_to_target(..., max_staleness_seconds: int | None = None)` and the CLI's `--max-staleness` defaults to `0`, converted to `None`. The effective default is *carry forever*, not *choose*. The flag also appears never to have been exercised in a committed artefact: STATUS-2026-09-whole-vs-focus sec. 4 names the experiment it would enable and records "Not run here".
3. **Two deferrals are still deferred, and both concern coarser bars.** ADR-003 sec. 4 never built the shift that would let a daily auxiliary contribute safely, and ADR-002 sec. 5 never measured the venue's 4-hour and daily anchors - every anchor figure in that ADR is the reference venue's. Consistently, all 11 canonical manifest entries are `_1H`.

The lesson ADR-002's own implementation status draws is the one to carry forward: three of its decisions were documented as done and were not - the canonical dataset, the realised volatility, the fixture the gitignore described - and *"a Status line saying 'built' is a claim like any other"*. An audit found all three; nothing in the gates could have.
