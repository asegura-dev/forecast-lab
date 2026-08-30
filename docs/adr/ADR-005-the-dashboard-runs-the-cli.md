# ADR-005 - The dashboard runs the command line; it does not reimplement it

- **Status:** **Plan** - decided now because it constrains what every command built from here on has to emit. Built in Phase 4.
- **Date:** 2026-08-23
- **Follows:** [ADR-001](ADR-001-hexagonal-architecture.md) - which makes `interfaces/` the only layer that decides what to read and where to write, and therefore makes a second interface a real architectural question rather than a styling one.
- **Context:** The project needs a Streamlit dashboard - the results are tables and distributions, and a reader who will not clone a repository will still open a page. The obvious way to build one is to import `forecast_lab.research` from a Streamlit script and call the functions directly. That is the way this decision refuses, and the reason is worth writing down before the first page exists rather than after the second copy of the logic has drifted.

## Decision

### 1. Every dashboard panel is a subprocess invocation of `forecast-lab <command> --json`

The dashboard builds an argument list, runs the console script, parses the JSON on stdout, and renders it. It imports nothing from `forecast_lab` except, at most, the package name it is invoking.

*Why:* the alternative creates a second composition root. `interfaces/cli.py` is currently the only module that resolves a path, decides which series answers `--target XAUUSD`, and hands frames to `research`. A Streamlit script that imports `research` directly must redo all of that - and it will redo it slightly differently. The version that drifts is the one nobody runs the gates against, and the failure mode is specific and ugly: the dashboard shows a number, the CLI shows a different number, and both are computed by code that passes its tests.

There is also a claim this repository makes and should be able to demonstrate. It calls itself a reproducible measurement instrument, and every published figure carries the command that produced it. If the dashboard is a second implementation, that claim covers the CLI and quietly excludes the page most readers will actually look at. If the dashboard *is* the commands, then the page is a live demonstration of the claim: what is on screen was produced by the same invocation printed underneath it.

### 2. Each panel shows the command it ran

Above or below every result, the literal argument list, copyable.

*Why:* it converts the dashboard from a thing that displays conclusions into a thing that teaches how to obtain them, and it makes any disagreement between page and terminal immediately checkable by the reader rather than only by the author. It costs one line of layout.

### 3. JSON is a contract, and it is versioned in the ADRs like any other

A command's `--json` payload has a stable shape. Renaming a key or changing a unit is a breaking change and goes in the CHANGELOG.

*Why:* the coupling this decision creates is real and should be named honestly. The dashboard is now coupled to the *output shape* of the CLI. That is a far better place for the coupling than the alternative - it is one narrow, explicit, testable surface instead of a diffuse dependency on every internal signature - but "better" is not "free". The payload is built by a dedicated function per command (`_baseline_payload`, and its siblings as they arrive) so that the contract has a single visible location rather than being scattered through print statements.

**Parsing the rendered table is not an option**, and it is worth saying because it is the shortcut that would otherwise appear. Rich chooses column widths from the terminal, wraps long cell values, and formats numbers for reading. A dashboard scraping that output would break on the first column that got wider, and it would break silently - a truncated number is still a number.

### 4. The dashboard does not fetch, ingest, or write

Read-only commands only. Anything that downloads or writes to `data/` stays a deliberate act at a terminal.

*Why:* a page that a stranger can load should not be able to start a multi-gigabyte download or overwrite a series on the machine hosting it. And the provenance argument from [ADR-002](ADR-002-data-source-and-symbol-set.md) sec. 7 depends on knowing when data changed and why; a button that refreshes it makes the manifest a record of whoever clicked last.

## Consequences

**A constraint on everything built from here.** Every analysis command must offer `--json`, and the payload must carry the numbers rather than sentences about them. That is true of `baseline`, `features`, `explore`, `train` and `validate`; **`align` was the one gap and closed on 2026-08-30**, ahead of Phase 4 - a rule with one standing exception is a convention. Cheap when applied at the moment a command is written, expensive when retrofitted across a dozen of them, which is why this is decided in Phase 1 for a thing built in Phase 4.

**A subprocess per panel is slower than a function call**, and the trade-off is accepted with a number attached: `baseline` runs in 0.8 seconds on the full reference series, which is well inside what a page can absorb behind a spinner. The moment a command exceeds that, the answer is to make the command faster or to cache its JSON keyed by the manifest hash - not to bypass it. Caching on the hash is the right key precisely because [ADR-002](ADR-002-data-source-and-symbol-set.md) already makes the hash the identity of the inputs.

**The dashboard becomes testable without Streamlit.** Its logic is "build an argument list, parse JSON", which a unit test can exercise directly. A dashboard that imported `research` would need Streamlit's own test harness to check anything at all.

> **Debt discharged 2026-08-30.** This paragraph originally continued: *"The payload shapes, however, are **not** pinned by anything yet: there is no test that exercises `interfaces/cli.py` ... That is a debt this ADR creates rather than one it inherits, and it has to close before the dashboard depends on them."* It is closed. `tests/unit/test_cli.py` holds **35 tests** over synthetic bars in a temporary directory - every command reachable from `--help`, every `--json` payload parsed, the documented exit codes (2 for a malformed request, 1 for a well-formed one the data cannot answer), and the keys the verdict rests on.

**What those tests deliberately do not assert:** the research numbers. Those belong to the module that computes them, and re-asserting an accuracy in a CLI test would create a second place to update whenever a measurement moves - which is precisely how a document goes stale while its gates stay green. The CLI tests check the *contract*; the unit tests check the *content*.

**The first test in that file is a memorial.** Early in this project a Typer application holding a single command promoted it to the root, so `forecast-lab symbols` failed with "unexpected extra argument" while 52 unit tests stayed green. The explicit callback that prevents it looks redundant enough to delete; now deleting it fails a gate.

**If this turns out to be wrong, the exit is visible.** The failure condition is a panel that genuinely needs an interactive object rather than a result - a live-refitting model, a slider that re-runs a fit at each tick. If that arrives, the answer is a new command that takes the parameter, not an import. If *that* stops working, this ADR gets superseded and the reason gets written down, which is the point of recording the decision at Plan status now.
