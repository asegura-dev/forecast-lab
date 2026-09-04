# ADR-005 - The dashboard runs the command line; it does not reimplement it

- **Status:** Accepted - **built** 2026-08-31 (`interfaces/runner.py`, `interfaces/presentation.py`, `interfaces/dashboard.py`, and the tests and layering guards described below). Decided at Plan status on 2026-08-23 because it constrained what every command built in between had to emit. **Audited and corrected the same day** - see *What the audit found*.
- **Date:** 2026-08-23, built 2026-08-31
- **Follows:** [ADR-001](ADR-001-hexagonal-architecture.md) - which makes `interfaces/` the only layer that decides what to read and where to write, and therefore makes a second interface a real architectural question rather than a styling one.
- **Context:** The project needs a Streamlit dashboard - the results are tables and distributions, and a reader who will not clone a repository will still open a page. The obvious way to build one is to import `forecast_lab.research` from a Streamlit script and call the functions directly. That is the way this decision refuses, and the reason is worth writing down before the first page exists rather than after the second copy of the logic has drifted.

## Decision

### 1. Every dashboard panel is a subprocess invocation of `forecast-lab <command> --json`

The dashboard builds an argument list, runs the console script, parses the JSON on stdout, and renders it. It imports nothing from `forecast_lab` except, at most, the package name it is invoking.

*Why:* the alternative creates a second composition root. `interfaces/cli.py` is currently the only module that resolves a path, decides which series answers `--target XAUUSD`, and hands frames to `research`. A Streamlit script that imports `research` directly must redo all of that - and it will redo it slightly differently. The version that drifts is the one nobody runs the gates against, and the failure mode is specific and ugly: the dashboard shows a number, the CLI shows a different number, and both are computed by code that passes its tests.

There is also a claim this repository makes and should be able to demonstrate. It calls itself a reproducible measurement instrument, and every published figure carries the command that produced it. If the dashboard is a second implementation, that claim covers the CLI and quietly excludes the page most readers will actually look at. If the dashboard *is* the commands, then the page is a live demonstration of the claim: what is on screen was produced by the same invocation printed underneath it.

### 2. Each panel shows the command it ran

Beside every result, the literal argument list, copyable.

The first implementation rendered it *above* while four docstrings and the page's own caption said "underneath". The decision permits either, so the code was compliant and the prose was wrong; both now say **beside**, which is what a reader sees.

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

**A subprocess per panel is slower than a function call**, and the trade-off is accepted with a number attached. **The number in the first version of this ADR was wrong**, and measuring it changed where the cost turned out to be:

| | Measured 2026-08-31 |
|---|---:|
| `baseline --json`, end to end | **2.9 s** |
| ...of which interpreter startup and imports | **2.8 s** |
| ...of which the analysis itself | **~0.11 s** |
| `validate` | ~40 s |
| `verdict` (18 configurations, two bootstraps) | ~90 s |

The claim of 0.8 seconds understated it by more than three times, and it pointed at the wrong thing: **the cost is a fixed startup toll paid per invocation, not work that grows with the data.** That makes caching on the manifest hash the right answer rather than a workaround - and it names a second option this ADR did not have, which is deferring the heavy imports in `cli.py` so a command that needs pandas pays for it and one that does not, does not. Caching on the hash is the right key precisely because [ADR-002](ADR-002-data-source-and-symbol-set.md) already makes the hash the identity of the inputs.

**The dashboard becomes testable without Streamlit.** Its logic is "build an argument list, parse JSON", which a unit test can exercise directly. A dashboard that imported `research` would need Streamlit's own test harness to check anything at all.

> **Debt discharged 2026-08-30.** This paragraph originally continued: *"The payload shapes, however, are **not** pinned by anything yet: there is no test that exercises `interfaces/cli.py` ... That is a debt this ADR creates rather than one it inherits, and it has to close before the dashboard depends on them."* It is closed. `tests/unit/test_cli.py` holds **35 tests** over synthetic bars in a temporary directory - every command reachable from `--help`, every `--json` payload parsed, the documented exit codes (2 for a malformed request, 1 for a well-formed one the data cannot answer), and the keys the verdict rests on.

**What those tests deliberately do not assert:** the research numbers. Those belong to the module that computes them, and re-asserting an accuracy in a CLI test would create a second place to update whenever a measurement moves - which is precisely how a document goes stale while its gates stay green. The CLI tests check the *contract*; the unit tests check the *content*.

**The first test in that file is a memorial.** Early in this project a Typer application holding a single command promoted it to the root, so `forecast-lab symbols` failed with "unexpected extra argument" while 52 unit tests stayed green. The explicit callback that prevents it looks redundant enough to delete; now deleting it fails a gate.

**Section 1 is now a gate, not a paragraph.** `tests/test_layering.py` walks both modules' ASTs: `runner` may import nothing from `forecast_lab` at all, and `dashboard` may import only `runner`. Checked by injecting `from forecast_lab.research import build_features` into the page and confirming the guard fails - a guard nobody has seen fail is a comment.

**The console script cannot be relied on to be executable.** Windows Smart App Control blocks the generated `forecast-lab.exe`: it is an unsigned shim with no reputation, and `uv sync` rewrites it, so installing an optional extra was enough to make every spawn fail with `WinError 4551`. `entry_point()` originally checked only that the file *existed*. It now falls back to `python -m forecast_lab.interfaces.cli` when the spawn is refused, and remembers, so the retry is paid once. The same policy blocks `streamlit.exe`, which is why the RUNBOOK invokes it as `python -m streamlit`.

**A consequence of that, visible on screen:** where the shim is blocked, the command line each panel displays is the `python -m` form. It is the same entry point and is what a reader on that machine would have to type, so sec. 2's claim survives - the line shown is the line that ran.

**Tables choose their renderer at run time**, and the reason is a correction.

The first version said this: *"`pyarrow` ships a native library that the same
application-control policy blocks outright - and unlike a `.pyd`, which clears after a few
attempts, it stayed blocked across five."* **That was true of five retries and false of two
days.** The library imports now. The mechanism is binary *reputation*, which accrues with a
release's age - already recorded in this repository for LightGBM and XGBoost, and not
connected to this case at the time.

So neither state is assumed. `arrow_available()` probes once per process: where it answers
yes the page renders a sortable `st.dataframe`, where it answers no it falls back to the
Markdown table. Both paths format through `presentation`, so the two cannot disagree about
how a p-value is written - which is the failure that made the module split necessary in the
first place. The fallback is tested by forcing the probe to fail, because this machine can
no longer exercise it.

**The page is launched the way everything else here is run: `forecast-lab dashboard`.**
Not in the original decision, and the omission was visible the moment it shipped - every
other capability in this project is a `forecast-lab <command>`, and the page was a long
path that worked only from the repository root. The command resolves the script beside
itself and spawns it; it never *imports* it, which would drag Streamlit into every
`forecast-lab --help` and invert the dependency this ADR exists to fix.

**It is refused from inside the page, by name.** `dashboard` serves this page, so a panel
invoking it would start a server that starts a server. It sits in `TERMINAL_ONLY` beside
the writing commands rather than merely being left out of the allow-list, because
"unknown command" would invite someone to add it.

**Streamlit is an optional extra**, not a dependency. `uv sync` stays light for the CLI - which is what the gates run and what every published figure comes from - and `uv sync --extra dashboard` adds the page. The runner has no Streamlit import at all, which is what lets its 23 tests run without it.

## What the audit found

Four independent reviews of the built page - one hunting runtime defects, one checking this
ADR clause by clause, one testing every page through Streamlit's headless harness, one
asking what the page was failing to say - returned **21 confirmed defects**. Three of them
were in guards and tests written to prevent exactly the thing they failed to prevent, and
those are the ones worth recording.

**The payload-contract test asserted nothing.** `test_the_payload_shapes_the_dashboard_reads_are_documented`
promised in its docstring to fail when a payload key was renamed, "rather than as a KeyError
in a browser". Its body asserted that a set literal three lines above it was non-empty. It
never touched `cli.py`, a payload, or the dashboard - so sec. 3's entire argument, that the
payload is *"one narrow, explicit, **testable** surface"*, rested on a tautology. The
replacement runs each command against synthetic bars and walks every path the dashboard
indexes, including nested ones; renaming `break_even` to `breakeven` now fails it.

**The layering guard was scoped by filename, and Streamlit walks around that.** It checked
`runner.py` and `dashboard.py` by name. But Streamlit discovers a `pages/` directory beside
the entrypoint and renders each file in it as a page of the same app - so
`interfaces/pages/verdict.py` could import `research` freely, with **no import statement in
`dashboard.py`** for a filename allow-list to see. Every gate would stay green while sec. 1
was dead. The guard is now **deny-by-default** across the whole `interfaces` package except
the two command-line entry points, and an unlisted module may import nothing. Verified by
planting that exact file and watching the guard fail - which also caught the first fix being
vacuous, because it scanned a path that did not exist.

**`dashboard.py` had no tests, and a comment in it said otherwise** - claiming the layering
test imported it, when that test only parses its text. Every rendering defect the audit found
lived there. It now has its own suite, skipped when the optional extra is absent.

**Read-only rested on statement ordering.** `explore` and `train` are read-only commands with
a writing *option*: `--figures DIR` creates the directory and saves PNGs into it. `Invocation`
validated the command word and nothing else, so the guarantee held only because both commands
return early on `--json` *before* reaching their figure block. `FORBIDDEN_OPTIONS` makes the
refusal structural, and a test asserts the option it guards still exists in the CLI.

**Two panels fetched a result and discarded it.** `verify` and `symbols` have no `--json`, so
`run()` returns their rendered text - and both call sites ignored the return value, leaving
the Data page showing a command line with nothing above it. Worse: `verify` exits 1 when the
bytes have drifted from the manifest, which this ADR's own module docstring names as the one
gap caching cannot cover, so the safety net **failed in precisely the case it existed for**.
Exit 1 from `verify` is now a report rather than a failure.

**Three figures were typed into the page** that the same panel already fetched from a command -
including a headline that read "nine of eighteen" while the Verdict page two clicks away
computed thirteen. A string literal is worse than a second implementation: the manifest-hash
cache expires every panel, and it cannot expire a sentence.

**And the corrections to this document:** its test counts were one low in two places (23 and
35 against 24 and 36), and `verdict` - the payload the dashboard indexes twenty keys deep -
was absent from the CLI test suite's command list entirely.

## The module split, added after the audit

`runner` invokes, `presentation` formats, `dashboard` lays out. Three modules, one
responsibility each, and the two that are not Streamlit are testable without it.

*Why it was not built this way:* the first version put formatting inside `st.markdown` calls,
which is the shortest path and looks harmless. It is not. A defect that printed
`2.00e-06`, `1.70e-05`, `2.34e-05` and `3.52e-05` **all as `0.0000`** - four configurations
made indistinguishable in the one table whose subject is statistical significance - could not
be caught by any test that did not start a browser. Moving formatting behind functions that
return strings made it a three-line unit test.

*The rule that follows:* a number's format follows from what it *means*, so a caller names the
meaning (`Style.SIGNIFICANCE`, `Style.PERCENT`) and never the precision.

## Charts, and where a chart specification belongs

A Vega-Lite specification is a dict - **pure data** - so building one is not rendering, and
it lives in `presentation` beside the formatting where a test can check it without a
browser. `dashboard` only draws what it is handed.

*Why that matters here more than usual:* every spec is wrapped in a `layer`. Streamlit
marshals a spec's **top-level** `data` and `datasets` through Arrow; a layered spec carries
its data on the children, where the marshaller does not look, so the whole thing is
serialised as plain JSON and drawn client-side. On a machine where the native library is
blocked, that is the difference between a chart and a blank page - and it is asserted by a
test rather than trusted to a comment, because the comment was true and unverifiable.

**The chart the project did not have.** `train --figures` and `explore --figures` write
sixteen committed PNGs, and **none of them covers `validate` or `verdict`** - no command
produced those numbers until this month. The Verdict page now draws the one image the whole
finding reduces to: skill on one axis, money on the other, every configuration to the right
of zero and below it. A real edge that costs more to collect than it is worth, in a single
picture rather than two tables a reader has to hold in their head at once.

**A theme is committed** (`.streamlit/config.toml`), and its palette is the one
`presentation` uses for charts. Two files that disagreed about what "this survived" looks
like would be worse than neither.

**Three page tests are marked `slow` and excluded from the default gate.** They spawn the
analysis commands, and the verdict page takes four minutes. A suite nobody waits for is a
suite nobody runs; `pytest -m slow` opts in, exactly as `-m network` already did.

## The record, and why it took a reader to notice

Nine payloads are committed under `docs/status/` - the sidecars every STATUS log and
FINDINGS quote. **The dashboard had them on disk and re-ran every command anyway**: two
minutes for the Verdict page, four with every configuration, on numbers that were already
there. Nobody had noticed because the panels *worked*; they were only slow, and slow is
easy to accept from something that is computing.

A sidebar control now offers **Published** or **Run now**. Published reads the sidecar and
renders in **0.6 seconds** instead of two to four minutes.

*Why this does not weaken sec. 1.* The published payload is the output of the same command,
committed - not a second implementation, and not a number typed into the page. The panel
still shows the command line, and adds which file the figure came from and when it was
generated. Reading a recorded result is what a research log is *for*.

*The rule that makes it safe:* a request is answered only by a payload whose own metadata
agrees - same target, same timeframe, same mode. `find` returns nothing rather than the
nearest thing, because serving gold's verdict to a reader who asked about the S&P would put
a number under a caption that does not describe it, which is the failure this repository is
a correction of arriving through a convenience. A test holds that.

*Matched, not tabulated.* Each sidecar carries its own `target`, `timeframe` and `mode`, and
which command wrote it is inferred from the sections only that command emits. A sidecar
committed later for another symbol is found without editing any code - and the signatures
are asserted against the real files, so a renamed key fails a gate rather than quietly
making every panel slow again.

*Where the record is silent, the option is withdrawn.* All nine cover gold at one hour. On
any other series the control disappears and a caption says why, rather than offering
something every panel then fails to deliver.

**If this turns out to be wrong, the exit is visible.** The failure condition is a panel that genuinely needs an interactive object rather than a result - a live-refitting model, a slider that re-runs a fit at each tick. If that arrives, the answer is a new command that takes the parameter, not an import. If *that* stops working, this ADR gets superseded and the reason gets written down, which is the point of recording the decision at Plan status now.
