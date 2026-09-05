# Chapter 6 - The architecture

This chapter answers a question that is not about statistics at all: **what stops the rest of this book from being wrong without anybody noticing?** Every earlier chapter makes a claim about a number, and those claims are checkable only because the code is arranged so that certain mistakes cannot be made quietly - a research function cannot read a file, a dashboard cannot become a second implementation, a new module cannot slip past the guard by not being on a list. This is the software design, and what each piece of it forbids.

## 1. Four layers, one arrow

```
interfaces --+--> ingest ----+
             |               +--> contracts
             +--> research --+
```

| Layer | May import of ours | What it is for |
|---|---|---|
| `contracts` | **nothing** | `errors`, `series`. The vocabulary of the domain. |
| `ingest` | `contracts` | `catalog`, `csv_reader`, `dukascopy`, `importer`, `manifest`. Knows bytes exist; always handed a path. |
| `research` | `contracts` | `align`, `labeling`, `splitting`, `baselines`, `costs`, `significance`, `walkforward`, `plots`, `features/`, `models/`. Pure computation over frames handed in. |
| `interfaces` | `contracts`, `ingest`, `research` | `cli` (2,472 lines), `__main__`, `runner`, `presentation`, `published`, `dashboard`. |

The middle column is not prose. It is `ALLOWED` in `tests/test_layering.py`, and `test_dependencies_point_inward` is parametrized over every one of the 38 `.py` files under `src/forecast_lab`, parsing each with `ast.parse` - nothing is imported - and failing with the file, the line number and the illegal arrow.

Two holes the obvious implementation leaves open are closed by name, both recorded in [ADR-001](../adr/ADR-001-hexagonal-architecture.md) sec. 4 as found by reading the guard's own code rather than trusting it. **Relative imports escape**: `from .. import ingest` parses as an `ImportFrom` with `module=None`, invisible to a guard that reads only `node.module`; `_imported_layers` reconstructs the target from `node.level` and `alias.name`. **Root modules are unguarded**: `_layer_of` returns `None` for a file at the package root, so it is neither attributed to a layer nor checkable against one - precisely where a leak would hide. Rather than special-case it, `test_no_python_modules_at_the_package_root` forbids the position, which is why `contracts` is a package and not a `contracts.py`.

## 2. `research` may not open files

The import table constrains which *modules* talk to each other. It says nothing about a module that hardcodes a path, and the rule that keeps an experiment reproducible is about I/O, not imports. So `test_research_never_reaches_for_data` is a separate test, and it is the rule this design exists to protect: **research receives frames; it does not go and get them.** A layer that can quietly re-read the world produces experiments that cannot be reproduced from stored inputs, and a result that cannot be reproduced cannot be falsified.

It works on the syntax tree, not on text, and that is load-bearing in both directions. `_io_name` flags a call to `open` or `urlopen`, an attribute call in `{read_csv, read_parquet, read_json, glob, rglob, iterdir, savefig}`, or an import of `{requests, httpx, urllib, pathlib, csv, sqlite3}`. Because it walks the AST, `research/__init__.py`'s own docstring - which says "no `open`, no reader, no HTTP client" - does not trip it, where a `grep` for `open(` would. In the other direction `.get` is deliberately **absent**: ADR-001 records that the first version included it and would have flagged every `dict.get` in the layer, and a guard that cries wolf on ordinary code trains everyone to ignore it.

`savefig` is on the list because of a rule that would otherwise be unenforceable: `research/plots.py` builds `Figure` objects and hands them back, and writing one is the composition root's job (`interfaces/cli.py::_write_figures`, [ADR-008](../adr/ADR-008-figures-are-built-in-memory.md)).

Three quarantines in the same file exist for a different reason: `mypy --strict` is only worth having while the untyped surface is small enough to audit, and every symbol crossing an untyped boundary arrives as `Any`. `ta` may be imported only by `research/features/technical.py`, the estimator stack only under `research/models/`, `matplotlib` only by the plotting module and the CLI that writes its output (see sec. 7); `arch` is deliberately *not* quarantined, because it ships `py.typed` (ADR-012). The guard also names its own limits: `importlib`, `__import__` and `sys.modules[...]` slip past an import-statement check. It guards against drift and honest mistakes, not determined circumvention.

## 3. Deny by default, and the guard that once guarded nothing

The most important property of the `interfaces` guard is its default. A new module under `interfaces/` is **forbidden until it is named**, not permitted until someone notices.

The first version was the other way round: it checked `runner.py` and `dashboard.py` by name. An audit found the walk-around. Streamlit discovers a `pages/` directory beside the entrypoint and renders each file in it as a page of the same app - so `interfaces/pages/verdict.py` importing `research` would need **no import statement in `dashboard.py`** for a filename allow-list to see. Every gate would stay green while [ADR-005](../adr/ADR-005-the-dashboard-runs-the-cli.md) sec. 1 was dead.

So the rule is inverted. `test_the_dashboard_cannot_reach_the_analysis_layer` walks `INTERFACES.rglob("*.py")`, skips only the two command-line entry points and `__init__.py`, and looks each remaining module up in `DASHBOARD_IMPORTS`. A module that is not there gets `set()` - an empty allow-list - so its first import from the package fails the gate instead of passing unseen. `test_a_new_module_under_interfaces_is_denied_by_default` closes the loop by failing on the mere *existence* of an unlisted module: a file that imports nothing yet must still be declared before it can grow. The table is short and every entry is a decision - `runner`, `presentation` and `published` may import **nothing** from `forecast_lab`; `dashboard` may import those three. When `published.py` arrived on 2026-09-02 it could not have shipped without appearing there.

An allow-by-default guard fails silently, which is the whole argument: its failure mode is a green gate, not a red one. This guard has been in exactly that state. A comment in `tests/test_layering.py` records it beside the `INTERFACES` constant - an earlier fix pointed one level off, **scanned a directory that did not exist, and passed vacuously**. Nothing about a passing test distinguishes "found no violations" from "looked nowhere", and it was caught only by planting the forbidden file and watching the guard stay green. A guard nobody has seen fail is a comment.

## 4. The CLI is the only entry point, and the dashboard runs it

`interfaces/cli.py` is the only module that resolves a path, decides which series answers `--target XAUUSD`, and hands frames to `research`. ADR-005 decided in Phase 1 - before a page existed - that the Streamlit dashboard would **invoke** `forecast-lab <command> --json` as a subprocess rather than import `research`. The alternative creates a second composition root, and the version that drifts is the one nobody runs the gates against: the page shows a number, the terminal shows a different one, and both are computed by code that passes its tests.

That is why every analysis command carries `--json`. `runner.py::JSON_CAPABLE` names the seven that do - `align`, `explore`, `features`, `baseline`, `train`, `validate`, `verdict` - and each payload is built by one dedicated function (`_baseline_payload`, `_validate_payload` and siblings), so the contract has a single visible location rather than being scattered through print statements. `symbols` and `verify` report to a human and exit; asking them for `--json` is a usage error, and the capability lives beside the allow-list so a panel cannot get it wrong.

`runner.py` is where the refusals are structural rather than incidental. `Invocation.__post_init__` validates before anything is spawned: `WRITING = {fetch, ingest}` is refused with a reason, `TERMINAL_ONLY = {dashboard}` by name because serving it from a panel would start a server that starts a server, and `FORBIDDEN_OPTIONS = {--figures}` because `explore` and `train` are read-only commands with a writing *option* - until that list existed, the guarantee rested on both functions returning early on `--json` before reaching their figure block, statement ordering rather than a gate. `REPORTING_EXITS = {"verify": {1}}` is the opposite correction: `verify` exits 1 on drift from the manifest, the single state it exists to detect, so treating that as an error made the panel fail in exactly the case it was built for.

**The published-result record**, `interfaces/published.py`, is the newest piece. Nine payloads sit committed under `docs/status/` - the sidecars every STATUS log and FINDINGS quote - and the dashboard was re-running every command anyway, two to four minutes on numbers already on disk. `catalogue()` reads them, `command_of()` infers which command wrote each from `SIGNATURES` (the sections only that builder emits), and a sidebar control offers **Published** (0.6 s) or **Run now**.

The rule that makes it safe is `find()` returning `None` rather than a near miss: a request is answered only by a payload whose own metadata agrees on target and timeframe, and on mode where both sides declare one. Serving gold's verdict to a reader who asked about the S&P would put a number under a caption that does not describe it - the failure this repository is a correction of, arriving through a convenience. `test_a_result_for_another_series_is_never_offered` and `test_a_result_for_another_mode_is_never_offered` hold it. None of this weakens the one-composition-root rule: a published payload is the output of the same command, committed - not a second implementation, and not a number typed into a page.

## 5. Errors, and why determinism is a correctness property

The hierarchy in `contracts/errors.py` is one base, `ForecastLabError`, with three contract violations beside it and sixteen more derived across `ingest` and `research`. It lives in `contracts` so `research` can reject a malformed frame without importing an outer layer, and nothing inside the engine swallows it: data that is wrong stops the run, because a pipeline that repairs its inputs silently produces results nobody can trace. `runner.py::RunnerError` and `published.py::PublishedError` are deliberately plain `RuntimeError`s - inheriting from `ForecastLabError` would be an import from `contracts`, which sec. 3 forbids those modules.

Anything that would silently produce a different number twice is treated as a bug, because a published figure that does not recompute is not evidence. Seeds are fixed at the point of use: `RANDOM_STATE = 42` for every estimator that takes one, `DEFAULT_SEED = 20260101` for the random baseline, `BOOTSTRAP_SEED` in `research/dependence.py` - which also exports `optimal_block` so `significance.py` cannot pick a block length by a different rule. `ingest/manifest.py::write` sorts entries and writes **no generation timestamp**, so identical data yields a byte-identical file. `ta==0.11.0` is pinned exactly for the same reason: it changes indicator values between releases.

The clearest case is a single argument. `research/models/catalogue.py::_random_forest` passes `n_jobs=1` where the original used `-1`. That is not a hyperparameter - `random_state` fixes every tree, so the forest is identical either way. What changes is the order in which 100 tree votes are summed, and floating-point addition is not associative. Measured: with `-1` probabilities differ by up to `3.3e-16` between runs and the JSON is not byte-reproducible; with `1` it is exact. The cost is 0.29 s against 3.04 s per fit, taking the command from six seconds to twelve. That trade is the repository's thesis expressed as a config value.

## 6. The three gates, and what the suite is organised around

```bash
uv run ruff check src tests
uv run python -m mypy --strict src tests
uv run python -m pytest -q
```

Run while writing this chapter, all three are green: ruff clean, mypy clean over 66 source files, **488 passed, 18 deselected in 23 s**. The `python -m` forms are not decoration - Smart App Control blocks the console scripts and the mypyc-compiled extensions inside the mypy wheel, removing a gate without saying so, which is why `pyproject.toml` sets `no-binary-package = ["mypy"]`.

506 tests are collected; the default run deselects the `network` (4) and `slow` (3) marks, because a suite nobody waits for is a suite nobody runs. The rest fall into three kinds:

- **Properties, not values.** `tests/unit/test_align.py` asserts that nothing from the future reaches a row and that no row exists which the target did not trade; `test_walkforward.py::test_no_fold_trains_on_its_own_future` and `test_splitting.py`'s purge tests make the same shape of claim. `test_contracts.py` pins what everything downstream may assume - a timestamp is never naive, a symbol is safe as a file name.
- **Contracts, not content.** `tests/unit/test_cli.py` checks that every command is reachable, that a malformed request exits 2 and an unanswerable one exits 1, and that each payload carries every key the dashboard indexes - walking nested paths, so a key missing from one configuration out of eighteen fails rather than hiding behind the first that has it. It deliberately does **not** re-assert research numbers.
- **Memorials.** `tests/regression/test_ffill_fabricates_labels.py` pins the defect at the centre of the re-analysis, on synthetic bars rather than vendor data. `test_cli.py::test_the_application_is_not_collapsed_into_a_single_command` pins the day a Typer app holding one command collapsed into a bare command, so `forecast-lab symbols` failed while 52 unit tests stayed green. `test_presentation.py::test_the_significance_of_a_p_value_survives` pins four p-values spanning an order of magnitude all printing as `0.0000`, in the one table whose subject is significance. And `test_a_new_module_under_interfaces_is_denied_by_default` pins the guard's own failure.

## 7. What it costs, and where the documents and the code disagree

The friction is real. There are more files than a script needs and a composition root that wires everything explicitly - `cli.py` is 2,472 lines largely because `research` is forbidden to fetch its own inputs. Every panel pays a subprocess: `baseline --json` measures **2.9 s end to end, of which 2.8 s is interpreter startup** and about 110 ms the analysis. The dashboard is coupled to the CLI's output *shape* - a narrower and more testable surface than a diffuse dependency on internal signatures, but not a free one, and the payload crosses that boundary as `Any`, so `mypy --strict` sees nothing beyond `json.loads`.

Four disagreements between what is written and what runs, found while checking this chapter:

1. **`docs/guides/engineering-conventions.md` rule 5 is stale.** It says `--json` exists on "`features`, `train` and `baseline` today" and that "`align` does not yet and should". `align`, `explore`, `validate` and `verdict` all have it; ADR-005 records that gap closing on 2026-08-30.
2. **`verdict` is missing from `COMMANDS` in `tests/unit/test_cli.py`**, whose own comment claims a command missing from that list is "worth failing over". Nothing cross-checks the list against the registered commands, so `forecast-lab verdict --help` is exercised by no test. ADR-005's audit reports exactly this omission as found; the payload test now covers `verdict`, the help test still does not.
3. **`train`'s payload keys are not pinned, and the dashboard indexes them.** `dashboard.py` reads `payload["selected_on_validation"]` and iterates `payload["scores"]` on `s["block"]`, but `train` is absent from `DASHBOARD_KEYS`, and `test_the_key_list_covers_the_commands_the_dashboard_runs` asserts only `set(DASHBOARD_KEYS) <= JSON_CAPABLE` - a direction that permits the gap - while its own comment says "the dashboard reads `train`". Renaming `selected_on_validation` would pass all three gates and break the Models page: precisely the failure ADR-005 sec. 3 says that test exists to prevent.
4. **A live consequence of the same gap.** `dashboard.py` renders unavailable models with `", ".join(payload["models_unavailable"])`, while `cli.py::_train_payload` builds that field as a list of `{"name", "reason"}` dicts, and `str.join` over dicts raises `TypeError`. It is unreachable where all six estimators load - the list is empty, so the branch is skipped - which is why every gate is green. It fires on exactly the Smart-App-Control machine this repository documents at length.

Smaller: the matplotlib quarantine's message says "matplotlib belongs in `research/plots.py`" while the check also exempts `interfaces/cli.py`, which imports pyplot to save figures - correct by ADR-008, but the message would misdirect. And `docs/INDEX.md` still says `test_cli.py` pins the payloads "with 35 tests" where collection reports 42.

**Was it worth it?** Yes, and the evidence is the shape of the failures. Every defect above lives where the guards do not reach - a stale sentence, an allow-list nobody cross-checked, a rendering path on the far side of `Any` - and not one can change a published number. The mistakes that *could* have, the forward fill and a second implementation of the analysis, are the two this architecture makes structurally hard. The price is a slower page and more wiring in one file. The return is that a reviewer can check any figure in this book by reading one command line and running it.
