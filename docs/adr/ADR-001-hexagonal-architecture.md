# ADR-001 - Four layers, one arrow inward, and the ports we deliberately do not build

- **Status:** Accepted - **built**. All four layers exist, the three gates are green, and every rule below is enforced by a test rather than by convention - including the one forbidding `research` to reach for data, which became a live gate when that layer gained its first module (ADR-003).
- **Date:** 2026-08-17
- **Context:** This repository re-engineers a notebook pipeline that predicts short-horizon market direction. The notebook's failures were not failures of algorithm choice - they were failures of **structure**: a merge that fabricated rows, features computed after alignment, model selection that read the test set, evaluation entangled with training. Every one of those is invisible in a linear script and mechanically preventable once the pieces have names and the dependencies point one way. The question this ADR settles is how much structure the project has actually earned: enough to make those failures impossible, and no more.

## Decision

### 1. Four layers, dependencies pointing inward

```
interfaces --+--> ingest ----+
             |               +--> contracts
             +--> research --+
```

```
src/forecast_lab/
    contracts/    Timeframe, SymbolSpec, SeriesMeta, RunConfig, errors
    ingest/       catalog, csv_reader, fetch, manifest
    research/     align, features/, labeling, evaluation/
    interfaces/   cli.py - the composition root
```

- **contracts** - validated data and the vocabulary of the domain. Imports nothing of ours.
- **ingest** - knows where bytes come from (a file on disk, an HTTP CDN) and turns them into series.
- **research** - alignment, features, labels, evaluation. Pure computation over data handed to it.
- **interfaces** - the CLI. The only place that imports concrete implementations, and the only place that touches the filesystem or the network.

*Why:* the golden rule is that **research never reaches for data**. It cannot open a file, cannot call an API, cannot know where anything lives. That is what makes an experiment reproducible from stored inputs, and an experiment that can quietly re-read the world is one whose results cannot be falsified.

**Trade-off:** more files than a script, and a composition root that has to wire things explicitly. For a one-off analysis this would be overhead. For a repository whose entire claim is that its numbers are trustworthy, the structure *is* the claim.

### 2. We name no ports, and that is the decision

A hexagonal design usually comes with a `ports/` package of abstract interfaces and a `storage/` layer behind them. **This one has neither**, and the omission is deliberate enough to be written down rather than merely noticed.

*Why:* an abstract interface earns its place when a **second implementation exists** - when there are two data sources to rank, or two persistence backends to swap. Here there is exactly one source feeding exactly one caller. An interface written now would be shaped around an imagined second case rather than a real one, and the odds of guessing that shape correctly are poor.

The distinction worth keeping: this is not an argument that ports are wrong. It is an argument that this project, today, has nothing to put behind one.

**Trade-off:** the day a second source appears, extracting the interface is a refactor. That refactor will be done with two real implementations in front of us instead of one imagined, which is the whole argument. The cost of waiting is an afternoon; the cost of guessing wrong is an abstraction shaped around a case that never arrived, and which everything downstream has already bent to fit.

### 3. Pydantic at the boundary; DataFrames inside

`contracts/` holds frozen, `extra="forbid"` Pydantic models for the run configuration, the symbol specification and the series metadata. Bars and the feature matrix travel as pandas DataFrames. Internal computation results are frozen dataclasses.

*Why:* validate once at the edge, then trust. A misnamed upstream field must **explode at the boundary**, not flow half-parsed into a result. But wrapping 51,000 bars across 11 symbols in Pydantic models would cost real time and buy nothing: pandas is the correct tool for numeric tabular data, and the bugs this project actually suffers from live in the frame's *shape and index*, which Pydantic would not see anyway.

**Trade-off:** the most dangerous object in the system, the feature matrix, is the least validated by the type system. That is why the guarantees about it are tests - no price levels survive, the two feature modes share an index, the alignment fabricates nothing - rather than annotations.

### 4. The dependency rule is a test, not a paragraph

`tests/test_layering.py` walks every module under `src/forecast_lab`, reads its imports statically (no importing, so it stays fast and side-effect free) and fails naming the file and the illegal arrow.

```python
ALLOWED = {
    "contracts":  frozenset(),
    "ingest":     frozenset({"contracts"}),
    "research":   frozenset({"contracts"}),
    "interfaces": frozenset({"contracts", "ingest", "research"}),
}
```

Two holes that the obvious implementation of this guard leaves open are closed here. Both were found by reading the guard's own code rather than trusting that it did what its name said:

- **Relative imports escape.** The natural implementation skips a node when `node.module is None`, so `from .. import ingest` is invisible to the guard. Resolved by reading `alias.name` when `module is None and level >= 2`.
- **Root modules are unguarded.** A `_layer_of()` helper returns `None` for a module sitting at the package root, so such a file is neither checked nor checkable, and it is exactly where a dependency leak would hide. The guard therefore fails on any `.py` at the package root other than `__init__.py`. This is also why **contracts is a package, not a `contracts.py`**: as a bare module it would sit in the blind spot *and* break the guard's own "every layer exists" assertion.

A third rule the import table cannot express, and which matters more here than either of them: **research may not open files.** No `open`, `read_csv`, `Path.glob`, or network client anywhere under `research/`. Enforced by its own test.

*Why:* the table constrains which *modules* talk to each other. It does not stop research from reading a path it hardcodes. The rule that keeps experiments reproducible is about I/O, not imports.

**Trade-off:** a static import check cannot see `importlib`, and a text scan for `open(` can be fooled. These are guards against drift and honest mistakes, not against determined circumvention.

### 5. Configuration lives in interfaces, if it ever appears

Phase 1 has no secrets and one path, so there is no settings module yet. When one is needed it goes in `interfaces/`, not at the package root.

*Why:* reading configuration **is** composition. And a root-level `settings.py` would sit in the guard's blind spot (sec. 4), and an unguarded module is exactly where an environment dependency leaks into research.

## Consequences

- Four packages, three arrows, and every layer has a real consumer today.
- What was cut is recorded rather than silently absent: no ports package, no storage layer, no settings library, no `.env.example`. In a research book, "not built yet, and why" carries the same weight as built.
- The layering guard runs in the same gate as the tests, so the architecture cannot erode quietly between reviews.
- Research being forbidden to read data means the CLI must load and pass frames explicitly. That is more wiring in one file, in exchange for the property that every research function is callable from a test with synthetic data and no filesystem.

## Implementation status

**Built (2026-08-18):** `contracts/` with `Timeframe`, `SymbolSpec`, `SeriesMeta` and the error hierarchy; `ingest/catalog.py`; `interfaces/cli.py` with the `symbols` command; `tests/test_layering.py` enforcing the dependency table and both holes of sec. 4. Three gates green, 52 tests passing.

**Live since 2026-08-22:** `research` gained its first module (ADR-003), so `test_research_never_reaches_for_data` stopped skipping and started enforcing. It immediately earned its keep in an unexpected direction - its list of I/O calls included `get`, which would have flagged every `dict.get` in the layer. A guard that cries wolf on ordinary code trains everyone to ignore it, so HTTP clients are now caught by their import instead.

**What building it taught, recorded because it is the kind of thing that gets re-learned:** the layering guard and the contract tests all passed while the CLI itself was broken - a Typer application holding a single command collapses into a bare command, so `forecast-lab symbols` failed with "unexpected extra argument". Fifty-two green tests said nothing about it. Running the thing is a separate act from testing it, and the acceptance criterion for every slice includes a real invocation for that reason.

**Deferred by design**, each to be revisited only when a live need appears: a source port (sec. 2), a storage layer (sec. 2), a settings module (sec. 5).
