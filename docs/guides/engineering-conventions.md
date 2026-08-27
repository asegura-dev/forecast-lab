# Engineering conventions

How this repository is built and kept honest. These are working rules, not aspirations: most of them exist because breaking them once produced a wrong number that looked right.

## The three gates

No change is finished until all three are green:

```bash
uv run ruff check src tests
uv run python -m mypy --strict src tests
uv run python -m pytest -q
```

The two `python -m` are not decoration. Windows Smart App Control blocks the console scripts and, worse, the mypyc-compiled extensions inside the mypy wheel, with `DLL load failed: an application control policy blocked this file`. That silently removes a gate on any machine with it enabled, so `pyproject.toml` builds mypy from source (`no-binary-package`). Type checking gets roughly three times slower, which on a project this size is invisible; a gate that does not run is not.

## Architecture - four layers, one arrow inward

```
interfaces --+--> ingest ----+
             |               +--> contracts
             +--> research --+
```

### The hard rules

1. **`contracts` imports nothing of ours.** It is the vocabulary of the domain.
2. **`research` never reaches for data.** No `open`, no `read_csv`, no `Path.glob`, no HTTP client. It receives DataFrames and does not know where they came from. A research layer that can quietly re-read the world produces experiments that cannot be reproduced from stored inputs - and a result that cannot be reproduced cannot be falsified.
3. **The CLI is the only composition root.** The only place that imports concrete implementations, and the only place that *decides* what to read and where to write. `ingest` performs the reading and writing - that is its job - but it is always handed a path rather than choosing one.
4. **No `.py` at the package root** except `__init__.py`. A module there sits outside the layering guard, which is precisely where a dependency leak hides.
5. **Every analysis command offers `--json`** - `features`, `train` and `baseline` today - built by a dedicated payload function rather than scattered through print statements. `align` does not yet and should: it produces a substantive result the dashboard will need. The dashboard runs these commands instead of reimplementing them ([ADR-005](../adr/ADR-005-the-dashboard-runs-the-cli.md)), and scraping a Rich table would break on the first column that got wider - silently, because a truncated number is still a number.

Rules 1, 2 and 4 are enforced by [`tests/test_layering.py`](../../tests/test_layering.py), along with the quarantines that keep untyped dependencies in one module each. Rule 3 is a convention: the guard checks that `research` performs no I/O, not that `interfaces` is the only caller deciding paths. A reviewer forgets on a Friday; a gate does not - so the distinction between what is guarded and what is merely agreed is worth stating.

**Ports are not created in advance** (see [ADR-001 sec. 2](../adr/ADR-001-hexagonal-architecture.md)). An abstract interface earns its place when a second implementation exists. One written in anticipation is over-engineering in good handwriting.

## Data rules

- **A timestamp is the bar's OPEN.** Everything about alignment depends on it: an auxiliary bar opening at `s <= t` closes at `s + delta <= t + delta`, which is the instant the target's bar at `t` closes and the decision is taken. A symbol delivered on a shifted grid would inject look-ahead into every row, silently. Ingestion therefore **fails loudly** on a grid that does not fit; it never aligns anyway.
- **Indicators are computed on each symbol's native grid**, and only then reindexed onto the target. The other order fabricates zero returns on stale rows, and because staleness correlates with the hour of day, a tree model learns a session clock disguised as a macro signal.
- **No raw price levels in the feature matrix.** And "scale-free" is not "stationary": the distribution shift between splits has to be measured, not assumed.
- **Selection happens on validation.** The test set is evaluated once and the report records that it was spent.
- **No result is published without its baselines** - today the majority class of *train* applied blind, persistence, and a seeded random draw. The economic ones (buy & hold, always-long, always-flat) arrive with the cost model and are not built yet. An accuracy of 51.9% with no baseline beside it is misleading; that is exactly how the original project's headline number came to be believed.

## Document as you build, in the same step

| What | Where | Shape |
|---|---|---|
| A real **decision** | `docs/adr/ADR-NNN-*.md` | Status / Date / Context, then Decision (each point with its *Why* and its **Trade-off**), Consequences, and **Implementation status** |
| The **narrative** | `docs/NN-*.md` | Numbered chapters; a `## Trade-off` section is mandatory and `## Summary` closes |
| The **result** of an experiment | `docs/status/STATUS-*.md` | What was tried, what came out, an honest verdict, and `## Caveats` |
| An **operational procedure** | `docs/guides/RUNBOOK-*.md` | The how, including the mistakes that taught it |
| **Notable** changes | `CHANGELOG.md` | Dated, `Added`/`Changed`/`Fixed`, newest first |
| One line per file | `docs/INDEX.md` | Updated whenever a document is created |

An ADR's Status is a living cycle: **Plan -> Accepted -> Built**. It is updated in the ADR itself as the work lands, and *Implementation status* also records what went wrong.

Cross-references use **"Ch. N sec. M"** and **"ADR-00N sec. M"**. There is no documentation compiler: these are plain Markdown files and the convention is a writing convention.

**One paragraph, one line.** Markdown source is not hard-wrapped at a column limit; a paragraph occupies a single line and the editor wraps it visually. The usual argument for wrapping is that a one-word change then touches one line instead of a whole paragraph in the diff. That is a real benefit, and it does not survive contact with a WYSIWYG editor, which re-flows the file on save and undoes the convention silently - and a convention the tooling reverts is worse than none, because the two styles end up mixed in the same repository. The rendered output is identical either way; only diff granularity differs. Prose is plain ASCII, with no typographic dashes, arrows or section signs, so it survives a terminal on any codepage and greps without escaping.

## Language

English for everything shipped, commit messages included. The log is read by whoever opens the repository - GitHub shows the newest commit message above the file list - so a message that explains a decision is only worth writing in a language its reader has.

## Epistemic rules

- **Anchor every claim in the actual data.** Verify before asserting. When closing a step, re-read what was written and confirm that every path, symbol and number is true.
- **Null results are findings** and are reported faithfully. So is "there was no defect here".
- **A large, visible edge is almost always your own error**, not a discovery. A spectacular number is a red flag, not a result.
- **Measure before concluding.** No blind change, and no hunch dressed as a diagnosis.
- Every lesson is documented where it will be needed again and, where it applies, turned into a **regression test**.

## Version control

- **When:** after each verified piece of work, with the three gates green.
- **Granularity:** one commit per logical unit. Unrelated changes are not mixed.
- **Messages** explain **what** changed and **why**. Nothing generic (`update`, `fix`, `wip`).
- **Never commit** `.env`, `.venv/` or `data/`. Check `git status` before every commit.
