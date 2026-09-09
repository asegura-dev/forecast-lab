# ADR-014 - Reproducing a result is a command, the same way verifying the data is

**Status:** Built 2026-09-09.

## Context

`verify` answers one half of the reproducibility claim: are the bytes on disk the ones a published number was computed from? It has existed since Phase 1 and it works.

Nothing answered the other half. Nine payloads sit committed under `docs/status/` - the figures every ADR, STATUS log and chapter of the book quotes - and **there was no way to ask whether they still follow from the data.** Checking one meant reconstructing its invocation by reading a STATUS log and hoping, because no payload recorded how it was made. Reproducing `model-comparison.json` required knowing it came from `--dir data/reference`, which was written down nowhere.

That gap is not theoretical. Three defects this project has already found lived in it:

- The **return-accounting error** changed every figure in `verdict-canonical.json`. Nothing could have listed what else moved; the blast radius had to be worked out by hand.
- **`features`, `explore` and `train` defaulted to a different dataset** than `validate` and `verdict` for a fortnight. Each command was correct about its own data, so nothing was inconsistent - and nothing could compare them.
- **`f1` and `negative_predictive_value` were added to the score table**, so every committed payload silently stopped matching what the code produces. Discovered only by regenerating them for an unrelated reason.

## Decision

**Every payload records the command line that produced it**, under a `run` key: the command, `sys.argv[1:]` verbatim, and when. **`forecast-lab reproduce` re-runs each committed payload from its own record and reports whether the result still follows.**

*Why `sys.argv` rather than a reconstruction from parameters.* A reconstruction is a second implementation of the argument parser, and it drifts: it would have gone on emitting `--dir data/reference` for three commands after their default changed, which is precisely the class of error the record exists to catch. What the operator actually typed cannot be wrong about what was run.

*Why re-run rather than recompute in-process.* The composition root is the CLI ([ADR-005](ADR-005-the-dashboard-runs-the-cli.md)), and what a reader can check is what a reader can type. Calling the payload builders directly would verify a path nobody uses.

*Why drift is reported in three kinds, not one.* "The payload changed" covers three situations that mean different things:

| | What it means | What to do |
|---|---|---|
| **rows grew** | the data extended; the committed figure describes an earlier snapshot | nothing - that is what pinning is for |
| **fields added** | the code records more than it did; no published number moved | regenerate when convenient |
| **a figure moved** | the same span now produces a different answer | stop and find out why |

Collapsing them would make the report fire on every routine addition, and **a check that fires on everything is one nobody reads** - the same failure `verify` had until it learned to tell a fetch from a rewrite ([ADR-002 sec. 7](ADR-002-data-source-and-symbol-set.md)).

*Why provenance is excluded from the comparison by construction.* `generated_at` is a clock reading. Two correct runs of one command differ in it and in nothing else, so a comparison that included it would report drift always.

## Consequences

**A simplification fell out.** `interfaces/published.py` inferred which command wrote a payload from a hand-kept table of key sets, with a test to keep the table honest. `reproduce` needed the real command line anyway - a key set cannot be re-run - so recording it made the table redundant, and it is gone. Declared beats inferred, and one mechanism beats two.

**All nine payloads were regenerated** to carry the record. Verified before installing: five reproduced byte-identically, two differed only in `generated_at`, and two differed only by the two metrics added since. **No published figure moved.**

**It is slow**, because it is honest: fifteen minutes for all nine, most of it `verdict` and `validate`. `--only <command>` scopes it.

**Drift is not an error exit.** The expected state after any `fetch` is drift, and an exit code that treats the normal case as failure is how a check gets removed from CI.

**What this does not do:** it re-runs commands, so it cannot detect that a *command* is wrong - only that its output changed. A defect present when the payload was committed and still present today reproduces perfectly. That is the limit of every reproducibility check, and it is why the gates and the reviews exist alongside it.
