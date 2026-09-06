# RUNBOOK - running this from nothing

From a fresh clone to a feature matrix, with what every number on screen means and what to do when a step fails. Each command here has been run; the outputs quoted are real.

The order matters and is not arbitrary: **nothing downstream works until data exists on disk**, because `data/` is deliberately not committed.

## 1. The environment

```
uv sync
uv run forecast-lab --help
```

`uv sync` builds the virtual environment from `uv.lock`, so everyone gets identical versions of everything - which is the only way a published number can be recomputed. Python itself comes from uv's managed builds rather than from whatever is on `PATH` (`python-preference = "only-managed"` in `pyproject.toml`); there is a Microsoft Store Python on many Windows machines and a pin that means two different things is not a pin.

Check the three gates run before doing anything else. If they are not green on a fresh clone, the problem is the environment rather than your change:

```
uv run ruff check src tests
uv run python -m mypy --strict src tests
uv run python -m pytest -q
```

Expect `488 passed, 18 deselected`. The 18 are opt-in by design: fifteen network tests (sec. 11) and three dashboard pages that spawn the analysis commands - `pytest -m slow` runs those, and one of them takes four minutes.

**Note the `python -m` in front of mypy and pytest.** It is not decoration - see sec. 12.

## 2. Getting data onto disk

Nothing below sec. 2 works until this is done. There are two sources and they are not interchangeable.

**The engine's source**, downloadable by anyone with no API key:

```
uv run forecast-lab fetch
uv run forecast-lab verify
```

`fetch` pulls **both sides of the order book** and writes mid prices plus the spread of every bar. The spread is not decoration: it is the transaction cost, and the break-even accuracy that decides whether any edge is worth having is computed from it. It takes a while for that reason - two requests per symbol per year.

**Or the reference exports**, if you are reproducing the original project's baseline:

```
uv run forecast-lab ingest --from <path to the exports>
uv run forecast-lab verify
```

`ingest` reads every file *before* copying it, so `data/` only ever holds series the rest of the pipeline may trust. It reports what it ignored and why - a scan that quietly skips half a directory is how somebody models a subset they did not choose.

**`verify` re-hashes what is on disk against `docs/status/data-manifest.json`**, which is committed while the data is not. Editing one digit of one price leaves the file exactly as long and changes every number computed from it; the hash is what notices. It is a command rather than a test on purpose - a test reading `data/` would skip itself in a clean clone and report green while guaranteeing nothing.

Then confirm what arrived:

```
uv run forecast-lab symbols
```

## 3. Putting the symbols on one timeline

```
uv run forecast-lab align --target XAUUSD --timeframe 1H
uv run forecast-lab align --target XAUUSD --timeframe 1H --json
```

**What you are looking at.** The target's own bars define the timeline and nothing may extend it. Auxiliary symbols are read onto it by carrying their last known value forward, and each one reports how often that happened and how old the value got.

On the canonical dataset - which is what this command reads with no `--dir` - expect roughly: **DXY carried on 10.1% of rows**, the two indices around 1.5%, and everything else under 0.2%, over **51,147 rows and 76 columns**. Those are facts about trading calendars, not defects. What would be a defect is losing track of them, because "the S&P is at 4,500" and "the S&P was at 4,500, sixteen hours ago" are different statements and only one is true.

The reference exports give different figures - VIX on 8.0%, DXY on 7.1%, crypto absent from 25.5% because it starts a year late - and this guide quoted **those** against the canonical command for a fortnight, which is its own small illustration of the point. VIX is not in the canonical dataset at all ([ADR-002](../adr/ADR-002-data-source-and-symbol-set.md) drops it), so the figure could not have appeared no matter how the command was run.

## 4. Describing the data before modelling it

```
uv run forecast-lab explore --target XAUUSD --timeframe 1H
uv run forecast-lab explore --target XAUUSD --timeframe 1H --figures docs/status/figures
```

**What to read, and why each one is there.** Every panel exists because the original project's EDA got the same question wrong in a way that is invisible from the code.

- **Two normality tests, not one.** Both reject, and they mean opposite things. The price fails for having tails *shorter* than a normal - it is bounded and was trending - which licenses nothing. Returns fail for tails reaching ten standard deviations, which is why a Sharpe ratio's textbook confidence interval will be wrong later.
- **The correlation table, read left to right.** `On levels` is what the original reported; `On returns` is what a model sees. Gold against the S&P goes from +0.919 to +0.139. Watch WTIUSD and USDJPY: they change sign between the two columns.
- **The yearly table.** The `Rose` column drifts about three points across the sample, which is more than the effect anyone is trying to detect - so which years fall in the test block matters.

## 5. Labels, splits, and the numbers a model must beat

```
uv run forecast-lab baseline --target XAUUSD --timeframe 1H --dir data/reference
```

**What every line means**, because this output is the point of the project:

- **`22,167 of 23,181 bars labelled`** - the horizon is a timestamp, not a position. The missing 1,013 (4.37%) are bars with no bar exactly one hour later, so `close.shift(-1)` there would reach across two hours, or across fifty, while reporting an hourly move.
- **`Across those gaps price rose 59.35% of the time, against 50.87%`** - the largest directional skew in this dataset. It is not a strategy: those are exactly the hours that pay overnight financing. It is reported because measuring something and setting it aside is different from never looking.
- **`majority-class (UP, from train) 51.46% ... specificity 0.00%`** - fitted on train and applied blind. The zero specificity is the signature to remember: it is what a constant looks like when only accuracy is reported.
- **`random (train frequencies) 51.70%`** - a seeded coin flip, scoring **above** the honest baseline. Nothing has been discovered; it is the plainest demonstration that at this sample size the noise is the size of everything being argued about.
- **`A rule fitted on this block would gain 2.01% for free`** - the distance between the honest baseline and an oracle. Larger than the effect anyone is trying to detect.

Break-even is **53.49%** against the spread measured at the venue, and **51.92%** under an assumed 1 bp round trip. Nothing available without a model reaches either. Every analysis command reads the canonical dataset by default, so `train` computes this from the `spread` column and says so; pass `--dir data/reference` and it says instead that no spread exists, assumes 1 bp, and states the average move it measured that assumption against. **Which dataset produced a number is always in the line that prints it** - it had to be, because for a fortnight `train` and `verdict` defaulted to different ones.

## 6. The feature matrix

```
uv run forecast-lab features --target XAUUSD --timeframe 1H
uv run forecast-lab features --target XAUUSD --timeframe 1H --mode whole
```

Expect **22,982 rows x 19 columns** in focus mode and **x 199** in whole - and, critically, **the same index in both**. The original project's two modes covered different rows (24,232 against 22,441), so comparing them mixed a change of feature set with a change of sample.

**Reading the table:**

- **`Moved by (p99)`** decides. Every column is rebuilt on prices multiplied by ten; a column that moves is a price level. The verdict is the 99th percentile of the per-row deviation, not the maximum - see the ADR for why.
- **`Worst row`** is reported and never decides. A large worst beside a tiny p99 means a numerical instability, not a price level.
- **`ADF p` and `KPSS p` are diagnostics, never gates.** At n = 5,000 the ADF rejects a unit root on almost anything, and both tests are invalid under the heteroskedasticity and regime change that characterise this data. Columns where the two disagree are counted and reported rather than resolved by picking a favourite.

The analysis commands also take `--json`, because the dashboard runs them rather than reimplementing them:

```
uv run forecast-lab features --target XAUUSD --timeframe 1H --json
```

To check the pipeline is genuinely symbol-agnostic, point it somewhere else:

```
uv run forecast-lab features --target BTCUSD --timeframe 1H --mode whole
```

## 7. Fitting the models

```
uv run forecast-lab train --target XAUUSD --timeframe 1H
uv run forecast-lab train --target XAUUSD --timeframe 1H --no-pca
```

Six estimators x three representations (raw, PCA at 95% and 90%), about twelve seconds.

**Reading the output.** Two tables: validation first, then test. Selection happens on the validation table and the test table is scored afterwards - the order on screen is the order of operations, not a layout choice.

- **`Edge`** is the column to read first: accuracy minus what predicting UP every time scores on the same block. Green is positive, red is negative. The literal `always-UP (from train)` row at the bottom is the same comparison, spelled out.
- **`Spec.`** at 0.00% with high recall means the model is a constant. The command says so explicitly when it happens.
- **`Repr.`** shows the retained component count in brackets for PCA rows - `pca-90 (6)` means 90% of the training variance needed six components.

Expect the run to select **Random Forest [raw]** and report it beating the baseline by about **0.15 points** on test - a positive edge that is nowhere near the 53.49% this data's own spread demands.

On the reference exports the same command selects **XGBoost [raw]** at **-0.21 points**:

```
uv run forecast-lab train --target XAUUSD --timeframe 1H --dir data/reference
```

**Both are the project's central finding, and the pair is more useful than either.** One dataset hands you a small positive edge and the other a small negative one; neither comes close to paying for itself, which is the point. [STATUS 2026-08-24](../status/STATUS-2026-08-models.md) works through what a single run does and does not establish.

Add `--figures` to write the charts:

```
uv run forecast-lab train --target XAUUSD --timeframe 1H --figures docs/status/figures
```

Eight PNGs, four per block. Start with `edge-<block>.png`: every configuration as a bar, with the constant predictor and the break-even accuracy as vertical lines. A bar left of the red line is a model that lost to a rule with no parameters. `confusion-<block>.png` is the other one worth reading closely - if a model's DOWN row and UP row look the same, it is predicting UP at the same rate whether price rose or fell, which is what no signal looks like.

**The output is byte-reproducible.** Running it twice on the same data gives an identical `--json` payload, hash for hash. That is deliberate, and it doubled the command's runtime: `Random Forest` is fitted single-threaded, because summing 100 tree votes across cores lands on a different last bit each run. If you ever see the hashes differ, something is wrong - start there rather than with the numbers.

If a model cannot be loaded, the command prints it and continues - see sec. 12 for the reason that happens on Windows.

## 8. Scoring across the whole history

`train` scores one held-out block. `validate` scores nine years of it, and prints what size of edge the design could have seen.

```
uv run forecast-lab validate --target XAUUSD --timeframe 1H
uv run forecast-lab validate --target XAUUSD --timeframe 1H --json
uv run forecast-lab validate --target XAUUSD --timeframe 1H --folds 8 --rolling
uv run forecast-lab validate --target XAUUSD --timeframe 1H --pca      # 18 configurations
```

Six estimators x five folds, about forty seconds - the extra time is the bootstrap. It needs the canonical data (`data/raw`, the default here) - the reference exports are too short to cut into useful folds and carry no spread, so the break-even falls back to the assumed 1 bp and the command says so.

**Reading the output.** Three parts, in the order the argument runs:

- **The fold table** shows each train/test pair with the months it covers. Every training block ends before its test block begins; the purged bars in the line above it are the ones removed at each boundary because a label there reaches into the test window.
- **The pooled table** is the result, and it carries two different tests. Read `accuracy` and `baseline` *together*, never `edge` alone: under walk-forward every model's edge turns positive, and that is **not** the models improving - accuracy falls 0.21 points against the single split while the baseline falls 0.67, because averaging five stretches moves the majority class nearer a half. Then read `short by`, which is the other test: whether the model pays for its own trading.
- **`flip` and `held` are why the break-even column differs per row.** A model that changes position every 5.7 bars crosses the spread half as often as one changing every 2.6, and faces a threshold more than a point lower. The rows are sorted by `short by`, not by accuracy - the two orderings are different, which is the point of [ADR-012](../adr/ADR-012-turnover-and-dependence-are-measured.md).
- **The power lines** state the minimum detectable effect over the bars actually scored, and how often this design would see an edge large enough to pay for costs. Expect `0.62%` and `100.0%`.

Expect the run to put **Naive Bayes** closest, at 50.77% against its own 51.23% break-even - short by 0.46 points, 1.86 standard errors, with no model of six closer. Note that it is *fifth* by accuracy: HistGradientBoosting scores 51.23% and still falls further short, because it trades twice as often. [STATUS 2026-08-28](../status/STATUS-2026-08-turnover.md) works through what that does and does not establish.

**Read the last line the command prints.** It reports the serial dependence measured by a stationary bootstrap - expect an inflation around `0.97x`, meaning the standard errors above are already honest. Earlier versions of this project printed a caveat there instead, asserting the figures were optimistic by an unmeasured factor. They were not, and measuring it is what settled that.

**`--pca` widens the comparison to eighteen configurations** - the six estimators on raw features and on PCA at 95% and 90% of training variance, which is what the original project compared. It is off by default because it triples the runtime, and measured, every PCA row scores below its raw counterpart *and* trades more often, so it is penalised twice. Use it to check that claim, not to find a winner.

**`--rolling` answers a different question.** The default expanding window trains on all history to date, which is what a deployment would have. A fixed-length rolling window asks whether recent history predicts better than distant - a hypothesis about regime change rather than a validation design. Use it to explore, not to report.

## 9. The verdict

`validate` asks whether each model clears its costs. This asks the two questions underneath, which turn out to disagree.

```
uv run forecast-lab verdict --target XAUUSD --timeframe 1H
uv run forecast-lab verdict --target XAUUSD --timeframe 1H --long-only
uv run forecast-lab verdict --target XAUUSD --timeframe 1H --no-pca --json
```

Eighteen configurations, five folds and two bootstraps - about ninety seconds.

**Reading the output.** Three numbered sections and a verdict:

- **1. Is there skill?** Pesaran-Timmermann against *independence*, not against a coin flip. The `independent` column is what the two marginals produce with no information passing between them - a constant predictor scores exactly zero against it. The `after Holm` column is the one that counts: sixteen configurations are significant at 5% and **nine survive** the correction for having tried eighteen.
- **2. Is it worth anything?** Strategy returns net of the venue's own spread, against a benchmark of holding the asset. Expect **0 of 18** on both counts.
- **3. Does the best survive having been the best?** Hansen SPA, Romano-Wolf StepM, and the Deflated Sharpe. Expect p = 0.761, nothing rejected, DSR 2.3e-10.

The verdict to expect: **a real directional edge, worth less than nothing.** Nine configurations carry a robust signal of about one point; traded, they turn buy-and-hold's +100.9% into -86.9%.

**`--long-only` is the friendlier framing** - long-or-flat instead of long-or-short, so half the turnover and a wrong call merely forgoes a move instead of taking it backwards. It exists so the negative result cannot be blamed on the harsher one. It does not change the count.

**What the verdict does not settle**, and the command says so on its last line: slippage, the rollover surcharge and the overnight swap are unmodelled. Each raises the bar, so they cannot rescue a negative result - but a positive one would have needed them first.

## 10. The dashboard

```
uv sync --extra dashboard
uv run forecast-lab dashboard
uv run forecast-lab dashboard --port 8080 --headless
```

Seven pages, each running the same commands you have been running and printing the command
line beside the result. Nothing here is a second implementation: `runner.py` imports
nothing from the package at all and `dashboard.py` imports only `runner`, which two guards
in the layering suite enforce ([ADR-005](../adr/ADR-005-the-dashboard-runs-the-cli.md)).

**The command resolves the script beside itself**, so it runs from any directory and
from an installed package rather than only from the repository root - and it invokes
`python -m streamlit` rather than the `streamlit` console script, for the same reason
sec. 1 uses `python -m mypy`: Smart App Control blocks unsigned shims. The dashboard
handles the equivalent problem for its own entry point automatically - if
`forecast-lab.exe` is blocked, it falls back to `python -m` and the command lines on
screen show that form.

**`forecast-lab dashboard` cannot be reached from inside the page.** It serves the page;
running it from a panel would start a server that starts a server, so the runner refuses
it by name.

**It is read-only.** `fetch` and `ingest` are refused by the runner itself, not merely
absent from the interface - and so is any option that writes, such as `--figures`. A page
a stranger can load cannot start a download or overwrite `data/` (ADR-005 sec. 4).

**Tables render as a sortable grid where `pyarrow` loads and as Markdown where it does
not**, which on Windows depends on whether Smart App Control has let the native library
through yet. Nothing to configure; the page probes once and picks.

**Start on Published.** The sidebar's `Results` control chooses between the payloads
committed under `docs/status/` - the ones this project's documents quote - and a fresh
run. Published renders the Verdict page in under a second; every panel says which file
its figures came from. The control only appears for a series the repository has a
committed result for, which today is gold at one hour.

**The Verdict page is the slow one when you ask it to recompute.** It runs `validate` and `verdict` end to end - about
two minutes over six configurations, four over eighteen. The sidebar's **Configurations**
control chooses: eighteen matches the published figures, six is raw features only and three
times faster. Results are cached on the manifest hash, so a second visit is instant and a
`fetch` invalidates everything.

**If a panel reports a failure, read the command line under it and run that.** The page is
those commands, so anything it cannot do you can reproduce in a terminal in one paste - which
is the whole point of showing the line.

## 11. The network tests

Excluded from the gates so the default run is hermetic - no network, any OS, fast. They hold the venue to its side of the contract: that hourly bars still open exactly on the hour (the invariant the whole alignment design rests on), that both sides combine into a positive spread, and that every mapped instrument still exists.

```
uv run python -m pytest -m network
```

Run them when the data source misbehaves or before trusting a fresh `fetch`.

## 12. When something fails

**`DLL load failed ... an application control policy blocked this file`** - Windows Smart App Control blocking an unsigned binary extension. Two forms:

- *On mypy*, permanently. Its published wheel ships mypyc-compiled `.pyd` extensions that SAC refuses outright, which silently removes one of the three gates. `pyproject.toml` pins a source build (`no-binary-package = ["mypy"]`), and mypy and pytest are invoked as `python -m` so the module is resolved from the environment rather than through a launcher shim.
- *On scipy, for the first few runs after a fresh `uv sync`*, then never again. SAC evaluates each unknown binary's reputation asynchronously and blocks it while it asks - **and it does so one file at a time**. Observed on a fresh clone: the first run failed on `_sparsetools`, `_moduleTNC`, `_zeros` and `pyduccfft` with 8 collection errors; the second failed on `rcont` alone with 2; the third passed all 218 with nothing reinstalled between them. **Re-run before diagnosing anything.**

**`error: Failed to spawn: forecast-lab ... (os error 4551)`** - the same policy, refusing the launcher itself. The `forecast-lab` console script is a generated `.exe` shim, and a freshly built binary has no reputation. Measured here: a shim installed days ago runs fine while one regenerated by `uv sync` minutes earlier stays blocked across repeated attempts - so unlike the extensions above, waiting is not a reliable fix.

Go through the interpreter instead, which is already a trusted binary:

```
uv run python -m forecast_lab.interfaces features --target XAUUSD --timeframe 1H
```

Every command works identically either way. Substitute `python -m forecast_lab.interfaces` for `forecast-lab` anywhere in this guide.

**`No such directory: data\raw`** - `data/` is gitignored, so a fresh clone has none. Go back to sec. 2. This also catches out a second clone of this repo on the same machine: the code is in git, the data is not.

**`No 1H series for XAUUSD in data\raw`** - you fetched into one directory and are reading from another. `fetch` writes to `data/raw`, `ingest --from <path>` writes to `data/reference`, and every command that *reads* a series takes `--dir`.

**`verify` says `EXTENDED`** - the common case, and not a fault. The series extends forward in time, so a fetch a week after the manifest was cut adds bars to every file. The command re-hashes the *recorded prefix* of each one and, when it matches, says so and exits 0: the bytes a published number came from are still there, at the front of the file. **The published figures stay pinned to the 2026-08-26 snapshot**, which is what the committed manifest describes, and anything you compute now simply uses more data than that.

This guide said all of that in prose for a fortnight while the command itself printed `CHANGED ... no longer reproducible` for the same event. A warning that appears on every routine fetch is one nobody reads, which is the whole reason the two cases were separated ([ADR-002 sec. 7](../adr/ADR-002-data-source-and-symbol-set.md)).

**`verify` says `REVISED` or `MISSING`** - this one matters. The bytes the manifest recorded are no longer on disk, so a published number cannot be reproduced from this data. Two causes:

1. **The venue restated a bar.** Rare, and worth looking at rather than accepting. Note that a revision can arrive *inside* a file that also grew - which is why the check re-hashes the prefix instead of trusting that longer means append-only.
2. **A file was edited.** Editing one digit of one price leaves the file exactly as long and changes every number computed from it.

**Do not update the manifest to match**, which records the discrepancy as the truth. To reproduce a published figure, restore the snapshot the manifest describes. To move the project forward onto newer data, regenerate the numbers *and* the manifest together, in one commit - never the manifest alone.

Checked on the real case: extending the sample by 52 bars moved the count of configurations with significant skill from 9 to 13 and left the economic verdict identical at 0 of 18. That the conclusion survives a data refresh is worth knowing; that the *numbers* move is why they are pinned.

**Feature counts changed and you did not change the code** - check the `ta` version. It is the only dependency pinned exactly (`ta==0.11.0`) because it changes indicator values between releases; `uv lock --upgrade` would walk past a range pin and move every published number without touching a line of ours.

**`LightGBM not run: blocked by an application control policy`** - the same policy again, and here it is *not* transient. LightGBM and XGBoost load a native DLL through `ctypes`, which the policy refuses outright for a binary with no established reputation. Measured: **4.7.0 and 3.4.1 are blocked; 4.6.0 and 3.0.5 load first try**. Those two versions are pinned exactly in `pyproject.toml` for that reason, so the failure mode to watch for is a well-meaning `uv lock --upgrade` silently removing the two most interesting models from the comparison. The command never hides it - it names every model it could not run.

**The OneDrive exports fail to read** - Files On-Demand leaves placeholder stubs on disk. Open the folder in Explorer and let it hydrate before pointing `ingest --from` at it.

## 13. What is not built yet

`verdict` is where the pipeline stops, and the research question is answered: a real directional edge, worth less than nothing. Nine of eighteen configurations survive Holm; none makes money.

Three things are still owed, each named in [ADR-011](../adr/ADR-011-power-before-verdict.md):

- **A single `FINDINGS` document** a reader can land on instead of assembling the answer from thirteen ADRs and eight STATUS logs.
- **The gap benchmark.** Price rises through the venue's pauses 56-59% of the time, which clears 53.49% on its face - but those are exactly the hours that pay overnight financing. Until the swap is modelled, it stays a measurement rather than a strategy.
