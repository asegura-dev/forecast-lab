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

Expect `299 passed, 15 deselected`. The 15 are the network tests, opt-in by design (see sec. 9).

**Note the `python -m` in front of mypy and pytest.** It is not decoration - see sec. 10.

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
```

**What you are looking at.** The target's own bars define the timeline and nothing may extend it. Auxiliary symbols are read onto it by carrying their last known value forward, and each one reports how often that happened and how old the value got.

Expect roughly: VIX carried on 8.0% of rows, DXY on 7.1%, the currencies under 0.2%, and crypto absent from 25.5% because it starts a year late. Those are facts about trading calendars, not defects. What would be a defect is losing track of them, because "the S&P is at 4,500" and "the S&P was at 4,500, sixteen hours ago" are different statements and only one is true.

## 4. Describing the data before modelling it

```
uv run forecast-lab explore --target XAUUSD --timeframe 1H --dir data/reference
uv run forecast-lab explore --target XAUUSD --timeframe 1H --dir data/reference --figures docs/status/figures
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

Break-even is **51.92%** under an assumed 1 bp round trip, and **53.49%** against the spread measured at the venue. Nothing available without a model reaches either. On a series carrying a `spread` column the `train` command computes this itself and says so; on the reference exports, which have none, it falls back to the assumption and labels it.

## 6. The feature matrix

```
uv run forecast-lab features --target XAUUSD --timeframe 1H --dir data/reference
uv run forecast-lab features --target XAUUSD --timeframe 1H --dir data/reference --mode whole
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
uv run forecast-lab features --target BTCUSD --timeframe 1H --dir data/reference --mode whole
```

## 7. Fitting the models

```
uv run forecast-lab train --target XAUUSD --timeframe 1H --dir data/reference
uv run forecast-lab train --target XAUUSD --timeframe 1H --dir data/reference --no-pca
```

Six estimators x three representations (raw, PCA at 95% and 90%), about twelve seconds.

**Reading the output.** Two tables: validation first, then test. Selection happens on the validation table and the test table is scored afterwards - the order on screen is the order of operations, not a layout choice.

- **`Edge`** is the column to read first: accuracy minus what predicting UP every time scores on the same block. Green is positive, red is negative. The literal `always-UP (from train)` row at the bottom is the same comparison, spelled out.
- **`Spec.`** at 0.00% with high recall means the model is a constant. The command says so explicitly when it happens.
- **`Repr.`** shows the retained component count in brackets for PCA rows - `pca-90 (6)` means 90% of the training variance needed six components.

Expect the run to select **XGBoost** and report it losing to the baseline by about 0.21 points. That is the project's central finding reproduced from data, and [STATUS 2026-08-24](../status/STATUS-2026-08-models.md) works through what it does and does not establish.

Add `--figures` to write the charts:

```
uv run forecast-lab train --target XAUUSD --timeframe 1H --figures docs/status/figures
```

Eight PNGs, four per block. Start with `edge-<block>.png`: every configuration as a bar, with the constant predictor and the break-even accuracy as vertical lines. A bar left of the red line is a model that lost to a rule with no parameters. `confusion-<block>.png` is the other one worth reading closely - if a model's DOWN row and UP row look the same, it is predicting UP at the same rate whether price rose or fell, which is what no signal looks like.

**The output is byte-reproducible.** Running it twice on the same data gives an identical `--json` payload, hash for hash. That is deliberate, and it doubled the command's runtime: `Random Forest` is fitted single-threaded, because summing 100 tree votes across cores lands on a different last bit each run. If you ever see the hashes differ, something is wrong - start there rather than with the numbers.

If a model cannot be loaded, the command prints it and continues - see sec. 10 for the reason that happens on Windows.

## 8. Scoring across the whole history

`train` scores one held-out block. `validate` scores nine years of it, and prints what size of edge the design could have seen.

```
uv run forecast-lab validate --target XAUUSD --timeframe 1H
uv run forecast-lab validate --target XAUUSD --timeframe 1H --json
uv run forecast-lab validate --target XAUUSD --timeframe 1H --folds 8 --rolling
```

Six estimators x five folds, about forty seconds - the extra time is the bootstrap. It needs the canonical data (`data/raw`, the default here) - the reference exports are too short to cut into useful folds and carry no spread, so the break-even falls back to the assumed 1 bp and the command says so.

**Reading the output.** Three parts, in the order the argument runs:

- **The fold table** shows each train/test pair with the months it covers. Every training block ends before its test block begins; the purged bars in the line above it are the ones removed at each boundary because a label there reaches into the test window.
- **The pooled table** is the result, and it carries two different tests. Read `accuracy` and `baseline` *together*, never `edge` alone: under walk-forward every model's edge turns positive, and that is **not** the models improving - accuracy falls 0.21 points against the single split while the baseline falls 0.67, because averaging five stretches moves the majority class nearer a half. Then read `short by`, which is the other test: whether the model pays for its own trading.
- **`flip` and `held` are why the break-even column differs per row.** A model that changes position every 5.7 bars crosses the spread half as often as one changing every 2.6, and faces a threshold more than a point lower. The rows are sorted by `short by`, not by accuracy - the two orderings are different, which is the point of [ADR-012](../adr/ADR-012-turnover-and-dependence-are-measured.md).
- **The power lines** state the minimum detectable effect over the bars actually scored, and how often this design would see an edge large enough to pay for costs. Expect `0.62%` and `100.0%`.

Expect the run to put **Naive Bayes** closest, at 50.77% against its own 51.23% break-even - short by 0.46 points, 1.86 standard errors, with no model of six closer. Note that it is *fifth* by accuracy: HistGradientBoosting scores 51.23% and still falls further short, because it trades twice as often. [STATUS 2026-08-28](../status/STATUS-2026-08-turnover.md) works through what that does and does not establish.

**Read the last line the command prints.** It reports the serial dependence measured by a stationary bootstrap - expect an inflation around `0.97x`, meaning the standard errors above are already honest. Earlier versions of this project printed a caveat there instead, asserting the figures were optimistic by an unmeasured factor. They were not, and measuring it is what settled that.

**`--rolling` answers a different question.** The default expanding window trains on all history to date, which is what a deployment would have. A fixed-length rolling window asks whether recent history predicts better than distant - a hypothesis about regime change rather than a validation design. Use it to explore, not to report.

## 9. The network tests

Excluded from the gates so the default run is hermetic - no network, any OS, fast. They hold the venue to its side of the contract: that hourly bars still open exactly on the hour (the invariant the whole alignment design rests on), that both sides combine into a positive spread, and that every mapped instrument still exists.

```
uv run python -m pytest -m network
```

Run them when the data source misbehaves or before trusting a fresh `fetch`.

## 10. When something fails

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

**`verify` reports a mismatch** - the bytes on disk are not the bytes a published number was computed from. Either the venue revised a bar, or a file was edited. Re-fetch and re-run rather than updating the manifest to match, which would be recording the discrepancy as the truth.

**Feature counts changed and you did not change the code** - check the `ta` version. It is the only dependency pinned exactly (`ta==0.11.0`) because it changes indicator values between releases; `uv lock --upgrade` would walk past a range pin and move every published number without touching a line of ours.

**`LightGBM not run: blocked by an application control policy`** - the same policy again, and here it is *not* transient. LightGBM and XGBoost load a native DLL through `ctypes`, which the policy refuses outright for a binary with no established reputation. Measured: **4.7.0 and 3.4.1 are blocked; 4.6.0 and 3.0.5 load first try**. Those two versions are pinned exactly in `pyproject.toml` for that reason, so the failure mode to watch for is a well-meaning `uv lock --upgrade` silently removing the two most interesting models from the comparison. The command never hides it - it names every model it could not run.

**The OneDrive exports fail to read** - Files On-Demand leaves placeholder stubs on disk. Open the folder in Explorer and let it hydrate before pointing `ingest --from` at it.

## 11. What is not built yet

`validate` is where the pipeline currently stops, and it now produces a verdict rather than a number: the closest model is **0.46 points** short of paying for its own turnover, measured over 40,587 bars by a design that resolves 0.62.

Three things are still owed, each named in [ADR-011](../adr/ADR-011-power-before-verdict.md):

- **The significance battery** - Romano-Wolf and Hansen SPA, for having scored many configurations. (The stationary bootstrap that was owed alongside them has been run; the dependence it was meant to correct turned out not to be there.)
- **The gap benchmark.** Price rises through the venue's pauses 56-59% of the time, which clears 53.49% on its face - but those are exactly the hours that pay overnight financing. Until the swap is modelled, it stays a measurement rather than a strategy.
- **The dashboard** ([ADR-005](../adr/ADR-005-the-dashboard-runs-the-cli.md), still Plan).
