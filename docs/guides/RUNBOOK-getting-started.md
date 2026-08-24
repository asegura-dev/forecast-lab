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

Expect `183 passed, 15 deselected`. The 15 are the network tests, opt-in by design (see sec. 6).

**Note the `python -m` in front of mypy and pytest.** It is not decoration - see sec. 7.

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

## 4. Labels, splits, and the numbers a model must beat

```
uv run forecast-lab baseline --target XAUUSD --timeframe 1H --dir data/reference
```

**What every line means**, because this output is the point of the project:

- **`22,167 of 23,181 bars labelled`** - the horizon is a timestamp, not a position. The missing 1,013 (4.37%) are bars with no bar exactly one hour later, so `close.shift(-1)` there would reach across two hours, or across fifty, while reporting an hourly move.
- **`Across those gaps price rose 59.35% of the time, against 50.87%`** - the largest directional skew in this dataset. It is not a strategy: those are exactly the hours that pay overnight financing. It is reported because measuring something and setting it aside is different from never looking.
- **`majority-class (UP, from train) 51.46% ... specificity 0.00%`** - fitted on train and applied blind. The zero specificity is the signature to remember: it is what a constant looks like when only accuracy is reported.
- **`random (train frequencies) 51.70%`** - a seeded coin flip, scoring **above** the honest baseline. Nothing has been discovered; it is the plainest demonstration that at this sample size the noise is the size of everything being argued about.
- **`A rule fitted on this block would gain 2.01% for free`** - the distance between the honest baseline and an oracle. Larger than the effect anyone is trying to detect.

Break-even against the friendliest cost assumption is **51.92%**. Nothing available without a model reaches it.

## 5. The feature matrix

```
uv run forecast-lab features --target XAUUSD --timeframe 1H --dir data/reference
uv run forecast-lab features --target XAUUSD --timeframe 1H --dir data/reference --mode whole
```

Expect **22,982 rows x 17 columns** in focus mode and **x 179** in whole - and, critically, **the same index in both**. The original project's two modes covered different rows (24,232 against 22,441), so comparing them mixed a change of feature set with a change of sample.

**Reading the table:**

- **`Moved by (p99)`** decides. Every column is rebuilt on prices multiplied by ten; a column that moves is a price level. The verdict is the 99th percentile of the per-row deviation, not the maximum - see sec. 7 for why.
- **`Worst row`** is reported and never decides. A large worst beside a tiny p99 means a numerical instability, not a price level.
- **`ADF p` and `KPSS p` are diagnostics, never gates.** At n = 5,000 the ADF rejects a unit root on almost anything, and both tests are invalid under the heteroskedasticity and regime change that characterise this data. Columns where the two disagree are counted and reported rather than resolved by picking a favourite.

Every command producing a result also takes `--json`, because the dashboard runs these commands rather than reimplementing them:

```
uv run forecast-lab features --target XAUUSD --timeframe 1H --json
```

To check the pipeline is genuinely symbol-agnostic, point it somewhere else:

```
uv run forecast-lab features --target BTCUSD --timeframe 1H --dir data/reference --mode whole
```

## 6. The network tests

Excluded from the gates so the default run is hermetic - no network, any OS, fast. They hold the venue to its side of the contract: that hourly bars still open exactly on the hour (the invariant the whole alignment design rests on), that both sides combine into a positive spread, and that every mapped instrument still exists.

```
uv run python -m pytest -m network
```

Run them when the data source misbehaves or before trusting a fresh `fetch`.

## 7. When something fails

**`DLL load failed ... an application control policy blocked this file`** - Windows Smart App Control blocking an unsigned binary extension. Two forms:

- *On mypy*, permanently. Its published wheel ships mypyc-compiled `.pyd` extensions that SAC refuses outright, which silently removes one of the three gates. `pyproject.toml` pins a source build (`no-binary-package = ["mypy"]`), and mypy and pytest are invoked as `python -m` so the module is resolved from the environment rather than through a launcher shim.
- *On scipy, for the first few runs after a fresh `uv sync`*, then never again. SAC evaluates each unknown binary's reputation asynchronously and blocks it while it asks - **and it does so one file at a time**. Observed on a fresh clone: the first run failed on `_sparsetools`, `_moduleTNC`, `_zeros` and `pyduccfft` with 8 collection errors; the second failed on `rcont` alone with 2; the third passed all 183 with nothing reinstalled between them. **Re-run before diagnosing anything.**

**`error: Failed to spawn: forecast-lab ... (os error 4551)`** - the same policy, refusing the launcher itself. The `forecast-lab` console script is a generated `.exe` shim, and a freshly built binary has no reputation. Measured here: a shim installed days ago runs fine while one regenerated by `uv sync` minutes earlier stays blocked across repeated attempts - so unlike the extensions above, waiting is not a reliable fix.

Go through the interpreter instead, which is already a trusted binary:

```
uv run python -m forecast_lab.interfaces features --target XAUUSD --timeframe 1H
```

Every command works identically either way. Substitute `python -m forecast_lab.interfaces` for `forecast-lab` anywhere in this guide.

**`No such directory: data\raw`** - `data/` is gitignored, so a fresh clone has none. Go back to sec. 2. This also catches out a second clone of this repo on the same machine: the code is in git, the data is not.

**`No 1H series for XAUUSD in data\raw`** - you fetched into one directory and are reading from another. `fetch` writes to `data/raw`, `ingest` writes to `data/reference`, and every command takes `--dir`.

**`verify` reports a mismatch** - the bytes on disk are not the bytes a published number was computed from. Either the venue revised a bar, or a file was edited. Re-fetch and re-run rather than updating the manifest to match, which would be recording the discrepancy as the truth.

**Feature counts changed and you did not change the code** - check the `ta` version. It is the only dependency pinned exactly (`ta==0.11.0`) because it changes indicator values between releases; `uv lock --upgrade` would walk past a range pin and move every published number without touching a line of ours.

**The OneDrive exports fail to read** - Files On-Demand leaves placeholder stubs on disk. Open the folder in Explorer and let it hydrate before pointing `ingest --from` at it.

## 8. What is not built yet

`features` is where the pipeline currently stops. Scaling, PCA, the models and their confusion matrices are the next slice; walk-forward validation, the cost model and the power analysis follow. The order is deliberate: the original analysis went wrong before any model was fitted, so the corrections come first and the models second.
