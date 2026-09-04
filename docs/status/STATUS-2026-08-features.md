# STATUS 2026-08-24 - The feature matrix, and a rule that real data corrected twice

- **Question:** can the matrix carry everything the original project's indicators carried, **without** carrying the price level - and can the repository prove the second half rather than assert it?
- **Verdict:** **Yes, and the proof turned out to be harder to write than the matrix.** All nineteen columns clear the scale-freedom gate, and the gate itself had to be corrected twice on contact with real data - once in its threshold, once in its statistic.
- **Command:** `forecast-lab features --target XAUUSD --timeframe 1H --dir data/reference`
- **Machine-readable output:** [features-policy.json](features-policy.json)
- **Inputs:** `reference/*.csv`, hashed in [data-manifest.json](data-manifest.json).

## 1. What the matrix contains

| Mode | Rows | Columns | Symbols |
|---|---:|---:|---:|
| focus | 22,982 | 19 | 1 |
| whole | 22,982 | 199 | 10 |

**The two modes share an index exactly**, and a test asserts it. The original project's did not: the two ran over 24,232 and 22,441 rows respectively.

> **Corrected 2026-09-02.** This log said the cause was crypto in the WHOLE set. It was not - the original excludes BTCUSD explicitly. Its indicator function ends with `dropna()` and is called **once per symbol in a loop**, so the 199-row warm-up of the 200-period moving average is paid once per contributor: `24,431 - 10x199 = 22,441` and `24,431 - 199 = 24,232`, both exact. The symptom was diagnosed correctly and the mechanism named wrongly; the real one is a better example of what this project is about. Every difference between them mixed a change of feature set with a change of sample, which is why its comparison of the two never meant anything.

199 warm-up rows are dropped, by position rather than by `dropna()`. The longest window is 200 bars, so a row before that would report a value computed from less history than it claims. `dropna()` was rejected because it would *also* delete rows a stale auxiliary left empty - real target bars, carrying real target data, which belong in the matrix marked rather than removed.

## 2. Fidelity to the original, checked against the notebook rather than remembered

The original `indicator_creator` produces **23 columns per symbol**. Every one is accounted for:

| Original | Here | Note |
|---|---|---|
| `SMA_200/50/20` + `Dist_SMA200/50/20` | `dist_sma_200/50/20` | the raw levels are dropped; the distances were already there |
| `EMA_12/26` + `Dist_EMA12/26` | `dist_ema_12/26` | same |
| `MACD`, `MACD_signal`, `MACD_diff` | same, divided by close | MACD carries the price's units |
| `ADX` | `adx_14` | |
| `RSI` | `rsi_14` | |
| `ROC` | `roc_12` | |
| `BB_HIGH`, `BB_MID`, `BB_LOW`, `BB_WIDTH` | `bb_wband`, `bb_pband` | three raw levels dropped; `pband` added |
| `ATR` | `atr_pct` | as a fraction of price |
| `Returns`, `Log_Returns` | `return`, `log_return` | |
| - | `range_pct` | new |
| `RealVol_24h` (in the other notebook) | `realised_vol_24`, `realised_vol_168` | the substitute for VIX that ADR-002 promised. **Not new**: `Proyecto_Final_Completo` already computed a 24-bar version - corrected 2026-09-02 |

**The original's defect was narrower than "it lacked scale-free features", and that makes it more instructive.** It computed `Dist_SMA200` and its siblings itself - exactly the normalisation used here. What it did not do was *remove what the correction replaced*: the distances and the raw `SMA_200`, `EMA_12`, `BB_HIGH`, `BB_MID`, `BB_LOW`, `MACD` and `ATR` all sat in the same matrix. **Twelve of its twenty-three per-symbol columns carry the price level, beside their own normalised versions.** Nobody forgot the correction. Nobody deleted the thing it corrected.

## 3. The scale-freedom gate, and the two times it was wrong

Every column is rebuilt on prices multiplied by ten. A column that moves is a price level.

**First failure - the threshold.** Set to `1e-9`, reasoning that a real violation moves by nine orders of magnitude so anything above numerical noise must be real. It flagged `bb_pband`, which is `(close - lower) / (upper - lower)` - dimensionless by construction, but dividing by a Bollinger band's width, which goes small in quiet stretches and amplifies floating-point error. Measured: worst such noise **1.3e-09**; a raw price level **9.0e+00**. Threshold moved to `1e-6`.

**Second failure - the statistic.** Then ADX was added and moved by **7.1e-03**, four orders above the new threshold, on a quantity that is also dimensionless by construction. The size was not the clue; the shape was:

| | Value |
|---|---:|
| Maximum deviation | 7.127e-03 |
| 99th percentile | 6.988e-10 |
| Median | 4.170e-15 |
| Rows deviating by more than 1e-6 | **134 of 23,181 (0.58%)** |
| Position of those rows | **consecutive** - 4586, 4587, 4588, ... |

Consecutive rows are the signature of a recursion. `ta`'s ADX is a Wilder smoothing built on comparisons of consecutive highs and lows; where two are equal to within floating-point error, rescaling breaks the tie the other way, and the recursion carries that one flipped comparison forward until it decays.

**So the verdict became the 99th percentile, and the maximum is still printed beside it.** Hiding the maximum would be the dishonest half of the decision - a large worst beside a tiny percentile is a real numerical instability, worth knowing about, simply not a price level. The percentile is safe for what the rule exists to catch because a genuine price level moves on *every* row: on a raw close, percentile and maximum agree to within 20%.

**The lesson is not about this threshold.** A number chosen by reasoning about magnitudes was wrong twice, for two unrelated reasons, and both times the data said so within seconds of being asked. The statistic mattered more than the number. Same failure as the two-anchor grid rule in [ADR-002](../adr/ADR-002-data-source-and-symbol-set.md) sec. 6.

## 4. The result

All nineteen columns: **scale-free**.

| Column | p99 | Worst row |
|---|---:|---:|
| `return`, `log_return` | 1.0e-14 | 1.0e-14 |
| `range_pct` | 5.4e-15 | 6.4e-15 |
| `dist_sma_20/50/200` | ~1e-14 | ~1e-14 |
| `dist_ema_12/26` | ~2e-14 | ~3e-14 |
| `rsi_14` | 8.2e-15 | 1.9e-14 |
| `roc_12` | ~4e-15 | ~4e-15 |
| `adx_14` | **7.0e-10** | **7.1e-03** |
| `macd`, `macd_signal`, `macd_diff` | ~6e-14 | ~1e-13 |
| `atr_pct` | 4.0e-15 | 7.2e-15 |
| `bb_wband` | 3.6e-11 | 8.8e-11 |
| `bb_pband` | 1.8e-10 | 1.3e-09 |
| `realised_vol_24`, `realised_vol_168` | ~1e-14 | ~2e-14 |

The `adx_14` row is the one to look at: the two columns differ by seven orders of magnitude, and the table says so instead of picking one.

## 5. Layer 2, reported and never decisive

ADF returns **p = 0.000 for almost every column**, which is exactly why it is not a gate. At this sample size it rejects a unit root on almost anything, and both ADF and KPSS are invalid under the heteroskedasticity and regime change that characterise this data. **Four columns have ADF and KPSS pointing opposite ways**, reported as the honest outcome for a series that is neither clearly stationary nor clearly a random walk.

And one column makes the case for reporting rather than gating better than any argument could: **`realised_vol_168` is the only column the ADF does not reject** (p = 0.126). Seven-day volatility is persistent - close to a random walk in levels - which is exactly what it should look like. A gate would have thrown out the feature this project built to replace VIX, on a test that is invalid for this data anyway.

**"Carries no price level" is necessary, not sufficient**, and this log should not be read as a clean bill of health. `dist_sma_200` shifts its mean from 0.0024 to 0.0068 between train and test; `atr_pct` rises 37% in test. Both are scale-free. Both clear layers 1 and 2. What catches them is distribution shift measured **between blocks**, which is the next piece of work.

## 6. A dependency defect, generalised

The previous slice found that `ta`'s `AverageTrueRange` returns **0.0** across its whole warm-up rather than NaN, even with `fillna=False`. Adding ADX showed it is not an isolated bug: **`ADXIndicator` does the same**, with 27 zeros and no NaN at all.

It is the Wilder-recursion family. A zero is not "no data" - it is nil volatility, or no trend whatever: readings the market can plausibly produce, which `dropna()` will not remove, which explode any normalisation dividing by them, and which a tree learns as a real regime coinciding with the start of the sample. The wrapper restores the NaN, and the test asserts **both** halves - that `ta` does this, and that the wrapper undoes it - so a future release fixing it makes the test fail and tells us the workaround can go.

## 7. What changed in the repository

- `research/features/` exists: `technical.py`, `stationarity.py`, `build.py`, with 31 tests.
- The `features` command exists, with `--json`.
- `tests/test_layering.py` gained a quarantine guard: `ta` may only be imported by `technical.py`, because an untyped dependency spreading across a codebase takes `mypy --strict`'s guarantees with it.
- [RUNBOOK](../guides/RUNBOOK-getting-started.md) exists - the first end-to-end guide, from a fresh clone to this matrix, with a failure section carrying the real stumbles rather than imagined ones.
- **ADX and ROC were missing** from the first version of this slice, and were found by reading the original notebook rather than by remembering it. The correction and how it was caught are the substance of this entry.
