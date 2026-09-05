# ADR-006 - Indicators before alignment, and no column that carries a price

- **Status:** Accepted - **built** (`research/features/`, the `features` command, and 31 tests, including two that pin a threshold this ADR got wrong twice, plus a quarantine guard in the layering suite).
- **Date:** 2026-08-24
- **Follows:** [ADR-003](ADR-003-target-anchored-alignment.md) - which establishes the timeline and the per-symbol staleness this decision depends on, and [ADR-004](ADR-004-labels-splits-and-baselines.md), which establishes what the matrix is being built to predict.
- **Context:** The feature matrix is where two of the original analysis's defects live, and neither is visible by reading the code. One is an ordering decision that looks like a formatting preference. The other is a whole family of columns that pass every test anyone thought to run, because the test that would catch them was never written.

## Decision

### 1. Indicators are computed on each symbol's native grid, before any reindexing

`build_features()` takes the raw per-symbol series and does the alignment itself, precisely so it cannot be handed an already-aligned panel.

*Why:* after alignment, a row where an auxiliary symbol is stale holds a copy of that symbol's previous price. A return computed there is **exactly zero** - not a small number, an artefact. An RSI drifts toward 50, an ATR contracts toward nothing, and none of it is an observation of the market.

What makes it dangerous rather than merely noisy is that the artefact is not scattered at random. A symbol is stale precisely when its venue is shut, so the pattern tracks the hour of the day almost perfectly. Measured on the reference panel, VIX is carried on **8.0%** of rows and DXY on **7.1%** ([ADR-003](ADR-003-target-anchored-alignment.md) sec. 2). A tree fed those columns learns a session clock, and a report calls it a macro signal.

Carrying a *finished* indicator forward is a different act and is legitimate: at 03:00 the last thing known about the S&P really was its 21:00 RSI. What is not legitimate is computing an indicator *from* carried prices, because those inputs were never observed at those instants.

**Trade-off, accepted and written down so it is not mistaken for a bug later:** `ta` windows are **positional, not temporal**, and there is no option to change that. With 1,013 gaps in the target series, an `RSI_14` on a Monday bar uses thirteen bars from Friday, and `SMA_200` can span nine calendar days. The obvious fix - reindexing onto a complete grid before computing - reintroduces exactly the defect this section exists to prevent. The positional window is the lesser evil and becomes an invariant of the study.

### 2. No column may carry a price level, and it is checked by measurement

Every column is rebuilt on prices multiplied by ten. A column that moves is a price level; one that does not, is not.

*Why the rule:* gold runs from **1,616 to 4,378** across this sample. A feature carrying that level puts the test block outside the support of the training data, so a model fitted on it is extrapolating rather than predicting. It also looked like the explanation for something the original reported without remarking on it - its PCA at 90% variance collapsed 63 features into **6 components** - though [STATUS 2026-08-24](../status/STATUS-2026-08-models.md) sec. 8 later showed that was too quick: with every level removed, PCA still retains only 6 components from 19 features, and [ADR-009](ADR-009-exploratory-analysis.md) sec. 5 measured why - ten near-duplicate pairs, one of them exact. The levels made the redundancy worse; indicators from one price series are intrinsically redundant regardless.

*Why by measurement rather than by a list of names:* a name list is a claim about what a function does. Rescaling is an observation of what it did. The difference is not academic - it is how `bollinger_wband` got classified. The planning documents listed it as scale-free on the assumption that it normalises internally. It does, `(high - low) / mid * 100`, and the probe confirms it: multiply the input by ten and the output is unchanged. But the assumption was carrying the classification until something checked, and the whole architecture of this project exists because an unchecked assumption became a published number.

So: raw OHLC never enters the matrix, moving averages enter as `close / sma - 1`, MACD is divided by close, ATR enters as `atr / close`, and the two Bollinger outputs are already dimensionless.

**And the original's mistake was narrower than it first appears, which makes it more instructive.** It was not that it lacked scale-free features - it computed `Dist_SMA200`, `Dist_SMA50`, `Dist_SMA20`, `Dist_EMA12` and `Dist_EMA26` itself, exactly the normalisation used here. The defect is that it kept **both**: the distances *and* the raw `SMA_200`, `EMA_12`, `BB_HIGH`, `BB_MID`, `BB_LOW`, `MACD` and `ATR` columns they were derived from, all in the same matrix. Twelve of its twenty-three per-symbol columns carry the price level, sitting beside their own normalised versions. Nobody forgot the correction; nobody removed what the correction replaced. That is a far commoner failure than ignorance, and no test in the pipeline could have caught it because none existed.

### 3. Warm-up is NaN, always - including where the library says otherwise

Every indicator's first `window - 1` rows are blanked by the wrapper, whatever the library returned.

*Why:* `ta` honours `fillna=False` for SMA, EMA, RSI, MACD and Bollinger. `AverageTrueRange` does not - it returns **0.0** across its entire warm-up. Measured on a 14-bar ATR, rows 0 through 12 come back as exactly zero, with zero NaN in the whole column.

A zero is worse than a NaN in every respect that matters here. It is not "no data", it is *nil volatility* - a reading the market can plausibly produce. `dropna()` will not remove it. Any normalisation dividing by ATR explodes on it. And a tree will learn it as a genuine low-volatility regime that happens to coincide with the start of the sample.

The test asserts **both halves**: that `ta` really does this, and that the wrapper undoes it. If a future release fixes the library, the first assertion fails and tells us the workaround can go - which is the reason to assert a dependency's misbehaviour rather than silently work around it.

### 4. ADF and KPSS are reported. They never decide

Both run on every column and both are printed. Neither can fail the command.

*Why:* treating them as a gate would be a false guarantee, and a false guarantee is worse than no guarantee because it stops people looking. The tests read the last 5,000 rows of each column (`max_rows`), and at that length the ADF rejects a unit root on almost anything - on the reference run it returns p = 0.000 for almost every column - and both tests are invalid under heteroskedasticity and regime change, which is the entire character of this data. Four columns have ADF and KPSS pointing opposite ways, and that is reported as the honest outcome for a series that is neither clearly stationary nor clearly a random walk, rather than resolved by picking whichever test agrees with the conclusion already wanted.

**"Carries no price level" is necessary, not sufficient**, and this is the section that says so out loud. `dist_sma_200` shifts its mean from 0.0024 to 0.0068 between train and test; `atr_pct` rises 37% in test. Both are scale-free. Both clear layers 1 and 2. What catches them is distribution shift measured **between blocks** - PSI, KS, and adversarial validation - which needs the splits and therefore belongs with the validation work rather than here.

### 5. WHOLE excludes the late-starting symbols, so the two modes share an index

`LATE_STARTERS` holds BTCUSD and ETHUSD; they contribute no columns to WHOLE, and either can still be the target.

*Why:* the original project ran WHOLE and FOCUS and compared them over different rows - **24,232 against 22,441** - so every difference between them mixed a change of feature set with a change of sample, and the comparison never meant anything.

> **Corrected 2026-09-02 ([STATUS](../status/STATUS-2026-09-parity.md) sec. 2.1).** This paragraph blamed crypto, *"whose history begins a year after everything else"*. It is not the cause: the original **excludes BTCUSD explicitly**, commented out with the reason. The real mechanism is sharper - its indicator function ends with `dropna()` and is called **once per symbol in a loop**, so the 199-row warm-up of the 200-period moving average is paid once per contributor: `24,431 - 10x199 = 22,441` and `24,431 - 199 = 24,232`, both exact. A `dropna()` inside a loop, invisible at the call site, costing 1,791 rows. The decision below is unchanged and better motivated: excluding crypto is right for its own reason, and sharing an index is what the test actually asserts. Here both modes produce **22,982 rows** on the reference data, and a test asserts the indices are identical.

Symbol order is sorted explicitly rather than taken from a set. The original used `list(set(...))`, whose iteration order varies between processes - which makes the column layout, and therefore any PCA fitted on it, irreproducible from one run to the next.

## Consequences

**The probe's threshold and its statistic were both wrong, and real data corrected them in that order.** This is the same failure as the two-anchor grid rule in [ADR-002](ADR-002-data-source-and-symbol-set.md) sec. 6, caught the same way, twice.

*First:* `SCALE_TOLERANCE` was `1e-9`, on the reasoning that a real violation moves by nine orders of magnitude so anything above numerical noise must be real. It flagged `bb_pband` - dimensionless by construction, but its denominator is a Bollinger band's width, which goes small in quiet stretches and amplifies floating-point error. Measured: the worst such noise is **1.3e-09**, and a raw price level moves by **9.0e+00**. The threshold became `1e-6`.

*Then ADX arrived and broke it again*, moving by **7.1e-03** - four orders above the new threshold, on a quantity that is dimensionless by construction. The diagnosis was in the shape of the deviation rather than its size: **134 rows out of 23,181 (0.58%), at consecutive positions**. ADX is a Wilder recursion over comparisons of consecutive highs and lows; where two are equal to within floating-point error, rescaling breaks the tie the other way, and the recursion carries that single flipped comparison forward through a long run of rows before it decays. Meanwhile the 99th percentile of the same deviation was **7.0e-10**.

So the verdict is now the **99th percentile**, not the maximum - and the maximum is still reported beside it, because hiding it would be the dishonest half of the decision. A column with a large `worst` and a tiny `change` has a genuine numerical instability worth knowing about; it is simply not a price level. The percentile is safe for the thing the rule exists to catch, because a column that carries the price level moves on *every* row: on a raw close, percentile and maximum agree to within 20%. Two tests pin both halves.

**The lesson generalises past this rule.** A threshold chosen by reasoning about magnitudes was wrong twice, for two different reasons, and both times the data said so within seconds of being asked. The statistic mattered more than the number.

**The warm-up is dropped by position, not by `dropna()`.** 199 rows, the longest window minus one. `dropna()` would additionally delete rows a *stale auxiliary* left empty - real target bars, carrying real target data, which belong in the matrix marked rather than removed.

**`ta` is quarantined in one module**, enforced by `tests/test_layering.py`: any other module importing it fails the gate. A dependency with no type information that spreads across a codebase takes the type checker's guarantees with it.

**The windows are unchanged from the original** - SMA 20/50/200, EMA 12/26, RSI 14, ATR 14, Bollinger 20. Changing them would count as a fresh trial in the deflated-Sharpe accounting, and the object of this work is to isolate the effect of the corrections rather than of a hyperparameter sweep.
