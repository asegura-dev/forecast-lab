# STATUS 2026-08-30 - A real edge, worth less than nothing

- **Question:** two of them, kept apart. Is the directional signal real once corrected for having tried eighteen configurations? And does trading it beat holding the asset?
- **Verdict:** **Yes and no, decisively both.** Nine of eighteen configurations survive Holm on Pesaran-Timmermann - the best at z = 4.61, p < 0.00001 - so the directional edge is not an artefact of searching. **Zero of eighteen make money.** The best loses **196.7%** over the scored period while holding gold returns **+69.8%**; Hansen's SPA puts the probability that anything beat the benchmark at p = 0.763, Romano-Wolf's StepM rejects nothing, and the Deflated Sharpe of the least-bad configuration is 0.0000. One point of directional accuracy costs more to collect than it is worth.
- **Command:** `forecast-lab verdict --target XAUUSD --timeframe 1H`
- **Machine-readable output:** [verdict-canonical.json](verdict-canonical.json)
- **Inputs:** `raw/*.csv` (Dukascopy, 2018-01 to 2026-08), hashed in [data-manifest.json](data-manifest.json). Figures are pinned to the **2026-08-26 snapshot**; `fetch` extends the series forward, so a later download changes every hash and `verify` will say so. 40,587 scored bars over five walk-forward folds, 18 configurations, long-or-short at the venue's measured median spread of 1.86 bps.

## 1. Is there skill?

Pesaran-Timmermann, against independence between prediction and outcome rather than against a coin flip. The six strongest:

| Configuration | Accuracy | Independent | Excess | z | p | After Holm |
|---|---:|---:|---:|---:|---:|---|
| **HistGradientBoosting** [raw] | 51.23% | 50.09% | **+1.14%** | **4.61** | <0.00001 | **yes** |
| Random Forest [raw] | 51.12% | 50.10% | +1.02% | 4.14 | 0.00002 | yes |
| LightGBM [raw] | 51.11% | 50.10% | +1.00% | 4.07 | 0.00002 | yes |
| HistGradientBoosting [pca-90] | 51.16% | 50.19% | +0.98% | 4.01 | 0.00003 | yes |
| XGBoost [raw] | 51.07% | 50.09% | +0.98% | 3.97 | 0.00004 | yes |
| Naive Bayes [pca-90] | 50.96% | 50.18% | +0.78% | 3.20 | 0.00069 | yes |

**16 of 18 are significant at 5%; 9 survive Holm across all eighteen.** That is a real signal, and it is the first thing in this project that has survived every correction aimed at it.

**Why the null is independence and not one half.** A predictor that always says UP on a series that rises 52% of the time scores 52%, beats a coin flip by two points, and knows nothing. The independence benchmark is what the two marginals produce on their own - measured here at 50.09% to 50.19% - and against it a constant predictor scores exactly zero. Testing against 0.5 would be the error this whole repository is a correction of, committed one last time on its own results.

## 2. Is it worth anything?

Net of the venue's own median spread, positions long-or-short, benchmark always-long:

| Configuration | bps/bar | Cumulative |
|---|---:|---:|
| Logistic Regression [raw] | -0.4847 | **-196.73%** |
| Naive Bayes [raw] | -0.5248 | -213.00% |
| Logistic Regression [pca-95] | -0.5272 | -213.96% |
| **always-long (benchmark)** | **+0.1719** | **+69.77%** |

**0 of 18 make money at all. 0 of 18 beat holding the asset.**

Under the friendlier long-or-flat framing - half the turnover, and a wrong call merely forgoes a move instead of taking it backwards - the answer does not change: the best loses **63.5%** against the benchmark's +69.8%, and it is still 0 of 18 on both counts. The result cannot be blamed on the harsher framing.

## 3. Does the best survive having been the best?

| Test | Result |
|---|---|
| Hansen SPA | p = **0.763** (lower 0.763, upper 1.000) - nothing beats the benchmark |
| Romano-Wolf StepM | rejects **nothing** |
| Deflated Sharpe on the least-bad | **0.0000** - does not survive |

The Deflated Sharpe is computed on Logistic Regression [raw]: Sharpe **-0.02123 per bar** (about -1.99 annualised), skew **-0.75**, kurtosis **28.3**, against an expected maximum of +0.00980 for eighteen trials. A negative Sharpe cannot survive deflation, and the fat left tail the EDA found in the returns is carried into the calculation rather than assumed away.

## 4. The finding

**There is a real directional edge and it is worth less than nothing.**

Nine configurations carry a statistically robust signal of roughly one point of directional accuracy, surviving a correction for having tried eighteen. That signal is worth less than the spread costs to collect: the same models, traded, turn +69.8% of buy-and-hold into -196.7%.

That is a more useful result than either half alone. "There is no signal" would have been wrong. "We found an edge" would have been true and dangerously incomplete. What the data supports is that **the market is not perfectly efficient at this horizon, and the inefficiency is smaller than the cost of exploiting it** - which is roughly what an efficient market with frictions is supposed to look like.

## 5. Why this null is credible

A battery that only ever returns "not significant" is indistinguishable from a broken one. Each test in the suite is shown a case where the effect is real and a case where it is not:

- Pesaran-Timmermann detects a constructed 53% predictor at z > 4 and reports exactly zero for a constant predictor scoring 52%.
- SPA and StepM name a model constructed to beat the benchmark, and name nothing when every model is noise.
- The Deflated Sharpe survives a genuine Sharpe of 0.3 per bar at eighteen trials and fails a negative one.

**And it survives a data refresh.** A `fetch` on 2026-08-30 extended the sample from 51,147 bars to 51,200 - 52 more hours, a tenth of a percent. Re-run on it, the count of configurations with significant skill rises from **9 to 13** and the economic verdict is unchanged at **0 of 18** with SPA p = 0.776. More data finds more of the same signal and still cannot make it pay, which is the shape a real-but-worthless edge should have. The figures above stay pinned to the 26 August snapshot because that is what the committed manifest describes; the refreshed run is reported here rather than substituted for them.

## 6. What this run does not settle

- **The swap, slippage and the rollover surcharge remain unmodelled.** Each raises the bar, so they cannot rescue a negative result - but a positive one would have needed them first.
- **The horizon.** Break-even scales with turnover; these models hold 2.6 to 5.7 bars and nothing here tests a strategy holding twenty. What is ruled out is the hourly directional trade.
- **The gap benchmark**, still owed since ADR-010.

## 7. Defects found by running this

**`arch` is not symmetric, and it fails silently.** Given the same DataFrame, `StepM.superior_models` returns column *names* and `SPA.better_models` returns positional *indices*. The first version of `significance.py` passed both through `str()`, so a superior model at position 4 came back as the label `'4'` - valid-looking and belonging to nothing. Only a test asserting a known-superior model **by name** would have caught it: the count was right and the identity was wrong.

**Two of this log's own tests were wrong before they were right.** A test meant to show that fat tails widen a Sharpe's interval first compared two samples whose Sharpe ratios differed, then held the Sharpe fixed and still failed - because in `1 - skew·SR + (kurtosis-1)/4·SR²`, **positive skew reduces the variance**. Fat tails alone do not cost certainty; negative skew with fat tails does, which is the shape gold returns actually have. Both errors are recorded in the test's docstring rather than quietly fixed.
