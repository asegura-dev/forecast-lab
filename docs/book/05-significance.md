# Chapter 5 - Significance, multiplicity and the verdict

Chapter 4 established what a prediction has to beat and whether the design could have seen it. This chapter is about the machinery that decides which of the eighteen surviving numbers are *findings*: the four tests in `src/forecast_lab/research/significance.py` - Pesaran-Timmermann against independence, Holm step-down over the family, Hansen's SPA with Romano-Wolf's StepM over the economic comparison, and the Probabilistic and Deflated Sharpe ratios - what each asks that the others cannot, and what they returned on the run recorded in `docs/status/verdict-canonical.json`. The answer they jointly produce is the point of the project: **nine of eighteen configurations carry a directional signal that survives every correction aimed at it, and not one of them makes money.** A result can be statistically real and economically worthless at the same time, and this is what that looks like.

---

## 1. Where the machinery actually lives

There is no `research/evaluation/` package in this repository, no `directional.py` and no `multiplicity.py`. All four tests, the Holm correction and the module constants sit in one 380-line file, `src/forecast_lab/research/significance.py`, re-exported through `research/__init__.py` and called from exactly one place: `_run_battery()` in `src/forecast_lab/interfaces/cli.py`, which backs the `verdict` command. The nineteen tests pinning it are in `tests/unit/test_significance.py`. Three constants govern reproducibility - `DEFAULT_REPS = 1_000`, `DEFAULT_SEED = 42`, `DEFAULT_ALPHA = 0.05` - the seed fixed because *a published p-value that moves between runs is not a measurement.*

## 2. Pesaran-Timmermann: what directional accuracy is tested *against*

The trap this test exists to avoid is the one the whole repository is a correction of. A predictor that always says UP on a series that rises 52% of the time scores **52% accuracy**, beats a coin flip by two points, passes a naive binomial test, and knows nothing. Testing against 0.5 rewards it; testing against *what the two marginals imply on their own* scores it at exactly zero.

`pesaran_timmermann(predicted, actual)` computes, with `P` the share of actual UP bars and `Q` the share of predicted UP bars:

```
accuracy    = mean(predicted == actual)
independent = P·Q + (1 - P)(1 - Q)
```

`independent` is the hit rate two independent series with those marginals would produce between them. The statistic compares the two and is asymptotically standard normal:

```
               accuracy - independent
z  =  ------------------------------------------
       sqrt( var(accuracy) - var(independent) )

var(accuracy)    = independent · (1 - independent) / n

var(independent) = (2P - 1)² · Q(1 - Q) / n
                 + (2Q - 1)² · P(1 - P) / n
                 + 4 · P · Q · (1 - P) · (1 - Q) / n²
```

Two details in that denominator are easy to get wrong and neither is visible in the output.

*Why the first variance is evaluated at `independent` and not at `accuracy`:* the hit rate's variance is computed **under the null**, which is what a test statistic requires. The module writes it literally as `independent * (1 - independent) / n`.

*Why the subtraction:* the independence benchmark is not a constant. It is itself estimated from the same two marginals and therefore carries sampling variance of its own. Subtracting it turns the ratio into a test of *dependence between prediction and outcome* rather than a test of accuracy against a fixed number. Skipping it - which a naive binomial test does - "makes every result look more significant than it is."

**The degenerate case is handled explicitly, and it is not a corner case.** For a constant predictor `Q = 1`, so `Q(1-Q) = 0`, the third term vanishes and the two variances become algebraically identical - both reduce to `P(1-P)/n` - so the denominator is exactly zero. `pesaran_timmermann` detects `denominator <= 0` and returns `statistic = 0.0, p_value = 1.0` rather than an infinity, which `test_a_constant_predictor_has_accuracy_and_no_skill` pins. The p-value is one-sided (`_upper_tail`, wrapping `scipy.stats.norm.sf`), for ADR-011's reason: the question is "is this better than chance", not "is this different from chance".

**What the benchmark measured to.** Across all eighteen configurations in `verdict-canonical.json` the independence benchmark ranges from **50.03%** (`Naive Bayes [raw]`) to **50.33%** (`Logistic Regression [pca-90]`). ADR-013 sec. 2 and `STATUS-2026-08-verdict.md` sec. 1 both quote the band as "50.09% to 50.19%", which is the range across the *six strongest rows the status log tabulates*, not across the family. The wider band is the honest one, and still short of the constant predictor's 50.26% majority-class rate - the two nulls are different objects.

## 3. The multiplicity problem, and why this project could not exempt itself

Eighteen configurations were scored. At a 5% size the probability that *at least one* of eighteen null tests clears the threshold is `1 - 0.95¹⁸ ≈ 60%`. Reporting the smallest of eighteen p-values as though it were the only question asked is therefore not a small distortion; it is closer to a coin flip dressed as a discovery.

This project made that argument by hand, against the analysis it re-engineers, before it had results of its own to protect. `docs/status/STATUS-2026-08-models.md` sec. 3 puts numbers on it: 18 configurations, 3,294 rows per block, a standard error of 0.8712% on one accuracy, so the **expected maximum of eighteen draws under the null is 1.82 sigma = +1.59%** - against a best observed of +1.85% (2.13 sigma), short of Holm's threshold of `z >= 2.77`. The log's sharpest line is the count, not the p-value: only **5 of 18** configurations had a positive edge on test, where a coin would give about nine. ADR-013 sec. 3 states the consequence - quoting its own smallest p-value out of eighteen "would be a double standard visible to any reader who got that far" - and that is why `holm()` exists in this codebase at all.

## 4. Holm step-down

`holm(p_values, *, alpha=DEFAULT_ALPHA)` implements the Holm-Bonferroni step-down procedure and returns a `dict[str, bool]` re-keyed in the caller's original order. The procedure, exactly as coded:

1. Sort the `m` p-values ascending: `p(1) <= p(2) <= ... <= p(m)`.
2. At rank `i = 0, 1, ..., m-1`, compare `p(i+1)` against `alpha / (m - i)`: the smallest is tested at `alpha/m`, the next at `alpha/(m-1)`, the largest at `alpha`.
3. **Step down:** at the first p-value that fails its threshold, stop - every larger p-value fails too, whatever its own threshold would have been. The loop's `surviving` flag is never raised again once lowered; that flag *is* the step-down, and `test_holm_steps_down_and_stops_at_the_first_failure` pins it.

*Why it controls the family-wise error rate:* under any configuration of true nulls, the first true null reached in sorted order faces a threshold no larger than `alpha` divided by the number of true nulls still remaining, so a union bound over that set totals at most `alpha`. The guarantee holds under arbitrary dependence among the tests - which matters, because eighteen models fitted on the same features over the same bars are anything but independent.

*Why Holm rather than Bonferroni:* Bonferroni compares every p-value against `alpha/m`. Holm's *first* threshold is `alpha/m` and every subsequent one is strictly larger, so Holm rejects everything Bonferroni rejects and sometimes more, at the same guarantee - uniformly more powerful, no less valid. The module puts it as "the weaker correction would be conceding detections for nothing", and `test_holm_is_never_weaker_than_bonferroni` pins a disagreement: p = 0.014 with m = 4 fails Bonferroni's 0.0125 and passes Holm's `0.05/3` at rank two.

**On this project's own data the two agree, and the chapter should say so.** Recomputed from `verdict-canonical.json`, Holm rejects 9 and Bonferroni rejects the same 9: the family's p-values are bimodal - nine at or below 0.00246, the next at 0.00670 - so the step-down bought nothing here. Holm remains the right default; it simply did not earn its keep on this run.

## 5. SPA and StepM: what Holm does not ask

Holm operates on p-values already computed. It knows nothing about the *joint distribution* of the eighteen statistics, so it must assume the worst and pay for it. Hansen's SPA and Romano and Wolf's StepM estimate that joint distribution from the data, by bootstrap - and they ask a different question in a different space: not "is this accuracy better than independence" but "does anything beat the benchmark, in returns".

`superior_predictive_ability(benchmark, models, ...)` wraps `arch.bootstrap.SPA` and `arch.bootstrap.StepM`:

- **SPA** returns one p-value for the composite null *"no model beats the benchmark"*, in Hansen's three flavours: `lower`, `consistent`, `upper`. `Superiority` keeps all three, and says why - a gap between `lower` and `upper` means the answer depends on how badly-performing models are handled, "which is a fact about the comparison, not the data."
- **StepM** does what SPA cannot: it *names* the superior models while holding the family-wise error rate, stepping down as Holm does but with bootstrap-estimated critical values instead of a union bound.

*Why a bootstrap is unavoidable:* the maximum of eighteen correlated loss differentials has no closed-form null distribution, and the resampling must preserve the serial dependence in the return series - hence a **stationary bootstrap**, whose block lengths are geometric draws with mean `b`. `block_size` is not guessed: left unset it calls `optimal_block()` in `research/dependence.py`, the Politis-White rule via `arch.bootstrap.optimal_block_length`, floored at `MINIMUM_BLOCK = 1.0`. That function is public for one stated reason - two modules disagreeing about how dependent the data is "would make their p-values incomparable."

*The sign convention, which is a live hazard.* Both `arch` classes take **losses**, where lower is better. `_run_battery()` therefore negates everything at the call site - `-benchmark.to_numpy()` and `{key: -value.to_numpy() ...}` - where the returns come from `directional_returns()` in `research/costs.py`. The docstring warns that "a sign error here would silently invert every conclusion": nothing downstream would look wrong.

*What the benchmark is.* Always-long, never zero: `_run_battery` builds it as `directional_returns(pd.Series(1.0, index=index), ...)`, the constant predictor expressed as a position. Measured against zero, a model would collect this sample's upward drift and report it as skill - an accuracy without its baseline, in different units.

**The `arch` asymmetry that bit this project.** Given the *same* `pandas.DataFrame`, the two APIs return different kinds of thing:

| Call | Returns |
|---|---|
| `StepM.superior_models` | column **names** |
| `SPA.better_models(alpha)` | positional **indices** |

The first version passed both through `str()`, so a genuinely superior model at position 4 was reported as the label `'4'` - a valid-looking string belonging to no model. The count was right and only the identity was wrong, which is why nothing but an assertion **by name** could have caught it. The fix is one line, `better = tuple(names[int(position)] for position in spa.better_models(alpha))`, guarded by `test_the_names_come_back_rather_than_positions` and recorded in ADR-013's Consequences, `STATUS-2026-08-verdict.md` sec. 7 and `FINDINGS.md`.

## 6. PSR and DSR

`deflated_sharpe(returns, *, trials, trial_sharpes=None)` implements Bailey and Lopez de Prado, correcting a bare Sharpe for two things it omits.

**Shape.** The usual interval assumes normal returns. Hourly gold is not normal - the EDA measured ten-sigma moves - so the variance of the Sharpe estimator under non-normality enters the denominator directly:

```
var(SR_hat) = 1 - skew·SR + (kurtosis - 1)/4 · SR²     (kurtosis non-excess: 3.0 is Gaussian)

PSR(threshold) = Phi(  (SR - threshold) · sqrt(n - 1) / sqrt(var(SR_hat))  )
```

`PSR(0)` is the probability the true Sharpe exceeds zero. A subtlety recorded in the docstring of `test_the_shape_gold_actually_has_widens_the_interval`, after two wrong versions of that test: **fat tails alone do not cost certainty.** Positive skew *reduces* the variance in that formula - one earlier fixture drew kurtosis 434 with skew +6.9, which more than cancelled the penalty. What costs certainty is negative skew *with* fat tails, which is the shape measured here.

**The search.** Deflation replaces the threshold 0 with the Sharpe the best of `trials` random strategies would be expected to reach:

```
E[max of N] = spread · [ (1 - gamma)·Phi⁻¹(1 - 1/N) + gamma·Phi⁻¹(1 - 1/(N·e)) ]

DSR = PSR( E[max of N] )
```

with `gamma = EULER_MASCHERONI = 0.5772156649015329` (`_expected_maximum_z`), and `spread` the cross-trial standard deviation of the Sharpes observed, `np.std(trial_sharpes, ddof=1)`. Without `trial_sharpes` the module falls back to `1/sqrt(n-1)` and labels the result approximate rather than hiding the substitution.

**What "trials" means here, precisely.** `_run_battery()` passes `trials=len(returns)` - the **eighteen configurations scored in this one run** - with `trial_sharpes` the eighteen per-bar Sharpes. It does not count the representations, modes, horizons and feature sets explored over the project's life. Bailey and Lopez de Prado's `N` is the number of trials in the whole research process, so **18 is a lower bound and the DSR as computed is, if anything, generous.**

DSR is nonetheless the harshest test here, for two structural reasons: it is the only one whose threshold is *raised* rather than zero, and `survives()` demands `deflated > 1 - alpha`, i.e. **greater than 0.95** - a posterior-style bar, not a p-value below 0.05. It also runs on `best_by_return`, the highest mean net return, not on the best z-statistic: skill and profit are ranked separately all the way down.

## 7. The verdict's cells, and what was actually pre-registered

The verdict is reported as a four-row table in ADR-013, `README.md` and `FINDINGS.md`:

| Question | Test |
|---|---|
| Is there skill? | Pesaran-Timmermann + Holm |
| Does anything make money? | Net returns at the measured spread |
| Does anything beat holding gold? | Hansen SPA |
| Does the best survive being the best? | Romano-Wolf StepM, DSR |

**The code does not have four cells.** `_print_verdict()` prints **three** numbered sections - the third merges SPA, StepM and DSR under "Does the best survive having been the best?" - followed by a verdict paragraph selected by **three** branches: `surviving and not beaters` gives "there is a real directional edge and it is worth less than nothing"; `beaters` gives "check it against the costs this project does not model"; the fallback gives "no skill and no profit". Every clause is derived from the run rather than asserted, after a defect in which an earlier version used `abs()` and reported a +76% best as a 76% loss the first time the command ran on an index. The four-cell table is a reporting shape imposed in prose.

**And the pre-registration claim needs care, because `FINDINGS.md` states it against itself.** What the git history supports is that the **thresholds** were fixed first: ADR-010's measured break-even and ADR-011's minimum detectable effect landed in `85c0d89` on 2026-08-28, while `significance.py`, ADR-013 and `verdict-canonical.json` landed together in `9cb292c` on 2026-08-31, two commits later. Checkable with `git log --diff-filter=A`. What was **not** pre-registered is the taxonomy of possible outcomes - the four cells themselves. `FINDINGS.md` says so in its own words: "The thresholds were; the list of verdicts they might produce was not. Reading the finding below as a pre-registered result would overstate it." The bar was set in advance; the shape of the answer was not.

## 8. What actually happened

Everything below is from `docs/status/verdict-canonical.json`: XAUUSD, 1H, horizon 1, focus mode, five folds, **40,587 scored bars**, **18 configurations**, long-or-short at the measured median round trip of **1.86 bps**.

| Quantity | Value | Field |
|---|---:|---|
| Nominally significant at 5% | **16 of 18** | derived from `skill.per_configuration[*].p_value` |
| Survive Holm | **9 of 18** | `skill.surviving_holm` |
| Best z | **4.6114** (`HistGradientBoosting [raw]`) | `skill.per_configuration` |
| Its accuracy / independence / excess | 51.23% / 50.09% / **+1.14%** | same |
| Its p-value | 2.00e-06 | same |
| Where Holm stops | `LightGBM [pca-90]`, p = 0.006697 vs threshold 0.005556 | derived |
| Configurations that make money | **0 of 18** | `profit.profitable` |
| Configurations that beat always-long | **0 of 18** | `profit.beating_benchmark` |
| Least-bad by return | `Logistic Regression [raw]`, -0.4741 bps/bar, **-86.87%** | `profit.per_configuration` |
| Benchmark | +0.1979 bps/bar, **+100.91%** cumulative | `profit.benchmark_*` |
| SPA p (lower / consistent / upper) | 0.761 / **0.761** / 1.000 | `multiplicity.spa_*` |
| SPA better models | none | `multiplicity.spa_better` |
| StepM rejections | none | `multiplicity.stepm_rejected` |
| DSR input: Sharpe / skew / kurtosis | -0.021233 per bar / -0.748 / 28.29 | `multiplicity.deflated_sharpe` |
| PSR (probability true Sharpe > 0) | **8.34e-06** | same |
| Expected maximum, 18 trials | +0.009804 | same |
| **DSR** | **2.270e-10** | same |

**The SPA bracket is the strongest available form of the economic null.** `lower` and `consistent` are both 0.761 and `upper` is 1.000. Because even the most *liberal* of Hansen's three - the one that discards badly-performing models before recentring - returns 0.761, no treatment of the poor models produces significance. That is what reporting all three is for.

**The DSR is not a marginal miss.** The Sharpe being deflated is **negative**, so its PSR against a threshold of zero is already 1.26e-05 (a z of -4.21); raising the threshold to the expected maximum of +0.009976 collapses it to 2.27e-10. The battery is not rigged to reject: the *same* Sharpe with its sign flipped, through the same formula, gives a DSR of 0.99. This one fails on the arithmetic, not on the correction.

**Two defects this chapter found, both now fixed.** Writing it required reading every quoted figure back against the sidecar, and two did not survive that.

The documents quoted "DSR = 0.0000" where the payload says 2.27e-10, because `_print_verdict` formatted it with `:.4f` - the exact defect `src/forecast_lab/interfaces/presentation.py::significance` exists to end, surviving in the one table whose subject is significance. It now uses that helper.

The second was larger. The headline read **"the best loses 196.7%"** - a loss larger than the capital available to lose it. The money path summed per-bar returns instead of compounding them, and the series it summed was logarithmic, so short positions were credited with more than they earned. Compounded properly, the least-bad configuration loses **86.87%** and holding the asset returns **+100.91%**; the verdict is unchanged, because `exp` is monotonic. The dashboard glossary had even carried an entry explaining why the figure passed 100% - every clause true, and it never asked why a *return* was being reported as a sum. [STATUS 2026-09-04](../status/STATUS-2026-09-return-accounting.md) records the arithmetic and the three tests that now pin it. **An anomaly that has acquired an explanation is harder to find than one that has not.**

**One thing the code does not do that a document implies.** ADR-011's Consequences say the pooled edge of +0.97% "survives a Holm correction across the six models (z >= 2.39)". `holm()` has exactly one call site, `_run_battery()` at `cli.py:1606`; `validate` does not call it. That correction is arithmetic done in prose - right, but not machinery.

## What the battery licenses, and what it does not

Nine configurations beat independence after a family-wise correction across all eighteen, the best at 4.61 standard errors, on a design Chapter 4 shows resolves 0.62 points at 80% power. That signal is real; "we found nothing" would be the wrong summary. The same eighteen, traded at the venue's quoted spread, turn +100.9% into -86.9%, with SPA at 0.761, StepM naming none and a DSR of 2.27e-10. Both statements come out of the same run over the same bars, and they do not contradict each other because they answer different questions - which is why `verdict` keeps them apart. **The edge is about one point of directional accuracy, and the spread costs more than one point of directional accuracy is worth.** ADR-013 states it in the sentence the project was built to be able to say: *there is a real directional edge and it is worth less than nothing.*

What this does not license: it is **not** a finding of no edge, and **not** a claim about lower-frequency strategies, whose break-even scales with a turnover nothing here measures beyond 5.7 bars. And a battery that only ever returned "not significant" would be indistinguishable from a broken one, which is why every test in `tests/unit/test_significance.py` is shown both a real effect and a null: PT detects a constructed 53% predictor at z > 4 and scores a constant predictor at zero; SPA and StepM name a model built to beat the benchmark and name nothing when every model is noise; the DSR clears a Sharpe of 0.3 at eighteen trials and fails a negative one. The instruments work. What they found is a market not perfectly efficient at this horizon, whose inefficiency is smaller than the cost of exploiting it.
