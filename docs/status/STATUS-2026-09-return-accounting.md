# STATUS 2026-09-04 - The headline number was wrong, and a glossary entry had explained it away

- **Question:** the book being written for `docs/` requires every figure to come from a committed sidecar. Reading `verdict-canonical.json` against the prose that quotes it turned up a number that cannot exist: **"the best loses 196.7%"**, a loss larger than the capital available to lose it.
- **Verdict:** **two compounding errors, both flattering the strategy, in the project's most-quoted sentence.** Returns were summed instead of compounded, and the series being summed was logarithmic, so shorts were credited with more than they earned. Corrected, buy-and-hold returns **+100.91%** and the least-bad configuration loses **86.87%**.
- **What does not change:** the verdict. **9 of 18** still survive Holm, **0 of 18** still make money, **0 of 18** still beat holding the asset. `exp` is monotonic, so no ordering moved.
- **Command:** `forecast-lab verdict --target XAUUSD --timeframe 1H --mode focus --folds 5 --json`

## 1. The two errors

**A sum of returns is not a return.** `_run_battery` reported `float(v.sum())` over 40,587 per-bar returns and printed it with a `%`. A sum ignores that the second bar trades the proceeds of the first, and - unlike a return - it has no floor. That is why the figure could pass -100% without anything raising.

**The series was logarithmic, and the short side was wrong in one direction.** `realised` was `np.log(close).diff()`. Summing `±log(1+r)` is exact for a long position and wrong for a short one: the log return of a short is `log(1 - r)`, not `-log(1 + r)`, and Jensen puts the error always on the same side. Measured on this series with a half-short strategy, **the approximation flattered the result by 7.4 points**.

| | Reported | Corrected |
|---|---:|---:|
| Buy and hold | +69.77% | **+100.91%** |
| Least-bad configuration (Logistic Regression [raw]) | -196.73% | **-86.87%** |
| Worst configuration (LightGBM [pca-95]) | -381.91% | **-97.98%** |
| Long-or-flat framing, least-bad | -63.5% | **-47.00%** |
| Hansen SPA | p = 0.763 | p = **0.761** |
| Deflated Sharpe | 1.556e-10 | **2.270e-10** |

The fix is two lines: `realised` becomes `close.pct_change()`, and `cumulative` becomes `_compounded()` - `expm1(log1p(r).sum())`, which is `cumprod` with the precision that 40,587 sequential multiplications lose.

## 2. Why it survived a fortnight, three green gates and four independent reviews

**Nothing was inconsistent.** Every document agreed with every other, because they all quoted the same sidecar. The gates check types, style and behaviour - none of them knows what a return is. The number was not contradicted anywhere; it was simply impossible, and impossibility is not something a test suite notices unless someone writes the test.

**And the anomaly had already been noticed and rationalised.** The dashboard glossary carried this entry:

> *"The sum of per-bar returns over the scored period - not a drawdown. It exceeds 100% because it is a sum over 40,000 bars, not a capital loss."*

Every clause is true. It correctly identifies that the figure is a sum, correctly notes that this is why it passes 100%, and correctly says it is not a drawdown. What it never asks is why a *return* is being reported as a sum. **The explanation was accurate and it closed the question** - a reader who hit the strange number was handed a reason to stop looking, which is worse than no note at all.

That is the most transferable lesson in this file: an anomaly that has acquired an explanation is harder to find than one that has not.

## 3. What is now pinned

Three tests in `tests/unit/test_cli.py`:

- a cumulative return is never below **-100%** (the bound is `>=`, because 0.5<sup>200</sup> underflows and total ruin prints as exactly -100%);
- two bars of +10% make **+21%**, not +20%;
- a **-100%** bar is absorbing, and nothing after it recovers the position.

The first is the one that would have caught this on the day it was written.

## 4. Related, and fixed with it

`_print_verdict` formatted the Deflated Sharpe with `:.4f`, printing **0.0000** for 2.27e-10 - the exact defect `interfaces/presentation.py::significance` exists to end, in the one table whose subject is significance. It now uses that helper, and the documents quote **2.3e-10** instead of a rounded zero.

## 5. What this does not fix

The correction makes the accounting exact for the position framing this project models: one unit of capital, no leverage, no sizing, fully invested each bar. It says nothing about the costs still unmodelled - the overnight swap above all, which [FINDINGS](../../FINDINGS.md) carries as an open hypothesis and no run here tests. **A more honest headline is still not a complete one.**
