"""Whether the small edge that survived everything else is real.

By the end of ADR-012 the project had one live positive result and one dead one. The dead
one is profitability: nothing clears the accuracy that pays for its own turnover, and the
closest misses by 0.46 points. The live one is **skill**: pooled over 40,587 bars every
model beats the constant predictor, the best by +0.97% at 3.91 standard errors, and the
dependence measurement removed the last reason to distrust that standard error.

Those are different questions and this module answers the second. It cannot rescue the
first - 0.46 points is arithmetic - but "there is a real edge too small to trade" and
"there is no edge" are different findings, and the project has no business asserting
either without testing it.

**Four tests, each answering something the others cannot.**

*Pesaran-Timmermann* asks whether directional accuracy exceeds what independence between
prediction and outcome would produce. It is the right null for this question and a
stricter one than "beat 50%": a model that predicts UP 90% of the time on a series that
rises 52% of the time scores 52% accuracy while carrying **no** information, and a
comparison against a half would call that a win. PT compares against `P·Q + (1-P)(1-Q)`
instead, which is what those two marginals imply on their own.

*Hansen's SPA* and *Romano-Wolf's StepM* both ask whether the **best of many** survives
having been the best of many. Eighteen configurations were scored; the maximum of eighteen
draws from noise is not centred on zero. This project already made that argument by hand
against the original analysis - an expected maximum of +1.59% under the null - and using
Holm on its own results while demanding better of others would be a double standard. SPA
gives one p-value for "did anything beat the benchmark"; StepM names which ones, holding
the familywise error rate.

*The Deflated Sharpe Ratio* asks the same multiplicity question in return space, where
skew and fat tails matter and where a Sharpe computed from 40,000 hourly bars looks far
more certain than it is.

**The benchmark is always-long, not zero.** A directional gold strategy must beat holding
gold, which rose through most of this sample. Testing against zero would let a model
collect a bull market and call it skill - the same error as scoring accuracy without its
baseline, moved into return space.

**What is deliberately not modelled**, unchanged from ADR-010: slippage, the rollover
surcharge, and the overnight swap. Each raises the bar, so a negative result here is safe
and a positive one would need them before it meant anything.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from forecast_lab.contracts import ForecastLabError

#: Euler-Mascheroni, in the expected-maximum formula behind the Deflated Sharpe Ratio.
EULER_MASCHERONI = 0.5772156649015329
#: Bootstrap replications for SPA and StepM. Enough for a p-value to two decimals.
DEFAULT_REPS = 1_000
#: Fixed, for the same reason `dependence` fixes one: a published p-value that moves
#: between runs is not a measurement.
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05


class SignificanceError(ForecastLabError):
    """A test cannot be run on what was provided."""


# --- Pesaran-Timmermann ----------------------------------------------------------------


@dataclass(frozen=True)
class DirectionalTest:
    """Whether prediction and outcome are dependent, rather than merely often equal."""

    n: int
    accuracy: float
    #: What accuracy the two marginals produce on their own, with no information passing
    #: between them. The number a naive "beat 50%" comparison silently replaces with 0.5.
    independent_accuracy: float
    statistic: float
    p_value: float

    @property
    def excess(self) -> float:
        return self.accuracy - self.independent_accuracy

    def significant(self, alpha: float = DEFAULT_ALPHA) -> bool:
        return self.p_value < alpha


def pesaran_timmermann(predicted: ArrayLike, actual: ArrayLike) -> DirectionalTest:
    """Test directional accuracy against independence, one-sided.

    The statistic is `(accuracy - independent) / sqrt(var(accuracy) - var(independent))`,
    asymptotically standard normal. The subtraction in the denominator is the part worth
    understanding: it removes the variance the independence benchmark carries *because it
    is itself estimated from the same marginals*, and omitting it - which a naive binomial
    test does - makes every result look more significant than it is.
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.size != a.size:
        raise SignificanceError(f"{p.size:,} predictions against {a.size:,} outcomes")
    n = p.size
    if n < 2:
        raise SignificanceError(f"at least two observations are needed, got {n}")

    py = float(a.mean())  # share of actual UP
    px = float(p.mean())  # share of predicted UP
    accuracy = float((p == a).mean())
    independent = py * px + (1 - py) * (1 - px)

    var_accuracy = independent * (1 - independent) / n
    var_independent = (
        ((2 * py - 1) ** 2) * px * (1 - px) / n
        + ((2 * px - 1) ** 2) * py * (1 - py) / n
        + 4 * py * px * (1 - py) * (1 - px) / (n**2)
    )
    denominator = var_accuracy - var_independent
    if denominator <= 0:
        # Degenerate: a constant predictor makes the two variances coincide. There is no
        # dependence to detect and saying so beats returning an infinity.
        return DirectionalTest(
            n=n,
            accuracy=accuracy,
            independent_accuracy=independent,
            statistic=0.0,
            p_value=1.0,
        )

    statistic = (accuracy - independent) / math.sqrt(denominator)
    return DirectionalTest(
        n=n,
        accuracy=accuracy,
        independent_accuracy=independent,
        statistic=statistic,
        p_value=_upper_tail(statistic),
    )


# --- Hansen SPA and Romano-Wolf StepM --------------------------------------------------


@dataclass(frozen=True)
class Superiority:
    """Whether anything beat the benchmark, once being the best of many is accounted for."""

    benchmark: str
    #: Model names in the order they were supplied, so the index sets below can be read.
    models: tuple[str, ...]
    #: Hansen's three p-values. `consistent` is the one to quote; `lower` and `upper`
    #: bracket it and are reported because a gap between them means the answer depends on
    #: how poor models are handled, which is a fact about the comparison, not the data.
    p_lower: float
    p_consistent: float
    p_upper: float
    #: Beat the benchmark at the stated size, per SPA.
    better: tuple[str, ...]
    #: Rejected by StepM, controlling the familywise error rate.
    stepwise: tuple[str, ...]
    alpha: float

    @property
    def any_survives(self) -> bool:
        return self.p_consistent < self.alpha


def superior_predictive_ability(
    benchmark: ArrayLike,
    models: Mapping[str, ArrayLike],
    *,
    block_size: int | None = None,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> Superiority:
    """Run SPA and StepM on **loss** series, where lower is better.

    Pass losses, not performance: `arch` follows Hansen's convention and a sign error here
    would silently invert every conclusion. `costs.directional_returns()` produces the
    return series; a loss is its negation.

    ``block_size`` is the stationary-bootstrap block length. Left unset it is chosen the
    same way `dependence` chooses one, so both modules answer to the same measurement
    rather than to two different guesses.
    """
    from arch.bootstrap import SPA, StepM

    if not models:
        raise SignificanceError("no models to compare against the benchmark")
    bench = np.asarray(benchmark, dtype=float)
    names = tuple(models)
    # A DataFrame rather than a bare array, so `StepM` reports column names. `SPA` reports
    # positions either way and is translated below; the asymmetry is `arch`'s, not a
    # choice made here.
    matrix = pd.DataFrame(
        {name: np.asarray(models[name], dtype=float) for name in names}, columns=list(names)
    )
    if len(matrix) != bench.size:
        raise SignificanceError(
            f"benchmark has {bench.size:,} observations, models have {len(matrix):,}"
        )
    if bench.size < 2:
        raise SignificanceError("at least two observations are needed")

    if block_size is None:
        from forecast_lab.research.dependence import optimal_block

        block_size = max(1, round(optimal_block(bench)))

    spa = SPA(bench, matrix, block_size=block_size, reps=reps, seed=seed)
    spa.compute()
    # The two `arch` APIs are NOT symmetric, and the difference is silent. Given the same
    # DataFrame, `StepM.superior_models` returns column *names* while `SPA.better_models`
    # returns positional *indices* - so passing its output through `str()` yields '4',
    # which is a valid-looking label belonging to no model. Found by a test that asserted
    # a known-superior model came back by name.
    better = tuple(names[int(position)] for position in spa.better_models(alpha))

    step = StepM(bench, matrix, size=alpha, block_size=block_size, reps=reps, seed=seed)
    step.compute()
    stepwise = tuple(str(name) for name in step.superior_models)

    return Superiority(
        benchmark="benchmark",
        models=names,
        p_lower=float(spa.pvalues["lower"]),
        p_consistent=float(spa.pvalues["consistent"]),
        p_upper=float(spa.pvalues["upper"]),
        better=better,
        stepwise=stepwise,
        alpha=alpha,
    )


# --- Probabilistic and Deflated Sharpe --------------------------------------------------


@dataclass(frozen=True)
class SharpeVerdict:
    """A Sharpe ratio, and how much of it survives its own uncertainty and the search."""

    n: int
    sharpe: float
    skew: float
    #: Non-excess kurtosis: 3.0 is Gaussian.
    kurtosis: float
    trials: int
    #: Probability the true Sharpe exceeds zero, given this sample's shape.
    probabilistic: float
    #: The Sharpe the best of `trials` random strategies would be expected to reach.
    expected_maximum: float
    #: Probability the true Sharpe exceeds that expected maximum - the honest one.
    deflated: float

    def survives(self, alpha: float = DEFAULT_ALPHA) -> bool:
        return self.deflated > 1 - alpha


def deflated_sharpe(
    returns: ArrayLike, *, trials: int, trial_sharpes: Sequence[float] | None = None
) -> SharpeVerdict:
    """Bailey and Lopez de Prado's PSR and DSR.

    Two corrections a bare Sharpe ratio omits, both of which matter here.

    **Shape.** The confidence interval around a Sharpe assumes normal returns. Hourly gold
    returns reach ten standard deviations - this project measured that in its own EDA - so
    the interval computed under normality is too narrow. Skew and kurtosis enter the
    denominator directly.

    **The search.** A Sharpe of 0.5 is unremarkable if it is the best of eighteen. The
    expected maximum under the null grows with the number of trials, and the DSR asks
    whether the observed Sharpe beats *that* rather than beats zero. ``trial_sharpes``
    supplies the spread across the configurations actually tried; without it a
    conventional variance is assumed and the result is stated as approximate.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        raise SignificanceError(f"a Sharpe ratio needs at least four returns, got {n}")
    if trials < 1:
        raise SignificanceError(f"there must be at least one trial, got {trials}")

    deviation = float(r.std(ddof=1))
    if deviation == 0.0:
        raise SignificanceError("a constant return series has no Sharpe ratio")
    sharpe = float(r.mean()) / deviation
    centred = (r - r.mean()) / deviation
    skew = float((centred**3).mean())
    kurtosis = float((centred**4).mean())

    def psr(threshold: float) -> float:
        # The variance of the Sharpe estimator under non-normality.
        variance = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
        if variance <= 0:
            raise SignificanceError("the Sharpe estimator's variance is not positive")
        return _lower_tail((sharpe - threshold) * math.sqrt(n - 1) / math.sqrt(variance))

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        spread = float(np.std(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        # Falls back to the variance a single Sharpe estimate carries, which is the right
        # order of magnitude and is labelled rather than hidden.
        spread = 1.0 / math.sqrt(n - 1)

    expected_maximum = spread * _expected_maximum_z(trials)
    return SharpeVerdict(
        n=n,
        sharpe=sharpe,
        skew=skew,
        kurtosis=kurtosis,
        trials=trials,
        probabilistic=psr(0.0),
        expected_maximum=expected_maximum,
        deflated=psr(expected_maximum),
    )


def _expected_maximum_z(trials: int) -> float:
    """Expected maximum of ``trials`` standard normals, to the usual two terms."""
    if trials == 1:
        return 0.0
    return (1 - EULER_MASCHERONI) * _inverse_normal(1 - 1 / trials) + EULER_MASCHERONI * (
        _inverse_normal(1 - 1 / (trials * math.e))
    )


# --- the scipy boundary -----------------------------------------------------------------


def _upper_tail(statistic: float) -> float:
    from scipy.stats import norm

    return float(norm.sf(statistic))


def _lower_tail(statistic: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(statistic))


def _inverse_normal(probability: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(probability))


# --- multiplicity, for the tests that do not carry their own ---------------------------


def holm(p_values: Mapping[str, float], *, alpha: float = DEFAULT_ALPHA) -> dict[str, bool]:
    """Holm-Bonferroni: which of many p-values survive, controlling familywise error.

    SPA and StepM handle multiplicity internally; Pesaran-Timmermann does not, and running
    it eighteen times and quoting the smallest p-value would be the exact error this
    project accused the original analysis of. Holm rather than Bonferroni because it is
    uniformly more powerful and no less valid, so the weaker correction would be conceding
    detections for nothing.
    """
    if not p_values:
        raise SignificanceError("no p-values to correct")
    if not 0 < alpha < 1:
        raise SignificanceError(f"alpha must be in (0, 1), got {alpha}")

    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    total = len(ordered)
    verdict: dict[str, bool] = {}
    surviving = True
    for rank, (name, p) in enumerate(ordered):
        # Step down: once one fails, every larger p-value fails too.
        if surviving and p >= alpha / (total - rank):
            surviving = False
        verdict[name] = surviving
    return {name: verdict[name] for name in p_values}
