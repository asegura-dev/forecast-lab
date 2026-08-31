"""How much of the sample is real, when neighbouring bars are not independent.

`power.py` computes every standard error as ``0.5 / sqrt(n)``, which assumes the n
outcomes are independent draws. [ADR-011](../../../docs/adr/ADR-011-power-before-verdict.md)
flagged that assumption as broken and said so in every place the figure appeared:
overlapping feature windows make neighbouring rows dependent, so the effective sample must
be smaller than the row count and the power "optimistic by an unmeasured factor".

**Measured, the factor is 0.97.** (0.99 on the highest-scoring model, 0.97 on the one that
comes closest to paying for itself - below one either way, so the naive figure was if
anything conservative.) The caveat was wrong, and the reason is worth more than the
correction. The features *are* strongly dependent - on the canonical series, RSI-14
autocorrelates at +0.93 from one bar to the next and realised volatility at +0.99. But the
quantity being averaged is not a feature. It is whether the model got the direction right,
and **that series carries essentially no serial dependence at all** (-0.020 at lag one on
40,587 bars, against a standard error of 0.005), because the thing it is tracking does not
either: the label itself autocorrelates at -0.025.

A near-coin-flip outcome inherits almost none of the dependence of its inputs. Strongly
autocorrelated features feeding a model with a fifth of a point of skill produce hits and
misses that are close to independent draws - and it is the hits and misses that are being
averaged. So `0.5 / sqrt(n)` was right, 100% power is literal rather than optimistic, and
the honest thing is to publish the number that dissolves a caveat this project made
prominently and repeatedly.

**Why it is still worth computing.** A negative result about one's own caveat is only
credible if the measurement was capable of finding the opposite. On a synthetic AR(1)
correctness series the same code returns an inflation of **3.64x** and an effective sample
of 1,513 out of 20,000, which is pinned by a test. The instrument works; the dependence is
not there.

**Folds are bootstrapped separately.** A pooled walk-forward series is five contiguous
stretches with years of gap between them. Resampling it as one series would let a block
straddle a boundary and treat 2020 as adjacent to 2022. Each segment is resampled on its
own and the variances are combined, which is also what makes the fold structure visible in
the output rather than averaged away.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from forecast_lab.contracts import ForecastLabError

#: Fixed so the figure is byte-reproducible. A bootstrap whose published number moves
#: between runs is not a measurement, and this project fits Random Forest single-threaded
#: for the same reason.
BOOTSTRAP_SEED = 42
#: Enough for a standard error to two significant figures; the cost is linear.
BOOTSTRAP_REPLICATIONS = 1_000
#: A stationary bootstrap draws block lengths from a geometric distribution with mean `b`,
#: so `1/b` has to be a probability. Politis-White can return less than one when it finds
#: no dependence at all - 0.1 on one fold here - and the honest reading of that is "resample
#: independently", which is what a block length of exactly one does.
MINIMUM_BLOCK = 1.0


class DependenceError(ForecastLabError):
    """The dependence in a series cannot be measured from what was provided."""


@dataclass(frozen=True)
class Dependence:
    """What the serial dependence in an outcome series costs the standard error."""

    n: int
    #: Bars per segment, in order - the fold structure, kept visible.
    segments: tuple[int, ...]
    #: Politis-White optimal stationary block length per segment. Around one means the
    #: series is indistinguishable from independent draws.
    blocks: tuple[float, ...]
    #: Lag-one autocorrelation, pooled within segments. The cheap diagnostic that agrees
    #: with the expensive one, and the first thing to look at if they ever disagree.
    lag_one: float
    naive_standard_error: float
    bootstrap_standard_error: float

    @property
    def inflation(self) -> float:
        """How much wider the true standard error is than the independent-draws one.

        Above 1 means dependence is costing resolution. At or below 1 means the naive
        figure was already honest - which is what this project measured.
        """
        return self.bootstrap_standard_error / self.naive_standard_error

    @property
    def effective_sample(self) -> float:
        """The number of independent draws this series is worth."""
        return (0.5 / self.bootstrap_standard_error) ** 2

    @property
    def material(self) -> bool:
        """Whether the correction changes anything a reader would act on.

        Ten percent is a judgement, stated rather than hidden: below it the MDE moves by
        less than the rounding this project reports its accuracies at.
        """
        return abs(self.inflation - 1.0) >= 0.10


def measure_dependence(
    segments: Sequence[ArrayLike],
    *,
    seed: int = BOOTSTRAP_SEED,
    replications: int = BOOTSTRAP_REPLICATIONS,
) -> Dependence:
    """Measure how much serial dependence inflates the standard error of a mean.

    ``segments`` are contiguous stretches of a 0/1 outcome series - one per walk-forward
    fold, or a single segment for one block; anything `numpy` can read is accepted. They
    are resampled independently and their variances combined, because a block straddling
    two folds would treat years apart as adjacent.
    """
    arrays = [np.asarray(segment, dtype=float) for segment in segments]
    arrays = [a for a in arrays if a.size > 0]
    if not arrays:
        raise DependenceError("no observations to measure dependence in")
    if replications < 2:
        raise DependenceError(f"a bootstrap needs at least two replications, got {replications}")

    n = int(sum(a.size for a in arrays))
    if n < 2:
        raise DependenceError(f"at least two observations are needed, got {n}")

    blocks: list[float] = []
    variance = 0.0
    for index, array in enumerate(arrays):
        block = optimal_block(array)
        blocks.append(block)
        standard_error = _bootstrap_standard_error(
            array, block=block, seed=seed + index, replications=replications
        )
        # Var of a weighted mean of independent segment means: sum of (n_i * se_i)^2,
        # divided by n^2 at the end.
        variance += (standard_error * array.size) ** 2

    return Dependence(
        n=n,
        segments=tuple(int(a.size) for a in arrays),
        blocks=tuple(blocks),
        lag_one=_pooled_lag_one(arrays),
        naive_standard_error=0.5 / math.sqrt(n),
        bootstrap_standard_error=math.sqrt(variance) / n,
    )


def optimal_block(series: ArrayLike) -> float:
    """Politis-White optimal block length, floored at one.

    Public because `significance` bootstraps the same series and must not choose its
    block length by a different rule - two modules disagreeing about how dependent the
    data is would make their p-values incomparable.

    Not reimplemented: the selection rule involves a spectral estimate and a correlogram
    cut-off, and `arch` is the standard implementation. What is *not* delegated is the
    resampling seed, which is fixed here so the published figure can be recomputed.
    """
    from arch.bootstrap import optimal_block_length

    array = np.asarray(series, dtype=float)

    if array.size < 3 or float(np.std(array)) == 0.0:
        # A constant series has no dependence to find and would divide by zero in the
        # correlogram. One is the correct answer, not a fallback.
        return MINIMUM_BLOCK
    try:
        selected = float(optimal_block_length(array)["stationary"].iloc[0])
    except (ValueError, ZeroDivisionError, IndexError) as exc:
        raise DependenceError(f"the block length could not be selected: {exc}") from exc
    return max(MINIMUM_BLOCK, selected)


def _mean_of(sample: np.ndarray) -> np.ndarray:
    """The statistic being bootstrapped, as the one-element array `apply` expects.

    Returning a bare float would be the obvious spelling and `apply` is typed against
    arrays; keeping the shape honest is cheaper than silencing the checker.
    """
    return np.atleast_1d(np.asarray(sample).mean())


def _bootstrap_standard_error(
    array: np.ndarray, *, block: float, seed: int, replications: int
) -> float:
    from arch.bootstrap import StationaryBootstrap

    # `arch` types the block size as `int`, but a *stationary* bootstrap draws block
    # lengths from a geometric distribution whose mean is real by construction - Politis
    # and White's selection rule returns 2.1 and 101.3, not integers. Rounding to satisfy
    # the annotation would change the published number to work around a wrong stub, so
    # the stub is overridden here instead and nowhere else.
    bootstrap = StationaryBootstrap(block, array, seed=seed)  # type: ignore[arg-type]
    draws = bootstrap.apply(_mean_of, replications)
    return float(np.std(np.asarray(draws), ddof=1))


def lag_one_autocorrelation(segments: Sequence[ArrayLike]) -> float:
    """Lag-one autocorrelation, never crossing a segment boundary.

    Public because the central claim of ADR-012 is a *contrast* between series - the
    features autocorrelate at +0.93, what is being averaged does not - and a contrast
    quoted from a one-off script is not reproducible. Callers pass whichever series
    they want the figure for.
    """
    return _pooled_lag_one([np.asarray(s, dtype=float) for s in segments])


def _pooled_lag_one(arrays: list[np.ndarray]) -> float:
    numerator = denominator = 0.0
    for array in arrays:
        if array.size < 2:
            continue
        centred = array - array.mean()
        numerator += float((centred[:-1] * centred[1:]).sum())
        denominator += float((centred * centred).sum())
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
