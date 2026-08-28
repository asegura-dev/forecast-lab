"""How large an effect a design can see, and whether it could have seen the one that matters.

This module answers the question that decides whether a negative result means anything.
An experiment that finds nothing has two possible explanations - there was nothing, or
it could not have seen it - and only a power calculation tells them apart. The original
project ran neither, which is why its inconclusive result was read as a finding and why
this one could have been read as a non-finding.

**And computing it changed this project's own conclusion.** For a week the argument was
that a single 70/15/15 split "cannot resolve the effect it exists to test": its minimum
detectable effect is 2.17 points against a break-even 1.92 points above chance. That was
true of the *assumed* break-even. Measured at the venue, the round trip is 1.86 bps and
the break-even is **53.49%** - so the effect worth detecting is **3.49 points**, not 1.92,
and every design here sees it easily:

| Design | Out-of-sample bars | MDE | Power for +3.49 pp |
|---|---:|---:|---:|
| Reference, single split | 3,294 | 2.17% | **99.1%** |
| Canonical, single split | 7,306 | 1.45% | **100.0%** |
| Canonical, walk-forward | 40,587 | 0.62% | 100.0% |

Only **1,268 bars** are needed to detect a profitable edge at 80% power. The walk-forward
design scores 40,587. (1,268 rather than 1,269 because the break-even is 53.4927% before
rounding, so the effect is fractionally larger than the 3.49 points quoted everywhere.)

That converts the verdict from "we could not see" into **"we looked with power to spare
and there was nothing"** - which is a result rather than a shrug. It still does not rule
out an edge of half a point; it rules out one worth having, which is the only kind the
question was about.

**What this does not model.** Overlapping feature windows make neighbouring bars
dependent, so the *effective* sample is smaller than the row count and every figure here
is optimistic by some unmeasured factor. The correction needs a stationary bootstrap and
belongs with the evaluation battery; it is named here so no reader takes 100% literally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from forecast_lab.contracts import ForecastLabError

#: Conventional defaults, stated rather than hidden in a signature.
DEFAULT_POWER = 0.80
DEFAULT_ALPHA = 0.05


class PowerError(ForecastLabError):
    """A power calculation cannot be made from what was provided."""


@dataclass(frozen=True)
class Design:
    """What one validation design can and cannot see."""

    label: str
    n: int
    power: float
    alpha: float
    one_sided: bool

    @property
    def standard_error(self) -> float:
        """SE of an accuracy near one half - the worst case, and the honest one.

        p(1-p) is maximised at p = 0.5, so using 0.5 rather than the observed accuracy
        gives the widest interval. On a question whose answers all sit within two points
        of a half, the difference is negligible and the conservative choice costs nothing.
        """
        return 0.5 / math.sqrt(self.n)

    @property
    def minimum_detectable_effect(self) -> float:
        """The smallest true edge this design would detect ``power`` of the time."""
        return (_z(1 - self.alpha if self.one_sided else 1 - self.alpha / 2) + _z(self.power)) * (
            self.standard_error
        )

    def power_for(self, effect: float) -> float:
        """The probability of detecting an edge of exactly ``effect``, if it is real."""
        if effect <= 0:
            return self.alpha
        from scipy.stats import norm

        critical = _z(1 - self.alpha if self.one_sided else 1 - self.alpha / 2)
        return float(norm.cdf(effect / self.standard_error - critical))

    def resolves(self, effect: float) -> bool:
        """Whether this design sees ``effect`` at its stated power."""
        return effect >= self.minimum_detectable_effect


def _z(probability: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(probability))


def design(
    n: int,
    *,
    label: str = "",
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    one_sided: bool = True,
) -> Design:
    """Describe a validation design of ``n`` out-of-sample observations.

    One-sided by default, because the question is "is this better than chance" rather
    than "is this different from chance". A two-sided test would demand a larger effect
    for the same power, and using one while asking a directional question is the kind of
    quiet conservatism that makes a negative result unfalsifiable.
    """
    if n < 2:
        raise PowerError(f"a design needs at least two observations, got {n}")
    if not 0 < power < 1:
        raise PowerError(f"power must be in (0, 1), got {power}")
    if not 0 < alpha < 1:
        raise PowerError(f"alpha must be in (0, 1), got {alpha}")
    return Design(label=label or f"n={n:,}", n=n, power=power, alpha=alpha, one_sided=one_sided)


def required_sample(
    effect: float,
    *,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    one_sided: bool = True,
) -> int:
    """How many observations are needed to detect ``effect`` at ``power``.

    The number to compute *before* running an experiment, and the one the original
    project never did. Measured here: detecting the edge that would clear the venue's
    real costs needs **1,268 bars** (3.4927 points, before the rounding to 3.49 that the
    prose uses). Detecting the 1.92 points implied by the optimistic cost assumption
    needs 4,193. Both are available from a single split of this data, which is why the
    negative result is informative rather than merely quiet.
    """
    if effect <= 0:
        raise PowerError(f"the effect must be positive, got {effect}")
    if not 0 < power < 1 or not 0 < alpha < 1:
        raise PowerError("power and alpha must each be in (0, 1)")

    critical = _z(1 - alpha if one_sided else 1 - alpha / 2)
    return int(((critical + _z(power)) * 0.5 / effect) ** 2) + 1
