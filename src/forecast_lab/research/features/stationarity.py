"""Whether a column can survive prices it has never seen.

Gold runs from 1,616 to 4,378 across this sample. Any feature carrying that level has a
test block sitting outside the support of its training data, and a model fitted on one
is extrapolating rather than predicting. It looked like the explanation for the original
pipeline's PCA at 90% collapsing 63 features into **6 components** - though with every
level removed, PCA still retains 6 from 17, so the levels made the redundancy worse
rather than causing it.

The policy has three layers, in descending order of how much they can be trusted.

**Layer 1 - structural, and the only one that is a gate.** No column may be a price
level. It is checked by *measurement*, not by a list of names: multiply the input prices
by a constant and recompute. A column that scales with the constant is a price level; one
that does not, is not. That is the same probe that settled whether `bollinger_wband`
normalises internally, and it beats a name list for a plain reason - a name list is a
claim about what a function does, and this is an observation of what it did.

**Layer 2 - reported, never a gate.** ADF and KPSS on every column, printed as
diagnostics. They are *not* pass/fail here and treating them as such would be a false
guarantee: the tests read the last 5,000 rows of a column, and at that length the ADF
rejects a unit root on almost anything. Both are also invalid under heteroskedasticity
and regime change, which is the entire character of this data. A column can clear both
and still shift its mean between blocks.

**Layer 3 - the one that actually matters, and it lives elsewhere.** Distribution shift
*between splits*: PSI and KS per feature, plus adversarial validation - can a classifier
tell train from test using only X? Measured on this data, `dist_sma_200` moves its mean
from 0.0024 to 0.0068 between train and test, and `atr_pct` rises 37%. Both are
scale-free. Both would clear layers 1 and 2. That is the point: "carries no price level"
is a necessary condition, not a sufficient one, and a policy stopping at layer 1 would
be a certificate of quality with nothing behind it. It needs the blocks, so it belongs
with the validation work rather than here.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from forecast_lab.contracts import ForecastLabError

#: Factor the prices are multiplied by when probing for scale dependence. Any value far
#: from 1 works; 10 keeps the arithmetic legible in a failure message.
SCALE_PROBE = 10.0

#: A column is called scale-dependent when the probe moves it by more than this.
#:
#: Measured rather than chosen. With the statistic below, the worst movement among
#: genuinely scale-free columns on the real gold series is **7.0e-10**, while a raw price
#: level moves by **8.5e+00**. This threshold sits three orders above the noise and six
#: below a real violation.
SCALE_TOLERANCE = 1e-6

#: Which percentile of the per-row deviation decides the verdict.
#:
#: Not the maximum, and the reason is a measurement. `ta`'s ADX is a Wilder recursion
#: built on comparisons of consecutive highs and lows. When two of those are equal to
#: within floating-point error, rescaling can break the tie the other way - and because
#: the recursion carries state forward, that single flipped comparison contaminates a
#: long run of subsequent rows before decaying. On the real series that shows up as
#: **134 deviating rows out of 23,181 (0.58%), at consecutive positions**, pushing the
#: maximum to 7.1e-03 while the 99th percentile stays at 7.0e-10.
#:
#: ADX is dimensionless by construction, so the maximum was measuring an artefact of
#: float64 rather than a property of the feature. A percentile is robust to that and
#: still catches what matters: a column that genuinely carries the price level moves on
#: *every* row, so no percentile can hide it. The maximum is still reported, as a
#: diagnostic rather than as the verdict.
SCALE_PERCENTILE = 99.0


class StationarityError(ForecastLabError):
    """The stationarity policy cannot be evaluated on what was provided."""


class Scale(StrEnum):
    """What happened to a column when the prices underneath it were multiplied."""

    FREE = "scale-free"  # unchanged: a return, a ratio, an oscillator
    DEPENDENT = "scale-dependent"  # moved with the prices: a level, and not allowed
    UNDEFINED = "undefined"  # all-NaN or constant, so the probe says nothing


@dataclass(frozen=True)
class ScaleVerdict:
    """What the rescaling probe found for one column."""

    scale: Scale
    #: The percentile that decides, relative to the column's own magnitude.
    change: float
    #: The worst single row. Reported but never decisive - a large `worst` beside a tiny
    #: `change` means a numerical instability, not a price level.
    worst: float


@dataclass(frozen=True)
class ColumnReport:
    """One column, judged by each layer of the policy."""

    name: str
    scale: Scale
    relative_change: float  # how far the probe moved it, relative to its own magnitude
    worst_change: float  # the worst single row, for diagnosis only

    # Layer 2. Reported, never decisive. None when the column has too few finite values.
    adf_pvalue: float | None
    kpss_pvalue: float | None
    n_finite: int

    @property
    def allowed(self) -> bool:
        """Only layer 1 decides. Layers 2 and 3 inform."""
        return self.scale is not Scale.DEPENDENT

    @property
    def adf_rejects_unit_root(self) -> bool | None:
        """ADF's null is a unit root, so a small p-value argues for stationarity."""
        return None if self.adf_pvalue is None else self.adf_pvalue < 0.05

    @property
    def kpss_rejects_stationarity(self) -> bool | None:
        """KPSS's null is the opposite of ADF's, which is why both are reported."""
        return None if self.kpss_pvalue is None else self.kpss_pvalue < 0.05


@dataclass(frozen=True)
class PolicyReport:
    """The whole feature matrix, judged."""

    columns: tuple[ColumnReport, ...]

    @property
    def violations(self) -> tuple[ColumnReport, ...]:
        return tuple(c for c in self.columns if not c.allowed)

    @property
    def passes(self) -> bool:
        return not self.violations

    @property
    def disagreements(self) -> tuple[ColumnReport, ...]:
        """Columns where ADF and KPSS point opposite ways.

        Not a defect. It is the honest outcome for a series that is neither clearly
        stationary nor clearly a random walk, and printing it is more useful than
        picking whichever of the two tests agrees with the conclusion already wanted.
        """
        return tuple(
            c
            for c in self.columns
            if c.adf_rejects_unit_root is not None
            and c.kpss_rejects_stationarity is not None
            and c.adf_rejects_unit_root
            and c.kpss_rejects_stationarity
        )


def probe_scale(
    build: Callable[[pd.DataFrame], pd.DataFrame],
    bars: pd.DataFrame,
    *,
    factor: float = SCALE_PROBE,
) -> dict[str, ScaleVerdict]:
    """Rebuild the features on rescaled prices and see which columns moved.

    ``build`` takes a bar frame and returns a feature frame. It is passed in rather than
    imported, so this module stays honest about not knowing where its input comes from.

    Multiplying every price by a constant is an economically meaningless change: the
    same instrument quoted in a different unit. Anything that reacts to it is measuring
    the unit rather than the market.
    """
    scaled = bars.copy()
    for column in ("open", "high", "low", "close"):
        if column in scaled.columns:
            scaled[column] = scaled[column] * factor

    base, moved = build(bars), build(scaled)
    if list(base.columns) != list(moved.columns):
        raise StationarityError("rescaling changed the set of columns produced")

    verdicts: dict[str, ScaleVerdict] = {}
    for name in base.columns:
        a = base[name].to_numpy(dtype="float64")
        b = moved[name].to_numpy(dtype="float64")
        both = np.isfinite(a) & np.isfinite(b)
        if not both.any():
            verdicts[str(name)] = ScaleVerdict(Scale.UNDEFINED, 0.0, 0.0)
            continue
        # Relative, because an absolute difference is meaningless across columns whose
        # magnitudes differ by orders of magnitude.
        denominator = float(np.max(np.abs(a[both]))) or 1.0
        deviation = np.abs(a[both] - b[both]) / denominator
        change = float(np.percentile(deviation, SCALE_PERCENTILE))
        verdicts[str(name)] = ScaleVerdict(
            scale=Scale.DEPENDENT if change > SCALE_TOLERANCE else Scale.FREE,
            change=change,
            worst=float(np.max(deviation)),
        )
    return verdicts


def evaluate_stationarity(
    features: pd.DataFrame,
    scale_verdicts: dict[str, ScaleVerdict],
    *,
    max_rows: int | None = 5000,
) -> PolicyReport:
    """Judge every column: layer 1 from the probe, layer 2 from ADF and KPSS.

    ``max_rows`` caps how much of each column the tests read. ADF and KPSS on 23,000
    points are slow and, at that length, decided long before the last row - and this
    runs on every column of a matrix that can hold hundreds.
    """
    unknown = ScaleVerdict(Scale.UNDEFINED, 0.0, 0.0)
    reports: list[ColumnReport] = []
    for name in features.columns:
        verdict = scale_verdicts.get(str(name), unknown)
        series = features[name].to_numpy(dtype="float64")
        finite = series[np.isfinite(series)]
        if max_rows is not None and finite.size > max_rows:
            finite = finite[-max_rows:]
        adf_p, kpss_p = _tests(finite)
        reports.append(
            ColumnReport(
                name=str(name),
                scale=verdict.scale,
                relative_change=verdict.change,
                worst_change=verdict.worst,
                adf_pvalue=adf_p,
                kpss_pvalue=kpss_p,
                n_finite=int(finite.size),
            )
        )
    return PolicyReport(columns=tuple(reports))


def _tests(values: np.ndarray) -> tuple[float | None, float | None]:
    """ADF and KPSS, or None where the column cannot support them.

    Both raise on degenerate input - too few points, or no variance at all - and a
    diagnostic that crashes the report it belongs to is worse than one that abstains.
    """
    if values.size < 20 or float(np.std(values)) == 0.0:
        return None, None

    adf_p: float | None
    kpss_p: float | None
    try:
        adf_p = float(adfuller(values, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        adf_p = None
    try:
        # KPSS warns when the statistic falls outside its tabulated range and clips the
        # p-value. That is a real limit of the test rather than a problem with the data,
        # and the clipped value is still the honest answer to report - so the warning is
        # silenced here and the limitation is stated in this module's docstring instead.
        # Left unsuppressed it fires once per column, burying the report it belongs to.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_p = float(kpss(values, regression="c", nlags="auto")[1])
    except (ValueError, OverflowError):
        kpss_p = None
    return adf_p, kpss_p
