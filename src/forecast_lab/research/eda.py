"""Describing the data before modelling it, and the tests worth running on it.

Exploratory analysis is where a project decides what it believes before any model
exists, so a mistake here survives every correction made downstream. The original
project's EDA contains three, and each is a correct procedure pointed at a question
that did not need asking.

**A normality test on the price.** It runs D'Agostino-Pearson on `XAUUSD_Close` and
reports that the data are not normal. A price series with a trend is not normal, and
nothing follows from establishing it - the price is not what any model here consumes.
The question with content is whether the *returns* are normal, which decides whether a
Sharpe ratio or a t-test on them means anything. Both are computed and reported here,
with the price test kept only so the contrast is visible.

**A t-test that cannot fail.** The original defines `Direction = (Price_Change > 0)` and
then compares `Price_Change` between the groups that condition created, reporting
p < 0.001 and "significant". By construction the UP group has positive changes and the
DOWN group negative ones. It is a tautology wearing a p-value, and it feeds the report's
interpretation section. The comparison this module runs instead is on variables known
*before* the outcome - which is the only version that could have been informative.

**Correlations on price levels.** The report reads gold against the S&P at +0.923 as an
economic finding. Two series that both trend upward correlate on levels regardless of any
relationship, and `plots.correlation_comparison` draws the same pair on returns for
contrast. That figure lives with the other charts; the number is computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from forecast_lab.contracts import ForecastLabError

#: Below this many observations the shape statistics are noise dressed as description.
MIN_OBSERVATIONS = 30


class EdaError(ForecastLabError):
    """A description cannot be computed from what was provided."""


@dataclass(frozen=True)
class Description:
    """The summary statistics of one series, as the original reported them."""

    name: str
    count: int
    mean: float
    median: float
    mode: float
    std: float
    variance: float
    minimum: float
    maximum: float
    range: float
    iqr: float
    skewness: float
    kurtosis: float
    first_value: float
    last_value: float
    minimum_at: pd.Timestamp | None
    maximum_at: pd.Timestamp | None

    @property
    def total_change(self) -> float:
        return self.last_value - self.first_value

    @property
    def total_change_pct(self) -> float:
        return (self.last_value / self.first_value - 1.0) * 100 if self.first_value else 0.0


@dataclass(frozen=True)
class NormalityTest:
    """One normality test, with what its rejection does and does not license."""

    name: str
    statistic: float
    p_value: float
    n: int

    @property
    def rejects_normality(self) -> bool:
        return self.p_value < 0.05


@dataclass(frozen=True)
class MoveCounts:
    """How often the series rose, fell, or did neither."""

    up: int
    down: int
    flat: int

    @property
    def total(self) -> int:
        return self.up + self.down + self.flat

    @property
    def up_share(self) -> float:
        return self.up / self.total if self.total else 0.0


def describe(series: pd.Series, *, name: str = "") -> Description:
    """Summary statistics, including the shape measures the original reported.

    Mode is included because the original reported it. On a continuous price it is
    close to meaningless - it picks whichever value happened to repeat - and it is kept
    only so the two summaries can be compared line for line.
    """
    values = series.dropna()
    if len(values) < MIN_OBSERVATIONS:
        raise EdaError(f"{name or series.name}: {len(values)} observations is too few to describe")

    modes = values.mode()
    quartiles = values.quantile([0.25, 0.75])
    # pandas-stubs types an element access as the union of every dtype a Series can
    # hold, so the numeric narrowing has to be explicit rather than inferred.
    numeric = values.to_numpy(dtype="float64")

    return Description(
        name=name or str(series.name or "series"),
        count=len(values),
        mean=float(values.mean()),
        median=float(values.median()),
        mode=float(modes.iloc[0]) if len(modes) else float("nan"),
        std=float(values.std()),
        variance=float(values.var()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        range=float(values.max() - values.min()),
        iqr=float(quartiles.iloc[1] - quartiles.iloc[0]),
        # `skew` and `kurtosis` return a scalar at runtime, but pandas-stubs types
        # them as the union of every dtype a Series can hold. A cast is the project
        # policy over `type: ignore`: it never goes "unused" when the stubs improve.
        skewness=cast(float, values.skew()),
        kurtosis=cast(float, values.kurtosis()),
        first_value=float(numeric[0]),
        last_value=float(numeric[-1]),
        minimum_at=_timestamp_of(values.idxmin()),
        maximum_at=_timestamp_of(values.idxmax()),
    )


def _timestamp_of(label: object) -> pd.Timestamp | None:
    return label if isinstance(label, pd.Timestamp) else None


def normality(series: pd.Series, *, name: str = "") -> NormalityTest:
    """D'Agostino-Pearson, the test the original used.

    Reported for whatever it is handed, but the reading depends entirely on *what* it is
    handed. On a trending price a rejection is uninformative - it would reject on any
    series with a drift. On returns it is worth having: a rejection there is what makes a
    Sharpe ratio's usual confidence interval wrong, and this data rejects it decisively.
    """
    from scipy.stats import normaltest

    values = series.dropna().to_numpy(dtype="float64")
    if values.size < MIN_OBSERVATIONS:
        raise EdaError(f"{name}: {values.size} observations is too few to test")

    statistic, p_value = normaltest(values)
    return NormalityTest(
        name=name or str(series.name or "series"),
        statistic=float(statistic),
        p_value=float(p_value),
        n=int(values.size),
    )


def count_moves(returns: pd.Series) -> MoveCounts:
    """Rises, falls and exact ties, counted separately.

    Ties get their own count rather than being folded into either direction, for the
    same reason the labeller names them FLAT: a strictly-greater comparison would
    deposit every one of them on the DOWN side.
    """
    values = returns.dropna()
    return MoveCounts(
        up=int((values > 0).sum()),
        down=int((values < 0).sum()),
        flat=int((values == 0).sum()),
    )


def by_year(close: pd.Series) -> pd.DataFrame:
    """Per-year mean price, volatility, range and direction share.

    The original plotted these four and read the volatility panel as a regime story. It
    is also the panel that shows why a single chronological split is fragile: if the
    years differ this much, the block a model is tested on decides a good part of its
    score.
    """
    frame = pd.DataFrame({"close": close.dropna()})
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise EdaError("a datetime index is required to group by year")

    frame["ret"] = frame["close"].pct_change()
    grouped = frame.groupby(frame.index.year)

    out = pd.DataFrame(
        {
            "bars": grouped.size(),
            "mean_price": grouped["close"].mean(),
            "min_price": grouped["close"].min(),
            "max_price": grouped["close"].max(),
            # Annualised from hourly observations, so the column is comparable with the
            # realised-volatility feature rather than being a different quantity with a
            # similar name.
            "volatility": grouped["ret"].std() * np.sqrt(24 * 365),
            "up_share": grouped["ret"].apply(lambda r: float((r.dropna() > 0).mean())),
        }
    )
    out["range"] = out["max_price"] - out["min_price"]
    return out


def correlation_pairs(
    prices: pd.DataFrame, *, target: str
) -> pd.DataFrame:
    """Correlation of every symbol with the target, on levels and on returns.

    The two columns exist to be compared. The original's EDA reports the levels number
    for gold against the S&P - **+0.923** - and reads it as an economic relationship.
    Two series that both trend upward produce that whatever their relationship, and the
    returns column is what a model at this horizon actually sees.
    """
    if target not in prices.columns:
        raise EdaError(f"{target} is not among the columns supplied")

    levels = prices.corr()[target].drop(target)
    returns = prices.pct_change().corr()[target].drop(target)

    out = pd.DataFrame({"on_levels": levels, "on_returns": returns})
    out["inflation"] = out["on_levels"].abs() - out["on_returns"].abs()
    return out.sort_values("on_levels", key=abs, ascending=False)


def compare_by_direction(
    frame: pd.DataFrame, labels: pd.Series, columns: list[str]
) -> pd.DataFrame:
    """Welch t-tests of each column between the rows that rose and those that fell.

    **The columns must be knowable before the outcome.** The original ran this over
    `Price_Change` after defining direction as `Price_Change > 0`, so the groups were
    built by the very condition being tested and the result was a certainty reported as
    a discovery. A guard here cannot detect that in general - whether a column leaks the
    answer is a fact about how it was built - so the caller passes the columns and the
    docstring carries the warning.

    Welch rather than Student: the two groups have no reason to share a variance, and
    assuming they do inflates significance when they do not.
    """
    from scipy.stats import ttest_ind

    aligned = labels.reindex(frame.index)
    up = frame.loc[aligned == 1.0]
    down = frame.loc[aligned == 0.0]
    if len(up) < MIN_OBSERVATIONS or len(down) < MIN_OBSERVATIONS:
        raise EdaError("both directions need enough observations to compare")

    rows = []
    for column in columns:
        if column not in frame.columns:
            raise EdaError(f"{column} is not in the frame")
        a, b = up[column].dropna(), down[column].dropna()
        statistic, p_value = ttest_ind(a, b, equal_var=False)
        rows.append(
            {
                "variable": column,
                "up_mean": float(a.mean()),
                "down_mean": float(b.mean()),
                "difference": float(a.mean() - b.mean()),
                "t_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def qq_points(series: pd.Series, *, sample: int | None = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Theoretical and observed quantiles for a normal Q-Q plot.

    Computed here rather than in the plotting module so `plots` needs no statistics
    library: a figure is a drawing of numbers, and producing the numbers is this layer's
    job.

    ``sample`` thins very long series by taking evenly spaced order statistics. A Q-Q
    plot of 23,000 points is a solid band that hides the tails, which are the only part
    anyone reads it for.
    """
    from scipy.stats import norm

    values = np.sort(series.dropna().to_numpy(dtype="float64"))
    if values.size < MIN_OBSERVATIONS:
        raise EdaError(f"{values.size} observations is too few for a Q-Q plot")

    if sample is not None and values.size > sample:
        picks = np.linspace(0, values.size - 1, sample).astype(int)
        values = values[picks]

    n = values.size
    # Blom's plotting positions: (i - 3/8) / (n + 1/4), which is the convention scipy's
    # probplot uses and keeps the extreme quantiles finite.
    positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical: np.ndarray = norm.ppf(positions)

    standardised = (values - values.mean()) / values.std() if values.std() else values
    return theoretical, standardised


def feature_correlations(features: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Each feature's correlation with the direction, ordered by absolute size.

    The original ranked these and printed a top twenty. The ranking is worth having, but
    the number to watch is the *largest* one: if the strongest single feature correlates
    with the answer at 0.03, no combination of them is going to produce a large edge, and
    that is knowable before a model is fitted.

    Point-biserial by construction - one side is binary - which is just Pearson's r
    computed against a 0/1 variable, so no separate routine is needed.
    """
    aligned = labels.reindex(features.index)
    keep = aligned.notna()
    if int(keep.sum()) < MIN_OBSERVATIONS:
        raise EdaError("too few labelled rows to correlate against")

    numeric = features.loc[keep].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corrwith(aligned[keep].astype(float))
    ranked = correlations.dropna().sort_values(key=abs, ascending=False)
    return cast(pd.Series, ranked)


def multicollinear_pairs(features: pd.DataFrame, *, threshold: float = 0.9) -> pd.DataFrame:
    """Pairs of features correlated with each other beyond ``threshold``.

    The original ran this at 0.9 and printed the offenders. It is the measurement behind
    a claim this repository has been making without evidence: that PCA collapses the
    matrix because technical indicators computed from one price series are near-duplicates
    of each other. A count of redundant pairs is that claim, quantified.

    Only the upper triangle is walked, so each pair appears once rather than twice.
    """
    if not 0.0 < threshold <= 1.0:
        raise EdaError(f"threshold must be in (0, 1], got {threshold}")

    numeric = features.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    matrix = numeric.corr().to_numpy()
    names = list(numeric.columns)

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = matrix[i, j]
            if np.isfinite(value) and abs(value) > threshold:
                rows.append({"left": names[i], "right": names[j], "correlation": float(value)})

    frame = pd.DataFrame(rows, columns=["left", "right", "correlation"])
    if len(frame):
        frame = frame.sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)
    return frame
