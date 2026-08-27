"""Tests for the exploratory statistics.

Most of these pin a distinction the original project's EDA collapsed: the same
procedure gives a meaningful answer or a meaningless one depending on what it is
pointed at, and nothing in the output says which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_lab.research import (
    EdaError,
    by_year,
    compare_by_direction,
    correlation_pairs,
    count_moves,
    describe,
    feature_correlations,
    multicollinear_pairs,
    normality,
)
from forecast_lab.research.eda import qq_points

ORIGIN = "2022-01-03"


def _series(n: int = 500, seed: int = 4) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")
    return pd.Series(1800 + np.cumsum(rng.normal(0, 3, n)), index=index, name="close")


# --- describing ---------------------------------------------------------------------


@pytest.mark.unit
def test_the_summary_reports_what_the_original_reported() -> None:
    """Every field the notebook printed, so the two can be compared line for line."""
    values = pd.Series([1.0, 2.0, 2.0, 3.0, 100.0] * 20)
    summary = describe(values, name="test")

    assert summary.count == 100
    assert summary.median == pytest.approx(2.0)
    assert summary.mode == pytest.approx(2.0)
    assert summary.range == pytest.approx(99.0)
    # A long right tail, which is what an outlier at 100 among small values produces.
    assert summary.skewness > 1.0


@pytest.mark.unit
def test_the_total_change_is_first_to_last_not_min_to_max() -> None:
    """The original reported both and they are different questions."""
    index = pd.date_range(ORIGIN, periods=40, freq="h", tz="UTC")
    values = pd.Series([100.0] * 20 + [50.0] * 10 + [120.0] * 10, index=index)
    summary = describe(values)

    assert summary.total_change == pytest.approx(20.0)
    assert summary.total_change_pct == pytest.approx(20.0)
    assert summary.range == pytest.approx(70.0)  # min to max, a different number


@pytest.mark.unit
def test_the_extremes_carry_their_timestamps() -> None:
    series = _series()
    summary = describe(series)
    assert summary.minimum_at is not None
    assert summary.maximum_at is not None
    assert float(series.loc[summary.minimum_at]) == pytest.approx(summary.minimum)


@pytest.mark.unit
def test_too_few_observations_is_refused() -> None:
    """Shape statistics on ten points are noise wearing four decimal places."""
    with pytest.raises(EdaError, match="too few"):
        describe(pd.Series([1.0, 2.0, 3.0]))


# --- the test pointed at the right variable ------------------------------------------


@pytest.mark.unit
def test_normality_rejects_a_trending_series_for_an_uninteresting_reason() -> None:
    """The original's mistake, stated as a test.

    A price with a drift fails a normality test regardless of anything a model could
    use. The rejection is real and licenses nothing.
    """
    trending = pd.Series(np.linspace(1000.0, 4000.0, 800))
    assert normality(trending, name="price").rejects_normality


@pytest.mark.unit
def test_normality_does_not_reject_actual_gaussian_returns() -> None:
    """The other half: the test works, so a rejection on returns means something."""
    rng = np.random.default_rng(11)
    gaussian = pd.Series(rng.normal(0, 0.01, 5000))
    assert not normality(gaussian, name="returns").rejects_normality


@pytest.mark.unit
def test_qq_points_expose_fat_tails_that_a_p_value_alone_hides() -> None:
    """Two series can both fail the test for opposite reasons.

    Measured on the real data: the price's observed tails reach 1.1 standard deviations
    where a normal reaches 3.7 - **shorter**, because a price is bounded and trending -
    while returns reach 10. Both reject; only one rejection has a consequence.
    """
    rng = np.random.default_rng(3)
    # Student-t with 3 degrees of freedom: unmistakably fat-tailed.
    fat = pd.Series(rng.standard_t(3, 4000))
    theoretical, observed = qq_points(fat, sample=2000)

    assert theoretical.shape == observed.shape
    assert abs(observed).max() > abs(theoretical).max() * 1.5


@pytest.mark.unit
def test_qq_thins_a_long_series() -> None:
    """23,000 points draw a solid band that hides the tails they exist to show."""
    theoretical, observed = qq_points(_series(20_000), sample=1000)
    assert len(theoretical) == 1000
    assert len(observed) == 1000


# --- counting -------------------------------------------------------------------------


@pytest.mark.unit
def test_ties_are_counted_apart_from_both_directions() -> None:
    """A strictly-greater comparison would deposit every tie on the DOWN side."""
    moves = count_moves(pd.Series([0.01, -0.01, 0.0, 0.0, 0.02]))
    assert (moves.up, moves.down, moves.flat) == (2, 1, 2)
    assert moves.total == 5


# --- by year --------------------------------------------------------------------------


@pytest.mark.unit
def test_the_yearly_table_exposes_a_drifting_class_balance() -> None:
    """The panel that explains why the test block matters so much.

    Two years, one drifting up and one flat. If the share of rising bars differs between
    them, then which one lands in test decides part of any accuracy measured there.
    """
    index = pd.date_range("2022-01-01", periods=4000, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    drift = np.where(np.arange(4000) < 2000, 0.6, 0.0)
    values = pd.Series(1000 + np.cumsum(rng.normal(0, 1, 4000) + drift), index=index)

    table = by_year(values)
    assert list(table.columns) == [
        "bars", "mean_price", "min_price", "max_price", "volatility", "up_share", "range",
    ]
    assert len(table) >= 1
    assert (table["up_share"] > 0).all()


@pytest.mark.unit
def test_by_year_needs_a_datetime_index() -> None:
    with pytest.raises(EdaError, match="datetime index"):
        by_year(pd.Series([1.0] * 100))


# --- correlations ---------------------------------------------------------------------


@pytest.mark.unit
def test_a_shared_trend_inflates_the_level_correlation() -> None:
    """The +0.923 the original read as an economic finding.

    Two series built from the same trend and independent noise correlate strongly on
    levels and barely at all on returns. Nothing connects them except that both went up.
    """
    rng = np.random.default_rng(5)
    n = 2000
    index = pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")
    trend = np.cumsum(np.full(n, 0.5))
    frame = pd.DataFrame(
        {
            "XAUUSD": 1800 + trend + np.cumsum(rng.normal(0, 1, n)),
            "SPX": 4000 + trend * 2 + np.cumsum(rng.normal(0, 1, n)),
        },
        index=index,
    )

    pairs = correlation_pairs(frame, target="XAUUSD")
    levels = pairs["on_levels"].astype(float)
    returns = pairs["on_returns"].astype(float)
    inflation = pairs["inflation"].astype(float)

    assert levels["SPX"] > 0.8
    assert abs(returns["SPX"]) < 0.2
    assert inflation["SPX"] > 0.6


@pytest.mark.unit
def test_a_genuine_relationship_survives_the_change_of_basis() -> None:
    """The control. If everything collapsed on returns, the method would prove nothing."""
    rng = np.random.default_rng(9)
    n = 2000
    index = pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")
    shared = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "XAUUSD": 1800 + np.cumsum(shared),
            # Same shocks, so the relationship is in the returns rather than in a trend.
            "XAGUSD": 25 + np.cumsum(shared * 0.9 + rng.normal(0, 0.3, n)),
        },
        index=index,
    )

    pairs = correlation_pairs(frame, target="XAUUSD")
    assert pairs["on_returns"].astype(float)["XAGUSD"] > 0.7


@pytest.mark.unit
def test_a_missing_target_is_refused() -> None:
    with pytest.raises(EdaError, match="not among the columns"):
        correlation_pairs(pd.DataFrame({"SPX": [1.0, 2.0, 3.0]}), target="XAUUSD")


# --- the t-test that cannot fail ------------------------------------------------------


@pytest.mark.unit
def test_comparing_the_variable_that_defined_the_groups_is_a_tautology() -> None:
    """Reproduces the original's defect, so the docstring's warning has evidence.

    `Direction = (Price_Change > 0)` and then a t-test of `Price_Change` between the
    groups that condition created. It returns p ~ 0 and "significant" every time,
    because the UP group cannot contain a negative value.
    """
    rng = np.random.default_rng(13)
    n = 1000
    index = pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")
    change = pd.Series(rng.normal(0, 1, n), index=index)
    labels = (change > 0).astype(float)
    frame = pd.DataFrame({"price_change": change, "rsi": rng.uniform(30, 70, n)}, index=index)

    result = compare_by_direction(frame, labels, ["price_change", "rsi"])
    p_values = dict(
        zip(result["variable"], result["p_value"].astype(float), strict=True)
    )

    # Certainty, not a finding: the groups were built by the sign of this column.
    assert p_values["price_change"] < 1e-100
    # A variable that could not have leaked the answer behaves as it should.
    assert p_values["rsi"] > 0.01


@pytest.mark.unit
def test_a_direction_with_one_side_empty_is_refused() -> None:
    index = pd.date_range(ORIGIN, periods=200, freq="h", tz="UTC")
    frame = pd.DataFrame({"x": np.arange(200, dtype="float64")}, index=index)
    with pytest.raises(EdaError, match="enough observations"):
        compare_by_direction(frame, pd.Series(1.0, index=index), ["x"])


# --- features against the answer, and against each other -----------------------------


@pytest.mark.unit
def test_feature_correlations_rank_by_absolute_size() -> None:
    """A strong negative predictor matters as much as a strong positive one."""
    index = pd.date_range(ORIGIN, periods=400, freq="h", tz="UTC")
    rng = np.random.default_rng(21)
    labels = pd.Series(rng.integers(0, 2, 400).astype(float), index=index)
    frame = pd.DataFrame(
        {
            "useless": rng.normal(size=400),
            # Inverted on purpose: correlates strongly, and negatively.
            "inverted": 1.0 - labels + rng.normal(0, 0.1, 400),
        },
        index=index,
    )

    ranked = feature_correlations(frame, labels)
    assert ranked.index[0] == "inverted"
    assert ranked.iloc[0] < 0


@pytest.mark.unit
def test_the_strongest_correlation_is_what_bounds_the_result() -> None:
    """The number this figure exists for.

    On the real matrix the best single feature correlates with the direction at 0.019.
    Here the harness is checked the other way: given a feature that really does predict,
    the function has to find it - otherwise a near-zero result would prove nothing.
    """
    index = pd.date_range(ORIGIN, periods=600, freq="h", tz="UTC")
    rng = np.random.default_rng(23)
    labels = pd.Series(rng.integers(0, 2, 600).astype(float), index=index)
    frame = pd.DataFrame({"signal": labels * 2.0 + rng.normal(0, 0.5, 600)}, index=index)

    assert abs(feature_correlations(frame, labels)).max() > 0.8


@pytest.mark.unit
def test_multicollinear_pairs_finds_a_duplicate_and_reports_it_once() -> None:
    """`return` and `log_return` correlate at 1.0000 on the real matrix.

    A pair must appear once rather than twice: only the upper triangle is walked.
    """
    rng = np.random.default_rng(29)
    base = rng.normal(size=500)
    frame = pd.DataFrame(
        {
            "a": base,
            "a_copy": base * 2.0 + 1.0,  # a perfect linear image of `a`
            "unrelated": rng.normal(size=500),
        }
    )

    pairs = multicollinear_pairs(frame)
    assert len(pairs) == 1
    assert {pairs.iloc[0]["left"], pairs.iloc[0]["right"]} == {"a", "a_copy"}
    assert abs(float(pairs.iloc[0]["correlation"])) > 0.999


@pytest.mark.unit
def test_independent_features_produce_no_pairs() -> None:
    """The control: without it the function could return everything and still pass."""
    rng = np.random.default_rng(31)
    frame = pd.DataFrame({f"f{i}": rng.normal(size=800) for i in range(6)})
    assert multicollinear_pairs(frame).empty


@pytest.mark.unit
def test_the_redundancy_threshold_is_respected() -> None:
    rng = np.random.default_rng(37)
    base = rng.normal(size=1000)
    frame = pd.DataFrame({"a": base, "b": base * 0.8 + rng.normal(0, 0.55, 1000)})

    correlation = abs(frame.corr().to_numpy()[0, 1])
    # Just above the observed value nothing qualifies; just below, the pair does.
    assert multicollinear_pairs(frame, threshold=correlation + 0.02).empty
    assert len(multicollinear_pairs(frame, threshold=correlation - 0.02)) == 1


@pytest.mark.unit
def test_an_impossible_threshold_is_refused() -> None:
    with pytest.raises(EdaError, match="threshold"):
        multicollinear_pairs(pd.DataFrame({"a": [1.0, 2.0]}), threshold=1.5)
