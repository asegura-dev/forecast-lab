"""Tests for pooled walk-forward scoring.

Two of these pin decisions rather than arithmetic. `test_folds_are_pooled_not_averaged`
fixes the choice of estimator - a size-weighted pool, not a mean of fold accuracies -
because the two agree whenever folds are equal-sized and diverge exactly when they are
not, which is where a silent regression would hide. And
`test_the_baseline_is_refitted_inside_every_fold` fixes the reason the walk-forward edge
looks better than the single-split one: the comparison moved, not the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_lab.research import (
    ValidationError,
    score_walk_forward,
    walk_forward,
)
from forecast_lab.research.models.catalogue import CATALOGUE
from forecast_lab.research.models.validation import FoldScore, PooledScore

ORIGIN = "2018-01-01"


def _frame(n: int, *, seed: int = 0) -> pd.DataFrame:
    index = pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=index)


def _labels(frame: pd.DataFrame, *, up_share: float = 0.5, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    drawn = (rng.random(len(frame)) < up_share).astype(float)
    return pd.Series(drawn, index=frame.index)


#: The cheapest estimator in the catalogue, and one with no native dependency that an
#: application control policy could refuse - so these tests run everywhere.
LOGISTIC = next(spec for spec in CATALOGUE if spec.name == "Logistic Regression")


# --- the two decisions ----------------------------------------------------------------


@pytest.mark.unit
def test_folds_are_pooled_not_averaged() -> None:
    """Weighted by rows, so an unequal fold cannot count as much as a full one.

    Averaging the fold accuracies would give 65% here; pooling gives 60%, because the
    weaker fold carries twice the rows. The two agree on equal-sized folds, which is
    why this uses unequal ones.
    """
    pooled = PooledScore(
        model="stub",
        representation="raw",
        folds=(
            FoldScore(number=1, n=100, accuracy=0.80, baseline_accuracy=0.5),
            FoldScore(number=2, n=200, accuracy=0.50, baseline_accuracy=0.5),
        ),
    )

    assert pooled.n == 300
    assert pooled.accuracy == pytest.approx(0.60)
    # And that is not the mean of the fold accuracies, which is 0.65.
    assert pooled.accuracy != pytest.approx((0.80 + 0.50) / 2)
    assert pooled.edge == pytest.approx(pooled.accuracy - 0.5)


@pytest.mark.unit
def test_the_baseline_is_refitted_inside_every_fold() -> None:
    """Each fold's baseline comes from that fold's own training block.

    Labels here flip majority halfway through the series: the early folds train on a
    mostly-DOWN stretch and the late ones on a mostly-UP stretch. A single global
    majority would give every fold the same baseline; refitting per fold must not.
    """
    frame = _frame(3_000)
    half = len(frame) // 2
    values = np.concatenate([np.zeros(half), np.ones(len(frame) - half)])
    # A few flips so no training block is single-class, which would refuse to fit.
    values[::37] = 1.0 - values[::37]
    labels = pd.Series(values, index=frame.index)

    scheme = walk_forward(pd.DatetimeIndex(frame.index), folds=5, horizon=1)
    scored = score_walk_forward(LOGISTIC, frame, labels, scheme)

    baselines = [fold.baseline_accuracy for fold in scored.folds]
    assert len(set(baselines)) > 1, "a refitted baseline cannot be constant across regimes"


# --- the mechanics --------------------------------------------------------------------


@pytest.mark.unit
def test_every_fold_is_scored_and_the_rows_add_up() -> None:
    frame = _frame(3_000)
    labels = _labels(frame)
    scheme = walk_forward(pd.DatetimeIndex(frame.index), folds=4, horizon=1)

    scored = score_walk_forward(LOGISTIC, frame, labels, scheme)

    assert len(scored.folds) == 4
    assert scored.skipped == ()
    assert scored.n == sum(fold.n for fold in scored.folds)
    assert scored.n <= scheme.test_rows
    # The pooled figure has to sit inside the range it was pooled from.
    assert scored.worst_fold <= scored.accuracy <= scored.best_fold


@pytest.mark.unit
def test_unlabelled_rows_are_dropped_rather_than_guessed() -> None:
    """The 4.37% whose horizon spans a gap have no direction to be right about."""
    frame = _frame(3_000)
    labels = _labels(frame)
    labels.iloc[::4] = np.nan
    scheme = walk_forward(pd.DatetimeIndex(frame.index), folds=4, horizon=1)

    scored = score_walk_forward(LOGISTIC, frame, labels, scheme)

    assert scored.n < scheme.test_rows
    assert scored.n == pytest.approx(scheme.test_rows * 0.75, rel=0.05)


@pytest.mark.unit
def test_the_representation_is_recorded() -> None:
    """A pooled score without its representation cannot be compared with another."""
    frame = _frame(3_000)
    labels = _labels(frame)
    scheme = walk_forward(pd.DatetimeIndex(frame.index), folds=3, horizon=1)

    raw = score_walk_forward(LOGISTIC, frame, labels, scheme)
    reduced = score_walk_forward(LOGISTIC, frame, labels, scheme, variance=0.95)

    assert raw.representation == "raw"
    assert reduced.representation == "pca-95"
    assert raw.key != reduced.key


@pytest.mark.unit
def test_a_fold_that_cannot_be_fitted_is_recorded_not_dropped() -> None:
    """Silence here would make a four-fold score look like a five-fold one.

    The first training block is single-class, which `fit_and_predict` refuses; the rest
    are mixed. The result must carry both the folds it scored and the one it could not.
    """
    frame = _frame(3_000)
    labels = _labels(frame)
    first_block = len(frame) // 6
    labels.iloc[:first_block] = 1.0

    scheme = walk_forward(pd.DatetimeIndex(frame.index), folds=5, horizon=1)
    scored = score_walk_forward(LOGISTIC, frame, labels, scheme)

    assert len(scored.skipped) == 1
    assert scored.skipped[0][0] == 1
    assert "one class" in scored.skipped[0][1]
    assert len(scored.folds) == 4


@pytest.mark.unit
def test_a_model_that_fits_nowhere_is_refused() -> None:
    """An empty result would pool to a division by zero, so it raises instead."""
    frame = _frame(1_200)
    labels = pd.Series(1.0, index=frame.index)  # single class everywhere
    scheme = walk_forward(pd.DatetimeIndex(frame.index), folds=3, horizon=1)

    with pytest.raises(ValidationError, match="could not be scored on any fold"):
        score_walk_forward(LOGISTIC, frame, labels, scheme)
