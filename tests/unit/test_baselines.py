"""Tests for the numbers a model has to beat.

The one that matters most is the last: a baseline fitted on the block it is scored
against is not a baseline. That distinction is the whole reason this module exists, and
the difference between the two is exactly the trap the original analysis fell into.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forecast_lab.research import (
    BaselineError,
    BaselineReport,
    Direction,
    Score,
    evaluate_baselines,
)

UP = float(Direction.UP.value)
DOWN = float(Direction.DOWN.value)
FLAT = float(Direction.FLAT.value)


def _labels(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.RangeIndex(len(values)), dtype="float64")


def _rule(report: BaselineReport, fragment: str) -> Score:
    return next(s for s in report.scores if fragment in s.name)


# --- the majority class -------------------------------------------------------------


@pytest.mark.unit
def test_the_majority_comes_from_train_not_from_the_block_scored() -> None:
    """The trap, stated as a test.

    Train mostly falls, so the honest rule predicts DOWN. The block mostly rises. A rule
    fitted on the block would score 80%; the honest one scores 20%, and the gap between
    them is a fact about the drift rather than about any model.
    """
    train = _labels([DOWN] * 8 + [UP] * 2)
    block = _labels([UP] * 8 + [DOWN] * 2)

    report = evaluate_baselines(train, block, block="test")
    majority = _rule(report, "majority-class")

    assert "DOWN" in majority.name
    assert majority.accuracy == pytest.approx(0.2)
    assert report.up_rate == pytest.approx(0.8)
    assert report.train_up_rate == pytest.approx(0.2)
    assert report.oracle_gap == pytest.approx(0.6)


@pytest.mark.unit
def test_always_up_has_perfect_recall_and_no_specificity() -> None:
    """The signature the original analysis failed to notice.

    Its winning model reported 100% recall on UP. That is not a model detecting rises;
    it is a model that has learned to say UP, and specificity says so instantly.
    """
    train = _labels([UP] * 7 + [DOWN] * 3)
    block = _labels([UP] * 6 + [DOWN] * 4)

    majority = _rule(evaluate_baselines(train, block, block="test"), "majority-class")

    assert majority.recall == pytest.approx(1.0)
    assert majority.specificity == pytest.approx(0.0)
    assert majority.accuracy == pytest.approx(0.6)


# --- the other rules ----------------------------------------------------------------


@pytest.mark.unit
def test_persistence_repeats_the_previous_bar() -> None:
    train = _labels([UP, DOWN] * 5)
    block = _labels([UP, UP, UP, UP])

    persistence = _rule(evaluate_baselines(train, block, block="test"), "persistence")
    # Seeded with DOWN, then right three times out of four.
    assert persistence.accuracy == pytest.approx(0.75)


@pytest.mark.unit
def test_the_random_rule_is_reproducible() -> None:
    """A seeded baseline is a fact about the data, not about the day it ran."""
    train = _labels([UP] * 6 + [DOWN] * 4)
    block = _labels([UP, DOWN] * 20)

    first = _rule(evaluate_baselines(train, block, block="test"), "random")
    second = _rule(evaluate_baselines(train, block, block="test"), "random")

    assert first.accuracy == second.accuracy


# --- what is excluded ---------------------------------------------------------------


@pytest.mark.unit
def test_flat_bars_are_not_scored() -> None:
    """A tie has no direction, so no rule can be right or wrong about it."""
    train = _labels([UP] * 5 + [DOWN] * 5)
    block = _labels([UP, FLAT, FLAT, DOWN])

    report = evaluate_baselines(train, block, block="test")
    assert report.n == 2


@pytest.mark.unit
def test_unlabelled_bars_are_not_scored() -> None:
    train = _labels([UP] * 5 + [DOWN] * 5)
    block = pd.Series([UP, float("nan"), DOWN], dtype="float64")

    report = evaluate_baselines(train, block, block="test")
    assert report.n == 2


@pytest.mark.unit
def test_a_block_with_nothing_to_score_is_refused() -> None:
    train = _labels([UP] * 5 + [DOWN] * 5)
    with pytest.raises(BaselineError, match="no decided labels"):
        evaluate_baselines(train, _labels([FLAT, FLAT]), block="test")
