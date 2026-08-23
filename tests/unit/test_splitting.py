"""Tests for cutting a series into blocks that do not touch.

The property that matters is the purge: a label reaches forward, so the boundary has to
reach back, or training data contains the answer to a validation question.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from forecast_lab.research import SplitError, temporal_split

ORIGIN = datetime(2022, 1, 3, tzinfo=UTC)


def _index(n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([ORIGIN + timedelta(hours=i) for i in range(n)])


@pytest.mark.unit
def test_the_blocks_run_in_order_and_do_not_overlap() -> None:
    split = temporal_split(_index(1000), horizon=0)

    assert split.train.index[-1] < split.validation.index[0]
    assert split.validation.index[-1] < split.test.index[0]
    assert split.train.rows + split.validation.rows + split.test.rows == 1000


@pytest.mark.unit
def test_the_fractions_are_respected() -> None:
    split = temporal_split(_index(1000), train=0.70, validation=0.15, horizon=0)
    assert (split.train.rows, split.validation.rows, split.test.rows) == (700, 150, 150)


@pytest.mark.unit
def test_the_test_block_absorbs_the_remainder() -> None:
    """The three shares must sum to the whole series, whatever the rounding does.

    Deriving test from the other two means a rounding error cannot silently drop rows
    on the floor.
    """
    split = temporal_split(_index(997), horizon=0)
    assert split.rows == 997


@pytest.mark.unit
def test_purging_removes_exactly_the_horizon_at_each_boundary() -> None:
    """A bar at the end of train carries an answer that lives inside validation.

    Leaving it in means training on the thing being predicted across the line.
    """
    horizon = 5
    split = temporal_split(_index(1000), horizon=horizon)

    assert split.purged == 2 * horizon  # one boundary before validation, one before test
    assert split.train.rows == 700 - horizon
    assert split.validation.rows == 150 - horizon
    assert split.test.rows == 150  # nothing follows test, so nothing is purged after it


@pytest.mark.unit
def test_nothing_is_purged_after_the_test_block() -> None:
    """No embargo, deliberately.

    The usual companion to purging discards a stretch after the test block, because in
    k-fold there is training data on both sides of it. In a single chronological split
    there is nothing after test, so an embargo would discard real data to guard against
    a leak that cannot happen.
    """
    split = temporal_split(_index(1000), horizon=10)
    assert split.test.index[-1] == _index(1000)[-1]


@pytest.mark.unit
def test_a_series_too_short_to_cut_is_refused() -> None:
    """Blocks smaller than the horizon they predict are not blocks."""
    with pytest.raises(SplitError, match="too few"):
        temporal_split(_index(5), horizon=3)


@pytest.mark.unit
def test_an_unsorted_index_is_refused() -> None:
    reversed_index = _index(100)[::-1]
    with pytest.raises(SplitError, match="sorted"):
        temporal_split(reversed_index)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("train", "validation"),
    [(0.0, 0.15), (1.0, 0.0), (0.9, 0.2), (0.7, -0.1)],
)
def test_impossible_fractions_are_refused(train: float, validation: float) -> None:
    with pytest.raises(SplitError, match="fractions"):
        temporal_split(_index(1000), train=train, validation=validation)
