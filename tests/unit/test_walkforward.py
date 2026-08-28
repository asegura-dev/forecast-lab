"""Tests for walk-forward validation.

The properties that matter are temporal: a fold may never train on its own future, and
the purge has to reach back at every boundary rather than only at the first one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forecast_lab.research import WalkForwardError, walk_forward

ORIGIN = "2018-01-01"


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")


# --- the temporal properties ---------------------------------------------------------


@pytest.mark.unit
def test_no_fold_trains_on_its_own_future() -> None:
    """The property the whole design exists for.

    Shuffled k-fold would pass every other test in this file and violate this one, which
    is why it is asserted directly rather than inferred from the row counts.
    """
    scheme = walk_forward(_index(6_000), folds=5, horizon=1)
    for fold in scheme:
        assert fold.train[-1] < fold.test[0]


@pytest.mark.unit
def test_the_test_blocks_move_forward_and_do_not_overlap() -> None:
    """Each fold scores a stretch no other fold scored."""
    scheme = walk_forward(_index(6_000), folds=5, horizon=1)
    ends = [fold.test[-1] for fold in scheme]
    starts = [fold.test[0] for fold in scheme]

    assert starts == sorted(starts)
    for earlier, later in zip(ends[:-1], starts[1:], strict=True):
        assert earlier < later


@pytest.mark.unit
def test_the_training_window_expands_by_default() -> None:
    """An expanding window is what a deployment would have: all history to date."""
    scheme = walk_forward(_index(6_000), folds=4, horizon=1)
    sizes = [fold.train_rows for fold in scheme]

    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    # Every fold starts at the beginning of the series.
    assert all(fold.train[0] == scheme.folds[0].train[0] for fold in scheme)


@pytest.mark.unit
def test_a_rolling_window_keeps_its_length_instead() -> None:
    """A different question - whether recent history predicts better than distant."""
    scheme = walk_forward(_index(6_000), folds=4, horizon=1, expanding=False)
    sizes = {fold.train_rows for fold in scheme}

    assert len(sizes) == 1
    # And it moves forward, unlike the expanding one.
    assert scheme.folds[-1].train[0] > scheme.folds[0].train[0]


@pytest.mark.unit
def test_every_boundary_is_purged_not_just_the_first() -> None:
    """A label at the last training bar reaches into the test block, in every fold."""
    horizon = 3
    scheme = walk_forward(_index(6_000), folds=5, horizon=horizon)

    assert all(fold.purged == horizon for fold in scheme)
    assert scheme.purged == horizon * len(scheme)


@pytest.mark.unit
def test_the_scheme_scores_far_more_bars_than_a_single_split() -> None:
    """The reason to do this at all.

    Measured on the canonical feature matrix of 50,948 rows: 42,455 bars fall in a test
    block against 7,306 from a single 70/15/15 split. After the unlabelled bars are
    dropped that is 40,587 actually scored, which takes the minimum detectable effect
    from 1.45 points to 0.62.
    """
    n = 50_948
    scheme = walk_forward(_index(n), folds=5, horizon=1)

    single_split_test = int(n * 0.15)
    assert scheme.test_rows > 4 * single_split_test


# --- refusals -------------------------------------------------------------------------


@pytest.mark.unit
def test_one_fold_is_refused() -> None:
    """A single fold is a single split, which already exists and is named differently."""
    with pytest.raises(WalkForwardError, match="at least two folds"):
        walk_forward(_index(1_000), folds=1)


@pytest.mark.unit
def test_blocks_smaller_than_the_horizon_are_refused() -> None:
    """A block that cannot outlast the label's reach is not a block."""
    with pytest.raises(WalkForwardError, match="not more than the horizon"):
        walk_forward(_index(30), folds=5, horizon=10)


@pytest.mark.unit
def test_an_unsorted_index_is_refused() -> None:
    with pytest.raises(WalkForwardError, match="sorted"):
        walk_forward(_index(1_000)[::-1], folds=3)


@pytest.mark.unit
def test_a_minimum_training_size_that_leaves_no_room_is_refused() -> None:
    with pytest.raises(WalkForwardError, match="no room"):
        walk_forward(_index(1_000), folds=5, min_train=900)


@pytest.mark.unit
def test_a_negative_horizon_is_refused() -> None:
    with pytest.raises(WalkForwardError, match="cannot be negative"):
        walk_forward(_index(1_000), folds=3, horizon=-1)


@pytest.mark.unit
def test_the_purge_guard_accounts_for_the_purge_itself() -> None:
    """Pins a bug found by running rather than by testing.

    The first version compared the training window *after* purging against the minimum
    required *before* it, so a horizon of one failed fold 1 every time - on the real
    series, 8,523 rows against a floor of 8,524. Any positive horizon must work.
    """
    for horizon in (1, 2, 5):
        scheme = walk_forward(_index(51_147), folds=5, horizon=horizon)
        assert len(scheme) == 5
        assert scheme.folds[0].train_rows > 0
