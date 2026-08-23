"""Tests for turning prices into the thing a model predicts.

Three decisions hide inside `close.shift(-1) > close`, and the original analysis made
all three without noticing. These pin them: which bar counts as "next", what a tie
means, and whether a threshold on the move can see the move it judges.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from forecast_lab.contracts import Timeframe
from forecast_lab.research import Direction, LabelingError, label_direction, rolling_dead_band

ORIGIN = datetime(2022, 1, 3, tzinfo=UTC)
H1 = Timeframe.H1


def _close(offsets: list[int], prices: list[float]) -> pd.Series:
    index = pd.DatetimeIndex([ORIGIN + timedelta(hours=o) for o in offsets])
    return pd.Series(prices, index=index, name="close")


# --- direction ----------------------------------------------------------------------


@pytest.mark.unit
def test_a_rise_is_up_and_a_fall_is_down() -> None:
    labels, _ = label_direction(_close([0, 1, 2], [100.0, 101.0, 100.5]), H1)
    assert labels["label"].tolist()[:2] == [Direction.UP.value, Direction.DOWN.value]


@pytest.mark.unit
def test_a_tie_is_flat_not_down() -> None:
    """`>` would sweep a tie into DOWN, and that is where the alignment bug landed.

    Forward-filled rows are all ties, so a strictly-greater comparison put the entire
    defect on one class. Naming FLAT makes a tie visible instead of silently directional.
    """
    labels, report = label_direction(_close([0, 1], [100.0, 100.0]), H1)
    assert labels["label"].iloc[0] == Direction.FLAT.value
    assert report.flat == 1
    assert report.down == 0


@pytest.mark.unit
def test_the_last_bar_has_no_answer() -> None:
    labels, report = label_direction(_close([0, 1], [100.0, 101.0]), H1)
    assert pd.isna(labels["label"].iloc[-1])
    assert report.labelled == 1


# --- the horizon, which is a timestamp and not a position ---------------------------


@pytest.mark.unit
def test_a_gap_is_not_the_next_bar() -> None:
    """The defect this module exists to prevent.

    Gold has 780 two-hour gaps and 183 weekend gaps. Shifting by position calls Monday's
    close "the next hour" in 4.37% of rows, mixing a far harder prediction problem into
    the same metric without saying so.
    """
    labels, report = label_direction(_close([0, 1, 50, 51], [100.0, 101.0, 120.0, 121.0]), H1)

    assert labels["label_valid"].tolist() == [True, False, True, False]
    assert pd.isna(labels["label"].iloc[1])
    assert report.gapped == 1


@pytest.mark.unit
def test_the_distance_actually_travelled_is_recorded() -> None:
    """A row must be able to say how far ahead its answer really lies.

    Bar 1 is followed by a 49-hour gap, and the column has to say 49 hours rather than
    the one hour the horizon asked for. It also pins the units: the first version divided
    by a hard-coded 1e9, which is only right when pandas happens to be storing
    nanoseconds, and here it is storing microseconds.
    """
    labels, _ = label_direction(_close([0, 1, 50], [100.0, 101.0, 120.0]), H1)
    ahead = labels["seconds_ahead"]

    assert ahead.iloc[0] == pytest.approx(3600.0)
    assert ahead.iloc[1] == pytest.approx(49 * 3600.0)
    assert pd.isna(ahead.iloc[2])  # nothing follows the last bar


@pytest.mark.unit
def test_what_happened_across_the_gap_is_reported_not_discarded() -> None:
    """Dropping the gapped rows silently would hide the subpopulation, not handle it.

    Here every gapped row rose while the labelled row fell, so the two distributions are
    opposites. On the real series they differ by 8 points, which is exactly the size of
    finding that a positional shift folds into a headline number without saying so.
    """
    close = _close([0, 1, 50, 51, 100], [100.0, 99.0, 120.0, 119.0, 140.0])
    _, report = label_direction(close, H1)

    assert report.gapped == 2
    assert (report.gapped_up, report.gapped_down) == (2, 0)
    assert report.gapped_up_rate == 1.0
    assert report.up_rate == 0.0  # the bars that did land all fell


@pytest.mark.unit
def test_a_longer_horizon_looks_further() -> None:
    labels, _ = label_direction(_close([0, 1, 2, 3], [100.0, 90.0, 105.0, 80.0]), H1, horizon=2)
    # From bar 0 the answer is bar 2 (105 > 100 = UP), not bar 1.
    assert labels["label"].iloc[0] == Direction.UP.value
    assert labels["future_close"].iloc[0] == 105.0


@pytest.mark.unit
def test_a_horizon_below_one_is_refused() -> None:
    with pytest.raises(LabelingError, match="at least one bar"):
        label_direction(_close([0, 1], [100.0, 101.0]), H1, horizon=0)


@pytest.mark.unit
def test_an_unsorted_series_is_refused() -> None:
    unsorted = _close([2, 0, 1], [102.0, 100.0, 101.0])
    with pytest.raises(LabelingError, match="sorted"):
        label_direction(unsorted, H1)


# --- the dead band ------------------------------------------------------------------


@pytest.mark.unit
def test_a_move_inside_the_dead_band_is_flat() -> None:
    """A move smaller than the cost of trading is noise, not an opportunity."""
    close = _close([0, 1, 2], [100.0, 100.2, 105.0])
    band = pd.Series([1.0, 1.0, 1.0], index=close.index)

    labels, _ = label_direction(close, H1, dead_band=band)
    assert labels["label"].iloc[0] == Direction.FLAT.value  # +0.2, inside the band
    assert labels["label"].iloc[1] == Direction.UP.value  # +4.8, outside


@pytest.mark.unit
def test_the_dead_band_must_match_the_series() -> None:
    close = _close([0, 1], [100.0, 101.0])
    with pytest.raises(LabelingError, match="indexed like"):
        label_direction(close, H1, dead_band=pd.Series([1.0]))


@pytest.mark.unit
def test_the_rolling_dead_band_cannot_see_the_move_it_judges() -> None:
    """The `shift(1)` is load-bearing.

    Without it the threshold would include the very bar whose move it is judging, which
    is selecting the test set on its own answer.
    """
    close = _close([0, 1, 2, 3, 4], [100.0, 101.0, 102.0, 103.0, 200.0])
    band = rolling_dead_band(close, window=2, min_periods=1)

    # The jump at bar 4 must not raise the threshold at bar 4 itself.
    assert pd.isna(band.iloc[0])
    assert band.iloc[4] == pytest.approx(1.0)


# --- the report ---------------------------------------------------------------------


@pytest.mark.unit
def test_the_up_rate_ignores_flat_bars() -> None:
    """A tie has no direction, so it cannot be counted for or against one."""
    _, report = label_direction(_close([0, 1, 2, 3], [100.0, 101.0, 101.0, 102.0]), H1)
    assert (report.up, report.down, report.flat) == (2, 0, 1)
    assert report.up_rate == 1.0
