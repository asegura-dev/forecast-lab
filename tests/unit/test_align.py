"""Tests for putting several symbols on one timeline.

Two properties carry the weight: nothing from the future may reach a row, and no row may
exist that the target did not trade. The rest is bookkeeping about how old each carried
value is, which matters because a value carried far enough stops describing the present.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from forecast_lab.contracts import Timeframe
from forecast_lab.research import STALENESS_SUFFIX, AlignmentError, align_to_target

ORIGIN = datetime(2022, 1, 3, tzinfo=UTC)


def _bars(offsets: list[int], prices: list[float], *, step_hours: int = 1) -> pd.DataFrame:
    index = pd.DatetimeIndex([ORIGIN + timedelta(hours=o * step_hours) for o in offsets])
    return pd.DataFrame({"close": prices}, index=index)


def _target() -> pd.DataFrame:
    return _bars([0, 1, 2, 3], [100.0, 101.0, 102.0, 103.0])


# --- the timeline -------------------------------------------------------------------


@pytest.mark.unit
def test_the_target_defines_the_timeline() -> None:
    aux = _bars([0, 1, 2, 3, 4, 5], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    panel = align_to_target("XAUUSD", {"XAUUSD": _target(), "EURUSD": aux}, Timeframe.H1)

    assert panel.rows == 4
    assert panel.target == "XAUUSD"
    assert list(panel.frame.index) == list(_target().index)


@pytest.mark.unit
def test_columns_are_prefixed_by_symbol() -> None:
    aux = _bars([0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0])
    panel = align_to_target("XAUUSD", {"XAUUSD": _target(), "EURUSD": aux}, Timeframe.H1)

    assert "XAUUSD_close" in panel.frame.columns
    assert "EURUSD_close" in panel.frame.columns
    assert panel.columns_for("EURUSD") == ["EURUSD_close", f"EURUSD_{STALENESS_SUFFIX}"]


@pytest.mark.unit
def test_the_target_carries_no_staleness_column() -> None:
    """It is the timeline, so it is never stale relative to itself."""
    aux = _bars([0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0])
    panel = align_to_target("XAUUSD", {"XAUUSD": _target(), "EURUSD": aux}, Timeframe.H1)
    assert f"XAUUSD_{STALENESS_SUFFIX}" not in panel.frame.columns


# --- the direction of time, which is the whole point --------------------------------


@pytest.mark.unit
def test_only_the_past_is_read() -> None:
    """The auxiliary trades on the half hour; each target row must see the EARLIER bar.

    If this ever read the later bar instead, every row would carry information from
    after its own decision point - and the only symptom would be a model that predicts
    slightly too well.
    """
    aux_index = pd.DatetimeIndex(
        [ORIGIN + timedelta(minutes=30) + timedelta(hours=h) for h in range(4)]
    )
    aux = pd.DataFrame({"close": [10.0, 11.0, 12.0, 13.0]}, index=aux_index)

    panel = align_to_target("XAUUSD", {"XAUUSD": _target(), "EURUSD": aux}, Timeframe.H1)
    values = panel.frame["EURUSD_close"].tolist()

    # At 00:00 nothing has traded yet; at 01:00 the 00:30 bar is the latest known.
    assert pd.isna(values[0])
    assert values[1:] == [10.0, 11.0, 12.0]


@pytest.mark.unit
def test_values_are_never_carried_backwards() -> None:
    """A price nobody knew yet cannot be filled into a time before it existed."""
    aux = _bars([2, 3], [12.0, 13.0])
    panel = align_to_target("XAUUSD", {"XAUUSD": _target(), "EURUSD": aux}, Timeframe.H1)

    values = panel.frame["EURUSD_close"].tolist()
    assert pd.isna(values[0]) and pd.isna(values[1])
    assert values[2:] == [12.0, 13.0]


@pytest.mark.unit
def test_a_longer_auxiliary_bar_is_refused() -> None:
    """The judgement the reader deliberately leaves to alignment (ADR-002 sec. 6).

    A daily bar has not closed when an hourly decision is taken, so its close describes
    hours that have not happened. No grid is clean enough to make that safe.
    """
    aux = _bars([0, 1], [10.0, 11.0], step_hours=24)
    with pytest.raises(AlignmentError, match="still open"):
        align_to_target(
            "XAUUSD",
            {"XAUUSD": _target(), "SPX": aux},
            Timeframe.H1,
            timeframes={"SPX": Timeframe.D1},
        )


@pytest.mark.unit
def test_a_shorter_auxiliary_bar_is_allowed() -> None:
    """Finer bars close before the target's decision, so they carry no future."""
    aux = _bars([0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0])
    panel = align_to_target(
        "XAUUSD",
        {"XAUUSD": _bars([0, 1], [100.0, 101.0], step_hours=4), "EURUSD": aux},
        Timeframe.H4,
        timeframes={"EURUSD": Timeframe.H1},
    )
    assert panel.rows == 2


# --- staleness ----------------------------------------------------------------------


@pytest.mark.unit
def test_staleness_is_measured_in_seconds() -> None:
    aux = _bars([0, 2], [10.0, 12.0])
    panel = align_to_target("XAUUSD", {"XAUUSD": _target(), "EURUSD": aux}, Timeframe.H1)

    # 00:00 fresh, 01:00 an hour old, 02:00 fresh again, 03:00 an hour old.
    assert panel.frame[f"EURUSD_{STALENESS_SUFFIX}"].tolist() == [0.0, 3600.0, 0.0, 3600.0]


@pytest.mark.unit
def test_staleness_is_per_symbol() -> None:
    """One column would average two very different stories.

    In the real data the index that trades a US session and the currency that trades
    around the clock go stale on completely different schedules; a single column would
    describe neither.
    """
    fresh = _bars([0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0])
    sparse = _bars([0], [20.0])
    panel = align_to_target(
        "XAUUSD", {"XAUUSD": _target(), "EURUSD": fresh, "SPX": sparse}, Timeframe.H1
    )

    assert panel.frame[f"EURUSD_{STALENESS_SUFFIX}"].max() == 0.0
    assert panel.frame[f"SPX_{STALENESS_SUFFIX}"].max() == 3 * 3600


@pytest.mark.unit
def test_a_staleness_limit_drops_the_value_instead_of_carrying_it() -> None:
    """Past the limit, the last known value stops describing the present.

    Leaving it in place would let a Friday close masquerade as a Monday price for the
    whole weekend, which a model would happily learn.
    """
    aux = _bars([0], [10.0])
    panel = align_to_target(
        "XAUUSD",
        {"XAUUSD": _target(), "EURUSD": aux},
        Timeframe.H1,
        max_staleness_seconds=3600,
    )

    values = panel.frame["EURUSD_close"].tolist()
    assert values[:2] == [10.0, 10.0]
    assert pd.isna(values[2]) and pd.isna(values[3])


@pytest.mark.unit
def test_coverage_reports_what_was_carried() -> None:
    aux = _bars([1], [11.0])
    panel = align_to_target("XAUUSD", {"XAUUSD": _target(), "EURUSD": aux}, Timeframe.H1)

    report = next(c for c in panel.coverage if c.symbol == "EURUSD")
    assert report.rows == 4
    assert report.missing == 1  # 00:00, before the auxiliary began
    assert report.stale == 2  # 02:00 and 03:00 carry a value older than one bar
    assert report.max_stale_seconds == 2 * 3600
    assert report.missing_fraction == pytest.approx(0.25)


# --- what gets refused --------------------------------------------------------------


@pytest.mark.unit
def test_a_missing_target_is_refused() -> None:
    with pytest.raises(AlignmentError, match="not among the series"):
        align_to_target("XAUUSD", {"EURUSD": _target()}, Timeframe.H1)


@pytest.mark.unit
def test_an_empty_target_is_refused() -> None:
    """With no bars there is no timeline, and an empty panel would hide that."""
    empty = pd.DataFrame({"close": []}, index=pd.DatetimeIndex([], tz=UTC))
    with pytest.raises(AlignmentError, match="empty"):
        align_to_target("XAUUSD", {"XAUUSD": empty}, Timeframe.H1)


@pytest.mark.unit
def test_a_naive_index_is_refused() -> None:
    naive = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2022-01-03"]))
    with pytest.raises(AlignmentError, match="timezone-naive"):
        align_to_target("XAUUSD", {"XAUUSD": naive}, Timeframe.H1)


@pytest.mark.unit
def test_an_unsorted_index_is_refused() -> None:
    """`searchsorted` silently returns nonsense on an unsorted index."""
    unsorted = _bars([2, 0, 1], [102.0, 100.0, 101.0])
    with pytest.raises(AlignmentError, match="not sorted"):
        align_to_target("XAUUSD", {"XAUUSD": unsorted}, Timeframe.H1)


@pytest.mark.unit
def test_a_duplicated_index_is_refused() -> None:
    duplicated = _bars([0, 0, 1], [100.0, 100.5, 101.0])
    with pytest.raises(AlignmentError, match="duplicated"):
        align_to_target("XAUUSD", {"XAUUSD": duplicated}, Timeframe.H1)


@pytest.mark.unit
def test_alignment_is_deterministic_in_column_order() -> None:
    """Column order must not depend on dictionary insertion order.

    The original notebook built its symbol list with `list(set(...))`, whose order varies
    between processes - so its column order, and therefore its PCA components, were not
    reproducible across runs.
    """
    aux_a = _bars([0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0])
    aux_b = _bars([0, 1, 2, 3], [20.0, 21.0, 22.0, 23.0])

    one = align_to_target(
        "XAUUSD", {"XAUUSD": _target(), "SPX": aux_b, "EURUSD": aux_a}, Timeframe.H1
    )
    two = align_to_target(
        "XAUUSD", {"EURUSD": aux_a, "XAUUSD": _target(), "SPX": aux_b}, Timeframe.H1
    )

    assert list(one.frame.columns) == list(two.frame.columns)
