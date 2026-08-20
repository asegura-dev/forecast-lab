"""Tests for the reading boundary.

Everything downstream trusts that a frame sits on a regular grid, is ordered, and
carries UTC timestamps. These tests pin that the reader *establishes* those properties
rather than assuming them - and, just as importantly, that it refuses a file it cannot
vouch for instead of quietly repairing it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from forecast_lab.contracts import GridError, SymbolSpec, Timeframe
from forecast_lab.ingest import read_series

XAU = SymbolSpec(name="XAUUSD")
H1 = Timeframe.H1

#: 2022-01-03 00:00 UTC, on the hour.
BASE = 1641168000


def _csv(path: Path, rows: list[str], header: str = "time,open,high,low,close") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def _bars(count: int, *, start: int = BASE, step: int = 3600) -> list[str]:
    return [f"{start + i * step},{100 + i},{101 + i},{99 + i},{100.5 + i}" for i in range(count)]


# --- the happy path -----------------------------------------------------------------


@pytest.mark.unit
def test_reads_a_well_formed_series(tmp_path: Path) -> None:
    frame, meta = read_series(_csv(tmp_path / "x.csv", _bars(3)), XAU, H1)

    assert meta.rows == 3
    assert list(frame.columns) == ["open", "high", "low", "close"]
    assert meta.first == datetime(2022, 1, 3, 0, tzinfo=UTC)
    assert meta.last == datetime(2022, 1, 3, 2, tzinfo=UTC)
    assert meta.anchors == frozenset({0})


@pytest.mark.unit
def test_the_index_is_utc_aware(tmp_path: Path) -> None:
    """A naive index is how a timezone bug enters and stays invisible."""
    _, meta = read_series(_csv(tmp_path / "x.csv", _bars(2)), XAU, H1)
    assert meta.first is not None
    assert meta.first.tzinfo is UTC


@pytest.mark.unit
def test_optional_columns_are_kept(tmp_path: Path) -> None:
    """Volume and spread exist for fetched data and not for the reference exports."""
    path = _csv(
        tmp_path / "x.csv",
        [f"{BASE},100,101,99,100.5,1.25,0.7"],
        header="time,open,high,low,close,volume,spread",
    )
    frame, _ = read_series(path, XAU, H1)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "spread"]


@pytest.mark.unit
def test_unknown_columns_are_dropped(tmp_path: Path) -> None:
    path = _csv(
        tmp_path / "x.csv",
        [f"{BASE},100,101,99,100.5,nonsense"],
        header="time,open,high,low,close,Extra",
    )
    frame, _ = read_series(path, XAU, H1)
    assert "extra" not in frame.columns


@pytest.mark.unit
def test_headers_are_case_and_space_insensitive(tmp_path: Path) -> None:
    """Exports arrive with whatever capitalisation the venue felt like."""
    path = _csv(
        tmp_path / "x.csv", [f"{BASE},100,101,99,100.5"], header=" Time , Open ,HIGH,low,Close"
    )
    frame, meta = read_series(path, XAU, H1)
    assert list(frame.columns) == ["open", "high", "low", "close"]
    assert meta.rows == 1


# --- timestamps ---------------------------------------------------------------------


@pytest.mark.unit
def test_iso_timestamps_are_accepted(tmp_path: Path) -> None:
    path = _csv(tmp_path / "x.csv", ["2022-01-03T00:00:00Z,100,101,99,100.5"])
    _, meta = read_series(path, XAU, H1)
    assert meta.first == datetime(2022, 1, 3, 0, tzinfo=UTC)


@pytest.mark.unit
def test_a_naive_string_is_read_as_utc(tmp_path: Path) -> None:
    """The only safe assumption: local time would make one file mean two things.

    Read as local, the same file would produce different bars on two machines - and the
    difference would never announce itself.
    """
    path = _csv(tmp_path / "x.csv", ["2022-01-03 00:00:00,100,101,99,100.5"])
    _, meta = read_series(path, XAU, H1)
    assert meta.first == datetime(2022, 1, 3, 0, tzinfo=UTC)


@pytest.mark.unit
def test_unparseable_timestamps_are_rejected(tmp_path: Path) -> None:
    path = _csv(tmp_path / "x.csv", ["not-a-date,100,101,99,100.5"])
    with pytest.raises(GridError, match="could not be parsed"):
        read_series(path, XAU, H1)


@pytest.mark.unit
def test_rows_are_sorted(tmp_path: Path) -> None:
    """Order on disk is the venue's business; order in memory is ours."""
    rows = _bars(3)
    path = _csv(tmp_path / "x.csv", [rows[2], rows[0], rows[1]])
    frame, meta = read_series(path, XAU, H1)
    assert frame.index.is_monotonic_increasing
    assert meta.first == datetime(2022, 1, 3, 0, tzinfo=UTC)


# --- what gets refused --------------------------------------------------------------


@pytest.mark.unit
def test_a_missing_time_column_is_rejected(tmp_path: Path) -> None:
    path = _csv(tmp_path / "x.csv", ["100,101,99,100.5"], header="open,high,low,close")
    with pytest.raises(GridError, match="no 'time' column"):
        read_series(path, XAU, H1)


@pytest.mark.unit
def test_a_missing_price_column_is_rejected(tmp_path: Path) -> None:
    path = _csv(tmp_path / "x.csv", [f"{BASE},100,101"], header="time,open,high")
    with pytest.raises(GridError, match="missing price column"):
        read_series(path, XAU, H1)


@pytest.mark.unit
def test_duplicate_timestamps_are_rejected(tmp_path: Path) -> None:
    """Which of the two bars is the truth? There is no defensible answer.

    `reindex` refuses a duplicated index outright, so accepting one here would only
    move the failure somewhere less informative.
    """
    rows = [*_bars(2), f"{BASE},999,999,999,999"]
    path = _csv(tmp_path / "x.csv", rows)
    with pytest.raises(GridError, match="duplicated timestamp"):
        read_series(path, XAU, H1)


# --- the grid, which is the whole point ---------------------------------------------


@pytest.mark.unit
def test_a_shifted_grid_is_still_a_grid(tmp_path: Path) -> None:
    """Bars every hour at :30 are regular - one anchor, just not zero."""
    path = _csv(tmp_path / "x.csv", _bars(3, start=BASE + 1800))
    _, meta = read_series(path, XAU, H1)
    assert meta.anchors == frozenset({1800})


@pytest.mark.unit
def test_two_anchors_are_allowed(tmp_path: Path) -> None:
    """A trading day that shifts with daylight saving yields exactly two offsets.

    That is a calendar fact, not corruption (ADR-002 sec. 5), so it must pass.
    """
    rows = _bars(2, start=BASE + 7200, step=14400) + _bars(2, start=BASE + 10800, step=14400)
    path = _csv(tmp_path / "x.csv", rows)
    _, meta = read_series(path, XAU, Timeframe.H4)
    assert meta.anchors == frozenset({7200, 10800})


@pytest.mark.unit
def test_three_anchors_are_described_not_rejected(tmp_path: Path) -> None:
    """Measured, not assumed: a US index on a European calendar really has three.

    An earlier version of this reader rejected anything above two offsets. Running it
    against real exports showed SPX and NDX daily bars sitting at three, because the US
    and European daylight-saving calendars do not switch on the same weekend. The
    threshold was measuring the calendar rather than a defect, so the reader now records
    the shape and lets alignment judge it.
    """
    rows = [
        f"{BASE + offset + day * 86400},100,101,99,100.5"
        for day, offset in enumerate((75600, 75600, 79200, 79200, 82800))
    ]
    _, meta = read_series(_csv(tmp_path / "x.csv", rows), XAU, Timeframe.D1)
    assert meta.anchors == frozenset({75600, 79200, 82800})


@pytest.mark.unit
def test_an_irregular_grid_is_described(tmp_path: Path) -> None:
    """Arbitrary offsets are recorded so alignment can refuse them with context."""
    rows = [f"{BASE + off},100,101,99,100.5" for off in (0, 137, 900, 2711, 3600, 4501)]
    _, meta = read_series(_csv(tmp_path / "x.csv", rows), XAU, H1)
    # Six bars, five distinct offsets: the one at +3600 lands back on anchor zero.
    assert meta.anchors == frozenset({0, 137, 900, 901, 2711})
    assert meta.short_gaps == 5


@pytest.mark.unit
def test_short_gaps_are_counted(tmp_path: Path) -> None:
    """A daylight-saving Monday is a 23-hour day, not an overlapping bar.

    Counting it, rather than raising on it, is what lets a daily series spanning two
    decades of calendar changes be imported at all.
    """
    rows = [f"{BASE + s},100,101,99,100.5" for s in (0, 86400, 169200, 255600)]
    _, meta = read_series(_csv(tmp_path / "x.csv", rows), XAU, Timeframe.D1)
    assert meta.short_gaps == 1


@pytest.mark.unit
def test_a_clean_intraday_series_has_no_short_gaps(tmp_path: Path) -> None:
    _, meta = read_series(_csv(tmp_path / "x.csv", _bars(5)), XAU, H1)
    assert meta.short_gaps == 0


@pytest.mark.unit
def test_gaps_do_not_break_the_grid(tmp_path: Path) -> None:
    """A weekend is a missing bar, not a misplaced one.

    The target series has 183 weekend gaps and 780 daily-break gaps; none of them make
    the grid irregular, and treating them as errors would reject every real series.
    """
    rows = [f"{BASE + h * 3600},100,101,99,100.5" for h in (0, 1, 2, 50, 51, 120)]
    path = _csv(tmp_path / "x.csv", rows)
    _, meta = read_series(path, XAU, H1)
    assert meta.rows == 6
    assert meta.anchors == frozenset({0})


@pytest.mark.unit
def test_an_empty_series_reads_as_empty(tmp_path: Path) -> None:
    path = _csv(tmp_path / "x.csv", [])
    _, meta = read_series(path, XAU, H1)
    assert meta.is_empty
    assert meta.first is None
    assert meta.anchors == frozenset()
