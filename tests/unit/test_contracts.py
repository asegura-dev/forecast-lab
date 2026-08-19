"""Tests for the domain vocabulary.

These pin the invariants the rest of the pipeline is allowed to assume: that a symbol
is safe to use as a file name, that a timeframe knows its own duration, and that a
timestamp is never naive. Each one exists because the alternative is a bug that stays
invisible until the results are wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from forecast_lab.contracts import (
    InvalidSymbolError,
    SeriesMeta,
    SymbolSpec,
    Timeframe,
    UnknownTimeframeError,
)

# --- Timeframe ----------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timeframe", "seconds"),
    [(Timeframe.H1, 3600), (Timeframe.H4, 14400), (Timeframe.D1, 86400)],
)
def test_timeframe_knows_its_duration(timeframe: Timeframe, seconds: int) -> None:
    assert timeframe.seconds == seconds


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["1H", "1h", " 4h ", "1d"])
def test_timeframe_parses_user_input_leniently(raw: str) -> None:
    """Case and stray whitespace come from humans and command lines, not from bugs."""
    assert Timeframe.parse(raw) in set(Timeframe)


@pytest.mark.unit
def test_unknown_timeframe_names_what_is_supported() -> None:
    """An error message that does not say what IS allowed wastes the reader's time."""
    with pytest.raises(UnknownTimeframeError) as exc:
        Timeframe.parse("30m")
    assert "1H" in str(exc.value)


# --- SymbolSpec ---------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", ["XAUUSD", "SPX", "BTCUSD", "E1"])
def test_valid_symbols_are_accepted(name: str) -> None:
    assert SymbolSpec(name=name).name == name


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "",  # empty
        "xauusd",  # lower case: not portable on a case-insensitive filesystem
        "NASDAQ:AAPL",  # a colon is illegal in a Windows path
        "BTC/USD",  # a slash would silently create a directory
        "A" * 16,  # too long to stay readable in a file name
        "XAU USD",  # whitespace
    ],
)
def test_symbols_that_would_break_a_filesystem_are_rejected(name: str) -> None:
    """Symbols become file names, so the boundary is the place to refuse them.

    Accepting `BTC/USD` here would not fail here - it would fail three steps later as
    a mysteriously missing file, or worse, silently create a nested directory.
    """
    with pytest.raises((InvalidSymbolError, ValidationError)):
        SymbolSpec(name=name)


@pytest.mark.unit
def test_symbol_is_frozen() -> None:
    symbol = SymbolSpec(name="XAUUSD")
    with pytest.raises(ValidationError):
        symbol.name = "SPX"  # type: ignore[misc]  # assigning to a frozen field


# --- SeriesMeta ---------------------------------------------------------------------


def _meta(**overrides: object) -> SeriesMeta:
    base: dict[str, object] = {
        "symbol": SymbolSpec(name="XAUUSD"),
        "timeframe": Timeframe.H1,
        "rows": 2,
        "first": datetime(2022, 1, 2, 23, tzinfo=UTC),
        "last": datetime(2022, 1, 3, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return SeriesMeta(**base)


@pytest.mark.unit
def test_naive_timestamps_are_rejected() -> None:
    """A naive datetime is the commonest way a timezone bug enters a pipeline."""
    with pytest.raises(ValidationError):
        _meta(first=datetime(2022, 1, 2, 23))  # a naive datetime: the point of the test


@pytest.mark.unit
def test_timestamps_are_normalised_to_utc() -> None:
    """Two venues in two zones must compare as one timeline."""
    madrid = timezone(timedelta(hours=2))
    meta = _meta(
        first=datetime(2022, 1, 3, 1, tzinfo=madrid),
        last=datetime(2022, 1, 3, 2, tzinfo=UTC),
    )
    assert meta.first is not None
    assert meta.first.tzinfo is UTC
    assert meta.first.hour == 23


@pytest.mark.unit
def test_a_span_must_run_forwards() -> None:
    with pytest.raises(ValidationError):
        _meta(first=datetime(2022, 1, 3, tzinfo=UTC), last=datetime(2022, 1, 2, tzinfo=UTC))


@pytest.mark.unit
def test_an_empty_series_has_no_span() -> None:
    with pytest.raises(ValidationError):
        _meta(rows=0)


@pytest.mark.unit
def test_a_half_specified_span_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _meta(last=None)


@pytest.mark.unit
def test_anchors_must_fall_inside_one_bar() -> None:
    """An anchor is an offset within a bar, so it cannot exceed the bar's own length."""
    with pytest.raises(ValidationError):
        _meta(anchors=frozenset({3600}))


@pytest.mark.unit
def test_two_anchors_are_legitimate() -> None:
    """A venue whose trading day shifts with daylight saving has two anchors.

    This is data, not corruption (ADR-002 sec. 5), which is why `anchors` is a set.
    """
    meta = _meta(timeframe=Timeframe.H4, anchors=frozenset({7200, 10800}))
    assert meta.anchors == frozenset({7200, 10800})
