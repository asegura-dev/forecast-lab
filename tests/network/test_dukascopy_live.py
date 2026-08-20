"""Opt-in live checks against the public venue.

Excluded from the default gates (`addopts = -m "not network"`); run deliberately with
`uv run pytest -m network`. The point is not to assert prices - those change - but to
prove the two structural claims the whole ingestion rests on, against the real service
rather than against a fixture that encodes what we already believe.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from forecast_lab.contracts import SymbolSpec, Timeframe
from forecast_lab.ingest import INSTRUMENTS, fetch_series

XAU = SymbolSpec(name="XAUUSD")

#: A window in the past, so the test does not depend on the market being open now.
START = datetime(2026, 8, 3, tzinfo=UTC)
END = datetime(2026, 8, 7, tzinfo=UTC)


@pytest.mark.network
def test_the_hourly_grid_sits_exactly_on_the_hour() -> None:
    """The invariant the whole alignment design rests on (ADR-002 sec. 6).

    A timestamp is the bar's open, and hourly bars open on the hour. If the venue ever
    shifted that grid, every forward-filled row downstream would carry look-ahead of the
    size of the shift, silently. This is cheap to check and catastrophic to assume.
    """
    fetched = fetch_series(XAU, Timeframe.H1, START, END)
    offsets = {int(ts.timestamp()) % 3600 for ts in fetched.frame.index}
    assert offsets == {0}, f"hourly bars are off the hour: {sorted(offsets)}"


@pytest.mark.network
def test_both_sides_are_combined_into_a_mid_and_a_spread() -> None:
    """The reason both sides are downloaded at all.

    The spread is the transaction cost, and the break-even accuracy that decides whether
    any edge is worth having is computed from it. Asserting only that it is positive
    keeps the test honest: its size is the market's business, not ours.
    """
    fetched = fetch_series(XAU, Timeframe.H1, START, END)
    frame = fetched.frame

    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "spread"]
    assert not frame.empty
    assert (frame["spread"] > 0).all(), "the ask must sit above the bid on every bar"
    assert (frame["high"] >= frame["low"]).all()


@pytest.mark.network
def test_the_spread_is_not_constant() -> None:
    """If it were, a constant would have done and this design would be over-engineering.

    Measured at the weekly open it is an order of magnitude wider than mid-session,
    which is exactly the variation a fixed cost assumption would have hidden.
    """
    fetched = fetch_series(XAU, Timeframe.H1, START, END)
    spread = fetched.frame["spread"]
    assert spread.max() > spread.min() * 2


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(INSTRUMENTS))
def test_every_mapped_symbol_still_exists(name: str) -> None:
    """The instrument map is the one place this project speaks the venue's vocabulary.

    A renamed or withdrawn instrument would otherwise surface as an empty series months
    later, and an empty series is indistinguishable from a closed market.
    """
    fetched = fetch_series(SymbolSpec(name=name), Timeframe.H1, START, END)
    assert not fetched.frame.empty
