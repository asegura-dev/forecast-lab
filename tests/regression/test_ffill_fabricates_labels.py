"""The defect at the centre of the re-analysis, pinned so it cannot come back.

The original pipeline merged every symbol with an outer join and forward-filled. That
creates a row at every timestamp *any* symbol traded, so on the hours when the
currencies were open and gold was not, gold's close became a copy of the previous bar.
A label built from ``close[t+1] > close[t]`` then compares a price with itself, returns
False, and manufactures a DOWN out of nothing.

Measured on the real exports, with gold as the target and the nine auxiliaries the
original project used:

===========================  ======  =======  =======  =======
strategy                      rows    ties      UP       DOWN
===========================  ======  =======  =======  =======
outer join + forward fill     24,430   1,288   48.53%   51.47%
target-anchored (correct)     23,180      38   51.15%   48.85%
===========================  ======  =======  =======  =======

Two things about that table matter more than the row count. The forward fill **never
creates an UP** - 11,857 either way - so every fabricated row lands on one side. And it
**flips which class is the majority**: the corrupted data says gold falls more often
than it rises, which is the baseline every model was then measured against.

The invariant these tests fix is the identity behind it: **each fabricated row adds
exactly one tie**. On the real data that reads 24,430 - 23,180 = 1,250 and
1,288 - 38 = 1,250. Here it is proven on synthetic bars, because a test that needs
19 MB of vendor data to run is a test that stops being run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from forecast_lab.contracts import Timeframe
from forecast_lab.research import align_to_target

HOUR = 3600


def _bars(offsets: list[int], prices: list[float]) -> pd.DataFrame:
    """A price series at the given hour offsets from a fixed origin."""
    origin = datetime(2022, 1, 3, tzinfo=UTC)
    index = pd.DatetimeIndex([origin + timedelta(hours=o) for o in offsets])
    return pd.DataFrame({"close": prices}, index=index)


def _ties(close: pd.Series) -> int:
    """Bars whose successor closes at exactly the same price."""
    values = close.to_numpy()
    return int(np.sum(values[1:] == values[:-1]))


def _naive_union_ffill(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The original approach, reproduced here so the contrast is measured, not asserted.

    It lives inside the test rather than in the package: this repository must not ship a
    function whose only purpose is to be wrong.
    """
    joined = None
    for symbol, frame in series.items():
        renamed = frame.add_prefix(f"{symbol}_")
        joined = renamed if joined is None else joined.join(renamed, how="outer")
    assert joined is not None
    return joined.ffill()


@pytest.mark.regression
def test_each_fabricated_row_adds_exactly_one_tie() -> None:
    """The identity that quantifies the damage.

    Gold trades every hour except 3 and 4; the currency trades throughout. The outer
    join therefore invents exactly two gold bars, and each one ties with the bar before
    it.
    """
    target = _bars([0, 1, 2, 5, 6], [100.0, 101.0, 102.0, 103.0, 104.0])
    auxiliary = _bars([0, 1, 2, 3, 4, 5, 6], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    series = {"XAUUSD": target, "EURUSD": auxiliary}

    naive = _naive_union_ffill(series)
    correct = align_to_target("XAUUSD", series, Timeframe.H1)

    fabricated = len(naive) - correct.rows
    added_ties = _ties(naive["XAUUSD_close"]) - _ties(correct.frame["XAUUSD_close"])

    assert fabricated == 2
    assert added_ties == fabricated, (
        "every row the merge invents copies the previous close, so it must add exactly "
        "one tie; if this drifts, the alignment has started filling the target"
    )


@pytest.mark.regression
def test_the_forward_fill_never_creates_an_up() -> None:
    """All the distortion lands on one side, which is why it moves the baseline.

    A label of ``close[t+1] > close[t]`` is strictly greater, so a tie counts as DOWN.
    The fabricated rows can therefore only ever add DOWNs - never a single UP - and the
    class balance shifts by exactly the number of rows invented.
    """
    target = _bars([0, 1, 2, 5, 6], [100.0, 101.0, 102.0, 103.0, 104.0])
    auxiliary = _bars([0, 1, 2, 3, 4, 5, 6], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    series = {"XAUUSD": target, "EURUSD": auxiliary}

    naive_close = _naive_union_ffill(series)["XAUUSD_close"].to_numpy()
    correct_close = align_to_target("XAUUSD", series, Timeframe.H1).frame["XAUUSD_close"].to_numpy()

    ups_naive = int(np.sum(naive_close[1:] > naive_close[:-1]))
    ups_correct = int(np.sum(correct_close[1:] > correct_close[:-1]))

    assert ups_naive == ups_correct


@pytest.mark.regression
def test_the_target_timeline_is_never_extended() -> None:
    """The property the whole correction rests on: no row the target did not trade.

    The auxiliary may open earlier, close later, and trade through gaps; none of that
    can add a bar to the target's own timeline.
    """
    target = _bars([2, 3, 4], [100.0, 101.0, 102.0])
    auxiliary = _bars([0, 1, 2, 3, 4, 5, 6], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

    panel = align_to_target("XAUUSD", {"XAUUSD": target, "EURUSD": auxiliary}, Timeframe.H1)

    assert panel.rows == 3
    assert list(panel.frame.index) == list(target.index)
    assert panel.frame["XAUUSD_close"].tolist() == [100.0, 101.0, 102.0]
