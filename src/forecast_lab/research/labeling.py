"""Turning prices into the thing a model is asked to predict.

A direction label looks like one line of code - ``close.shift(-1) > close`` - and that
line contains three decisions, each of which the original analysis got wrong by not
noticing it was making one.

**Which bar is "next".** Shifting by position assumes the next row is one interval
later. On the target series it is not: gold has 780 two-hour gaps where the venue
pauses, 183 weekend gaps of about fifty hours, and a handful longer still. In 4.37% of
rows, ``close.shift(-1)`` is not the next hour - it is Monday. Those are a different
prediction problem with several times the variance, quietly mixed into the same metric.
So the horizon is expressed as a **timestamp**, and every row records how far ahead its
answer actually lies.

**What a tie means.** ``>`` is strictly greater, so a bar closing exactly where it
opened counts as DOWN. That sounds pedantic until the alignment is wrong: forward-filled
rows are *all* ties, so the defect lands entirely on one class. The real series has 38
genuine ties out of 23,180 - rare, but not zero, and never silently swept into a
direction.

**Whether the move is worth acting on.** A move smaller than the cost of trading is not
an opportunity, and a model rewarded for predicting it is being trained on noise. An
optional dead band marks those rows, and it is computed **causally** - from information
available before the bar - because a threshold derived from the move it is judging would
select the test set on the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from forecast_lab.contracts import ForecastLabError, Timeframe


class LabelingError(ForecastLabError):
    """The label cannot be built without inventing something."""


class Direction(IntEnum):
    """What the price did over the horizon.

    ``FLAT`` exists so that a tie, or a move inside the dead band, is a stated outcome
    rather than a silent DOWN.
    """

    DOWN = 0
    UP = 1
    FLAT = 2


@dataclass(frozen=True)
class LabelReport:
    """What the labelling produced, and what it had to leave out."""

    rows: int
    labelled: int  # rows whose horizon lands on a real bar at the right distance
    up: int
    down: int
    flat: int
    gapped: int  # rows whose next bar is further away than the horizon asks

    # What price did across those gaps. Not labels - the stretch they span is not the
    # one being predicted - but the distribution is reported rather than discarded,
    # because discarding it is how the notebook hid a 4.37% subpopulation inside its
    # headline accuracy.
    gapped_up: int
    gapped_down: int

    @property
    def up_rate(self) -> float:
        decided = self.up + self.down
        return self.up / decided if decided else 0.0

    @property
    def gapped_up_rate(self) -> float:
        """How often price rose across a gap rather than across the stated horizon.

        Worth its own number: if it differs materially from ``up_rate``, then the rows
        a positional shift silently folds in are a different problem, and any metric
        that mixes them is averaging two things.
        """
        decided = self.gapped_up + self.gapped_down
        return self.gapped_up / decided if decided else 0.0

    @property
    def gapped_fraction(self) -> float:
        return self.gapped / self.rows if self.rows else 0.0


def label_direction(
    close: pd.Series,
    timeframe: Timeframe,
    *,
    horizon: int = 1,
    dead_band: pd.Series | None = None,
) -> tuple[pd.DataFrame, LabelReport]:
    """Label each bar with what the price does ``horizon`` bars ahead.

    Returns a frame carrying the label, the realised return, how far ahead the answer
    actually lies, and whether that distance is the one asked for.

    ``dead_band``, when given, is a per-row threshold on the absolute return below which
    the move is called FLAT. It must be causal - computed from bars at or before each
    row - because a threshold derived from the move it judges would be selecting on the
    answer.
    """
    if horizon < 1:
        raise LabelingError(f"the horizon must be at least one bar, got {horizon}")
    index = pd.DatetimeIndex(close.index)
    if not index.is_monotonic_increasing:
        raise LabelingError("the series must be sorted before it can be labelled")

    # The answer is looked up by timestamp, not by position. Position would silently
    # treat Monday's close as "the next hour" across every weekend.
    wanted = index + pd.Timedelta(seconds=timeframe.seconds * horizon)
    positions = index.searchsorted(wanted, side="left")
    found = positions < len(index)
    safe = np.where(found, positions, 0)

    # A bar exists at that position, but it is only the answer if it sits exactly where
    # the horizon asks. Anything further is the far side of a gap.
    landed = np.where(found, index[safe] == wanted, False)

    values = close.to_numpy(dtype="float64")
    future = np.where(landed, values[safe], np.nan)
    change = future - values

    # The move to whatever bar actually comes next, at whatever distance. This is what
    # `close.shift(-1)` computes, and keeping it alongside `change` is what makes the
    # difference between the two measurable instead of asserted.
    next_change = np.where(found, values[safe] - values, np.nan)

    label = np.full(len(index), Direction.FLAT.value, dtype="float64")
    label = np.where(change > 0, Direction.UP.value, label)
    label = np.where(change < 0, Direction.DOWN.value, label)

    if dead_band is not None:
        if not dead_band.index.equals(close.index):
            raise LabelingError("the dead band must be indexed like the price series")
        band = dead_band.to_numpy(dtype="float64")
        label = np.where(np.abs(change) < band, Direction.FLAT.value, label)

    label = np.where(landed, label, np.nan)

    # Subtract as timestamps and ask the result for seconds. Casting to int64 and
    # dividing by a hard-coded 1e9 assumes nanosecond resolution, and pandas stores
    # whatever unit the source implied - microseconds here, milliseconds for a fetched
    # series. That assumption already produced a staleness figure a thousand times too
    # small once; it does not get to happen twice.
    ahead = np.where(found, (index[safe] - index).total_seconds(), np.nan)

    frame = pd.DataFrame(
        {
            "label": label,
            "future_close": future,
            "change": change,
            "next_change": next_change,
            "seconds_ahead": ahead,
            "label_valid": landed,
        },
        index=index,
    )

    return frame, _report(frame, landed, next_change)


def _report(frame: pd.DataFrame, landed: np.ndarray, next_change: np.ndarray) -> LabelReport:
    label = frame["label"]
    # Rows where a later bar exists but not at the distance asked for: the horizon spans
    # a gap, so the answer would describe a longer stretch than it claims to.
    gapped = ~landed & frame["seconds_ahead"].notna().to_numpy()
    across = next_change[gapped]
    return LabelReport(
        rows=len(frame),
        labelled=int(landed.sum()),
        up=int((label == Direction.UP.value).sum()),
        down=int((label == Direction.DOWN.value).sum()),
        flat=int((label == Direction.FLAT.value).sum()),
        gapped=int(gapped.sum()),
        gapped_up=int((across > 0).sum()),
        gapped_down=int((across < 0).sum()),
    )


def rolling_dead_band(
    close: pd.Series, *, window: int, multiple: float = 1.0, min_periods: int | None = None
) -> pd.Series:
    """A causal threshold: a multiple of recent absolute change, computed from the past.

    ``shift(1)`` is the load-bearing part. Without it the window would include the very
    bar whose move is being judged, and the threshold would be a function of the answer -
    which is how a test set gets selected on its own outcome.
    """
    moves = close.diff().abs()
    rolling = moves.rolling(window=window, min_periods=min_periods or window)
    return rolling.mean().shift(1) * multiple
