"""Cutting a series into train, validation and test without letting them touch.

Three properties, in descending order of how often they are got wrong.

**The cut is chronological.** Shuffling a time series lets a model train on Thursday to
predict Wednesday. Everyone knows this; it is stated because the rest of the module only
makes sense once it is assumed.

**A label reaches forward, so the boundary must reach back.** A bar at the last row of
train carries an answer that lives `horizon` bars into validation. Without removing it,
training data contains the thing being predicted on the other side of the line. The
removal is called **purging**, and here it is exactly `horizon` bars - no more.

**No embargo, and that is deliberate.** The usual companion to purging discards a
stretch *after* the test block, because in k-fold cross-validation there is training
data on both sides of it. In a single chronological split there is nothing after test,
so an embargo would remove real data to protect against a leak that cannot occur. It
arrives with combinatorial cross-validation or not at all.

What purging cannot fix is worth naming: overlapping feature windows make neighbouring
rows dependent, so the *effective* sample is smaller than the row count. That inflates
nothing here, but it does mean every confidence interval computed downstream has to
account for it. That is the evaluation layer's problem, not this one's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from forecast_lab.contracts import ForecastLabError


class SplitError(ForecastLabError):
    """The series cannot be cut into usable blocks."""


@dataclass(frozen=True)
class Block:
    """One contiguous stretch of the timeline."""

    name: str
    index: pd.DatetimeIndex

    @property
    def rows(self) -> int:
        return len(self.index)

    @property
    def first(self) -> pd.Timestamp | None:
        return self.index[0] if len(self.index) else None

    @property
    def last(self) -> pd.Timestamp | None:
        return self.index[-1] if len(self.index) else None


@dataclass(frozen=True)
class TemporalSplit:
    """Train, validation and test, in order, with nothing shared between them."""

    train: Block
    validation: Block
    test: Block
    purged: int

    @property
    def blocks(self) -> tuple[Block, Block, Block]:
        return (self.train, self.validation, self.test)

    @property
    def rows(self) -> int:
        return sum(b.rows for b in self.blocks) + self.purged


def temporal_split(
    index: pd.DatetimeIndex,
    *,
    train: float = 0.70,
    validation: float = 0.15,
    horizon: int = 1,
) -> TemporalSplit:
    """Cut ``index`` chronologically, purging ``horizon`` bars at each boundary.

    The test share is whatever the other two leave, so the three always sum to the whole
    series and a rounding error cannot silently drop a block of rows.
    """
    if not 0 < train < 1 or not 0 <= validation < 1 or train + validation >= 1:
        raise SplitError(
            f"train={train} and validation={validation} must be positive fractions "
            "leaving room for a test block"
        )
    if horizon < 0:
        raise SplitError(f"the horizon cannot be negative, got {horizon}")
    if len(index) < 3 * (horizon + 1):
        raise SplitError(
            f"{len(index)} rows is too few to cut into three blocks with a purge of "
            f"{horizon}; the result would be blocks smaller than the horizon they predict"
        )
    if not index.is_monotonic_increasing:
        raise SplitError("the index must be sorted before it can be cut chronologically")

    n = len(index)
    train_end = int(n * train)
    validation_end = int(n * (train + validation))

    # The last `horizon` bars of each block carry an answer that lives in the next one,
    # so they belong to neither and are dropped.
    blocks = (
        Block("train", index[: max(train_end - horizon, 0)]),
        Block("validation", index[train_end : max(validation_end - horizon, train_end)]),
        Block("test", index[validation_end:]),
    )
    for block in blocks:
        if block.rows == 0:
            raise SplitError(f"the {block.name} block came out empty; adjust the fractions")

    return TemporalSplit(
        train=blocks[0],
        validation=blocks[1],
        test=blocks[2],
        purged=n - sum(b.rows for b in blocks),
    )
