"""Scoring a model on many stretches of history instead of one.

A single chronological split scores a model on one block, and that block has a character:
the reference test set is a rally where gold rose 30%, and its share of rising bars
differs from the training block's by more than the effect anyone is trying to detect.
An accuracy measured there is partly a statement about which months it landed on.

Walk-forward answers the same question over several disjoint blocks. Each fold trains on
everything before its test window and scores the window that follows, so every prediction
is still made from the past only - and the folds together cover **5.6 times** the bars a
single split can score: 40,587 labelled bars against 7,306, which takes the minimum
detectable effect from 1.45 points to 0.62.

**What it fixes and what it does not.** It removes the dependence on one arbitrary block
and it shrinks the standard error, which is worth having. It does *not* remove the
dependence between neighbouring rows - overlapping feature windows make the effective
sample smaller than the row count in every design here - and it does not turn the folds
into independent experiments, because they share training data. Averaging fold accuracies
and quoting a t-statistic over them would be wrong for exactly that reason; the folds are
pooled instead, and the spread across them is reported as a description rather than as an
interval.

**Expanding, not rolling, by default.** An expanding window trains on all available
history, which is what someone deploying this would do. A rolling window of fixed length
answers a different question - whether recent history predicts better than distant - and
that is a hypothesis about regime change rather than a validation design. Both are
available; the default is the one that matches the deployment.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from forecast_lab.contracts import ForecastLabError

#: Folds by default. Five gives blocks large enough to score and few enough that each
#: still trains on a substantial history - with nine years of hourly bars, each test
#: window is around a year.
DEFAULT_FOLDS = 5


class WalkForwardError(ForecastLabError):
    """The series cannot be cut into walk-forward folds."""


@dataclass(frozen=True)
class Fold:
    """One train/test pair, in time order, with nothing shared between them."""

    number: int
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    purged: int

    @property
    def train_rows(self) -> int:
        return len(self.train)

    @property
    def test_rows(self) -> int:
        return len(self.test)


@dataclass(frozen=True)
class WalkForward:
    """Every fold, plus what the scheme discarded to keep them separate."""

    folds: tuple[Fold, ...]
    expanding: bool
    horizon: int

    @property
    def test_rows(self) -> int:
        """Total bars scored - the number that decides the design's resolution."""
        return sum(fold.test_rows for fold in self.folds)

    @property
    def purged(self) -> int:
        return sum(fold.purged for fold in self.folds)

    def __iter__(self) -> Iterator[Fold]:
        return iter(self.folds)

    def __len__(self) -> int:
        return len(self.folds)


def walk_forward(
    index: pd.DatetimeIndex,
    *,
    folds: int = DEFAULT_FOLDS,
    horizon: int = 1,
    expanding: bool = True,
    min_train: int | None = None,
) -> WalkForward:
    """Cut ``index`` into ``folds`` train/test pairs, each testing on the block that follows.

    The first fold trains on the initial stretch and tests the next; each subsequent fold
    moves the boundary forward by one test block. ``horizon`` bars are purged from the end
    of every training window, because a label at the last training bar reaches into the
    test block - the same reason the single split purges, applied once per fold.

    No embargo, for the same reason as the single split: nothing follows a test block that
    is used for training in the *same* fold. A later fold trains on it, which is correct -
    that is history by then - and is precisely what makes this walk-forward rather than
    k-fold on shuffled data.
    """
    if folds < 2:
        raise WalkForwardError(f"walk-forward needs at least two folds, got {folds}")
    if horizon < 0:
        raise WalkForwardError(f"the horizon cannot be negative, got {horizon}")
    if not index.is_monotonic_increasing:
        raise WalkForwardError("the index must be sorted before it can be cut chronologically")

    n = len(index)
    # Each fold needs a test block, and the first needs a training block before it, so
    # the series is divided into folds + 1 stretches.
    block = n // (folds + 1)
    if block <= horizon:
        raise WalkForwardError(
            f"{n:,} rows split into {folds + 1} blocks leaves {block} per block, which is "
            f"not more than the horizon of {horizon}"
        )

    floor = min_train if min_train is not None else block
    if floor > n - folds * block:
        raise WalkForwardError(
            f"a minimum training size of {floor:,} leaves no room for {folds} test blocks"
        )

    built: list[Fold] = []
    for number in range(folds):
        train_end = block * (number + 1)
        test_end = min(train_end + block, n)
        if test_end - train_end <= 0:
            break

        # The window a deployment would have: everything so far, or the most recent
        # `floor` bars when rolling.
        start = 0 if expanding else max(0, train_end - floor)
        train = index[start : max(train_end - horizon, start)]
        # Compared against `floor - horizon`, not `floor`: purging always removes the
        # last `horizon` bars, so measuring the purged window against the unpurged
        # minimum fails every time the horizon is positive. Found by running it on real
        # data, where fold 1 came out at 8,523 rows against a floor of 8,524.
        if len(train) < floor - horizon:
            raise WalkForwardError(
                f"fold {number + 1} has {len(train):,} training rows, below the "
                f"{floor - horizon:,} required after purging"
            )

        built.append(
            Fold(
                number=number + 1,
                train=train,
                test=index[train_end:test_end],
                purged=train_end - len(train) - start,
            )
        )

    if not built:
        raise WalkForwardError("no fold could be built from this index")
    return WalkForward(folds=tuple(built), expanding=expanding, horizon=horizon)
