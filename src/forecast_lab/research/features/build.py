"""Assembling the feature matrix, in the order that does not fabricate data.

The order is the whole content of this module, and getting it backwards is defect D9.

**Indicators first, on each symbol's own grid. Reindexing second.** The other way round
looks identical and is not. After reindexing, a row where an auxiliary symbol is stale
holds a copy of that symbol's previous price - so a return computed there is exactly
zero, an RSI drifts toward 50, and an ATR contracts toward nothing. None of that is a
market observation; it is an artefact of the alignment. And it is not randomly scattered
either: staleness tracks the hour of the day almost perfectly, because a symbol is stale
precisely when its venue is shut. A tree fed those columns learns the clock and the
report calls it a macro signal.

**Two modes, and the reason they exist.** The original project ran WHOLE (every symbol)
and FOCUS (the target alone) and compared them over different rows - 24,232 against
22,441 - so comparing them was never valid. Both modes here share an index, and a test
asserts it, so the comparison means something for the first time.

The cause of that mismatch is worth knowing, because an audit found this docstring had it
wrong. It was not crypto starting late: the original excludes BTCUSD explicitly. Its
indicator function ends with `dropna()` and runs once per symbol in a loop, so a 199-row
warm-up is paid once per contributor - ten times for WHOLE, once for FOCUS. Crypto is
still excluded here, for its own reason (it would cost a year of history), and stays
available as a *target*.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from forecast_lab.contracts import Timeframe
from forecast_lab.research.align import align_to_target
from forecast_lab.research.features.technical import LONGEST_WINDOW, FeatureError, indicators

#: Excluded from WHOLE because their history starts a year after everything else, so
#: including them would cost 25.5% of the rows - and it was the original project's own
#: eventual choice, for the same reason.
LATE_STARTERS = frozenset({"BTCUSD", "ETHUSD"})


class Mode(StrEnum):
    """Which symbols contribute features."""

    FOCUS = "focus"  # the target only
    WHOLE = "whole"  # the target plus every auxiliary that spans the sample


@dataclass(frozen=True)
class FeatureMatrix:
    """The matrix, and what it cost to build."""

    frame: pd.DataFrame
    target: str
    timeframe: Timeframe
    mode: Mode
    symbols: tuple[str, ...]
    rows_before_warmup: int
    warmup_dropped: int

    @property
    def rows(self) -> int:
        return len(self.frame)

    @property
    def columns(self) -> int:
        return len(self.frame.columns)


def build_features(
    target: str,
    series: Mapping[str, pd.DataFrame],
    timeframe: Timeframe,
    *,
    mode: Mode = Mode.FOCUS,
    timeframes: Mapping[str, Timeframe] | None = None,
    max_staleness_seconds: int | None = None,
    drop_warmup: bool = True,
) -> FeatureMatrix:
    """Build the feature matrix for ``target``, indicators first and reindexing second.

    ``series`` maps a symbol to its bars on that symbol's own grid - which is what makes
    the correct order possible at all. Handing this function an already-aligned panel
    would reintroduce D9, so it takes the raw series and does the alignment itself.
    """
    if target not in series:
        raise FeatureError(f"no bars supplied for the target {target}")

    contributors = _contributors(target, series, mode)

    # Step 1: indicators on native grids. Each symbol's frame keeps its own index, its
    # own gaps and its own trading calendar - which is the only state in which its
    # returns are the ones the market actually printed.
    #
    # No prefix here: `align_to_target` namespaces every column by symbol when it builds
    # the panel, and prefixing twice produces `GOLD_GOLD_rsi_14`.
    computed = {symbol: indicators(series[symbol]) for symbol in contributors}

    # Step 2: only now onto the target's timeline. Carrying a finished indicator forward
    # is honest - it is what was last known - whereas computing one on carried prices is
    # not, because the inputs were never observed at those instants.
    panel = align_to_target(
        target,
        computed,
        timeframe,
        timeframes=timeframes,
        max_staleness_seconds=max_staleness_seconds,
    )

    frame = panel.frame
    before = len(frame)
    dropped = 0
    if drop_warmup:
        # The first LONGEST_WINDOW - 1 rows cannot have a full window behind them for
        # every indicator. Dropping by position rather than by dropna() is deliberate:
        # dropna() would also delete rows a *stale auxiliary* left empty, which are real
        # rows carrying real target data and belong in the matrix, marked.
        frame = frame.iloc[max(LONGEST_WINDOW - 1, 0) :]
        dropped = before - len(frame)

    return FeatureMatrix(
        frame=frame,
        target=target,
        timeframe=timeframe,
        mode=mode,
        symbols=tuple(contributors),
        rows_before_warmup=before,
        warmup_dropped=dropped,
    )


def _contributors(target: str, series: Mapping[str, pd.DataFrame], mode: Mode) -> Sequence[str]:
    """Which symbols contribute columns, in a deterministic order.

    Sorted explicitly. The original used `list(set(...))`, whose order varies between
    processes - which makes the column layout, and therefore every PCA component fitted
    on it, irreproducible from one run to the next.
    """
    if mode is Mode.FOCUS:
        return [target]
    others = sorted(s for s in series if s != target and s not in LATE_STARTERS)
    return [target, *others]
