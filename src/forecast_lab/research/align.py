"""Putting several symbols on one timeline without inventing a single bar.

This is the correction at the centre of the whole re-analysis, so it is worth stating
what went wrong before stating what is done instead.

The original pipeline merged every symbol with an outer join and forward-filled the
result. That produces a row at every timestamp *any* symbol traded - including the
1,250 hours when the currencies were open and gold was not. On those rows gold's close
is a copy of the previous bar, so ``close[t+1] > close[t]`` compares a price with
itself, returns False, and manufactures a DOWN label out of nothing. Measured on the
real exports, the damage is not subtle: it turns 1,288 bars into exact ties, and flips
which class is the majority - the corrupted dataset says gold falls 51.47% of the time
while the real series says it rises 51.15%. Every model was then compared against a
baseline that did not exist.

The fix is to stop treating all symbols as equals. **The target's own bars are the
timeline.** Auxiliary symbols are read onto it by carrying their last known value
forward, which is legitimate - that value *was* the last thing known at that instant -
and the target is never filled, so no row exists that the target did not trade.

Two things make that safe rather than merely tidy:

- **A timestamp is the bar's open** (ADR-002 sec. 6). An auxiliary bar opening at or
  before `t` closes at or before `t + one interval`, which is exactly when the decision
  at `t` is taken. Nothing from the future can arrive.
- **Which is only true while the auxiliary interval is not longer than the target's.**
  A daily bar carried onto an hourly row is still *open* when the hourly decision is
  taken, so its close is information from the future. That is refused here, loudly.

Carrying a value forward is honest but not free: after a while "the last known value"
stops describing the present. So how stale each symbol is on each row is measured and
kept, per symbol, and a caller may set a limit beyond which the value becomes missing
instead of misleading.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_lab.contracts import ForecastLabError, Timeframe

#: Suffix of the column recording how old each auxiliary reading is, in seconds.
STALENESS_SUFFIX = "staleness_s"


class AlignmentError(ForecastLabError):
    """Two series cannot be put on one timeline without inventing information."""


@dataclass(frozen=True)
class SymbolAlignment:
    """How well one auxiliary symbol covered the target's timeline."""

    symbol: str
    rows: int
    missing: int  # rows before this symbol's history begins, or beyond the staleness limit
    stale: int  # rows carrying a value from an earlier bar than the row itself
    max_stale_seconds: int

    @property
    def stale_fraction(self) -> float:
        return self.stale / self.rows if self.rows else 0.0

    @property
    def missing_fraction(self) -> float:
        return self.missing / self.rows if self.rows else 0.0


@dataclass(frozen=True)
class AlignedPanel:
    """The target's timeline, with every symbol read onto it."""

    frame: pd.DataFrame
    target: str
    timeframe: Timeframe
    coverage: tuple[SymbolAlignment, ...]

    @property
    def rows(self) -> int:
        return len(self.frame)

    def columns_for(self, symbol: str) -> list[str]:
        return [c for c in self.frame.columns if c.startswith(f"{symbol}_")]


def align_to_target(
    target: str,
    series: Mapping[str, pd.DataFrame],
    timeframe: Timeframe,
    *,
    timeframes: Mapping[str, Timeframe] | None = None,
    max_staleness_seconds: int | None = None,
) -> AlignedPanel:
    """Put every symbol on the target's timeline, inventing nothing.

    ``series`` maps symbol name to its bars, indexed by timezone-aware UTC timestamps.
    ``timeframes`` gives each symbol's interval when they differ from ``timeframe``;
    omitted, every symbol is assumed to share the target's.

    Raises ``AlignmentError`` if the target is absent, if a symbol's index is unusable,
    or if an auxiliary's bars are longer than the target's - the one case where carrying
    a value forward would import the future.
    """
    if target not in series:
        raise AlignmentError(f"the target {target!r} is not among the series: {sorted(series)}")

    target_frame = series[target]
    index = _checked_index(target, target_frame)
    if len(index) == 0:
        raise AlignmentError(f"{target}: the target series is empty, so there is no timeline")

    intervals = dict(timeframes or {})
    columns: dict[str, pd.Series] = {}
    coverage: list[SymbolAlignment] = []

    # The target goes on unchanged. It defines the timeline; it is never filled.
    for name in target_frame.columns:
        columns[f"{target}_{name}"] = pd.Series(target_frame[name].to_numpy(), index=index)

    for symbol in sorted(s for s in series if s != target):
        aux_interval = intervals.get(symbol, timeframe)
        _reject_coarser(symbol, aux_interval, timeframe)

        aux = series[symbol]
        aux_index = _checked_index(symbol, aux)
        aligned, staleness, missing_mask = _carry_forward(
            aux, aux_index, index, max_staleness_seconds
        )

        for name in aux.columns:
            columns[f"{symbol}_{name}"] = aligned[name]
        columns[f"{symbol}_{STALENESS_SUFFIX}"] = staleness

        coverage.append(
            SymbolAlignment(
                symbol=symbol,
                rows=len(index),
                missing=int(missing_mask.sum()),
                # Any age above zero means this symbol did not trade on this bar and
                # its previous value is standing in. That is the thing worth counting;
                # a threshold here would only hide the smallest, commonest case.
                stale=int((staleness.fillna(0) > 0).sum()),
                max_stale_seconds=int(staleness.max()) if staleness.notna().any() else 0,
            )
        )

    return AlignedPanel(
        frame=pd.DataFrame(columns, index=index),
        target=target,
        timeframe=timeframe,
        coverage=tuple(coverage),
    )


def _reject_coarser(symbol: str, aux: Timeframe, target: Timeframe) -> None:
    """Refuse an auxiliary whose bars outlast the target's.

    This is the judgement the reader deliberately does not make (ADR-002 sec. 6),
    because only here are the target and its interval known. A daily bar carried onto
    an hourly row has not closed when that hour's decision is taken: its high, low and
    close describe hours that have not happened yet. No amount of grid cleanliness
    makes that safe, and the symptom - a model that predicts slightly too well - is
    exactly the one nobody investigates.
    """
    if aux.seconds > target.seconds:
        raise AlignmentError(
            f"{symbol} has {aux.value} bars and the target has {target.value}. A longer "
            f"bar is still open when the target's decision is taken, so carrying it "
            f"forward would import the future. Resample or drop the symbol."
        )


def _checked_index(symbol: str, frame: pd.DataFrame) -> pd.DatetimeIndex:
    """The frame's index, once it is provably a sorted, unique, UTC timeline."""
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise AlignmentError(f"{symbol}: the index is timezone-naive; it must be UTC")
    if not index.is_monotonic_increasing:
        raise AlignmentError(f"{symbol}: the index is not sorted")
    if index.has_duplicates:
        raise AlignmentError(f"{symbol}: the index has duplicated timestamps")
    return index


def _carry_forward(
    aux: pd.DataFrame,
    aux_index: pd.DatetimeIndex,
    target_index: pd.DatetimeIndex,
    max_staleness_seconds: int | None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Read an auxiliary onto the target's timeline, and measure how old each value is.

    For each target timestamp `t`, take the last auxiliary bar opening at or before `t`.
    ``searchsorted(t, side="right") - 1`` is that bar, and the ``- 1`` is what makes it
    "at or before" rather than "at or after" - the difference between reading the past
    and reading the future.

    Rows before the auxiliary's history begins have no such bar and become missing,
    never back-filled: a value that did not exist yet cannot be carried backwards to a
    time when nobody knew it.
    """
    positions = aux_index.searchsorted(target_index, side="right") - 1
    known = positions >= 0
    safe = np.where(known, positions, 0)

    # Subtract the timestamps and ask the result for seconds, rather than casting to
    # int64 and dividing. The integer view of a DatetimeIndex is in whatever resolution
    # pandas chose - nanoseconds here, milliseconds for the fetched series - so a fixed
    # divisor is right for one and a thousand times wrong for the other.
    elapsed = (target_index - aux_index[safe]).total_seconds()
    age = pd.Series(
        np.where(known, elapsed, np.nan),
        index=target_index,
        dtype="float64",
    )

    missing = ~known
    if max_staleness_seconds is not None:
        # Beyond the limit the last known value stops describing the present, so it is
        # dropped rather than left to masquerade as current.
        missing = missing | (age.to_numpy() > max_staleness_seconds)

    values = {
        name: pd.Series(
            np.where(missing, np.nan, aux[name].to_numpy()[safe]),
            index=target_index,
            dtype="float64",
        )
        for name in aux.columns
    }
    return pd.DataFrame(values, index=target_index), age.mask(missing), missing
