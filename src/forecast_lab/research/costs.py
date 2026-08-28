"""What it costs to act on a prediction, and the accuracy that would pay for it.

Every accuracy in this project has been compared against **51.92%**, a break-even
carried from a planning document that assumed a one basis point round trip. The venue's
spread was downloaded with every bar from the first day and never used. Measured over
51,147 hourly bars of gold it is **1.86 bps at the median** - and the break-even that
follows is **53.49%**, more than a point and a half above the figure the repository has
been quoting.

That gap is not a detail. The best accuracy any configuration produced, on either
dataset, was 53.31% - and that one was already established as the best of eighteen coin
flips. Against a measured cost, **nothing tested comes within two points of paying for
itself.**

**The arithmetic.** A directional strategy that flips position at rate `f` pays the round
trip on each flip and captures the move otherwise. Setting expected profit to zero:

    p = 0.5 + f * c / (2 * E|r|)

with `c` the round-trip cost, `E|r|` the mean absolute return per bar, and `f` the share
of bars on which the position changes. At `f = 0.5` - a model with no persistence, which
is what a coin flip and every model measured here produce - it reduces to
`0.5 + c / (4 * E|r|)`.

**What this module does not model yet**, and each one raises the bar rather than lowering
it: slippage beyond the quoted spread, the rollover surcharge, and the overnight swap
that gold pays on the long side and triples on Wednesdays. The swap matters especially
for the gap finding - price rises through the venue's pauses 56-59% of the time, and
those are precisely the hours financing is charged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_lab.contracts import ForecastLabError

#: Position changes per bar for a model with no persistence. A predictor whose output is
#: independent from bar to bar flips half the time, which is what every model measured in
#: this project does - their autocorrelation of predictions is indistinguishable from
#: zero. A trend follower would sit lower and pay less; that is a parameter, not a
#: constant, and it is exposed as one.
DEFAULT_FLIP_RATE = 0.5

#: Cost assumed by the planning analysis, kept so the two can be compared in one table.
ASSUMED_ROUND_TRIP_BPS = 1.0


class CostError(ForecastLabError):
    """A cost model cannot be built from what was provided."""


@dataclass(frozen=True)
class SpreadSummary:
    """The distribution of the quoted spread, in basis points of price."""

    bars: int
    median_bps: float
    mean_bps: float
    p95_bps: float
    max_bps: float
    #: Mean absolute return per bar, in basis points - the size of what a correct
    #: prediction earns before costs.
    mean_absolute_return_bps: float

    @property
    def cost_to_move_ratio(self) -> float:
        """Median round trip as a share of the average move it has to be paid out of."""
        return self.median_bps / self.mean_absolute_return_bps


@dataclass(frozen=True)
class BreakEven:
    """The accuracy at which a strategy stops losing money, under one cost assumption."""

    label: str
    round_trip_bps: float
    flip_rate: float
    mean_absolute_return_bps: float

    @property
    def accuracy(self) -> float:
        return 0.5 + self.flip_rate * self.round_trip_bps / (2 * self.mean_absolute_return_bps)

    @property
    def edge_required(self) -> float:
        """Points above chance, which is the number to compare an edge against."""
        return self.accuracy - 0.5


def summarise_spread(bars: pd.DataFrame) -> SpreadSummary:
    """Describe the quoted spread and the move it has to be paid out of.

    ``bars`` needs `close` and `spread`, which is what `fetch` writes: it downloads both
    sides of the book and records the difference per bar. The reference exports carry no
    spread column at all, which is one more reason the canonical dataset had to exist.
    """
    for column in ("close", "spread"):
        if column not in bars.columns:
            raise CostError(f"a '{column}' column is required to model costs")

    frame = bars[["close", "spread"]].dropna()
    if frame.empty:
        raise CostError("no bars carry both a close and a spread")

    close = frame["close"].to_numpy(dtype="float64")
    spread = frame["spread"].to_numpy(dtype="float64")
    if (close <= 0).any():
        raise CostError("a non-positive close makes the spread unmeasurable in basis points")

    in_bps = spread / close * 10_000
    returns = np.diff(close) / close[:-1]

    return SpreadSummary(
        bars=len(frame),
        median_bps=float(np.median(in_bps)),
        mean_bps=float(np.mean(in_bps)),
        p95_bps=float(np.percentile(in_bps, 95)),
        max_bps=float(np.max(in_bps)),
        mean_absolute_return_bps=float(np.mean(np.abs(returns)) * 10_000),
    )


def break_even(
    round_trip_bps: float,
    mean_absolute_return_bps: float,
    *,
    label: str = "",
    flip_rate: float = DEFAULT_FLIP_RATE,
) -> BreakEven:
    """The accuracy that pays for one round trip.

    Raises rather than returning an absurd number when the move cannot cover the cost:
    an accuracy above 1.0 is not a threshold, it is the arithmetic saying the strategy is
    impossible at that frequency, and reporting it as a percentage invites someone to
    read it as merely demanding.
    """
    if mean_absolute_return_bps <= 0:
        raise CostError("the mean absolute return must be positive")
    if not 0 < flip_rate <= 1:
        raise CostError(f"the flip rate must be in (0, 1], got {flip_rate}")

    result = BreakEven(
        label=label or f"{round_trip_bps:.2f} bps",
        round_trip_bps=round_trip_bps,
        flip_rate=flip_rate,
        mean_absolute_return_bps=mean_absolute_return_bps,
    )
    if result.accuracy >= 1.0:
        raise CostError(
            f"a {round_trip_bps:.2f} bps round trip cannot be recovered from a "
            f"{mean_absolute_return_bps:.2f} bps average move at this frequency"
        )
    return result


def break_even_table(
    summary: SpreadSummary, *, flip_rate: float = DEFAULT_FLIP_RATE
) -> tuple[BreakEven, ...]:
    """Every cost assumption worth reporting, in one table.

    The assumed figure is kept first so that the difference between what was published
    and what was measured is visible rather than replaced quietly. A number this project
    has quoted eleven times does not get to change without the old value beside it.
    """
    move = summary.mean_absolute_return_bps
    return (
        break_even(ASSUMED_ROUND_TRIP_BPS, move, label="assumed (planning)", flip_rate=flip_rate),
        break_even(summary.median_bps, move, label="measured, median", flip_rate=flip_rate),
        break_even(summary.mean_bps, move, label="measured, mean", flip_rate=flip_rate),
        break_even(summary.p95_bps, move, label="measured, 95th pct", flip_rate=flip_rate),
    )


def net_of_costs(
    returns: pd.Series, positions: pd.Series, *, round_trip_bps: float
) -> pd.Series:
    """Strategy returns after paying the round trip on every position change.

    ``positions`` holds the position carried *into* each bar, so the cost of changing it
    is charged on the bar where the change happens. Getting that off by one credits a
    strategy with a move it paid to enter after the move was over - which is the same
    class of error as labelling by position instead of by timestamp.
    """
    aligned = positions.reindex(returns.index).ffill().fillna(0.0)
    gross = aligned * returns
    # A flip from -1 to +1 crosses the spread twice; abs() of the difference gets that
    # right, where counting "did it change" would charge one round trip for two.
    turnover = aligned.diff().abs().fillna(abs(aligned.iloc[0]) if len(aligned) else 0.0)
    charged = turnover / 2.0 * (round_trip_bps / 10_000)
    return gross - charged
