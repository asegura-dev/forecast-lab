"""Tests for the cost model.

The first one is the reason the module exists: every accuracy in this project was
compared against a break-even derived from an assumed cost, while the venue's measured
spread sat unused in a column that `fetch` had been downloading with every bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_lab.research import (
    ASSUMED_ROUND_TRIP_BPS,
    BreakEven,
    CostError,
    break_even,
    break_even_table,
    net_of_costs,
    summarise_spread,
)

ORIGIN = "2022-01-03"


def _bars(n: int = 500, *, spread: float = 0.4, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")
    close = pd.Series(2000 + np.cumsum(rng.normal(0, 3, n)), index=index)
    return pd.DataFrame({"close": close, "spread": spread}, index=index)


# --- the arithmetic -------------------------------------------------------------------


@pytest.mark.unit
def test_a_costless_strategy_breaks_even_at_chance() -> None:
    """The sanity anchor: with nothing to pay, any edge above 50% is profit."""
    assert break_even(0.0, 13.0).accuracy == pytest.approx(0.5)


@pytest.mark.unit
def test_the_break_even_rises_with_the_cost() -> None:
    """Monotone, and the shape matters: it is linear in the cost."""
    cheap = break_even(1.0, 13.35).accuracy
    dear = break_even(2.0, 13.35).accuracy
    assert dear > cheap
    # Twice the cost is twice the distance from chance.
    assert (dear - 0.5) == pytest.approx(2 * (cheap - 0.5))


@pytest.mark.unit
def test_the_measured_cost_demands_more_than_the_assumed_one() -> None:
    """The finding this module was written to expose.

    Measured on 51,147 hourly bars of gold: a median spread of 1.86 bps against the
    1.00 bps the planning analysis assumed, and a mean absolute move of 13.35 bps. The
    break-even moves from 51.87% to **53.49%** - past every accuracy this project has
    produced, including the best of eighteen configurations.
    """
    move = 13.35
    assumed = break_even(ASSUMED_ROUND_TRIP_BPS, move)
    measured = break_even(1.86, move)

    assert assumed.accuracy == pytest.approx(0.5187, abs=1e-4)
    assert measured.accuracy == pytest.approx(0.5349, abs=1e-4)
    # The best accuracy any configuration reached, on either dataset.
    assert measured.accuracy > 0.5331


@pytest.mark.unit
def test_a_lower_flip_rate_pays_less() -> None:
    """A trend follower holds through more bars and crosses the spread less often."""
    flippy = break_even(2.0, 13.0, flip_rate=0.5).accuracy
    patient = break_even(2.0, 13.0, flip_rate=0.1).accuracy
    assert patient < flippy
    assert (patient - 0.5) == pytest.approx((flippy - 0.5) / 5)


@pytest.mark.unit
def test_an_unrecoverable_cost_is_refused_rather_than_reported() -> None:
    """An accuracy above 1.0 is not a demanding threshold, it is an impossibility.

    Reporting "104%" invites someone to read it as merely hard. Raising says what the
    arithmetic actually found: at this frequency, that cost cannot be recovered.
    """
    with pytest.raises(CostError, match="cannot be recovered"):
        break_even(round_trip_bps=200.0, mean_absolute_return_bps=13.0)


@pytest.mark.unit
@pytest.mark.parametrize("flip", [0.0, -0.1, 1.5])
def test_an_impossible_flip_rate_is_refused(flip: float) -> None:
    with pytest.raises(CostError, match="flip rate"):
        break_even(1.0, 13.0, flip_rate=flip)


@pytest.mark.unit
def test_a_zero_move_is_refused() -> None:
    with pytest.raises(CostError, match="must be positive"):
        break_even(1.0, 0.0)


# --- summarising the spread -----------------------------------------------------------


@pytest.mark.unit
def test_the_spread_is_reported_in_basis_points_of_price() -> None:
    """A spread in USD means nothing without the price it is a fraction of.

    0.40 USD on a 2,000 USD instrument is 2 bps; the same 0.40 on gold at 4,000 is 1.
    """
    bars = pd.DataFrame(
        {"close": [2000.0] * 100, "spread": [0.4] * 100},
        index=pd.date_range(ORIGIN, periods=100, freq="h", tz="UTC"),
    )
    summary = summarise_spread(bars)
    assert summary.median_bps == pytest.approx(2.0)
    assert summary.bars == 100


@pytest.mark.unit
def test_the_summary_carries_the_move_the_cost_is_paid_from() -> None:
    """Cost alone is not a threshold; it becomes one relative to the average move."""
    summary = summarise_spread(_bars(800))
    assert summary.mean_absolute_return_bps > 0
    assert 0 < summary.cost_to_move_ratio < 1


@pytest.mark.unit
def test_a_missing_spread_column_is_refused() -> None:
    """The reference exports have no spread at all, which is why `fetch` downloads both
    sides of the book."""
    with pytest.raises(CostError, match="'spread' column"):
        summarise_spread(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))


@pytest.mark.unit
def test_the_table_keeps_the_assumed_figure_beside_the_measured_ones() -> None:
    """A number quoted eleven times does not get replaced without its old value shown."""
    table = break_even_table(summarise_spread(_bars(600)))
    assert table[0].label == "assumed (planning)"
    assert all(isinstance(entry, BreakEven) for entry in table)
    # Median, mean and p95 of a constant spread coincide; the ordering must still hold.
    assert table[3].accuracy >= table[1].accuracy


# --- charging the cost ----------------------------------------------------------------


@pytest.mark.unit
def test_holding_a_position_costs_nothing_after_entering_it() -> None:
    """Turnover, not exposure, is what pays the spread."""
    index = pd.date_range(ORIGIN, periods=10, freq="h", tz="UTC")
    returns = pd.Series(0.001, index=index)
    always_long = pd.Series(1.0, index=index)

    net = net_of_costs(returns, always_long, round_trip_bps=10.0)
    # One entry charged on the first bar, nothing thereafter.
    assert net.iloc[0] < net.iloc[1]
    assert net.iloc[1:].nunique() == 1


@pytest.mark.unit
def test_a_flip_costs_twice_a_single_entry() -> None:
    """Going from short to long crosses the spread on both legs.

    Counting "did the position change" rather than how far it moved would charge one
    round trip for two, and a strategy that flips constantly would look half as expensive
    as it is.
    """
    index = pd.date_range(ORIGIN, periods=4, freq="h", tz="UTC")
    returns = pd.Series(0.0, index=index)

    entered = net_of_costs(returns, pd.Series([0.0, 1.0, 1.0, 1.0], index=index),
                           round_trip_bps=10.0)
    flipped = net_of_costs(returns, pd.Series([0.0, -1.0, 1.0, 1.0], index=index),
                           round_trip_bps=10.0)

    assert flipped.iloc[2] == pytest.approx(2 * entered.iloc[1])


@pytest.mark.unit
def test_costs_can_turn_a_winning_gross_return_into_a_losing_net_one() -> None:
    """The whole point of the module, in one assertion.

    A strategy that is right slightly more often than not, flipping every bar, loses
    money once the spread is charged. That is the gap between an edge and a profit.
    """
    rng = np.random.default_rng(17)
    n = 4000
    index = pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")
    returns = pd.Series(rng.normal(0, 0.0013, n), index=index)
    # Right 52% of the time - above chance, below the 53.49% break-even.
    correct = rng.random(n) < 0.52
    positions = pd.Series(np.where(correct, np.sign(returns), -np.sign(returns)), index=index)

    gross = (positions * returns).sum()
    net = net_of_costs(returns, positions, round_trip_bps=1.86).sum()

    assert gross > 0
    assert net < 0
