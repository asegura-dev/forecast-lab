"""Tests for assembling the matrix.

The first two are the ones that matter: the order of operations (defect D9) and the
shared index between modes, which the original project never had and therefore never
had a valid comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_lab.contracts import Timeframe
from forecast_lab.research import FeatureError, Mode, build_features
from forecast_lab.research.features import LATE_STARTERS, LONGEST_WINDOW

H1 = Timeframe.H1


def _bars(n: int, *, start: str = "2022-01-03", every: int = 1, seed: int = 4) -> pd.DataFrame:
    """``every`` > 1 gives a symbol that trades less often - a stale auxiliary."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq=f"{every}h", tz="UTC")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=index)
    return pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close}, index=index
    )


# --- the order of operations --------------------------------------------------------


@pytest.mark.unit
def test_a_stale_auxiliary_does_not_get_a_fabricated_zero_return() -> None:
    """Defect D9, stated as a test.

    The auxiliary trades every third hour. Computing its return *after* reindexing would
    produce an exact zero on the two rows out of three where its price is a carried
    copy - and since those rows are the same hours every day, a tree learns the clock.
    Computing before means the carried value is a real return from its own last bar.
    """
    target = _bars(300)
    sparse = _bars(100, every=3, seed=7)

    matrix = build_features(
        "GOLD", {"GOLD": target, "AUX": sparse}, H1, mode=Mode.WHOLE, drop_warmup=False
    )
    returns = matrix.frame["AUX_return"].dropna()

    # If the order were wrong, roughly two thirds of these would be exactly zero.
    zeros = float((returns == 0.0).mean())
    assert zeros < 0.05, f"{zeros:.1%} of carried returns are exactly zero"


@pytest.mark.unit
def test_the_target_contributes_its_own_indicators() -> None:
    matrix = build_features("GOLD", {"GOLD": _bars(300)}, H1)
    assert any(str(c).startswith("GOLD_") for c in matrix.frame.columns)
    assert "GOLD_rsi_14" in matrix.frame.columns


# --- the two modes ------------------------------------------------------------------


@pytest.mark.unit
def test_whole_and_focus_share_an_index() -> None:
    """The comparison the original project ran but could not justify.

    Its WHOLE included crypto, which starts a year late, so the two modes covered
    different rows - 24,232 against 22,441 - and every difference between them mixed a
    change of feature set with a change of sample.
    """
    series = {"GOLD": _bars(300), "SPX": _bars(300, seed=11), "EURUSD": _bars(300, seed=12)}

    focus = build_features("GOLD", series, H1, mode=Mode.FOCUS)
    whole = build_features("GOLD", series, H1, mode=Mode.WHOLE)

    assert focus.frame.index.equals(whole.frame.index)
    assert whole.columns > focus.columns


@pytest.mark.unit
def test_whole_excludes_the_late_starters() -> None:
    """Crypto costs 25.5% of the rows and buys a symbol that starts a year late."""
    late = next(iter(LATE_STARTERS))
    series = {"GOLD": _bars(300), "SPX": _bars(300, seed=11), late: _bars(300, seed=13)}

    whole = build_features("GOLD", series, H1, mode=Mode.WHOLE)
    assert late not in whole.symbols
    assert not any(str(c).startswith(f"{late}_") for c in whole.frame.columns)


@pytest.mark.unit
def test_a_late_starter_may_still_be_the_target() -> None:
    """Excluded as a feature source, available as the thing being predicted."""
    late = next(iter(LATE_STARTERS))
    matrix = build_features(late, {late: _bars(300)}, H1)
    assert matrix.target == late


@pytest.mark.unit
def test_the_symbol_order_is_deterministic() -> None:
    """The original used `list(set(...))`, whose order varies between processes - which
    makes the column layout, and any PCA fitted on it, irreproducible."""
    series = {"GOLD": _bars(250), "ZZZ": _bars(250, seed=2), "AAA": _bars(250, seed=3)}
    first = build_features("GOLD", series, H1, mode=Mode.WHOLE).symbols
    second = build_features("GOLD", dict(reversed(list(series.items()))), H1, mode=Mode.WHOLE)
    assert first == second.symbols == ("GOLD", "AAA", "ZZZ")


# --- warm-up ------------------------------------------------------------------------


@pytest.mark.unit
def test_the_warmup_is_dropped_by_position_not_by_dropna() -> None:
    """dropna() would also delete rows a stale auxiliary left empty - real target rows
    that belong in the matrix, marked."""
    matrix = build_features("GOLD", {"GOLD": _bars(400)}, H1)
    assert matrix.warmup_dropped == LONGEST_WINDOW - 1
    assert matrix.rows == matrix.rows_before_warmup - matrix.warmup_dropped


@pytest.mark.unit
def test_the_warmup_can_be_kept() -> None:
    matrix = build_features("GOLD", {"GOLD": _bars(400)}, H1, drop_warmup=False)
    assert matrix.warmup_dropped == 0


@pytest.mark.unit
def test_a_missing_target_is_refused() -> None:
    with pytest.raises(FeatureError, match="no bars supplied"):
        build_features("GOLD", {"SPX": _bars(100)}, H1)
