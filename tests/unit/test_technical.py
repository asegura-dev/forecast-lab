"""Tests for the indicator wrappers.

The first one is the reason this module exists rather than calling `ta` directly. The
rest pin the properties the feature matrix downstream takes for granted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

from forecast_lab.research import Scale, indicators, probe_scale
from forecast_lab.research.features import (
    ADX_WINDOW,
    ATR_WINDOW,
    RSI_WINDOW,
    SMA_WINDOWS,
    FeatureError,
)

ORIGIN = "2022-01-03"


def _bars(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")
    close = pd.Series(1800 + np.cumsum(rng.normal(0, 3, n)), index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + rng.uniform(0.5, 3, n),
            "low": close - rng.uniform(0.5, 3, n),
            "close": close,
        },
        index=index,
    )


# --- the reason the wrapper exists --------------------------------------------------


@pytest.mark.unit
def test_the_wilder_indicators_emit_zero_not_nan_during_warmup() -> None:
    """The single most load-bearing line in the wrapper.

    `ta` honours `fillna=False` for SMA, EMA, RSI, MACD and Bollinger. It does not for
    its Wilder recursions - AverageTrueRange and ADXIndicator both return **0.0** across
    their whole warm-up, with no NaN anywhere in the column.

    A zero is not "no data". It is nil volatility, or no trend at all: readings the
    market can plausibly produce. `dropna()` will not remove them, any normalisation
    dividing by them explodes, and a tree learns them as a real regime that happens to
    coincide with the start of the sample.

    Both halves are asserted - that the library does this, and that the wrapper undoes
    it - so that a future `ta` release fixing it makes this test fail and tells us the
    workaround can go.
    """
    bars = _bars()

    atr = AverageTrueRange(
        bars["high"], bars["low"], bars["close"], window=ATR_WINDOW, fillna=False
    ).average_true_range()
    adx = ADXIndicator(
        bars["high"], bars["low"], bars["close"], window=ADX_WINDOW, fillna=False
    ).adx()

    for raw in (atr, adx):
        assert int(raw.isna().sum()) == 0
        assert raw.iloc[: ATR_WINDOW - 1].eq(0.0).all()

    wrapped = indicators(bars)
    assert wrapped["atr_pct"].iloc[: ATR_WINDOW - 1].isna().all()
    assert pd.notna(wrapped["atr_pct"].iloc[ATR_WINDOW - 1])
    # ADX smooths a smoothed quantity, so its warm-up is about twice the window.
    assert wrapped[f"adx_{ADX_WINDOW}"].iloc[: 2 * ADX_WINDOW - 1].isna().all()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("column", "window"),
    [
        ("dist_sma_20", SMA_WINDOWS[0]),
        ("dist_sma_200", SMA_WINDOWS[2]),
        ("rsi_14", RSI_WINDOW),
        ("atr_pct", ATR_WINDOW),
        ("bb_wband", 20),
    ],
)
def test_every_indicator_blanks_its_warmup(column: str, window: int) -> None:
    """No column may report a value computed from less history than it claims."""
    frame = indicators(_bars())
    assert frame[column].iloc[: window - 1].isna().all()


# --- scale freedom, checked by measurement ------------------------------------------


@pytest.mark.unit
def test_no_column_moves_when_the_price_is_rescaled() -> None:
    """Quoting the same instrument in a different unit must change nothing.

    This is layer 1 of the stationarity policy, exercised directly on the indicators.
    Gold runs 1,616 to 4,378 in the real sample, so a column carrying the price level
    puts the test block outside the training data's support.
    """
    bars = _bars()
    for name, verdict in probe_scale(indicators, bars).items():
        assert verdict.scale is Scale.FREE, f"{name} moved by {verdict.change:.2e}"


@pytest.mark.unit
def test_the_probe_catches_a_price_level() -> None:
    """The guard has to fail on a bad column, or it guarantees nothing.

    Every real column passes, which is exactly why a test that only checks passing
    columns is worthless: it would still pass if the probe were `return True`.
    """
    bars = _bars()

    def with_a_raw_price(frame: pd.DataFrame) -> pd.DataFrame:
        out = indicators(frame)
        out["close_level"] = frame["close"]  # the mistake, made on purpose
        return out

    scales = probe_scale(with_a_raw_price, bars)
    assert scales["close_level"].scale is Scale.DEPENDENT
    assert scales["rsi_14"].scale is Scale.FREE


# --- refusals -----------------------------------------------------------------------


@pytest.mark.unit
def test_missing_columns_are_refused() -> None:
    with pytest.raises(FeatureError, match="missing column"):
        indicators(pd.DataFrame({"close": [1.0, 2.0]}))


@pytest.mark.unit
def test_an_unsorted_series_is_refused() -> None:
    bars = _bars(50)
    with pytest.raises(FeatureError, match="sorted"):
        indicators(bars.iloc[::-1])


@pytest.mark.unit
def test_a_non_positive_price_is_refused_rather_than_logged() -> None:
    """log(0) is -inf, which propagates silently through every mean downstream."""
    bars = _bars(50)
    bars.loc[bars.index[10], "close"] = 0.0
    with pytest.raises(FeatureError, match="non-positive"):
        indicators(bars)


@pytest.mark.unit
def test_the_prefix_namespaces_every_column() -> None:
    """WHOLE mode puts several symbols in one frame; collisions would be silent."""
    frame = indicators(_bars(60), prefix="XAUUSD")
    assert all(str(c).startswith("XAUUSD_") for c in frame.columns)
