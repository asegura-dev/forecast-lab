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
    REALISED_VOL_WINDOWS,
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


# --- realised volatility, the feature that replaced VIX -----------------------------


@pytest.mark.unit
def test_realised_volatility_needs_a_full_window_before_it_reports() -> None:
    """`min_periods` equals the window, deliberately.

    A partial window would report a confident annualised figure computed from three
    bars. This is the feature that justified dropping VIX (ADR-002 sec. 4), so it does
    not get to be sloppier than the thing it replaced.
    """
    frame = indicators(_bars(600))
    for window in REALISED_VOL_WINDOWS:
        column = frame[f"realised_vol_{window}"]
        assert column.iloc[:window].isna().all()
        assert pd.notna(column.iloc[window])


@pytest.mark.unit
def test_realised_volatility_is_rolling_and_never_expanding() -> None:
    """An expanding window changes what the column means as the sample grows.

    At bar t it would have seen t observations and at bar t+1 it would have seen t+1, so
    early values are noisier than late ones for a reason that has nothing to do with the
    market. Asserted by doubling the volatility of the second half and checking the
    column follows: an expanding window would still be dominated by the calm first half.
    """
    rng = np.random.default_rng(17)
    n = 800
    index = pd.date_range(ORIGIN, periods=n, freq="h", tz="UTC")
    steps = rng.normal(0, 1, n)
    steps[n // 2 :] *= 6.0
    close = pd.Series(1800 + np.cumsum(steps), index=index)
    bars = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close}, index=index
    )

    column = indicators(bars)["realised_vol_24"]
    calm = float(column.iloc[n // 2 - 50 : n // 2 - 10].mean())
    wild = float(column.iloc[-40:].mean())
    assert wild > calm * 3, "the window is not tracking the regime it sits in"


@pytest.mark.unit
def test_realised_volatility_survives_a_change_of_price_unit() -> None:
    """It is a standard deviation of *log returns*, so the unit cancels."""
    bars = _bars(600)
    verdicts = probe_scale(indicators, bars)
    for window in REALISED_VOL_WINDOWS:
        assert verdicts[f"realised_vol_{window}"].scale is Scale.FREE


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
