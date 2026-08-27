"""Technical indicators, wrapped so their edges are visible.

The indicators themselves come from `ta` - reimplementing an RSI would be reinventing a
wheel that thousands of people have already debugged. What this module adds is the three
guarantees `ta` does not give, each of which corresponds to a way the original analysis
produced a number that looked right.

**Warm-up is NaN, always.** `ta` accepts `fillna=False` and mostly honours it - SMA, EMA,
RSI, MACD and Bollinger all return NaN until their window fills. `AverageTrueRange` does
not: it returns **0.0** for the first `window - 1` rows. Measured on a 14-bar ATR, rows 0
through 12 come back as exactly zero. A zero is not "no data"; it is *nil volatility*, a
perfectly plausible reading that no `dropna()` will remove, that makes any normalisation
dividing by ATR explode, and that a tree can learn as a genuine low-volatility regime.
The wrapper restores those NaN. That is the single most load-bearing line here.

**Indicators are computed on each symbol's native grid, before any reindexing.** The
other order is the defect: on a row where an auxiliary symbol is stale, its price is a
copy of the previous bar, so its return is a manufactured zero, its RSI drifts to 50 and
its ATR contracts. VIX is carried on 8.0% of rows and DXY on 7.1% (ADR-003 sec. 2) - and
because staleness correlates almost perfectly with the hour of day, a tree fed those
columns learns a session clock and reports it as a macro signal.

**`ta` is quarantined here.** This is the only module permitted to import it or to carry
a `cast` around it, and a test enforces that. A dependency with no type information that
leaks across a codebase takes the type checker's guarantees with it.

One property is accepted rather than fixed, and it is written down because it will look
like a bug later: **`ta` windows are positional, not temporal.** With 1,013 gaps in the
target series, an `RSI_14` on a Monday bar uses thirteen bars from Friday, and `SMA_200`
can span nine calendar days. There is no option to change it. The alternative -
reindexing onto a complete grid before computing - would reintroduce exactly the defect
the previous paragraph describes, so the positional window is the lesser evil and is
recorded as an invariant of the study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, roc
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from forecast_lab.contracts import ForecastLabError

#: The windows the original project used. Kept unchanged on purpose: altering them would
#: count as a fresh trial in the deflated-Sharpe accounting, and the point of this work is
#: to isolate the effect of the corrections rather than of a hyperparameter sweep.
SMA_WINDOWS = (20, 50, 200)
EMA_WINDOWS = (12, 26)
RSI_WINDOW = 14
ROC_WINDOW = 12
ADX_WINDOW = 14
ATR_WINDOW = 14
BOLLINGER_WINDOW = 20
#: MACD's slow EMA is 26 bars and its signal line smooths that over another 9.
MACD_WINDOW = 26 + 9

#: Windows for realised volatility, in hourly bars: one day and seven days.
#:
#: Chosen from measurement rather than taste. The probe that dropped VIX compared it
#: against annualised rolling realised volatility of the S&P at four windows, and the
#: correlation with VIX's *level* was +0.645 at 24h, +0.744 at 5d, **+0.754 at 7d** and
#: +0.755 at 21d - flat past a week, so 7 days buys the tracking at the shortest warm-up
#: (STATUS 2026-08-18). The 24-hour window is kept beside it because a session-scale
#: measure answers a different question from a week-scale one.
REALISED_VOL_WINDOWS = (24, 24 * 7)

#: Hourly bars in a year, for annualising. The venue trades around the clock on
#: weekdays, so this is 24 x 365 rather than a 252-day equity convention - the series
#: being measured has weekend gaps, not weekend zeros.
BARS_PER_YEAR = 24 * 365

#: The longest window any indicator uses. Everything before this many bars is warm-up.
LONGEST_WINDOW = max(
    *SMA_WINDOWS,
    *EMA_WINDOWS,
    RSI_WINDOW,
    ROC_WINDOW,
    ADX_WINDOW,
    ATR_WINDOW,
    BOLLINGER_WINDOW,
    MACD_WINDOW,
    *REALISED_VOL_WINDOWS,
)

_REQUIRED = ("open", "high", "low", "close")


class FeatureError(ForecastLabError):
    """Indicators cannot be computed from what was provided."""


def indicators(bars: pd.DataFrame, *, prefix: str = "") -> pd.DataFrame:
    """Compute the indicator set for one symbol, on that symbol's own grid.

    ``bars`` must carry `open`, `high`, `low` and `close` and be sorted. The result is
    indexed exactly like ``bars``, with warm-up rows as NaN rather than as a value.

    Call this **before** aligning onto a target timeline, never after: on a stale row the
    price is a copy, so a return computed there is a zero the market never printed.
    """
    missing = [c for c in _REQUIRED if c not in bars.columns]
    if missing:
        raise FeatureError(f"{prefix or 'series'}: missing column(s) {', '.join(missing)}")
    index = pd.DatetimeIndex(bars.index)
    if not index.is_monotonic_increasing:
        raise FeatureError(f"{prefix or 'series'}: bars must be sorted before indicators")

    close, high, low = bars["close"], bars["high"], bars["low"]
    columns: dict[str, pd.Series] = {}

    # --- scale-free by construction ---------------------------------------------------
    # A return, a ratio and an oscillator survive gold going from 1,616 to 4,378. A price
    # level does not: the test block would sit outside the support of the training data,
    # which is why the original pipeline's PCA collapsed 63 features into 6 components.
    columns["return"] = close.pct_change()
    columns["log_return"] = _log_return(close)
    columns["range_pct"] = (high - low) / close

    # Realised volatility: the standard deviation of log returns over a trailing window,
    # annualised. This is what replaced VIX (ADR-002 sec. 4) - VIX's hourly history
    # starts in 2022-10 and keeping it would have cost 55% of the sample for a feature
    # correlating -0.011 with the target, while this tracks the *level* of the same fear
    # regime at +0.754 and exists across the whole window.
    #
    # `min_periods` equals the window, so a partial window reports nothing rather than a
    # confident number computed from three bars. Rolling, never expanding: an expanding
    # window at bar t has seen a different amount of history than at bar t+1, which makes
    # the column's meaning drift across the sample.
    log_returns = columns["log_return"]
    for window in REALISED_VOL_WINDOWS:
        rolling = log_returns.rolling(window=window, min_periods=window).std()
        columns[f"realised_vol_{window}"] = rolling * np.sqrt(BARS_PER_YEAR)

    for window in SMA_WINDOWS:
        sma = _warm(SMAIndicator(close, window=window, fillna=False).sma_indicator(), window)
        # The distance to the average, not the average. Same information, no price level.
        columns[f"dist_sma_{window}"] = close / sma - 1.0
    for window in EMA_WINDOWS:
        ema = _warm(EMAIndicator(close, window=window, fillna=False).ema_indicator(), window)
        columns[f"dist_ema_{window}"] = close / ema - 1.0

    columns[f"rsi_{RSI_WINDOW}"] = _warm(
        RSIIndicator(close, window=RSI_WINDOW, fillna=False).rsi(), RSI_WINDOW
    )
    # Rate of change: already a percentage, so it needs no normalisation.
    columns[f"roc_{ROC_WINDOW}"] = _warm(roc(close, window=ROC_WINDOW, fillna=False), ROC_WINDOW)
    # ADX measures trend *strength* on a 0-100 scale, independent of direction and of
    # the price level. Its warm-up is roughly twice the window, because it smooths a
    # smoothed quantity - and like ATR it is a Wilder recursion, so the wrapper's blank
    # matters here too.
    columns[f"adx_{ADX_WINDOW}"] = _warm(
        ADXIndicator(high, low, close, window=ADX_WINDOW, fillna=False).adx(), 2 * ADX_WINDOW
    )

    macd = MACD(close, fillna=False)
    # MACD is a difference of two moving averages, so it carries the price's units and
    # scales with it. Dividing by close makes it comparable across the sample and across
    # symbols; the raw form never enters the matrix.
    for name, series in (
        ("macd", macd.macd()),
        ("macd_signal", macd.macd_signal()),
        ("macd_diff", macd.macd_diff()),
    ):
        columns[name] = _warm(series, MACD_WINDOW) / close

    columns["atr_pct"] = (
        _warm(
            AverageTrueRange(high, low, close, window=ATR_WINDOW, fillna=False)
            .average_true_range(),
            ATR_WINDOW,
        )
        / close
    )

    bands = BollingerBands(close, window=BOLLINGER_WINDOW, fillna=False)
    # `bollinger_wband` is already normalised by the moving average - verified, not
    # assumed: it computes (high - low) / mid * 100, and multiplying the input price by
    # ten leaves it unchanged (ratio 1.0000). `bollinger_pband` is a position within the
    # band, which is dimensionless by construction.
    columns["bb_wband"] = _warm(bands.bollinger_wband(), BOLLINGER_WINDOW)
    columns["bb_pband"] = _warm(bands.bollinger_pband(), BOLLINGER_WINDOW)

    frame = pd.DataFrame(columns, index=index)
    if prefix:
        frame.columns = pd.Index([f"{prefix}_{name}" for name in frame.columns])
    return frame


def _warm(series: pd.Series, window: int) -> pd.Series:
    """Blank the first ``window - 1`` rows, whatever the library put there.

    Idempotent where `ta` already returns NaN, and load-bearing where it does not:
    `AverageTrueRange` reports 0.0 across its whole warm-up even with `fillna=False`.
    A zero survives `dropna`, divides badly, and reads as a real low-volatility regime.
    """
    out = series.copy()
    out.iloc[: max(window - 1, 0)] = float("nan")
    return out


def _log_return(close: pd.Series) -> pd.Series:
    """Log return without importing numpy's error handling into the feature matrix.

    A non-positive price is not a small number, it is a broken row - and it would come
    out of `log` as `-inf`, which propagates silently through every downstream mean.
    """
    ratio = close / close.shift(1)
    invalid = ratio <= 0
    if bool(invalid.any()):
        raise FeatureError(
            f"{int(invalid.sum())} bar(s) imply a non-positive price ratio; "
            "a log return cannot be defined there"
        )
    return pd.Series(np.log(ratio.to_numpy(dtype="float64")), index=close.index)
