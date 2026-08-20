"""Fetching bars from Dukascopy's public CDN.

The venue serves one OHLC series per side of the book. This module downloads both and
writes a single series carrying the **mid** prices plus the **spread of each bar**,
because the spread is not decoration - it is the transaction cost, and the whole
break-even calculation that decides whether a directional edge is worth anything rests
on it. Measured on real bars it moves between 1.27 and 1.78 bps within a single day; a
constant would hide exactly the variation that matters.

Two properties this module must preserve, both load-bearing:

- **The timestamp is the bar's open** (ADR-002 sec. 6), and the venue's hourly grid sits
  exactly on the hour. That is what makes forward-fill alignment free of look-ahead.
- **A failure is per symbol, not per run.** Downloading eleven symbols across eight years
  is slow enough that losing all of it to one bad response would be its own defect.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import pandas as pd

from forecast_lab.contracts import ForecastLabError, SymbolSpec, Timeframe

#: Our symbol -> the venue's instrument identifier. Names are what this project calls
#: things; the right-hand side is the venue's vocabulary and is not used anywhere else.
INSTRUMENTS: Final[dict[str, str]] = {
    "XAUUSD": "INSTRUMENT_FX_METALS_XAU_USD",
    "XAGUSD": "INSTRUMENT_FX_METALS_XAG_USD",
    "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    "GBPUSD": "INSTRUMENT_FX_MAJORS_GBP_USD",
    "USDJPY": "INSTRUMENT_FX_MAJORS_USD_JPY",
    "SPX": "INSTRUMENT_IDX_AMERICA_E_SANDP_500",
    "NDX": "INSTRUMENT_IDX_AMERICA_E_NQ_100",
    "DXY": "INSTRUMENT_IDX_AMERICA_DOLLAR_IDX_USD",
    "WTIUSD": "INSTRUMENT_CMD_ENERGY_E_LIGHT",
    "BTCUSD": "INSTRUMENT_VCCY_BTC_USD",
    "ETHUSD": "INSTRUMENT_VCCY_ETH_USD",
    # VIX exists at the venue but its hourly history starts only in 2022-10, which would
    # cost 55% of the sample for a feature measuring -0.011 against the target
    # (ADR-002 sec. 3). Kept here so it can be fetched deliberately, out of the default.
    "VIX": "INSTRUMENT_IDX_AMERICA_VOL_IDX_USD",
}

#: The modelling set: everything except VIX (ADR-002 sec. 3).
DEFAULT_SYMBOLS: Final = tuple(s for s in INSTRUMENTS if s != "VIX")

#: The window the symbol set allows (ADR-002 sec. 3).
DEFAULT_START: Final = datetime(2018, 1, 1, tzinfo=UTC)

#: One request per calendar year. The venue caps a single response, and a year of hourly
#: bars sits comfortably under it while keeping the number of requests small.
_CHUNK_DAYS: Final = 365

#: Decimals kept when writing. Far beyond any instrument's quoted precision (five
#: for FX, three for metals), so it only removes floating-point noise.
_WRITE_PRECISION: Final = 10

#: A pause between requests. This is a free public CDN; hammering it is both rude and
#: the fastest way to get a project blocked.
_PAUSE_SECONDS: Final = 0.3


class FetchError(ForecastLabError):
    """The venue could not be reached, or returned nothing usable."""


@dataclass(frozen=True)
class Fetched:
    """One downloaded series, ready to be written."""

    symbol: SymbolSpec
    timeframe: Timeframe
    frame: pd.DataFrame  # open/high/low/close (mid), volume, spread


def fetch_series(
    symbol: SymbolSpec,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    *,
    pause: float = _PAUSE_SECONDS,
) -> Fetched:
    """Download one series, both sides, and combine them into mid prices and a spread.

    Raises ``FetchError`` when the symbol is unknown to the venue or the window comes
    back empty - never returns a half-built frame, because a silently short series is
    indistinguishable downstream from a market that was closed.
    """
    instrument = _instrument_for(symbol)
    bid = _download(instrument, timeframe, start, end, side="bid", pause=pause)
    ask = _download(instrument, timeframe, start, end, side="ask", pause=pause)

    if bid.empty or ask.empty:
        raise FetchError(
            f"{symbol}: the venue returned no bars between "
            f"{start:%Y-%m-%d} and {end:%Y-%m-%d}"
        )

    # Only bars present on both sides can produce a mid or a spread. In practice the two
    # indexes agree; an inner join makes that an assertion rather than a hope.
    common = bid.index.intersection(ask.index)
    bid, ask = bid.loc[common], ask.loc[common]

    prices = ["open", "high", "low", "close"]
    frame = (bid[prices] + ask[prices]) / 2.0
    # Volume is reported per side and the two differ slightly; the mean keeps the
    # treatment symmetric with the prices rather than arbitrarily preferring one book.
    frame["volume"] = (bid["volume"] + ask["volume"]) / 2.0
    # The spread at the close is the one that matters: it is the cost of acting on the
    # decision this bar produces.
    frame["spread"] = ask["close"] - bid["close"]

    return Fetched(symbol=symbol, timeframe=timeframe, frame=frame.sort_index())


def write_series(fetched: Fetched, destination: Path) -> Path:
    """Write a fetched series as CSV, in the schema the reader expects."""
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{fetched.symbol.name}_{fetched.timeframe.value}.csv"

    # Averaging two finite decimals leaves binary noise in the thirteenth place, which
    # would be written out verbatim as 4083.1099999999997. Rounding well below any
    # instrument's quoted precision removes the artefact without touching a real digit.
    out = fetched.frame.round(_WRITE_PRECISION)
    # Seconds since the epoch, matching the reference exports, so one reader handles
    # both and neither format becomes a special case.
    out.insert(0, "time", [int(ts.timestamp()) for ts in pd.DatetimeIndex(out.index)])
    out.to_csv(path, index=False)
    return path


def _instrument_for(symbol: SymbolSpec) -> str:
    name = INSTRUMENTS.get(symbol.name)
    if name is None:
        known = ", ".join(sorted(INSTRUMENTS))
        raise FetchError(f"{symbol} is not mapped to a venue instrument; known: {known}")

    from dukascopy_python import instruments

    value = getattr(instruments, name, None)
    if value is None:
        raise FetchError(f"{symbol}: the venue no longer publishes {name}")
    return str(value)


def _download(
    instrument: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    *,
    side: str,
    pause: float,
) -> pd.DataFrame:
    """Fetch one side, a year at a time, and concatenate.

    Chunking is not an optimisation: a single request for eight years of hourly bars
    exceeds what the venue returns, and the truncation is silent.
    """
    import dukascopy_python as dk

    interval = {
        Timeframe.H1: dk.INTERVAL_HOUR_1,
        Timeframe.H4: dk.INTERVAL_HOUR_4,
        Timeframe.D1: dk.INTERVAL_DAY_1,
    }[timeframe]
    offer_side = dk.OFFER_SIDE_BID if side == "bid" else dk.OFFER_SIDE_ASK

    parts: list[pd.DataFrame] = []
    for window_start, window_end in _windows(start, end):
        try:
            part = dk.fetch(instrument, interval, offer_side, window_start, window_end)
        except Exception as exc:
            raise FetchError(f"{instrument} {side} {window_start:%Y-%m-%d}: {exc}") from exc
        if part is not None and not part.empty:
            parts.append(_normalise(part))
        time.sleep(pause)

    if not parts:
        return pd.DataFrame()
    joined = pd.concat(parts)
    return joined[~joined.index.duplicated(keep="first")].sort_index()


def _windows(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    step = timedelta(days=_CHUNK_DAYS)
    cursor = start
    while cursor < end:
        yield cursor, min(cursor + step, end)
        cursor += step


def _normalise(frame: Any) -> pd.DataFrame:
    """Lower-case the columns and guarantee a UTC index, whatever the client returned."""
    out = pd.DataFrame(frame).copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    index = pd.DatetimeIndex(out.index)
    out.index = index.tz_localize(UTC) if index.tz is None else index.tz_convert(UTC)
    return out
