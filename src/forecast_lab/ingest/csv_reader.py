"""Reading a price series off disk, and refusing the ones that would poison a model.

This is the boundary. Everything downstream trusts that a frame handed to it sits on a
regular grid, is strictly ordered in time, and carries timezone-aware UTC timestamps -
so this module is where those properties are *established*, loudly, rather than hoped
for.

The load-bearing invariant is that **a timestamp is the bar's open** (ADR-002 sec. 6).
Alignment later forward-fills auxiliary symbols onto the target's index, and that is
only free of look-ahead because a bar opening at or before `t` closes at or before
`t + one interval` - exactly when the decision at `t` is taken. A series delivered on a
shifted or irregular grid would break that silently, in every row, with no symptom
except results that are slightly too good.

What is rejected here is only what is **unambiguously** broken: a missing price column,
an unparseable timestamp, a duplicated one. Never repaired - repairing this far upstream
means guessing what the venue meant, and the guess is invisible by the time it becomes a
number in a report.

Grid shape is *described*, not judged. An earlier version rejected any series with more
than two grid offsets; running it against real exports showed that a US index quoted by a
European venue legitimately has three, because the two daylight-saving calendars do not
switch on the same weekend. The threshold was measuring the calendar, not a defect.
Whether a grid property is harmful depends on the target series and the horizon, and
only alignment knows those - so the reader records the facts and lets alignment decide.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Final

import pandas as pd

from forecast_lab.contracts import GridError, SeriesMeta, SymbolSpec, Timeframe

#: The columns every series must have. `volume` and `spread` are optional: the venue
#: provides them, the reference exports do not.
REQUIRED: Final = ("open", "high", "low", "close")
OPTIONAL: Final = ("volume", "spread")

#: Above this many distinct grid offsets, a series is worth a second look. It is a
#: reporting threshold, not a rejection one: one offset is a clean feed, two a venue
#: whose day shifts with daylight saving, three a US instrument quoted on a European
#: calendar. Hundreds would be a broken grid - but that is a judgement alignment makes.
NOTEWORTHY_ANCHORS: Final = 2


def read_series(
    path: Path, symbol: SymbolSpec, timeframe: Timeframe
) -> tuple[pd.DataFrame, SeriesMeta]:
    """Read one series file, validate it, and describe it.

    Returns the bars indexed by timezone-aware UTC timestamp, plus the metadata a
    caller needs to reason about the series without reading it again.

    Raises ``GridError`` only for what cannot be interpreted at all: a missing price
    column, an unparseable timestamp, a duplicated one.
    """
    frame = _load(path)
    frame.index = _to_utc_index(frame, path)
    frame = frame.drop(columns=["time"]).sort_index()

    # Re-wrap after sorting: pandas types a frame's index as Index[Any], and every
    # check below is about it being a DatetimeIndex specifically.
    index = pd.DatetimeIndex(frame.index)
    _reject_duplicates(index, path)

    meta = SeriesMeta(
        symbol=symbol,
        timeframe=timeframe,
        rows=len(frame),
        first=index[0].to_pydatetime() if len(index) else None,
        last=index[-1].to_pydatetime() if len(index) else None,
        anchors=_anchors(index, timeframe),
        short_gaps=_short_gaps(index, timeframe),
    )
    return frame, meta


def _load(path: Path) -> pd.DataFrame:
    """Read the raw table and check that the columns we need are present."""
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    if "time" not in frame.columns:
        raise GridError(f"{path.name}: no 'time' column; found {list(frame.columns)}")
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise GridError(f"{path.name}: missing price column(s) {missing}")

    keep = ["time", *REQUIRED, *(c for c in OPTIONAL if c in frame.columns)]
    return frame[keep]


def _to_utc_index(frame: pd.DataFrame, path: Path) -> pd.DatetimeIndex:
    """Turn the time column into a timezone-aware UTC index.

    Two encodings are accepted because both occur in the wild: an integer UNIX epoch in
    seconds, and an ISO-8601 string. A naive string is *assumed* UTC rather than local
    time - the only safe assumption, since local time depends on the machine that reads
    the file, which would make the same file mean different things on two computers.
    """
    raw = frame["time"]
    try:
        if pd.api.types.is_numeric_dtype(raw):
            index = pd.to_datetime(raw, unit="s", utc=True)
        else:
            index = pd.to_datetime(raw, utc=True, format="mixed")
    except (ValueError, TypeError) as exc:
        # pandas raises rather than yielding NaT, so the failure has to be caught to
        # be re-stated in this project's own vocabulary.
        raise GridError(f"{path.name}: a timestamp could not be parsed ({exc})") from exc

    if index.isna().any():
        bad = int(index.isna().sum())
        raise GridError(f"{path.name}: {bad} timestamp(s) could not be parsed")
    return pd.DatetimeIndex(index)


def _reject_duplicates(index: pd.DatetimeIndex, path: Path) -> None:
    """A repeated timestamp is ambiguous, and `reindex` refuses it outright.

    Which of the two bars is the truth? There is no answer the reader can defend, so it
    does not pick one.
    """
    duplicated = index.duplicated()
    if bool(duplicated.any()):
        n = int(duplicated.sum())
        raise GridError(
            f"{path.name}: {n} duplicated timestamp(s), first at {index[duplicated][0]}"
        )


def _anchors(index: pd.DatetimeIndex, timeframe: Timeframe) -> frozenset[int]:
    """The distinct offsets at which bars sit within their interval.

    A clean hourly feed gives ``{0}``. A venue whose day starts at 22:00 UTC and shifts
    with daylight saving gives two. A US index quoted on a European calendar gives three,
    because the two calendars switch on different weekends - measured, not assumed.
    """
    if len(index) == 0:
        return frozenset()
    seconds = timeframe.seconds
    return frozenset(int(ts.timestamp()) % seconds for ts in index)


def _short_gaps(index: pd.DatetimeIndex, timeframe: Timeframe) -> int:
    """Consecutive bars closer together than one nominal interval.

    On intraday series this should be zero and anything else is suspicious. On daily
    series it counts short sessions - a daylight-saving Monday is 23 hours long - so it
    is information rather than an error. The distinction is why this is counted and
    reported instead of raised.
    """
    if len(index) < 2:
        return 0
    seconds = timeframe.seconds
    stamps = [int(ts.timestamp()) for ts in index]
    return sum(1 for a, b in pairwise(stamps) if b - a < seconds)
