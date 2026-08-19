"""The exception hierarchy, in `contracts` so every layer may raise and catch it.

It lives here rather than in `ingest` because `research` must be able to reject a
malformed frame without importing an outer layer (ADR-001 sec. 1). An exception type
is part of the vocabulary of the domain, not of the thing that happens to raise it.

The distinction that matters: these are *contract violations*, raised when data does
not mean what the pipeline requires it to mean. They are never caught and swallowed
inside the engine. Data that is wrong must stop the run, because a pipeline that
repairs its inputs silently produces results nobody can trace.
"""

from __future__ import annotations


class ForecastLabError(Exception):
    """Base class for every error this project raises deliberately."""


class InvalidSymbolError(ForecastLabError):
    """A symbol name is not usable as a stable identifier.

    Symbols become directory and file names, so anything a case-insensitive or
    reserved-name filesystem would mangle has to be rejected at the boundary rather
    than discovered as a missing file three steps later.
    """


class UnknownTimeframeError(ForecastLabError):
    """A timeframe string does not name a supported bar interval."""


class GridError(ForecastLabError):
    """A series does not sit on the regular grid its timeframe implies.

    The load-bearing invariant of the whole alignment design is that a timestamp is
    the bar's *open* (ADR-002 sec. 6). A series on a shifted or irregular grid would
    inject look-ahead into every aligned row, so it is rejected loudly instead of
    being aligned anyway.
    """
