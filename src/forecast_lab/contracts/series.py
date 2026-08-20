"""What a price series is, and how it is identified.

Three contracts, validated once at the boundary and trusted thereafter (ADR-001 sec. 3):

- ``Timeframe`` - a bar interval, and the number of seconds it spans. The duration is
  the useful part: the grid check and every horizon calculation are arithmetic on it.
- ``SymbolSpec`` - a symbol name that is safe to use as an identifier and as a path
  component.
- ``SeriesMeta`` - what is known about one ``(symbol, timeframe)`` series without
  loading it: how many bars, over what span, and on which grid offsets it sits.

``SeriesMeta`` deliberately carries ``anchors`` as a *set*. A venue's higher-timeframe
bars are anchored to its own trading day, and that anchor moves with daylight saving
time, so a single offset cannot describe a real series (ADR-002 sec. 5).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from forecast_lab.contracts.errors import InvalidSymbolError, UnknownTimeframeError

# Symbols become file and directory names. Windows is case-insensitive and rejects a
# handful of names outright, so anything outside this shape is refused at the boundary
# rather than found missing later.
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,15}$")


class Timeframe(StrEnum):
    """A bar interval. The value is the canonical spelling used in file names."""

    H1 = "1H"
    H4 = "4H"
    D1 = "1D"

    @property
    def seconds(self) -> int:
        """Span of one bar, in seconds."""
        return {Timeframe.H1: 3600, Timeframe.H4: 14400, Timeframe.D1: 86400}[self]

    @classmethod
    def parse(cls, raw: str) -> Timeframe:
        """Parse a timeframe from user input, case-insensitively.

        Raises ``UnknownTimeframeError`` rather than ``ValueError`` so a caller can
        catch every contract violation of this project through one base class.
        """
        try:
            return cls(raw.strip().upper())
        except ValueError as exc:
            supported = ", ".join(t.value for t in cls)
            raise UnknownTimeframeError(
                f"unknown timeframe {raw!r}; supported: {supported}"
            ) from exc


class SymbolSpec(BaseModel):
    """A market symbol, usable as an identifier and as a path component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str

    @field_validator("name", mode="after")
    @classmethod
    def _check_shape(cls, value: str) -> str:
        if not _SYMBOL_RE.match(value):
            raise InvalidSymbolError(
                f"symbol {value!r} must be 1-15 characters of A-Z and 0-9; "
                "symbols are used as file names, so punctuation and case are not portable"
            )
        return value

    def __str__(self) -> str:
        return self.name


class SeriesMeta(BaseModel):
    """What is known about one series without loading its bars.

    Two fields describe the grid rather than judging it, because measurement showed a
    threshold could not tell a defect from a calendar (ADR-002 sec. 6):

    - ``anchors`` - every distinct ``timestamp % timeframe.seconds`` observed. A clean
      hourly feed yields ``{0}``. A US index quoted by a European venue yields three,
      because the two daylight-saving calendars do not switch on the same weekend.
    - ``short_gaps`` - consecutive bars closer together than one nominal interval. On a
      daily series this counts short sessions, not overlaps: a "day" is a session, not
      24 hours.

    Neither is an error here. Whether they matter depends on the target and the horizon,
    which only alignment knows.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: SymbolSpec
    timeframe: Timeframe
    rows: int = Field(ge=0)
    first: datetime | None = None
    last: datetime | None = None
    anchors: frozenset[int] = frozenset()
    short_gaps: int = Field(default=0, ge=0)

    @field_validator("first", "last", mode="after")
    @classmethod
    def _require_utc(cls, value: datetime | None) -> datetime | None:
        """Timestamps are timezone-aware UTC everywhere, with no exceptions.

        A naive datetime is the single most common way a timezone bug enters a
        pipeline, and it is invisible until the results are wrong.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware; got a naive datetime")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if (self.first is None) != (self.last is None):
            raise ValueError("first and last must both be set, or both be absent")
        if self.first is not None and self.last is not None and self.first > self.last:
            raise ValueError(f"first ({self.first}) is after last ({self.last})")
        if self.rows == 0 and self.first is not None:
            raise ValueError("an empty series cannot have a time span")
        for anchor in self.anchors:
            if not 0 <= anchor < self.timeframe.seconds:
                raise ValueError(
                    f"anchor {anchor} is outside one {self.timeframe} bar "
                    f"({self.timeframe.seconds} s)"
                )
        return self

    @property
    def is_empty(self) -> bool:
        return self.rows == 0
