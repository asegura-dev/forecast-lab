"""Discovering which series exist on disk, from their file names alone.

This module deliberately does **not** open the files. Knowing what is available is a
different question from knowing what is inside, it is a hundred times cheaper, and
keeping them apart means ``forecast-lab symbols`` stays instant on a directory of any
size. Row counts and time spans arrive with the reader, in the next slice.

The file-name pattern is a **configurable regular expression with a default**, not a
positional glob. A glob that splits on underscores and takes the second-to-last field
looks general and is not: ``BINANCE_BTC_USDT_1H.csv`` would yield a symbol of
``USDT``. A named-group regex says exactly what it means, and a venue with a different
convention supplies its own instead of forcing a rename.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from forecast_lab.contracts import InvalidSymbolError, SymbolSpec, Timeframe, UnknownTimeframeError

#: Matches ``XAUUSD_1H.csv`` and ``ANYPREFIX_XAUUSD_1H.csv``. Must expose the named
#: groups ``symbol`` and ``timeframe``; everything else in the name is ignored.
DEFAULT_PATTERN = re.compile(
    r"^(?:[A-Za-z0-9.-]+_)*?(?P<symbol>[A-Z0-9]{1,15})_(?P<timeframe>1[HD]|4H)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SeriesFile:
    """One discovered series file: what it is, and where it lives."""

    symbol: SymbolSpec
    timeframe: Timeframe
    path: Path

    @property
    def key(self) -> tuple[str, str]:
        """Stable sort and lookup key."""
        return (self.symbol.name, self.timeframe.value)


def parse_name(path: Path, pattern: re.Pattern[str] = DEFAULT_PATTERN) -> SeriesFile | None:
    """Read ``(symbol, timeframe)`` off a file name, or return ``None`` if it is not ours.

    Returning ``None`` rather than raising is the right shape here: a data directory
    legitimately contains files that are not series, and the caller reports them as
    skipped. A name that matches the pattern but carries an unusable symbol or an
    unsupported timeframe *does* raise, because that is a malformed member of the set
    rather than a non-member.
    """
    match = pattern.match(path.name)
    if match is None:
        return None
    symbol = SymbolSpec(name=match.group("symbol").upper())
    timeframe = Timeframe.parse(match.group("timeframe"))
    return SeriesFile(symbol=symbol, timeframe=timeframe, path=path)


@dataclass(frozen=True)
class CatalogReport:
    """Everything a scan found, including what it refused and why.

    Skipped files are carried rather than dropped. A run that silently ignores half a
    directory is how somebody ends up modelling a subset they did not choose - so the
    CLI prints what it ignored, and that is only possible if the scan reports it.
    """

    series: tuple[SeriesFile, ...]
    skipped: tuple[tuple[Path, str], ...]

    @property
    def symbols(self) -> tuple[SymbolSpec, ...]:
        seen: dict[str, SymbolSpec] = {}
        for item in self.series:
            seen.setdefault(item.symbol.name, item.symbol)
        return tuple(seen[name] for name in sorted(seen))

    def for_symbol(self, symbol: str) -> tuple[SeriesFile, ...]:
        return tuple(s for s in self.series if s.symbol.name == symbol.upper())


def scan(directory: Path, pattern: re.Pattern[str] = DEFAULT_PATTERN) -> CatalogReport:
    """Discover every series file directly inside ``directory``.

    The scan is **not recursive**, on purpose. A data directory usually sits next to
    other data directories, and a recursive walk would happily ingest whatever is
    nearby - unrelated CSVs from another project, or an old copy of the same series.
    Descending has to be an explicit choice, not a default.

    Results are sorted, so the column order of anything built downstream is
    reproducible across runs and machines. Directory iteration order is not.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    found: list[SeriesFile] = []
    skipped: list[tuple[Path, str]] = []
    for path in _entries(directory):
        try:
            item = parse_name(path, pattern)
        except (InvalidSymbolError, UnknownTimeframeError) as exc:
            skipped.append((path, str(exc)))
            continue
        if item is None:
            skipped.append((path, "name does not match the series pattern"))
        else:
            found.append(item)

    return CatalogReport(
        series=tuple(sorted(found, key=lambda s: s.key)),
        skipped=tuple(sorted(skipped, key=lambda s: s[0].name)),
    )


def _entries(directory: Path) -> Iterator[Path]:
    """Files directly inside ``directory``, sorted, ignoring subdirectories."""
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if path.is_file():
            yield path
