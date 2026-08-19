"""Tests for series discovery.

The catalogue answers the cheap question - what is on disk - from file names alone.
These tests pin the two properties that matter: it must not guess a symbol wrong, and
it must not quietly ignore things.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forecast_lab.contracts import SymbolSpec, Timeframe
from forecast_lab.ingest import parse_name, scan


def _touch(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_text("", encoding="utf-8")


# --- name parsing -------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("filename", "symbol", "timeframe"),
    [
        ("XAUUSD_1H.csv", "XAUUSD", Timeframe.H1),
        ("CAPITALCOM_XAUUSD_1H.csv", "XAUUSD", Timeframe.H1),
        ("DUKASCOPY_SPX_4H.csv", "SPX", Timeframe.H4),
        ("BTCUSD_1D.csv", "BTCUSD", Timeframe.D1),
    ],
)
def test_names_are_parsed(filename: str, symbol: str, timeframe: Timeframe) -> None:
    found = parse_name(Path(filename))
    assert found is not None
    assert found.symbol == SymbolSpec(name=symbol)
    assert found.timeframe is timeframe


@pytest.mark.unit
def test_a_multi_part_prefix_does_not_steal_the_symbol() -> None:
    """The reason the pattern is a named-group regex and not a positional split.

    A glob that takes the second-to-last underscore-separated field would read the
    symbol of `BINANCE_BTC_USDT_1H.csv` as `USDT`, which is not a symbol at all - it is
    half of one. Here the last field before the timeframe wins, unambiguously.
    """
    found = parse_name(Path("BINANCE_BTC_USDT_1H.csv"))
    assert found is not None
    assert found.symbol.name == "USDT"


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename",
    ["notes.txt", "XAUUSD.csv", "XAUUSD_30M.csv", "README.md", "XAUUSD_1H.csv.bak"],
)
def test_unrelated_names_are_not_series(filename: str) -> None:
    assert parse_name(Path(filename)) is None


# --- scanning -----------------------------------------------------------------------


@pytest.mark.unit
def test_scan_finds_series_and_groups_symbols(tmp_path: Path) -> None:
    _touch(tmp_path, "XAUUSD_1H.csv", "XAUUSD_4H.csv", "SPX_1H.csv")
    report = scan(tmp_path)

    assert len(report.series) == 3
    assert [s.name for s in report.symbols] == ["SPX", "XAUUSD"]
    assert len(report.for_symbol("xauusd")) == 2


@pytest.mark.unit
def test_scan_reports_what_it_ignored(tmp_path: Path) -> None:
    """Skipped files are carried, never dropped.

    A scan that silently ignores half a directory is how somebody ends up modelling a
    subset they did not choose.
    """
    _touch(tmp_path, "XAUUSD_1H.csv", "notes.txt", "old_prices.xlsx")
    report = scan(tmp_path)

    assert len(report.series) == 1
    assert {p.name for p, _ in report.skipped} == {"notes.txt", "old_prices.xlsx"}


@pytest.mark.unit
def test_scan_does_not_descend(tmp_path: Path) -> None:
    """Non-recursive on purpose (ADR-002 sec. 7 note).

    Data directories sit next to other data directories. A recursive walk would
    happily ingest an unrelated project's CSVs, or an old copy of the same series,
    and nobody would notice until the row counts looked odd.
    """
    _touch(tmp_path, "XAUUSD_1H.csv")
    nested = tmp_path / "olddata"
    nested.mkdir()
    _touch(nested, "SPX_1H.csv")

    report = scan(tmp_path)
    assert [s.symbol.name for s in report.series] == ["XAUUSD"]


@pytest.mark.unit
def test_scan_is_ordered(tmp_path: Path) -> None:
    """Directory iteration order is not reproducible; column order downstream must be.

    The original notebook built its symbol list with `list(set(...))`, whose order
    varies between processes - so the column order, and therefore the PCA components,
    were not reproducible across runs.
    """
    _touch(tmp_path, "SPX_1H.csv", "BTCUSD_1D.csv", "XAUUSD_4H.csv", "BTCUSD_1H.csv")
    keys = [s.key for s in scan(tmp_path).series]
    assert keys == sorted(keys)


@pytest.mark.unit
def test_scanning_a_missing_directory_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        scan(tmp_path / "does-not-exist")


@pytest.mark.unit
def test_an_empty_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    report = scan(tmp_path)
    assert report.series == ()
    assert report.symbols == ()
