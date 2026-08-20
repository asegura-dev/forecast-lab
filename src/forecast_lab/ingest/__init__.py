"""Where bytes come from, and how they become series.

This layer knows about file names, directories and (later) HTTP. Nothing above it does:
`research` receives frames and never learns where they were read from (ADR-001 sec. 1).

`manifest` is re-exported as a module rather than by its functions: `manifest.read(...)`
says what it reads, while a bare `read` imported from a package does not.
"""

from forecast_lab.ingest import manifest
from forecast_lab.ingest.catalog import (
    DEFAULT_PATTERN,
    CatalogReport,
    SeriesFile,
    parse_name,
    scan,
)
from forecast_lab.ingest.csv_reader import NOTEWORTHY_ANCHORS, read_series
from forecast_lab.ingest.dukascopy import (
    DEFAULT_START,
    DEFAULT_SYMBOLS,
    INSTRUMENTS,
    Fetched,
    FetchError,
    fetch_series,
    write_series,
)
from forecast_lab.ingest.importer import ImportReport, import_directory

__all__ = [
    "DEFAULT_PATTERN",
    "DEFAULT_START",
    "DEFAULT_SYMBOLS",
    "INSTRUMENTS",
    "NOTEWORTHY_ANCHORS",
    "CatalogReport",
    "FetchError",
    "Fetched",
    "ImportReport",
    "SeriesFile",
    "fetch_series",
    "import_directory",
    "manifest",
    "parse_name",
    "read_series",
    "scan",
    "write_series",
]
