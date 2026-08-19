"""Where bytes come from, and how they become series.

This layer knows about file names, directories and (later) HTTP. Nothing above it
does: `research` receives frames and never learns where they were read from
(ADR-001 sec. 1).
"""

from forecast_lab.ingest.catalog import (
    DEFAULT_PATTERN,
    CatalogReport,
    SeriesFile,
    parse_name,
    scan,
)

__all__ = [
    "DEFAULT_PATTERN",
    "CatalogReport",
    "SeriesFile",
    "parse_name",
    "scan",
]
