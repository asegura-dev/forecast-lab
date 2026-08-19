"""The vocabulary of the domain: validated data, and the errors it can raise.

This package imports nothing else of ours (ADR-001 sec. 1). Everything above it may
depend on these names; they depend on nothing, which is what keeps the dependency
arrow pointing one way.

Only what a live caller needs is defined here. ``RunConfig`` and the dataset
descriptors arrive with the slices that use them, not in anticipation.
"""

from forecast_lab.contracts.errors import (
    ForecastLabError,
    GridError,
    InvalidSymbolError,
    UnknownTimeframeError,
)
from forecast_lab.contracts.series import SeriesMeta, SymbolSpec, Timeframe

__all__ = [
    "ForecastLabError",
    "GridError",
    "InvalidSymbolError",
    "SeriesMeta",
    "SymbolSpec",
    "Timeframe",
    "UnknownTimeframeError",
]
