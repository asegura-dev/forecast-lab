"""Turning payload values into strings a reader can act on.

Pure functions, no Streamlit, no subprocess, no imports from this package. That is the
point rather than a stylistic preference: the first version of the dashboard formatted
inside `st.markdown` calls, so **the formatting could not be tested without a browser** -
and a defect hid there for exactly as long as that was true.

**The defect this module exists because of.** Every non-percentage float was rendered as
`{:,.4f}`. On the verdict table that made `2.00e-06`, `1.70e-05`, `3.52e-05` and `2.34e-05`
all read as `0.0000`: four configurations with p-values spanning an order of magnitude,
indistinguishable, in the one table whose entire subject is statistical significance. The
Deflated Sharpe of `1.56e-10` printed as `0.0000` in the caption beside it.

The fix is not a wider format. It is that **a number's format follows from what it means**,
so a caller names the meaning and never the precision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Below this, a fixed four-decimal format prints zero and loses the reader's information.
SCIENTIFIC_BELOW = 1e-4


class Style:
    """What a value *is*, from which how to print it follows.

    Strings rather than an enum because they are written as literals at every call site and
    an enum would buy nothing but imports.
    """

    PERCENT = "percent"
    """A fraction of one, shown as a signed percentage: accuracies, edges, flip rates."""

    SIGNIFICANCE = "significance"
    """A p-value or a probability that may be tiny. Switches to scientific rather than
    rounding away the difference between 2e-06 and 2e-05."""

    RATIO = "ratio"
    """A dimensionless number read at four decimals: a z-statistic, an autocorrelation."""

    COUNT = "count"
    """An integer, with thousands separators."""

    BARS = "bars"
    """A duration in bars, at one decimal - 5.7 reads better than 5.6818."""

    BPS = "bps"
    """Basis points, at two decimals."""

    TEXT = "text"
    """Anything else, rendered as-is."""


def cell(value: Any, style: str = Style.TEXT) -> str:
    """One value, formatted according to what it means.

    ``None`` renders as an em dash rather than "None", because a missing number and the
    string "None" look different to a reader and identical to a naive formatter.
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if style == Style.COUNT:
        return f"{int(value):,}"
    if isinstance(value, str):
        return value
    if style == Style.PERCENT:
        return f"{float(value):+.2%}"
    if style == Style.SIGNIFICANCE:
        return significance(float(value))
    if style == Style.RATIO:
        return f"{float(value):.4f}"
    if style == Style.BARS:
        return f"{float(value):.1f}"
    if style == Style.BPS:
        return f"{float(value):+.2f} bps"
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def significance(value: float) -> str:
    """A p-value, without rounding away the thing it measures.

    Fixed notation down to `SCIENTIFIC_BELOW`, scientific below it. The alternative -
    more decimal places - would print `0.0000020002` and be no easier to compare.
    """
    if value == 0.0:
        return "0"
    if abs(value) < SCIENTIFIC_BELOW:
        return f"{value:.2e}"
    return f"{value:.5f}"


def percent(value: float) -> str:
    """A signed percentage. Signed because most of these are differences from a baseline
    and an unsigned -0.46% would be read as a gain."""
    return f"{value:+.2%}"


def accuracy(value: float) -> str:
    """An accuracy, which is a level rather than a difference, so it carries no sign."""
    return f"{value:.2%}"


def markdown_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, str] | None = None
) -> str:
    """Render rows as a Markdown table, returning the string rather than drawing it.

    Markdown rather than `st.dataframe` is forced rather than chosen: both Streamlit table
    widgets serialise through Arrow, and `pyarrow` ships a native library that this
    machine's application-control policy blocks outright - measured across five retries, it
    does not clear the way a `.pyd` does. The consolation is real, though: a Markdown table
    pastes into a document, and the largest result set here is eighteen rows.

    Returning a string is what makes the formatting testable at all.
    """
    if not rows:
        return ""
    styles = styles or {}
    columns = list(rows[0])
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| "
        + " | ".join(cell(row.get(column), styles.get(column, Style.TEXT)) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def rows_from(
    mapping: Mapping[str, Mapping[str, Any]], *, key_column: str
) -> list[dict[str, Any]]:
    """Flatten a payload's `{name: {...}}` section into rows a table can take.

    Several payloads key their per-configuration results by name; every page that renders
    one was writing the same comprehension. One helper, no coupling to any particular
    payload shape.
    """
    return [{key_column: name, **dict(fields)} for name, fields in mapping.items()]


def shell_path(path: str) -> str:
    """A path that survives being pasted into a shell.

    `str(Path("data/raw"))` is `data\\raw` on Windows, and the command lines this project
    displays are meant to be copied - into bash, which reads that as `dataraw`. Forward
    slashes work in both shells and in the CLI's own argument parsing.
    """
    return path.replace("\\", "/")
