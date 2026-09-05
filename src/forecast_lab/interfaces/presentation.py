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


def unavailable_note(entries: Sequence[Mapping[str, Any]]) -> str:
    """Name the estimators that did not load, and why.

    Lives here rather than inline in the page because the inline version was wrong for as
    long as it existed: it joined the entries as if they were strings, and each is a
    `{"name", "reason"}` mapping. Nothing caught it. The branch is dead whenever all six
    estimators import - which is every machine this was developed on - and it fires on
    exactly the one the RUNBOOK documents, where Smart App Control blocks an unsigned
    native DLL. The reader most in need of the message was the one who got a stack trace.

    `runner.run()` returns `Any`, so the type checker could not see it either. Moving the
    expression into this module puts it where a unit test can reach it without a browser.
    """
    return ", ".join(
        f"{entry['name']} ({entry['reason']})" if entry.get("reason") else str(entry["name"])
        for entry in entries
    )


def markdown_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, str] | None = None
) -> str:
    """Render rows as a Markdown table, returning the string rather than drawing it.

    The fallback for when `st.dataframe` cannot be used, and a decent thing in its own
    right: a Markdown table pastes into a document, and the largest result set here is
    eighteen rows.

    **A correction worth keeping.** An earlier version of this docstring said `pyarrow`
    "does not clear the way a `.pyd` does - measured across five retries". That was true of
    five retries and false of two days: the library imports now. The mechanism is binary
    *reputation*, which accrues with a release's age, and this project had already recorded
    it for LightGBM and XGBoost. So Markdown is no longer forced - it is the fallback, and
    `dashboard.table` picks whichever the machine can actually render.

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


# --- chart specifications ----------------------------------------------------------------

#: The project's palette, kept in one place so a chart and a table cannot disagree about
#: what "this survived" looks like. Restrained on purpose: two accents against a neutral,
#: because a page about a null result should not look like a dashboard selling one.
INK = "#2b3a4a"
MUTED = "#8fa3b8"
POSITIVE = "#2e7d5b"
NEGATIVE = "#b3453a"
RULE = "#c2452f"


def _layered(*views: dict[str, Any]) -> dict[str, Any]:
    """Wrap views in a `layer`, which is what keeps a chart off the Arrow path.

    Streamlit marshals a spec's **top-level** `data` and `datasets` keys through pyarrow.
    A layered spec carries its data on the children instead, where the marshaller does not
    look, so the whole thing is serialised as plain JSON and drawn client-side. That matters
    on a machine whose application-control policy blocks the native library - and it costs
    nothing on one where it does not.
    """
    return {"layer": list(views), "config": {"axis": {"labelColor": INK, "titleColor": INK}}}


def _rule(value: float, axis: str) -> dict[str, Any]:
    return {
        "data": {"values": [{"at": value}]},
        "mark": {"type": "rule", "color": RULE, "strokeDash": [4, 4], "size": 1},
        "encoding": {axis: {"field": "at", "type": "quantitative"}},
    }


def bar_spec(
    rows: Sequence[Mapping[str, Any]],
    *,
    category: str,
    value: str,
    rule: float | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Horizontal bars, sorted by value, with an optional reference line.

    Horizontal because the categories here are configuration names - "HistGradientBoosting
    [pca-90]" is unreadable rotated.
    """
    views: list[dict[str, Any]] = [
        {
            "data": {"values": list(rows)},
            "mark": {"type": "bar", "cornerRadiusEnd": 2},
            "encoding": {
                "y": {"field": category, "type": "nominal", "sort": "-x", "title": None},
                "x": {"field": value, "type": "quantitative", "title": title or value},
                "color": {
                    "condition": {"test": f"datum['{value}'] >= 0", "value": POSITIVE},
                    "value": NEGATIVE,
                },
                "tooltip": [
                    {"field": category, "type": "nominal"},
                    {"field": value, "type": "quantitative", "format": ".4f"},
                ],
            },
        }
    ]
    if rule is not None:
        views.append(_rule(rule, "x"))
    return _layered(*views)


def scatter_spec(
    rows: Sequence[Mapping[str, Any]],
    *,
    x: str,
    y: str,
    label: str,
    highlight: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
) -> dict[str, Any]:
    """Two questions on two axes - the shape of this project's whole finding.

    Skill on one axis and money on the other, every configuration a point. They sit to the
    right of zero and below it: a real edge that costs more to collect than it is worth.
    No committed figure covers this, because no command produced it until the verdict did.
    """
    encoding: dict[str, Any] = {
        "x": {"field": x, "type": "quantitative", "title": x_title or x},
        "y": {"field": y, "type": "quantitative", "title": y_title or y},
        "tooltip": [
            {"field": label, "type": "nominal"},
            {"field": x, "type": "quantitative", "format": ".4f"},
            {"field": y, "type": "quantitative", "format": ".2%"},
        ],
    }
    if highlight:
        encoding["color"] = {
            "condition": {"test": f"datum['{highlight}']", "value": POSITIVE},
            "value": MUTED,
        }
    return _layered(
        {
            "data": {"values": list(rows)},
            "mark": {"type": "point", "filled": True, "size": 90, "opacity": 0.85},
            "encoding": encoding,
        },
        _rule(0.0, "x"),
        _rule(0.0, "y"),
    )


def lines_spec(
    rows: Sequence[Mapping[str, Any]],
    *,
    x: str,
    y: str,
    series: str,
    rule: float | None = None,
    y_title: str | None = None,
) -> dict[str, Any]:
    """One line per series - how far an answer moves between folds.

    The spread across folds is a description, never an interval: the folds share training
    data, so they are not independent experiments and their variation is not a sampling
    distribution.
    """
    views: list[dict[str, Any]] = [
        {
            "data": {"values": list(rows)},
            "mark": {"type": "line", "point": {"size": 40}, "strokeWidth": 1.5},
            "encoding": {
                "x": {"field": x, "type": "ordinal", "title": x},
                "y": {
                    "field": y,
                    "type": "quantitative",
                    "title": y_title or y,
                    "scale": {"zero": False},
                },
                "color": {"field": series, "type": "nominal", "title": None},
                "tooltip": [
                    {"field": series, "type": "nominal"},
                    {"field": x, "type": "ordinal"},
                    {"field": y, "type": "quantitative", "format": ".2%"},
                ],
            },
        }
    ]
    if rule is not None:
        views.append(_rule(rule, "y"))
    return _layered(*views)


def carries_no_top_level_data(spec: Mapping[str, Any]) -> bool:
    """Whether a spec avoids the keys Streamlit routes through Arrow.

    Exposed so a test can assert it rather than a comment claiming it. `data` and
    `datasets` at the top level are the two the marshaller pulls out; anything nested
    inside `layer` is left alone.
    """
    return "data" not in spec and "datasets" not in spec
