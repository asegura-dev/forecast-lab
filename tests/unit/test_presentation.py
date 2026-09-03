"""Tests for formatting, which is the point of `presentation` existing at all.

The first version of the dashboard formatted inside `st.markdown` calls, so none of this
could be checked without a browser - and `test_the_significance_of_a_p_value_survives`
pins the defect that hid there: four p-values spanning an order of magnitude all printed
as `0.0000`, in the one table whose subject is statistical significance.

Every test here runs with no Streamlit, no subprocess and no data.
"""

from __future__ import annotations

from typing import Any

import pytest

from forecast_lab.interfaces.presentation import (
    NEGATIVE,
    POSITIVE,
    SCIENTIFIC_BELOW,
    Style,
    accuracy,
    bar_spec,
    carries_no_top_level_data,
    cell,
    lines_spec,
    markdown_table,
    percent,
    rows_from,
    scatter_spec,
    shell_path,
    significance,
)

# --- the defect this module exists because of --------------------------------------------


@pytest.mark.unit
def test_the_significance_of_a_p_value_survives() -> None:
    """The real p-values from a verdict run, which `{:.4f}` made indistinguishable.

    All four are `0.0000` under a fixed four-decimal format. A reader comparing
    configurations in the significance table could not tell them apart at all.
    """
    measured = [2.0001871680506356e-06, 1.7018e-05, 2.34e-05, 3.5198e-05]

    rendered = [significance(value) for value in measured]

    assert len(set(rendered)) == 4, f"p-values collapsed to {set(rendered)}"
    assert all("e-0" in text for text in rendered)
    # And the old behaviour is what it is being compared against.
    assert len({f"{value:,.4f}" for value in measured}) == 1


@pytest.mark.unit
def test_a_deflated_sharpe_of_one_in_ten_billion_is_not_zero() -> None:
    """`1.56e-10` printed as `0.0000` in the caption beside the table."""
    assert significance(1.5557694840213364e-10) == "1.56e-10"


@pytest.mark.unit
def test_ordinary_probabilities_stay_readable() -> None:
    """Scientific notation everywhere would be worse, not better."""
    assert significance(0.763) == "0.76300"
    assert significance(0.05439) == "0.05439"
    assert significance(0.0) == "0"


@pytest.mark.unit
def test_the_switch_to_scientific_happens_at_the_stated_threshold() -> None:
    """Asserted so the boundary cannot drift silently."""
    assert "e-" not in significance(SCIENTIFIC_BELOW * 1.1)
    assert "e-" in significance(SCIENTIFIC_BELOW * 0.9)


# --- a format follows from what a number means ---------------------------------------------


@pytest.mark.unit
def test_a_difference_carries_its_sign_and_a_level_does_not() -> None:
    """`-0.46%` read as a gain when the sign was dropped; an accuracy needs no `+`."""
    assert percent(-0.0046) == "-0.46%"
    assert percent(0.0097) == "+0.97%"
    assert accuracy(0.5123) == "51.23%"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "style", "expected"),
    [
        (0.5123, Style.PERCENT, "+51.23%"),
        (4.6114, Style.RATIO, "4.6114"),
        (40587, Style.COUNT, "40,587"),
        (5.6818, Style.BARS, "5.7"),
        (1.86, Style.BPS, "+1.86 bps"),
        (2e-06, Style.SIGNIFICANCE, "2.00e-06"),
        (True, Style.TEXT, "yes"),
        (False, Style.TEXT, "no"),
        ("Naive Bayes [raw]", Style.TEXT, "Naive Bayes [raw]"),
    ],
)
def test_each_style_formats_what_it_names(value: object, style: str, expected: str) -> None:
    assert cell(value, style) == expected


@pytest.mark.unit
def test_a_missing_value_is_a_dash_rather_than_the_word_none() -> None:
    """A gap and the string "None" look different to a reader and identical to a
    formatter that calls `str()`."""
    assert cell(None, Style.PERCENT) == "-"
    assert cell(None) == "-"


@pytest.mark.unit
def test_a_boolean_is_never_formatted_as_a_number() -> None:
    """`bool` is an `int` in Python, so a count style would render `True` as `1`."""
    assert cell(True, Style.COUNT) == "yes"
    assert cell(False, Style.PERCENT) == "no"


# --- the table -------------------------------------------------------------------------------


@pytest.mark.unit
def test_a_table_renders_a_header_a_rule_and_one_row_each() -> None:
    rows = [{"model": "a", "accuracy": 0.51}, {"model": "b", "accuracy": 0.49}]

    lines = markdown_table(rows, {"accuracy": Style.PERCENT}).splitlines()

    assert lines[0] == "| model | accuracy |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| a | +51.00% |"
    assert len(lines) == 4


@pytest.mark.unit
def test_a_column_without_a_style_still_renders() -> None:
    """A caller adding a column must not have to update a style map to see it."""
    assert "| a | 1.5000 |" in markdown_table([{"x": "a", "y": 1.5}])


@pytest.mark.unit
def test_a_missing_key_in_a_later_row_does_not_shift_the_columns() -> None:
    """Columns come from the first row; a row lacking one gets a dash, not a gap."""
    lines = markdown_table([{"a": 1, "b": 2}, {"a": 3}]).splitlines()

    assert lines[3] == "| 3 | - |"


@pytest.mark.unit
def test_an_empty_table_is_empty_rather_than_a_header_with_no_rows() -> None:
    assert markdown_table([]) == ""


@pytest.mark.unit
def test_rows_from_flattens_a_keyed_section() -> None:
    """Several payloads key results by name; every page was writing this comprehension."""
    payload = {"Naive Bayes": {"accuracy": 0.5077}, "LightGBM": {"accuracy": 0.5111}}

    rows = rows_from(payload, key_column="configuration")

    assert rows[0] == {"configuration": "Naive Bayes", "accuracy": 0.5077}
    assert [r["configuration"] for r in rows] == ["Naive Bayes", "LightGBM"]


# --- the copyable command line -----------------------------------------------------------------


@pytest.mark.unit
def test_a_windows_path_survives_being_pasted_into_a_shell() -> None:
    """ADR-005 sec. 2 promises a line a reader can copy. `data\\raw` is `dataraw` in bash."""
    assert shell_path("forecast-lab align --dir data\\raw --json") == (
        "forecast-lab align --dir data/raw --json"
    )


@pytest.mark.unit
def test_a_posix_path_is_left_alone() -> None:
    line = "forecast-lab align --dir data/raw --json"

    assert shell_path(line) == line


# --- chart specifications ------------------------------------------------------------------


#: Enough of a verdict payload to build every chart from.
CONFIGURATIONS = [
    {
        "configuration": "HistGradientBoosting [raw]",
        "skill": 0.0114,
        "money": -2.795,
        "survives": True,
    },
    {
        "configuration": "Logistic Regression [raw]",
        "skill": 0.0038,
        "money": -1.967,
        "survives": False,
    },
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec",
    [
        bar_spec(CONFIGURATIONS, category="configuration", value="skill", rule=0.0),
        scatter_spec(
            CONFIGURATIONS, x="skill", y="money", label="configuration", highlight="survives"
        ),
        lines_spec(
            [{"fold": 1, "accuracy": 0.51, "configuration": "a"}],
            x="fold",
            y="accuracy",
            series="configuration",
            rule=0.5,
        ),
    ],
)
def test_no_spec_puts_its_data_where_streamlit_would_marshal_it(spec: dict[str, Any]) -> None:
    """The property that lets these render where `pyarrow` is unavailable.

    Streamlit pulls a spec's **top-level** `data` and `datasets` through Arrow. A layered
    spec carries data on its children instead. This is asserted rather than commented,
    because the comment was true and unverifiable and the difference showed up as a blank
    page on the machine this was built on.
    """
    assert carries_no_top_level_data(spec)
    assert spec["layer"], "a layer is what moves the data off the top level"
    assert all("data" in view for view in spec["layer"])


@pytest.mark.unit
def test_a_reference_line_is_a_layer_of_its_own() -> None:
    """So the rule has its own one-row dataset rather than being encoded into the bars."""
    spec = bar_spec(CONFIGURATIONS, category="configuration", value="skill", rule=0.0)

    assert len(spec["layer"]) == 2
    assert spec["layer"][1]["mark"]["type"] == "rule"
    assert spec["layer"][1]["data"]["values"] == [{"at": 0.0}]


@pytest.mark.unit
def test_a_chart_without_a_rule_has_one_layer() -> None:
    assert len(bar_spec(CONFIGURATIONS, category="configuration", value="skill")["layer"]) == 1


@pytest.mark.unit
def test_the_scatter_rules_both_axes() -> None:
    """Skill on one axis and money on the other, with zero marked on each - which is the
    whole finding: right of one line, below the other."""
    spec = scatter_spec(CONFIGURATIONS, x="skill", y="money", label="configuration")

    axes = {next(iter(view["encoding"])) for view in spec["layer"][1:]}
    assert axes == {"x", "y"}


@pytest.mark.unit
def test_the_highlight_colours_by_a_boolean_field() -> None:
    spec = scatter_spec(
        CONFIGURATIONS, x="skill", y="money", label="configuration", highlight="survives"
    )

    condition = spec["layer"][0]["encoding"]["color"]["condition"]
    assert condition["test"] == "datum['survives']"
    assert condition["value"] == POSITIVE


@pytest.mark.unit
def test_bars_are_coloured_by_sign_rather_than_by_category() -> None:
    """A negative edge and a positive one must not look alike on a page about whether an
    edge exists."""
    encoding = bar_spec(CONFIGURATIONS, category="configuration", value="skill")["layer"][0][
        "encoding"
    ]

    assert encoding["color"]["condition"]["value"] == POSITIVE
    assert encoding["color"]["value"] == NEGATIVE


@pytest.mark.unit
def test_every_spec_carries_a_tooltip() -> None:
    """Eighteen points on a scatter are unreadable without one."""
    for spec in (
        bar_spec(CONFIGURATIONS, category="configuration", value="skill"),
        scatter_spec(CONFIGURATIONS, x="skill", y="money", label="configuration"),
        lines_spec(
            [{"fold": 1, "accuracy": 0.5, "configuration": "a"}],
            x="fold",
            y="accuracy",
            series="configuration",
        ),
    ):
        assert spec["layer"][0]["encoding"]["tooltip"]


@pytest.mark.unit
def test_a_fold_axis_is_ordinal_and_the_accuracy_axis_does_not_force_zero() -> None:
    """Five folds are categories, not a quantity; and an accuracy scale anchored at zero
    would compress every difference this project is about into nothing."""
    spec = lines_spec(
        [{"fold": 1, "accuracy": 0.51, "configuration": "a"}],
        x="fold",
        y="accuracy",
        series="configuration",
    )
    encoding = spec["layer"][0]["encoding"]

    assert encoding["x"]["type"] == "ordinal"
    assert encoding["y"]["scale"]["zero"] is False
