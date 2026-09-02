"""Tests for the dashboard's layout, which had none.

An audit found that `dashboard.py` was exercised by nothing - and that a comment in the
file claimed otherwise, saying the layering test imported it when that test only parses its
text. Every rendering defect the audit turned up lived here for exactly that reason: two
panels that fetched a result and discarded it, a headline typed into the source that the
next page contradicted, and sidebar state that vanished on the way to another page.

**Streamlit is an optional extra**, so these skip when it is absent rather than making the
default install heavier. The gates run `uv sync` without the extra; a machine that has it
gets these too.

**Deliberately not asserted here:** research numbers. Those belong to the modules that
compute them, and the payload contract lives in `test_cli.py`. These check that the page
*renders* - which is the only thing `runner` and `presentation` cannot check for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="the dashboard is an optional extra")

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src" / "forecast_lab" / "interfaces" / "dashboard.py"

#: The pages that render a document and run no command, so they are fast and hermetic.
DOCUMENT_PAGES = ("Findings", "Reasoning")


def _app(timeout: int = 120) -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=timeout)


# --- the page renders at all --------------------------------------------------------------


@pytest.mark.unit
def test_the_app_starts_without_an_exception() -> None:
    """The cheapest possible check, and the one nothing performed before.

    A syntax error, a bad import, a widget built with an argument Streamlit removed - all
    of these left every other gate green while the page was a traceback.
    """
    app = _app().run()

    assert not app.exception, [str(e.message) for e in app.exception]
    assert app.title[0].value


@pytest.mark.unit
def test_every_page_is_reachable_from_the_sidebar() -> None:
    """The radio's options are the pages; a page added to one and not the other is a page
    that exists and cannot be opened."""
    app = _app().run()

    from forecast_lab.interfaces.dashboard import PAGES

    assert tuple(app.sidebar.radio[0].options) == PAGES


@pytest.mark.unit
@pytest.mark.parametrize("page", DOCUMENT_PAGES)
def test_a_document_page_renders_without_running_a_command(page: str) -> None:
    app = _app().run()
    app.sidebar.radio[0].set_value(page).run()

    assert not app.exception, [str(e.message) for e in app.exception]
    assert not app.error
    assert app.markdown, f"{page} rendered no text"


# --- what the audit found -------------------------------------------------------------------


@pytest.mark.unit
def test_the_findings_page_renders_the_document_from_its_title() -> None:
    """An earlier version split on the first `---` and dropped the title and lead paragraph,
    then opened with a hard-coded headline that the Verdict page later contradicted - nine
    of eighteen in prose against thirteen computed."""
    app = _app().run()
    app.sidebar.radio[0].set_value("Findings").run()

    body = "\n".join(block.value for block in app.markdown)
    assert body.lstrip().startswith("# Findings")
    # And no figure is typed into the page: the only numbers come from the document.
    from forecast_lab.interfaces.dashboard import ROOT

    assert body.strip() == (ROOT / "FINDINGS.md").read_text(encoding="utf-8").strip()


@pytest.mark.unit
def test_the_reasoning_page_renders_one_document_rather_than_every_document() -> None:
    """Twenty-one expanders pushed 200 KB of Markdown through every rerun."""
    app = _app().run()
    app.sidebar.radio[0].set_value("Reasoning").run()

    assert len(app.selectbox) == 1
    assert len(app.selectbox[0].options) > 10
    assert not app.expander


@pytest.mark.unit
def test_the_sidebar_choices_survive_a_visit_to_another_page() -> None:
    """Pins the defect that made a page compute against a dataset the reader did not pick.

    The widgets were unkeyed, and `main()` returns before creating them on the document
    pages - so Streamlit garbage-collected the choice, and coming back reset it to the
    default with only the command line to reveal it.
    """
    app = _app().run()
    app.sidebar.radio[0].set_value("Exploration").run()
    if len(app.sidebar.selectbox[0].options) < 2:
        pytest.skip("only one dataset on this machine")

    chosen = app.sidebar.selectbox[0].options[1]
    app.sidebar.selectbox[0].set_value(chosen).run()
    app.sidebar.radio[0].set_value("Findings").run()
    app.sidebar.radio[0].set_value("Exploration").run()

    assert app.session_state["dataset"] == chosen


@pytest.mark.unit
def test_the_mode_control_appears_only_where_the_command_takes_it() -> None:
    """`align` and `explore` have no `--mode`; a control that visibly does nothing costs
    trust, and a reader who changes it and sees the same command line loses it."""
    from forecast_lab.interfaces.dashboard import TAKES_MODE

    app = _app().run()
    app.sidebar.radio[0].set_value("Exploration").run()
    assert "mode" not in app.session_state

    app.sidebar.radio[0].set_value("Features").run()
    assert "mode" in app.session_state
    assert "Features" in TAKES_MODE


@pytest.mark.unit
def test_every_command_shown_would_survive_a_paste_into_a_shell() -> None:
    """ADR-005 sec. 2 promises a copyable line; `data\\raw` is `dataraw` in bash."""
    app = _app(timeout=300).run()
    app.sidebar.radio[0].set_value("Data").run()

    shown = [
        block.value
        for block in app.code
        if block.value.startswith(("forecast-lab", "python"))
    ]
    assert shown, "the Data page rendered no command line"
    assert not any("\\" in line for line in shown), shown


@pytest.mark.unit
def test_the_data_page_renders_the_output_of_the_commands_it_runs() -> None:
    """Three agents found this independently: `verify` and `symbols` have no `--json`, so
    `run()` returns their rendered text - and both call sites discarded it, leaving two of
    the three panels showing a command line and nothing else."""
    app = _app(timeout=300).run()
    app.sidebar.radio[0].set_value("Data").run()

    assert not app.exception, [str(e.message) for e in app.exception]
    rendered = [block.value for block in app.code] + [block.value for block in app.success]
    assert any("Symbol" in text for text in rendered), "the symbols table is missing"
    # `verify` reports either OK or a list of CHANGED files; both are output, neither is
    # an error, and the panel must show whichever it got.
    assert any(
        text.startswith(("OK", "CHANGED")) or "manifest" in text for text in rendered
    ), "the verify report is missing"
