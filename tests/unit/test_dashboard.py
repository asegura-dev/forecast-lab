"""Tests for the dashboard's layout, which had none.

An audit found that `dashboard.py` was exercised by nothing - and that a comment in the
file claimed otherwise, saying the layering test imported it when that test only parses its
text. Every rendering defect the audit turned up lived here for exactly that reason: two
panels that fetched a result and discarded it, a headline typed into the source that the
next page contradicted, and sidebar state that vanished on the way to another page.

**Streamlit is an optional extra**, so these skip when it is absent rather than making the
default install heavier. CI installs it deliberately (`uv sync --extra dashboard`): without
it these would skip and the run would report green over a dozen tests it never executed.

**They read a synthetic tree, not `data/`.** Three of them drove the sidebar, which builds
its options by globbing the data directory - so they passed on a machine that had fetched and
failed in a clean clone, where `data/` deliberately does not exist. A clean-clone run found
that before CI did; `FORECAST_LAB_ROOT` is what lets them point somewhere hermetic.

**Three of them are marked `slow`** and excluded from the default run: they spawn the
analysis commands, which fit models, and one takes four minutes. Opt in with
`pytest -m slow`. The rest render documents and finish in under a second, which is what
keeps the gate worth running on every change.

**Deliberately not asserted here:** research numbers. Those belong to the modules that
compute them, and the payload contract lives in `test_cli.py`. These check that the page
*renders* - which is the only thing `runner` and `presentation` cannot check for it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit", reason="the dashboard is an optional extra")

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src" / "forecast_lab" / "interfaces" / "dashboard.py"

#: The pages that render a document and run no command, so they are fast and hermetic.
DOCUMENT_PAGES = ("Findings", "Reasoning")


def _app(timeout: int = 120) -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=timeout)


REPOSITORY = Path(__file__).resolve().parents[2]

#: Enough of a series file for the sidebar to offer a symbol and an interval. The dashboard
#: reads only the *names* to build those options, so the bars need not be plausible - and
#: making them plausible would invite a later test to compute something from them, which is
#: how a fixture stops being a fixture.
_BARS = "time,open,high,low,close,volume,spread\n1514847600,1,1,1,1,1,0.1\n"


@pytest.fixture
def synthetic_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tree the dashboard can read without a download.

    Pins a defect a clean clone found and no machine with `data/` on it could: three tests
    drove the sidebar, which builds its options by globbing `data/`, so they passed here and
    failed on a fresh checkout - where that directory does not exist, deliberately. CI would
    have gone red on its first run.

    The committed payloads are copied rather than invented, because what the verdict test
    asserts is that **the published figures** reach the page. A fabricated payload would let
    the test pass while the real record was unreachable.
    """
    # Both datasets, because one of these tests changes the Dataset control and would
    # otherwise skip itself for want of a second option - and a test that skips is
    # decoration, which is this repository's own argument against a provenance *test*.
    for dataset in ("raw", "reference"):
        directory = tmp_path / "data" / dataset
        directory.mkdir(parents=True)
        (directory / "XAUUSD_1H.csv").write_text(_BARS, encoding="utf-8")
    status = tmp_path / "docs" / "status"
    status.mkdir(parents=True)
    for payload in (REPOSITORY / "docs" / "status").glob("*.json"):
        (status / payload.name).write_bytes(payload.read_bytes())

    monkeypatch.setenv("FORECAST_LAB_ROOT", str(tmp_path))
    return tmp_path


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
def test_the_sidebar_choices_survive_a_visit_to_another_page(synthetic_root: Path) -> None:
    """Pins the defect that made a page compute against a dataset the reader did not pick.

    The widgets were unkeyed, and `main()` returns before creating them on the document
    pages - so Streamlit garbage-collected the choice, and coming back reset it to the
    default with only the command line to reveal it.
    """
    app = _app().run()
    app.sidebar.radio[0].set_value("Exploration").run()
    # Asserted, not skipped. This used to skip when only one dataset was on disk, which the
    # fixture now guarantees is never true - so a skip here could only mean the fixture had
    # stopped working, and would hide that rather than report it.
    assert len(app.sidebar.selectbox[0].options) >= 2, "the fixture should offer two datasets"

    chosen = app.sidebar.selectbox[0].options[1]
    app.sidebar.selectbox[0].set_value(chosen).run()
    app.sidebar.radio[0].set_value("Findings").run()
    app.sidebar.radio[0].set_value("Exploration").run()

    assert app.session_state["dataset"] == chosen


@pytest.mark.unit
def test_the_mode_control_appears_only_where_the_command_takes_it(
    synthetic_root: Path,
) -> None:
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
@pytest.mark.slow
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
@pytest.mark.slow
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


# --- rendering on a machine that cannot use Arrow ---------------------------------------


@pytest.mark.unit
def test_a_table_falls_back_to_markdown_when_arrow_is_unavailable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The path this machine cannot exercise today, and could two days ago.

    `pyarrow` ships a native library that Windows Smart App Control blocked until the
    release accrued reputation. An earlier version of this project concluded from five
    retries that it never clears; it cleared overnight. So neither state is assumed - the
    capability is probed, and both branches have to work.
    """
    from forecast_lab.interfaces import dashboard

    def _blocked(_name: str) -> None:
        raise OSError("blocked by an application control policy")

    dashboard.arrow_available.cache_clear()
    # Patched on `importlib.util` itself rather than reached through the module under test,
    # which would require re-exporting an import just to make a test type-check.
    monkeypatch.setattr(importlib.util, "find_spec", _blocked)
    try:
        assert dashboard.arrow_available() is False
    finally:
        dashboard.arrow_available.cache_clear()


@pytest.mark.unit
def test_the_capability_is_probed_rather_than_assumed() -> None:
    """Whichever way it answers on this machine, it must answer from a probe."""
    from forecast_lab.interfaces.dashboard import arrow_available

    assert isinstance(arrow_available(), bool)


@pytest.mark.unit
@pytest.mark.slow
def test_the_verdict_page_draws_the_finding_rather_than_only_tabulating_it() -> None:
    """The scatter is the one image no committed figure covers, because no command
    produced it until `verdict` did: skill on one axis, money on the other, every
    configuration to the right of zero and below it."""
    app = _app(timeout=1200).run()
    app.sidebar.radio[0].set_value("Verdict").run()
    app.sidebar.selectbox("breadth").set_value("6 - raw features only, faster").run()

    assert not app.exception, [str(e.message) for e in app.exception]
    charts = app.get("vega_lite_chart")
    assert len(charts) >= 3, "expected fold stability, the scatter, and the edge bars"

    marks: set[str] = set()
    for element in charts:
        raw: Any = getattr(element, "spec", None)
        spec: dict[str, Any] = raw if isinstance(raw, dict) else json.loads(raw)
        # Every chart keeps its data off the top level, so a blocked Arrow cannot blank it.
        assert "data" not in spec and "datasets" not in spec
        for view in spec["layer"]:
            mark = view["mark"]
            marks.add(mark if isinstance(mark, str) else mark["type"])

    assert {"point", "line", "bar", "rule"} <= marks


# --- the record, rather than a run --------------------------------------------------------


@pytest.mark.unit
def test_the_verdict_page_serves_the_published_result_instantly(
    synthetic_root: Path,
) -> None:
    """The worst problem this page had: it re-ran everything.

    Nine payloads sit committed under `docs/status/` - the ones FINDINGS and the STATUS
    logs quote - and every panel spawned a subprocess anyway, two minutes for this page and
    four with every configuration. The published path reads them, and each panel says which
    file it came from so a reader can tell a committed figure from a fresh one.
    """
    app = _app(timeout=120).run()
    app.sidebar.radio[0].set_value("Verdict").run()

    assert not app.exception, [str(e.message) for e in app.exception]
    committed = [c.value for c in app.caption if "Committed result" in c.value]
    assert len(committed) >= 2, "both panels should come from the record"
    assert any("verdict-canonical.json" in c for c in committed)
    assert all("not recomputed now" in c for c in committed)
    # And the numbers are the published ones.
    assert any(m.value == "0 of 18" for m in app.metric)


@pytest.mark.unit
def test_a_series_with_no_committed_result_falls_back_to_running() -> None:
    """The record covers gold at one hour. Offering the option elsewhere would promise
    something every panel then failed to deliver, so it is withdrawn and said so."""
    from forecast_lab.interfaces.dashboard import ROOT
    from forecast_lab.interfaces.published import available_for

    assert available_for(ROOT, target="XAUUSD", timeframe="1H")
    assert not available_for(ROOT, target="SPX", timeframe="1H")
