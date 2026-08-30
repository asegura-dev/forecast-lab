"""Tests for the command line, which had none until now.

**Why this file exists.** 322 tests passed while nothing exercised the composition root,
and this repository has already been burned by exactly that: early on, a Typer app holding
a single command treated it as the root, so `forecast-lab symbols` failed with "unexpected
extra argument" while 52 unit tests stayed green. Testing the layer beneath the CLI does
not test the CLI. The first two tests here are that lesson, made permanent.

**What is asserted, and what deliberately is not.** These check the *contract* - that every
command is reachable, that refusals exit with the documented code rather than a traceback,
and that `--json` emits parseable JSON with the keys ADR-005 promises the dashboard. They
do **not** re-assert the research numbers: those belong to the module that computes them,
and duplicating them here would mean two places to update when a measurement changes,
which is how documentation goes stale.

Everything runs on synthetic bars written to a temporary directory. No network, no `data/`,
no dependence on which dataset happens to be on the machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from forecast_lab.interfaces.cli import app

runner = CliRunner()

#: Every command the application exposes. A command missing from this list is either new
#: and untested, or deleted and still documented - both worth failing over.
COMMANDS = (
    "symbols",
    "fetch",
    "align",
    "explore",
    "features",
    "train",
    "validate",
    "baseline",
    "ingest",
    "verify",
)

#: The commands ADR-005 sec. 5 requires to emit machine-readable output.
JSON_COMMANDS = ("align", "explore", "features", "baseline")

#: Long enough for the longest indicator window to warm up and for a split to have blocks.
BARS = 2_600


def _write_series(directory: Path, symbol: str, *, seed: int, spread: bool = True) -> None:
    """A plausible hourly series: a random walk with a positive floor and an OHLC that
    actually brackets the close, because the reader checks and a fixture that cannot be
    read tests nothing."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=BARS, freq="h", tz="UTC")
    close = 1_500 + np.cumsum(rng.normal(0, 1.5, BARS))
    frame = pd.DataFrame(
        {
            "time": index,
            "open": close + rng.normal(0, 0.5, BARS),
            "high": close + np.abs(rng.normal(0, 1.0, BARS)),
            "low": close - np.abs(rng.normal(0, 1.0, BARS)),
            "close": close,
            "volume": rng.integers(100, 10_000, BARS),
        }
    )
    if spread:
        frame["spread"] = np.abs(rng.normal(0.30, 0.08, BARS))
    frame.to_csv(directory / f"{symbol}_1H.csv", index=False)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Two symbols, so `whole` mode and the alignment report have something to align."""
    directory = tmp_path / "raw"
    directory.mkdir()
    _write_series(directory, "XAUUSD", seed=1)
    _write_series(directory, "SPX", seed=2)
    return directory


# --- the lesson: every command is reachable ------------------------------------------


@pytest.mark.unit
def test_the_application_is_not_collapsed_into_a_single_command() -> None:
    """Pins the defect that 52 passing tests missed.

    Typer promotes a lone command to the application root, and the explicit callback that
    prevents it looks redundant enough to delete. If it is ever removed, `--help` stops
    listing subcommands and this fails.
    """
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output, f"{command} is missing from --help"


@pytest.mark.unit
@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_has_reachable_help(command: str) -> None:
    """A command whose options cannot be parsed is broken however well its layer works."""
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0, result.output


# --- the ADR-005 contract -------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("command", JSON_COMMANDS)
def test_json_output_is_parseable(command: str, data_dir: Path) -> None:
    """ADR-005 sec. 5: the dashboard runs these rather than reimplementing them.

    `train` and `validate` are excluded here only because fitting six estimators on every
    invocation would dominate the suite's runtime; `validate` has its own test below.
    """
    result = runner.invoke(
        app, [command, "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(data_dir), "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, dict)


@pytest.mark.unit
def test_align_json_reports_the_staleness_it_had_to_carry(data_dir: Path) -> None:
    """The coverage report is the point of the command, so its shape is pinned."""
    result = runner.invoke(
        app, ["align", "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(data_dir), "--json"]
    )

    payload = json.loads(result.output)
    assert payload["target"] == "XAUUSD"
    assert payload["rows"] == BARS
    assert {c["symbol"] for c in payload["coverage"]} == {"SPX"}
    for cover in payload["coverage"]:
        assert set(cover) >= {"symbol", "missing", "stale", "max_stale_seconds"}


@pytest.mark.unit
def test_validate_json_carries_what_the_verdict_rests_on(data_dir: Path) -> None:
    """The keys ADR-012 argues from: per-model turnover, its break-even, and dependence."""
    result = runner.invoke(
        app,
        ["validate", "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(data_dir), "--folds",
         "2", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["models"], "no model was scored"
    for model in payload["models"]:
        assert set(model) >= {"flip_rate", "bars_held", "break_even", "clears_break_even"}
    assert set(payload["dependence"]) >= {"inflation", "effective_sample", "material"}
    assert payload["scheme"]["folds"] == 2


# --- refusals: the documented exit codes ----------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("command", ["align", "features", "baseline", "explore", "validate"])
def test_a_symbol_that_is_not_there_exits_one(command: str, data_dir: Path) -> None:
    """Exit 1 is "your request was well-formed and the data does not support it"."""
    result = runner.invoke(
        app, [command, "--target", "NOTHING", "--timeframe", "1H", "--dir", str(data_dir)]
    )

    assert result.exit_code == 1, result.output


@pytest.mark.unit
@pytest.mark.parametrize("command", ["align", "features", "baseline", "validate"])
def test_an_unknown_timeframe_exits_two(command: str, data_dir: Path) -> None:
    """Exit 2 is "your request was malformed" - the usage error, distinct from exit 1."""
    result = runner.invoke(
        app, [command, "--target", "XAUUSD", "--timeframe", "7H", "--dir", str(data_dir)]
    )

    assert result.exit_code == 2, result.output


@pytest.mark.unit
@pytest.mark.parametrize("command", ["features", "validate", "train"])
def test_an_unknown_mode_exits_two(command: str, data_dir: Path) -> None:
    result = runner.invoke(
        app,
        [command, "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(data_dir), "--mode",
         "sideways"],
    )

    assert result.exit_code == 2, result.output


@pytest.mark.unit
def test_a_directory_that_does_not_exist_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["align", "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(tmp_path / "absent")],
    )

    assert result.exit_code == 2, result.output


@pytest.mark.unit
def test_an_empty_directory_is_refused_rather_than_producing_an_empty_report(
    tmp_path: Path,
) -> None:
    """A report over zero symbols is not a result, and must not read like one."""
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(
        app, ["align", "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(empty)]
    )

    assert result.exit_code != 0
    assert "XAUUSD" in result.output


# --- the tables render ----------------------------------------------------------------


@pytest.mark.unit
def test_symbols_lists_what_is_on_disk(data_dir: Path) -> None:
    result = runner.invoke(app, ["symbols", "--dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "XAUUSD" in result.output
    assert "SPX" in result.output


@pytest.mark.unit
def test_baseline_prints_all_three_rules_a_model_has_to_beat(data_dir: Path) -> None:
    """ADR-004: three baselines, not one, and each with specificity beside its accuracy.

    The majority-class rule scores well and has 0% specificity - it is a constant wearing
    an accuracy - so a table showing accuracy alone would make it look like a contender.
    """
    result = runner.invoke(
        app, ["baseline", "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(data_dir)]
    )

    assert result.exit_code == 0, result.output
    for rule in ("majority-class", "persistence", "random"):
        assert rule in result.output, f"{rule} baseline is missing"
    assert "Specificity" in result.output


@pytest.mark.unit
def test_validate_names_each_model_and_its_own_break_even(data_dir: Path) -> None:
    """The correction ADR-012 made, visible in the output rather than only in the payload."""
    result = runner.invoke(
        app,
        ["validate", "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(data_dir), "--folds",
         "2"],
    )

    assert result.exit_code == 0, result.output
    assert "break-even" in result.output
    assert "flip" in result.output


@pytest.mark.unit
def test_a_series_without_a_spread_column_says_so_rather_than_inventing_one(
    tmp_path: Path,
) -> None:
    """The reference exports carry no spread, so the threshold falls back and must say it.

    Silently switching between a measured and an assumed break-even would make two runs
    incomparable without either of them saying so (ADR-010 sec. 4).
    """
    directory = tmp_path / "nospread"
    directory.mkdir()
    _write_series(directory, "XAUUSD", seed=3, spread=False)
    result = runner.invoke(
        app,
        ["validate", "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(directory), "--folds",
         "2"],
    )

    assert result.exit_code == 0, result.output
    assert "assumed" in result.output.lower()
