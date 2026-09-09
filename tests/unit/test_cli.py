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

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from forecast_lab.interfaces.cli import app

#: The package source, for the tests that read the dashboard rather than trusting a list.
SRC = Path(__file__).resolve().parents[2] / "src" / "forecast_lab"

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
    "dashboard",
)

#: The commands ADR-005 sec. 1 requires to emit machine-readable output.
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
    """ADR-005 sec. 3: the dashboard runs these rather than reimplementing them.

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


# --- the payload contract the dashboard depends on -------------------------------------


#: Every key the dashboard indexes, by command. Nested paths are dotted; a `[]` segment
#: means "every element of this list" and a `{}` segment means "every value of this dict".
#:
#: This list is checked against payloads produced by running the commands, which is the
#: whole point. Its first version asserted that a set literal three lines above it was
#: non-empty - it named `cli.py` in its docstring, touched nothing, and would have let
#: `break_even` be renamed to `breakeven` with every gate green and a KeyError waiting in
#: the browser. An audit found it; this is the replacement.
DASHBOARD_KEYS: dict[str, tuple[str, ...]] = {
    "align": (
        "target", "rows", "columns", "first", "last",
        "coverage[].symbol", "coverage[].rows", "coverage[].missing",
        "coverage[].missing_fraction", "coverage[].stale", "coverage[].stale_fraction",
        "coverage[].max_stale_seconds",
    ),
    "explore": (
        "correlations{}.on_levels", "correlations{}.on_returns", "correlations{}.inflation",
        "by_year{}.bars", "by_year{}.up_share", "by_year{}.volatility",
        "moves.up", "moves.down", "moves.up_share",
        "normality.price.p_value", "normality.returns.p_value",
    ),
    "features": (
        "rows", "columns", "warmup_dropped", "first", "last", "mode", "symbols",
        "policy_passes", "violations",
        "features[].name", "features[].scale", "features[].worst_change",
        "features[].adf_pvalue", "features[].kpss_pvalue",
    ),
    "validate": (
        "models[].model", "models[].representation", "models[].accuracy",
        "models[].baseline_accuracy", "models[].flip_rate", "models[].bars_held",
        "models[].break_even",
        "power.walk_forward_mde", "power.single_split_mde",
        "power.power_for_break_even", "power.bars_required",
        "dependence.inflation", "dependence.effective_sample",
        "dependence.lag_one_autocorrelation",
        "serial_dependence_contrast.label",
        "serial_dependence_contrast.most_autocorrelated_features",
        "scheme.blocks[].fold", "scheme.blocks[].train_rows", "scheme.blocks[].test_rows",
    ),
    "verdict": (
        "configurations", "scored_bars", "folds",
        "costs.round_trip_bps", "costs.source", "costs.position",
        "skill.surviving_holm",
        "skill.per_configuration{}.accuracy",
        "skill.per_configuration{}.independent_accuracy",
        "skill.per_configuration{}.excess",
        "skill.per_configuration{}.statistic",
        "skill.per_configuration{}.p_value",
        "skill.per_configuration{}.survives_holm",
        "profit.profitable", "profit.beating_benchmark", "profit.benchmark_cumulative",
        "profit.per_configuration{}.cumulative",
        "multiplicity.spa_p_consistent", "multiplicity.spa_p_lower",
        "multiplicity.spa_p_upper", "multiplicity.stepm_rejected",
        "multiplicity.deflated_sharpe.deflated",
        "multiplicity.deflated_sharpe.configuration",
        "multiplicity.deflated_sharpe.sharpe_per_bar",
        "multiplicity.deflated_sharpe.skew",
        "multiplicity.deflated_sharpe.kurtosis",
        "multiplicity.deflated_sharpe.expected_maximum",
        "multiplicity.deflated_sharpe.trials",
    ),
    "train": (
        "selected_on_validation", "test_edge",
        "models_unavailable",
        "scores[].model", "scores[].representation", "scores[].block",
        "scores[].accuracy", "scores[].baseline_accuracy", "scores[].edge",
        "scores[].auc", "scores[].brier", "scores[].predicted_up_rate",
    ),
}

#: Commands the dashboard runs whose output it prints rather than indexes. There are no
#: payload keys to pin, and each needs a reason - so that a seventh command cannot land
#: here by default, which is how `train` went unpinned.
RENDERED_AS_TEXT: dict[str, str] = {
    "symbols": "prints the inventory table as the CLI formats it; no payload is read",
    "verify": "prints the manifest comparison; the dashboard reads only its exit code",
}

#: Extra arguments some commands need to reach the sections the dashboard reads.
EXTRA_ARGUMENTS: dict[str, list[str]] = {
    "validate": ["--folds", "2"],
    "verdict": ["--folds", "2", "--no-pca", "--reps", "50"],
}


def _resolve(payload: object, path: str) -> None:
    """Walk a dotted path, asserting each segment exists.

    `[]` descends into every element of a list, `{}` into every value of a dict - so a
    key missing from one configuration out of eighteen fails rather than hiding behind
    the first one that has it.
    """
    if not path:
        return
    head, _, rest = path.partition(".")
    if head.endswith("[]"):
        name = head[:-2]
        assert isinstance(payload, dict) and name in payload, f"missing list `{name}`"
        items = payload[name]
        assert isinstance(items, list) and items, f"`{name}` is not a non-empty list"
        for item in items:
            _resolve(item, rest)
    elif head.endswith("{}"):
        name = head[:-2]
        assert isinstance(payload, dict) and name in payload, f"missing mapping `{name}`"
        values = payload[name]
        assert isinstance(values, dict) and values, f"`{name}` is not a non-empty mapping"
        for value in values.values():
            _resolve(value, rest)
    else:
        assert isinstance(payload, dict) and head in payload, f"missing key `{head}`"
        _resolve(payload[head], rest)


@pytest.mark.unit
@pytest.mark.parametrize("command", sorted(DASHBOARD_KEYS))
def test_the_payload_carries_every_key_the_dashboard_indexes(
    command: str, data_dir: Path
) -> None:
    """ADR-005 sec. 3 calls the payload a contract and its testable surface the reason to
    prefer it over an import. This is that test: run the command, walk each path the
    dashboard uses, and fail here rather than as a KeyError in a browser.
    """
    arguments = [
        command, "--target", "XAUUSD", "--timeframe", "1H", "--dir", str(data_dir),
        *EXTRA_ARGUMENTS.get(command, []), "--json",
    ]
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    for path in DASHBOARD_KEYS[command]:
        _resolve(payload, path)


@pytest.mark.unit
def test_the_key_list_covers_the_commands_the_dashboard_runs() -> None:
    """Every command the dashboard runs must have its payload keys pinned here.

    This assertion used to run the other way - `set(DASHBOARD_KEYS) <= JSON_CAPABLE` - which
    checks that nothing listed is bogus and says nothing about what is missing. `train` was
    missing for as long as the list existed, while `dashboard.py` indexed
    `selected_on_validation` and `scores[].block` directly: renaming either passed all three
    gates and broke the Models page. The comment beside the old assertion even said the
    dashboard reads `train`.

    So the list of commands is **read out of the dashboard** rather than restated. A page
    added tomorrow that runs a seventh command fails here until its payload is pinned, which
    is the deny-by-default posture `tests/test_layering.py` takes for imports.
    """
    from forecast_lab.interfaces.runner import JSON_CAPABLE

    source = ast.parse((SRC / "interfaces" / "dashboard.py").read_text(encoding="utf-8"))
    invoked = {
        node.args[0].value
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Invocation"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }

    assert invoked, "no command call sites found - the extraction stopped working"
    assert not (set(DASHBOARD_KEYS) & set(RENDERED_AS_TEXT)), "a command is in both lists"
    assert set(DASHBOARD_KEYS) <= JSON_CAPABLE, "a pinned command has no --json to pin"

    unclassified = invoked - set(DASHBOARD_KEYS) - set(RENDERED_AS_TEXT)
    assert not unclassified, (
        f"the dashboard runs commands nobody classified: {sorted(unclassified)}. "
        "Pin the payload keys in DASHBOARD_KEYS, or record in RENDERED_AS_TEXT why there "
        "are none to pin."
    )


# --- the headline number, and the arithmetic that got it wrong ---------------------------------


@pytest.mark.unit
def test_a_cumulative_return_can_never_pass_total_loss() -> None:
    """The defect this pins, in one property.

    The money path used to report `float(v.sum())` over per-bar returns and print it as a
    percentage, so the headline read "the best loses 196.7%" - a loss larger than the
    capital available to lose it, sitting in the README for a fortnight. The dashboard
    glossary had even grown an entry explaining why the figure passed 100%, which is a
    defect being described rather than investigated.

    No sequence of unlevered positions can lose more than everything. The bound is `>=`
    rather than `>`: 0.5 ** 200 is smaller than the least positive double, so total ruin is
    reported as exactly -100% instead of the -99.999...% arithmetic would give. That is the
    right answer to print, and the wrong one to assert a strict inequality against.
    """
    from forecast_lab.interfaces.cli import _compounded

    ruinous = pd.Series([-0.5] * 200)

    assert _compounded(ruinous) >= -1.0


@pytest.mark.unit
def test_a_cumulative_return_reinvests_rather_than_adds() -> None:
    """Two bars of +10% make +21%, not +20%: the second bar trades the first one's proceeds."""
    from forecast_lab.interfaces.cli import _compounded

    assert _compounded(pd.Series([0.1, 0.1])) == pytest.approx(0.21)


@pytest.mark.unit
def test_a_cumulative_return_is_exact_for_the_loss_that_ends_it() -> None:
    """-100% on any bar is total ruin, and nothing after it can recover the position."""
    from forecast_lab.interfaces.cli import _compounded

    assert _compounded(pd.Series([-1.0, 5.0, 5.0])) == pytest.approx(-1.0)


# --- which dataset a bare command reads --------------------------------------------------------

#: Every command's default data directory, and why it is that one. Two commands read the
#: prior project's exports because that is their subject; everything else reads the
#: canonical dataset `fetch` downloads.
#:
#: The list exists because the defaults had drifted apart silently. `features`, `explore`
#: and `train` were built when reference was the only data there was, and kept pointing at
#: it after `fetch` arrived - so a reader following the RUNBOOK got a break-even of 51.92%
#: from `train` and 53.49% from `verdict` and had no way to know they were different
#: datasets. Nothing was inconsistent; the two numbers simply answered different questions
#: without saying so.
DEFAULT_DIRECTORIES: dict[str, tuple[str, str]] = {
    "symbols": ("directory", "raw"),
    "fetch": ("destination", "raw"),
    "align": ("directory", "raw"),
    "explore": ("directory", "raw"),
    "features": ("directory", "raw"),
    "train": ("directory", "raw"),
    "validate": ("directory", "raw"),
    "verdict": ("directory", "raw"),
    "baseline": ("directory", "reference"),  # its subject is the original's published baseline
    "ingest": ("destination", "reference"),  # it writes those exports there
}


@pytest.mark.unit
def test_every_command_reads_the_dataset_this_list_says_it_does() -> None:
    """A silent split between datasets is the defect this pins.

    Read from the source rather than by invoking each command, because the question is what
    a *default* is - and the moment a test passes `--dir` to find out, it is no longer
    asking about the default.
    """
    source = ast.parse((SRC / "interfaces" / "cli.py").read_text(encoding="utf-8"))
    found: dict[str, tuple[str, str]] = {}
    for node in ast.walk(source):
        if not isinstance(node, ast.FunctionDef):
            continue
        command = next(
            (
                str(d.args[0].value)
                for d in node.decorator_list
                if isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr == "command"
                and d.args
                and isinstance(d.args[0], ast.Constant)
            ),
            None,
        )
        if command is None:
            continue
        names = [argument.arg for argument in node.args.args]
        offset = len(names) - len(node.args.defaults)
        for index, default in enumerate(node.args.defaults):
            if not isinstance(default, ast.Name) or not default.id.endswith("_DIR"):
                continue
            found[command] = (
                names[offset + index],
                "reference" if "REFERENCE" in default.id else "raw",
            )

    assert found == DEFAULT_DIRECTORIES, (
        "a command's default dataset changed. Every entry here is a claim the RUNBOOK and "
        "the committed sidecars depend on, so update both before updating this list."
    )


# --- reproduce: do the published numbers still follow from the data? ---------------------------


@pytest.mark.unit
def test_a_payload_that_matches_shows_no_drift() -> None:
    from forecast_lab.interfaces.cli import _drift

    payload = {"rows": 10, "edge": 0.01, "run": {"command": "train", "argv": []}}
    fresh = {"rows": 10, "edge": 0.01, "run": {"command": "train", "argv": ["x"]}}

    assert _drift(payload, fresh) is None


@pytest.mark.unit
def test_provenance_never_counts_as_drift() -> None:
    """Two correct runs of one command differ in their clocks and in nothing else.

    `generated_at` sits at the top level of two payloads for historical reasons. Excluding it
    is the correct semantics rather than a convenience: a check that fired on it would fire
    on every run, and a check that always fires is one nobody reads.
    """
    from forecast_lab.interfaces.cli import _drift

    payload = {"edge": 0.01, "generated_at": "2026-08-30T00:00:00+00:00", "run": {}}
    fresh = {"edge": 0.01, "generated_at": "2026-09-09T00:00:00+00:00", "run": {}}

    assert _drift(payload, fresh) is None


@pytest.mark.unit
def test_more_rows_is_reported_as_the_data_extending() -> None:
    """The benign case, and the one that must not read like a broken result."""
    from forecast_lab.interfaces.cli import _drift

    reason = _drift({"rows": 50948, "edge": 0.01}, {"rows": 51000, "edge": 0.02})

    assert reason is not None
    assert "the data extended" in reason
    assert "50,948" in reason and "51,000" in reason


@pytest.mark.unit
def test_a_new_field_is_reported_as_an_addition_not_a_moved_figure() -> None:
    """The case that actually arrived: `f1` and `negative_predictive_value` were added to
    the score table, so every committed payload differed while no published number had
    moved. Reporting that as a changed result would teach an operator to re-cut on sight."""
    from forecast_lab.interfaces.cli import _drift

    reason = _drift(
        {"rows": 10, "scores": [{"model": "RF", "accuracy": 0.51}]},
        {"rows": 10, "scores": [{"model": "RF", "accuracy": 0.51, "f1": 0.55}]},
    )

    assert reason is not None
    assert "unchanged" in reason and "gained fields" in reason


@pytest.mark.unit
def test_a_moved_figure_is_never_mistaken_for_an_addition() -> None:
    """The one the distinction must not let through."""
    from forecast_lab.interfaces.cli import _drift

    reason = _drift(
        {"rows": 10, "scores": [{"model": "RF", "accuracy": 0.51}]},
        {"rows": 10, "scores": [{"model": "RF", "accuracy": 0.62, "f1": 0.55}]},
    )

    assert reason is not None
    assert "a published figure moved" in reason


@pytest.mark.unit
def test_dropping_a_field_is_not_an_addition() -> None:
    """A payload that stopped reporting something has changed what it says."""
    from forecast_lab.interfaces.cli import _only_additions

    assert not _only_additions({"a": 1, "b": 2}, {"a": 1})


@pytest.mark.unit
def test_reordering_a_list_is_not_an_addition() -> None:
    """The score table is sorted, so its order is part of what the payload states."""
    from forecast_lab.interfaces.cli import _only_additions

    assert not _only_additions([{"m": "a"}, {"m": "b"}], [{"m": "b"}, {"m": "a"}])
