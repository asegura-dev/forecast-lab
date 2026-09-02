"""Tests for the dashboard's runner.

[ADR-005](../../docs/adr/ADR-005-the-dashboard-runs-the-cli.md) says the dashboard becomes
testable without Streamlit because its logic is "build an argument list, parse JSON". This
file is the cash value of that claim: everything below runs with no browser, no server and
no Streamlit installed.

The two that carry the decision are `test_a_command_that_writes_is_refused` and
`test_json_is_parsed_rather_than_the_rendered_table`. The first is ADR-005 sec. 4 - a page
a stranger can load must not be able to start a download. The second is sec. 3 - rich picks
column widths from the terminal and truncates, so a dashboard scraping its output would
break silently on the first number that got wider.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from forecast_lab.interfaces.runner import (
    FORBIDDEN_OPTIONS,
    JSON_CAPABLE,
    READ_ONLY,
    REPORTING_EXITS,
    TERMINAL_ONLY,
    WRITING,
    Invocation,
    RunnerError,
    entry_point,
    run,
    series_options,
)

#: A stand-in for the CLI: prints whatever it is told and exits with the code it is given.
STUB = """
import json, sys
mode = sys.argv[1]
if mode == "payload":
    print(json.dumps({"ok": True, "args": sys.argv[2:]}))
elif mode == "noise":
    print("warning: something on stdout that is not JSON")
    print(json.dumps({"ok": True}))
elif mode == "text":
    print("a rendered table, not a payload")
elif mode == "broken":
    print("{not json at all")
elif mode == "fail":
    print("the reason it failed", file=sys.stderr)
    sys.exit(2)
elif mode == "empty":
    pass
elif mode == "slow":
    import time; time.sleep(30)
elif mode == "drift":
    print("CHANGED raw/XAUUSD_1H.csv: size 1 != 2 recorded")
    print("11 changed, 0 missing.")
    sys.exit(1)
elif mode == "refuse":
    print("No 4H series for XAUUSD")
    sys.exit(1)
elif mode == "winerror":
    print("XGBoost not run: [WinError 126] The specified module could not be found")
    print(json.dumps({"ok": True}))
elif mode == "bothstreams":
    print("the real reason: No 4H series for XAUUSD")
    print("a warning nobody asked for", file=sys.stderr)
    sys.exit(1)
"""


@pytest.fixture
def stub(tmp_path: Path) -> tuple[str, ...]:
    """An executable that behaves like the CLI without being it."""
    script = tmp_path / "stub.py"
    script.write_text(STUB, encoding="utf-8")
    return (sys.executable, str(script))


def _with(stub: tuple[str, ...], mode: str, command: str = "baseline") -> Invocation:
    return Invocation(command, options=(("--mode-arg", mode),), _executable=(*stub, mode))


# --- the decisions ---------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("command", sorted(WRITING))
def test_a_command_that_writes_is_refused(command: str) -> None:
    """ADR-005 sec. 4. The refusal says *why*, not "unknown command"."""
    with pytest.raises(RunnerError, match="read-only"):
        Invocation(command)


@pytest.mark.unit
def test_the_dashboard_cannot_launch_itself() -> None:
    """`forecast-lab dashboard` serves this page.

    Reaching it from a panel would spawn a server that spawns a server, so it is refused
    by name rather than merely left out of the allow-list - the reason matters more than
    the refusal, because "unknown command" would invite someone to add it.
    """
    with pytest.raises(RunnerError, match="server that starts a server"):
        Invocation("dashboard")

    assert {"dashboard"} == TERMINAL_ONLY
    assert not READ_ONLY & TERMINAL_ONLY


@pytest.mark.unit
def test_an_unknown_command_is_refused_before_a_process_starts() -> None:
    """Building the invocation validates it, so nothing is ever spawned speculatively."""
    with pytest.raises(RunnerError, match="unknown command"):
        Invocation("rm-rf")


@pytest.mark.unit
def test_json_is_parsed_rather_than_the_rendered_table(stub: tuple[str, ...]) -> None:
    """ADR-005 sec. 3: the payload is the contract; the table is for people."""
    payload = run(_with(stub, "payload"))

    assert payload["ok"] is True
    assert "--json" in payload["args"]


@pytest.mark.unit
def test_the_command_shown_is_the_command_that_ran() -> None:
    """ADR-005 sec. 2. `display` is derived from the same parts as `argv`, not written twice."""
    invocation = Invocation("validate", options=series_options("XAUUSD", "1H"), flags=("--pca",))

    assert invocation.arguments == [
        "validate", "--target", "XAUUSD", "--timeframe", "1H", "--pca", "--json",
    ]
    assert invocation.display.endswith(" ".join(invocation.arguments))
    assert invocation.argv[-len(invocation.arguments):] == invocation.arguments


# --- the allow-list ----------------------------------------------------------------------


@pytest.mark.unit
def test_the_allow_list_and_the_writing_set_do_not_overlap() -> None:
    assert not READ_ONLY & WRITING
    assert JSON_CAPABLE <= READ_ONLY


@pytest.mark.unit
@pytest.mark.parametrize("command", ["symbols", "verify"])
def test_a_command_without_a_payload_does_not_get_json(command: str) -> None:
    """Found by running it: `symbols --json` exits 2, because it reports and exits.

    The capability lives beside the allow-list rather than in each panel, so a caller
    cannot get it wrong by default - and asking for it explicitly is refused with a reason.
    """
    assert "--json" not in Invocation(command).arguments
    with pytest.raises(RunnerError, match="no --json"):
        Invocation(command, json_output=True)


@pytest.mark.unit
def test_a_command_with_a_payload_gets_json_without_being_asked() -> None:
    assert Invocation("baseline").arguments[-1] == "--json"


# --- failures are reported, never half-rendered -------------------------------------------


@pytest.mark.unit
def test_a_non_zero_exit_raises_with_the_reason_attached(stub: tuple[str, ...]) -> None:
    with pytest.raises(RunnerError, match="the reason it failed"):
        run(_with(stub, "fail"))


@pytest.mark.unit
def test_output_that_is_not_json_raises_rather_than_returning_none(
    stub: tuple[str, ...],
) -> None:
    """A dashboard that renders half a payload shows numbers nobody can source."""
    with pytest.raises(RunnerError, match="did not return JSON"):
        run(_with(stub, "broken"))


@pytest.mark.unit
def test_no_output_at_all_raises(stub: tuple[str, ...]) -> None:
    with pytest.raises(RunnerError, match="no output to parse"):
        run(_with(stub, "empty"))


@pytest.mark.unit
def test_a_command_that_hangs_is_cut_off(stub: tuple[str, ...]) -> None:
    with pytest.raises(RunnerError, match="exceeded"):
        run(_with(stub, "slow"), timeout=1)


@pytest.mark.unit
def test_a_missing_executable_says_so(tmp_path: Path) -> None:
    invocation = Invocation("baseline", _executable=(str(tmp_path / "absent"),))

    with pytest.raises(RunnerError, match="could not find"):
        run(invocation)


# --- the noise a real console emits --------------------------------------------------------


@pytest.mark.unit
def test_a_warning_before_the_payload_does_not_break_the_parse(stub: tuple[str, ...]) -> None:
    """`uv` prints an environment warning and the CLI prints yellow lines for a model it
    cannot load. Both are legitimate, neither is JSON, and the payload still has to parse."""
    assert run(_with(stub, "noise"))["ok"] is True


@pytest.mark.unit
def test_the_child_is_told_to_emit_utf8(stub: tuple[str, ...], monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pins a defect found by running, not by reading.

    `subprocess.run(text=True)` decodes with the *parent's* locale codec - cp1252 on this
    machine - and the CLI emits characters outside it, so the reader thread died with a
    UnicodeDecodeError and the payload was lost. That is the same failure `cli.py`'s own
    docstring warns about for redirected output, arriving from the other side.
    """
    captured: dict[str, Any] = {}
    original = subprocess.run

    def _spy(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _spy)
    run(_with(stub, "payload"))

    assert captured["encoding"] == "utf-8"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"


@pytest.mark.unit
def test_the_entry_point_is_resolvable() -> None:
    """Either the console script or `python -m` - both are the same CLI."""
    resolved = entry_point()

    assert resolved
    assert len(resolved) == 1 or resolved[1:] == ("-m", "forecast_lab.interfaces.cli")


# --- against the real CLI, once ---------------------------------------------------------

#: The module form rather than `entry_point()`, deliberately. The console script is an
#: unsigned shim that Smart App Control blocks after every reinstall on this machine, and a
#: test asserting "the CLI has these commands" should not fail because of a signature
#: policy. The fallback that handles that case is pinned separately below.
CLI = (sys.executable, "-m", "forecast_lab.interfaces.cli")




@pytest.mark.unit
def test_the_real_cli_answers_a_help_request() -> None:
    """One end-to-end check that the entry point this module resolves actually exists.

    Deliberately `--help` rather than an analysis command: this suite is hermetic and must
    not depend on whether `data/` happens to be populated on the machine running it.
    """
    completed = subprocess.run(
        [*CLI, "--help"], capture_output=True, text=True, encoding="utf-8"
    )

    assert completed.returncode == 0
    assert "verdict" in completed.stdout


@pytest.mark.unit
def test_every_allowed_command_exists_in_the_cli() -> None:
    """The allow-list cannot drift from the application it is a list of."""
    completed = subprocess.run(
        [*CLI, "--help"], capture_output=True, text=True, encoding="utf-8"
    )

    for command in sorted(READ_ONLY | WRITING | TERMINAL_ONLY):
        assert command in completed.stdout, f"{command} is in the runner but not in the CLI"


@pytest.mark.unit
def test_a_payload_round_trips_through_json(stub: tuple[str, ...]) -> None:
    payload = run(_with(stub, "payload"))

    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.unit
def test_a_blocked_console_script_falls_back_to_the_module_form(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pins a defect the CLI tests found on the machine this project is developed on.

    Windows Smart App Control blocks the generated `forecast-lab.exe` - it is unsigned and
    a reinstall makes it new again, so `uv sync` is enough to break it with `WinError 4551`.
    `entry_point()` checked only that the file *existed*, which would have left the
    dashboard broken here while every other gate stayed green.

    The first spawn raises `OSError`; the second, through `python -m`, must succeed.
    """
    from forecast_lab.interfaces import runner

    monkeypatch.setattr(runner, "_SCRIPT_BLOCKED", False)
    monkeypatch.setattr(runner, "which", lambda _name: "C:/blocked/forecast-lab.exe")

    attempts: list[list[str]] = []
    original = subprocess.run

    def _spy(argv, **kwargs):  # type: ignore[no-untyped-def]
        attempts.append(list(argv))
        if "forecast-lab.exe" in argv[0]:
            raise OSError(4551, "an application control policy blocked this file")
        emit = "import json; print(json.dumps({'ok': True}))"
        return original([sys.executable, "-c", emit], **kwargs)

    monkeypatch.setattr(subprocess, "run", _spy)
    payload = runner.run(Invocation("baseline"))

    assert payload == {"ok": True}
    assert len(attempts) == 2, "it should retry exactly once"
    assert attempts[0][0].endswith("forecast-lab.exe")
    assert attempts[1][:3] == [sys.executable, "-m", "forecast_lab.interfaces.cli"]


@pytest.mark.unit
def test_a_failure_that_is_not_the_shim_is_not_retried(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The retry is for a blocked shim, not a general "try again" - one attempt, then raise."""
    from forecast_lab.interfaces import runner

    monkeypatch.setattr(runner, "_SCRIPT_BLOCKED", True)  # already on the module form
    def _explode(*_a: Any, **_k: Any) -> Any:
        raise OSError("disk on fire")

    monkeypatch.setattr(subprocess, "run", _explode)

    with pytest.raises(RunnerError, match="could not be started"):
        runner.run(Invocation("baseline"))


# --- what an audit found, pinned --------------------------------------------------------


@pytest.mark.unit
def test_an_option_that_writes_is_refused_even_on_an_allowed_command() -> None:
    """`explore` and `train` are read-only commands with a writing option.

    `--figures DIR` creates the directory and saves PNGs into it. Until this refusal
    existed, ADR-005 sec. 4's guarantee rested on the fact that both commands return early
    on `--json` *before* reaching their figure block - an accident of statement ordering
    inside two functions rather than a gate.
    """
    for command in ("explore", "train"):
        with pytest.raises(RunnerError, match="writes to disk"):
            Invocation(command, options=(("--figures", "C:/anywhere"),))
        with pytest.raises(RunnerError, match="writes to disk"):
            Invocation(command, flags=("--figures",))


@pytest.mark.unit
def test_the_forbidden_options_are_options_the_cli_really_has() -> None:
    """A guard against an option that no longer exists is a guard against nothing."""
    completed = subprocess.run(
        [*CLI, "explore", "--help"], capture_output=True, text=True, encoding="utf-8"
    )

    for option in FORBIDDEN_OPTIONS:
        assert option in completed.stdout, f"{option} is guarded but not in the CLI"


@pytest.mark.unit
def test_verifys_report_of_drift_is_a_report_and_not_a_failure(stub: tuple[str, ...]) -> None:
    """`verify` exits 1 when the bytes on disk have drifted from the manifest.

    That is the single state it exists to detect, so treating exit 1 as an error made the
    provenance panel fail in exactly the case it was built for - and it is the live state
    of this checkout whenever someone has run `fetch`.
    """
    assert REPORTING_EXITS["verify"] == {1}

    invocation = Invocation("verify", _executable=(*stub, "drift"))
    assert run(invocation).startswith("CHANGED")


@pytest.mark.unit
def test_exit_one_still_means_failure_everywhere_else(stub: tuple[str, ...]) -> None:
    """The exemption is per command; "unanswerable" must keep meaning that elsewhere."""
    with pytest.raises(RunnerError, match="exited 1"):
        run(Invocation("align", _executable=(*stub, "refuse")))


@pytest.mark.unit
def test_a_bracket_in_the_noise_does_not_break_the_parse(stub: tuple[str, ...]) -> None:
    """Pins a defect an audit found by reading the CLI rather than the runner.

    When an estimator cannot load for a reason other than an application-control policy,
    the CLI prints the operating system's own words - `[WinError 126] The specified module
    could not be found` - unescaped. A scan for the first `[` starts parsing there, and a
    perfectly good run reports "did not return JSON".
    """
    payload = run(Invocation("train", _executable=(*stub, "winerror")))

    assert payload == {"ok": True}


@pytest.mark.unit
def test_both_streams_reach_the_reader(stub: tuple[str, ...]) -> None:
    """`stderr or stdout` let one stray warning win outright and hide the CLI's own
    explanation, so a reader saw a numpy message where the reason was "No 4H series"."""
    with pytest.raises(RunnerError) as caught:
        run(Invocation("align", _executable=(*stub, "bothstreams")))

    assert "the real reason" in str(caught.value)
    assert "a warning nobody asked for" in str(caught.value)


@pytest.mark.unit
def test_the_working_directory_is_passed_through(stub: tuple[str, ...], tmp_path: Path) -> None:
    """Every path this project passes is relative, so a caller launched from elsewhere gets
    "No 1H series" on every panel while the page itself looks healthy."""
    captured: dict[str, Any] = {}
    original = subprocess.run

    def _spy(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "run", _spy)
        run(_with(stub, "payload"), cwd=str(tmp_path))

    assert captured["cwd"] == str(tmp_path)
