"""Building a command line, running it, and parsing what comes back.

This is the whole of the dashboard that is worth testing, which is why it lives apart from
the Streamlit script rather than inside it. ADR-005 decided in Phase 1 that the dashboard
would **invoke** `forecast-lab <command> --json` as a subprocess rather than importing
`research` and calling the functions directly, so the page and the terminal cannot drift
apart. Everything here follows from that.

**It imports nothing from this package.** Not `research`, not `ingest`, not even
`contracts`. That is the decision made executable: a module that cannot reach the analysis
layer cannot accidentally become a second implementation of it. A test in the layering
suite enforces it.

**It refuses to run anything that writes, or that serves this page.** `fetch` and
`ingest` download and overwrite;
ADR-005 sec. 4 keeps those at a terminal, because a page a stranger can load should not be
able to start a multi-gigabyte download or make the provenance manifest a record of
whoever clicked last. The allow-list is here rather than in the page, so a new panel cannot
quietly widen it.

**On the cost.** ADR-005 accepted the subprocess overhead with a number attached and the
number was wrong - see its Consequences for the correction. Measured on this machine, a
`baseline --json` run takes **2.9 seconds**, of which **2.8 is interpreter startup and
imports**: the analysis itself is about 110 ms. The fixed cost is paid per invocation and
does not grow with the data, which is what makes caching on the manifest hash the right
answer rather than a workaround.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from shutil import which
from typing import Any

#: Commands the dashboard may run. Everything here reads and reports; nothing downloads or
#: writes (ADR-005 sec. 4). `fetch` and `ingest` are absent on purpose and their absence is
#: asserted by a test, so removing this guard fails a gate rather than passing review.
READ_ONLY: frozenset[str] = frozenset(
    {
        "symbols",
        "verify",
        "align",
        "explore",
        "features",
        "baseline",
        "train",
        "validate",
        "verdict",
    }
)

#: Commands that write to disk or the network. Named rather than merely omitted, so the
#: refusal can say *why* instead of "unknown command".
WRITING = frozenset({"fetch", "ingest"})

#: Commands that exist at a terminal and have no meaning inside a panel. `dashboard`
#: serves this page: reaching it from here would spawn a server that spawns a server.
TERMINAL_ONLY = frozenset({"dashboard"})

#: Options that make a read-only command write. `explore --figures DIR` and
#: `train --figures DIR` create the directory and save PNGs into it, so the command
#: word is not the whole story. Until this existed, ADR-005 sec. 4's guarantee rested
#: on the fact that both commands return early on `--json` *before* reaching their
#: figure block - an accident of statement ordering inside two functions, not a gate.
FORBIDDEN_OPTIONS = frozenset({"--figures"})

#: The subset that emits a machine-readable payload. `symbols` and `verify` report to a
#: human and exit, so asking them for `--json` is a usage error - found by running it,
#: which returned exit 2 and an empty parse. The capability lives beside the allow-list
#: rather than in each caller, so a panel cannot get it wrong.
JSON_CAPABLE = frozenset(
    {"align", "explore", "features", "baseline", "train", "validate", "verdict"}
)

#: Exit codes that carry a report rather than a failure. `verify` exits 1 when the
#: bytes on disk have drifted from the manifest - which is the single state it exists
#: to detect, so treating it as an error meant the panel failed in exactly the case it
#: was built for. Keyed by command, because exit 1 means "unanswerable" everywhere
#: else and must keep meaning that.
REPORTING_EXITS: dict[str, frozenset[int]] = {"verify": frozenset({1})}

#: Generous: `verdict` fits eighteen configurations across five folds and bootstraps twice.
DEFAULT_TIMEOUT_SECONDS = 600


class RunnerError(RuntimeError):
    """A command could not be built, run, or parsed.

    Deliberately not a `ForecastLabError`: importing the package's error hierarchy would
    be an import from `contracts`, and this module's whole point is importing nothing.
    """


@dataclass(frozen=True)
class Invocation:
    """One command line, in a form that can be both executed and shown to a reader."""

    command: str
    #: `--target XAUUSD` style pairs, in the order they should appear.
    options: tuple[tuple[str, str], ...] = ()
    #: Bare switches such as `--pca` or `--long-only`.
    flags: tuple[str, ...] = ()
    #: Whether to append `--json`. Left unset it follows `JSON_CAPABLE`, which is what a
    #: caller almost always wants; setting it explicitly is for the rare panel that
    #: wants the rendered text of a command that could have given a payload.
    json_output: bool | None = None
    _executable: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if self.command in WRITING:
            raise RunnerError(
                f"{self.command!r} writes to disk or the network, and the dashboard is "
                "read-only (ADR-005 sec. 4). Run it at a terminal."
            )
        if self.command in TERMINAL_ONLY:
            raise RunnerError(
                f"{self.command!r} serves this page; running it from inside the page would "
                "start a server that starts a server. Run it at a terminal."
            )
        if self.command not in READ_ONLY:
            raise RunnerError(
                f"unknown command {self.command!r}; expected one of {sorted(READ_ONLY)}"
            )
        writing = {name for name, _ in self.options} & FORBIDDEN_OPTIONS
        if writing or set(self.flags) & FORBIDDEN_OPTIONS:
            offending = sorted(writing | (set(self.flags) & FORBIDDEN_OPTIONS))
            raise RunnerError(
                f"{offending} writes to disk, and the dashboard is read-only "
                "(ADR-005 sec. 4). Committed figures are rendered from the repository."
            )
        if self.json_output is None:
            object.__setattr__(self, "json_output", self.command in JSON_CAPABLE)
        elif self.json_output and self.command not in JSON_CAPABLE:
            raise RunnerError(
                f"{self.command!r} has no --json; it reports to a human and exits"
            )

    @property
    def arguments(self) -> list[str]:
        """Everything after the executable."""
        parts = [self.command]
        for name, value in self.options:
            parts.extend([name, str(value)])
        parts.extend(self.flags)
        if self.json_output:
            parts.append("--json")
        return parts

    @property
    def argv(self) -> list[str]:
        """What is actually executed, executable included."""
        return [*(self._executable or entry_point()), *self.arguments]

    @property
    def display(self) -> str:
        """The copyable line a panel shows beneath its result (ADR-005 sec. 2).

        Derived from `argv` rather than written separately, so the line on screen is the
        line that ran. Where the console script is not on PATH this shows the `python -m`
        form, which is the same entry point and is what a reader would have to type.
        """
        executable = self._executable or entry_point()
        head = "forecast-lab" if len(executable) == 1 else "python -m forecast_lab.interfaces.cli"
        return " ".join([head, *self.arguments])


#: Set once the console script has been shown to be unspawnable, so the fallback below is
#: taken directly rather than after a failed attempt on every single invocation.
_SCRIPT_BLOCKED = False

#: The module form, which needs no shim and therefore no reputation.
_MODULE_FORM: tuple[str, ...] = (sys.executable, "-m", "forecast_lab.interfaces.cli")


def entry_point() -> tuple[str, ...]:
    """How to invoke the CLI from here.

    Prefers the installed console script, so the line a panel displays is the line a reader
    can paste. Falls back to `python -m` when it is not on PATH - inside an unactivated
    virtual environment - or when it is present and **cannot be spawned**.

    That second case is not hypothetical. On Windows with Smart App Control the generated
    `forecast-lab.exe` shim is an unsigned binary with no reputation, and a reinstall makes
    it new again: `uv sync` rewrites it and the next spawn fails with `WinError 4551`,
    "an application control policy blocked this file". Checking only that the file exists
    would leave the dashboard broken on the machine this project is developed on. The two
    forms are the same entry point, so the fallback costs nothing but a longer line on
    screen.
    """
    if _SCRIPT_BLOCKED:
        return _MODULE_FORM
    script = which("forecast-lab")
    return (script,) if script else _MODULE_FORM


def run(
    invocation: Invocation,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    environment: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> Any:
    """Run ``invocation`` and return its parsed JSON payload.

    Raises rather than returning a partial result on any failure, because a dashboard that
    renders half a payload shows numbers whose provenance nobody can state.

    ``cwd`` matters more than it looks. Every path this project passes is relative -
    `--dir data/raw`, and the CLI's own default manifest at `docs/status/data-manifest.json`
    - so a caller launched from anywhere but the repository root gets "No 1H series for
    XAUUSD" on every panel while the page itself looks healthy. Found by testing that the
    server *starts* from another directory, which proved less than it appeared to.
    """
    try:
        completed = subprocess.run(
            invocation.argv,
            capture_output=True,
            text=True,
            # UTF-8 explicitly, and the child told to use it too. `text=True` alone decodes
            # with the parent's *locale* codec - cp1252 on Windows - and the CLI emits
            # characters outside it, so the reader thread dies with a UnicodeDecodeError
            # and the payload is lost. That is the same defect `cli.py`'s own docstring
            # warns about for redirected output, arriving from the parent's side; found by
            # running this against `symbols`, not by reading it.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            cwd=cwd,
            env=_child_environment(environment),
        )
    except FileNotFoundError as exc:
        raise RunnerError(
            f"could not find the forecast-lab entry point: {exc}. Is the package installed?"
        ) from exc
    except OSError as exc:
        # The console script exists and the operating system refused to start it - Smart
        # App Control, most likely. Retry through the module form, which needs no shim, and
        # remember so the next call does not pay for the failure again.
        if invocation._executable or entry_point() == _MODULE_FORM:
            raise RunnerError(f"`{invocation.display}` could not be started: {exc}") from exc
        global _SCRIPT_BLOCKED
        _SCRIPT_BLOCKED = True
        return run(invocation, timeout=timeout, environment=environment, cwd=cwd)
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"`{invocation.display}` exceeded {timeout}s") from exc

    reporting = REPORTING_EXITS.get(invocation.command, frozenset())
    if completed.returncode != 0 and completed.returncode not in reporting:
        # BOTH streams, concatenated. `stderr or stdout` let one stray warning - a numpy
        # RuntimeWarning, a sklearn ConvergenceWarning - win outright and hide the CLI's
        # own explanation, so the reader saw "invalid value encountered in divide" where
        # the actual reason was "No 4H series for XAUUSD".
        detail = "\n".join(
            stream.strip()
            for stream in (completed.stdout, completed.stderr)
            if stream and stream.strip()
        )
        raise RunnerError(
            f"`{invocation.display}` exited {completed.returncode}: {_tail(detail)}"
        )
    if not invocation.json_output:
        return completed.stdout

    if not completed.stdout.strip():
        raise RunnerError(f"`{invocation.display}` produced no output to parse")
    try:
        return _payload_from(completed.stdout)
    except RunnerError as exc:
        raise RunnerError(
            f"`{invocation.display}` did not return JSON: {_tail(completed.stdout)}"
        ) from exc


def _child_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    """The child's environment, with its output encoding pinned.

    `PYTHONIOENCODING` makes the CLI encode as UTF-8 regardless of the console codepage it
    inherits, which is the other half of the fix above: decoding correctly does not help if
    the child encoded with cp1252 and dropped a character on the way out.
    """
    inherited = dict(os.environ if environment is None else environment)
    inherited["PYTHONIOENCODING"] = "utf-8"
    return inherited


def _payload_from(output: str) -> Any:
    """Find and decode the JSON payload, ignoring whatever a console printed around it.

    **Not a scan for the first bracket.** That was the first version, and it is reachable:
    when an estimator cannot be loaded for a reason other than an application-control
    policy, the CLI prints the operating system's own words - `[WinError 126] The specified
    module could not be found` - and a scan for the first `[` starts parsing there. The
    payload is then "not JSON" and a working run reports a failure.

    Instead each line that could begin a JSON value is tried with `raw_decode`, which
    stops at the end of the first complete value and tells us it succeeded. Noise before
    the payload cannot match; noise after it is ignored by construction.

    Still not table-scraping, which ADR-005 sec. 3 forbids for good reason - rich chooses
    column widths from the terminal and truncates. This decodes the declared payload; it
    only has to find where it starts.
    """
    decoder = json.JSONDecoder()
    offset = 0
    for line in output.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped[:1] in "{[":
            try:
                value, _ = decoder.raw_decode(output[offset + (len(line) - len(stripped)) :])
            except json.JSONDecodeError:
                offset += len(line)
                continue
            return value
        offset += len(line)
    raise RunnerError("no JSON payload found in the output")


def _tail(text: str, limit: int = 400) -> str:
    """The end of a message, which is where a traceback keeps its reason."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"...{collapsed[-limit:]}"


def series_options(
    target: str, timeframe: str, directory: str | None = None
) -> tuple[tuple[str, str], ...]:
    """The three options nearly every analysis command takes, in a consistent order."""
    options: list[tuple[str, str]] = [("--target", target), ("--timeframe", timeframe)]
    if directory:
        options.append(("--dir", directory))
    return tuple(options)


def describe(invocations: Sequence[Invocation]) -> list[str]:
    """The command lines behind a page, for a panel that runs more than one."""
    return [invocation.display for invocation in invocations]
