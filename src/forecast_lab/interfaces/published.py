"""The results this repository has already computed and committed.

Every analysis command writes a `--json` payload, and nine of those payloads are committed
under `docs/status/` as the sidecars the STATUS logs and FINDINGS quote. They are the
project's record: the numbers a reader can check a published claim against, pinned to the
data snapshot the manifest describes.

**The dashboard had them on disk and re-ran every command anyway** - two minutes for a
verdict page, four with every configuration, on results that were already sitting there.
This module is what lets a panel offer the published figure instead, instantly, and say
which one it is showing.

**Matched, not hard-coded.** Each sidecar carries the `target`, `timeframe` and `mode` that
produced it, so a request is answered by finding a payload whose own metadata agrees rather
than by a table mapping filenames to commands. A sidecar committed later for another symbol
is found without touching this file - and a sidecar that does not match is not offered,
which is the behaviour that matters: a published figure for gold must never be served to
someone asking about the S&P.

**Each payload names the command that produced it**, so nothing here has to infer it. That
was not always true: this module used to carry a table of key sets, one per builder, and a
test to keep the table honest. `reproduce` needed the actual command line anyway - a key set
cannot be re-run - so recording it made the table redundant rather than merely convenient.

No Streamlit, no subprocess, and nothing imported from this package - the same rule
`runner` and `presentation` follow, enforced across the whole `interfaces` package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Where committed payloads live, relative to the repository root.
SIDECAR_DIRECTORY = Path("docs") / "status"

#: Where a payload records the command that produced it. Written by `cli._emit`.
PROVENANCE_KEY = "run"


class PublishedError(RuntimeError):
    """A committed result could not be found or read.

    Deliberately not a `ForecastLabError`: importing the package's error hierarchy would be
    an import from `contracts`, and this module imports nothing from the package.
    """


@dataclass(frozen=True)
class Published:
    """One committed payload, with enough about itself to be labelled on screen."""

    command: str
    path: Path
    payload: Any
    target: str
    timeframe: str
    mode: str | None
    generated_at: str | None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def provenance(self) -> str:
        """What a panel says above a published figure.

        Always says it is committed rather than live, because the whole hazard of offering
        both is a reader who cannot tell which they are looking at.
        """
        when = f", generated {self.generated_at[:10]}" if self.generated_at else ""
        scope = f", mode {self.mode}" if self.mode else ""
        return (
            f"Committed result from `{self.name}` - {self.target} at {self.timeframe}"
            f"{scope}{when}. Pinned to the data the manifest describes, not recomputed now."
        )


def command_of(payload: Mapping[str, Any]) -> str | None:
    """Which command wrote this payload, as the payload itself records.

    This used to infer the answer from the sections only one builder emits - a table of key
    sets kept by hand, with a test asserting it still matched the real files. It worked, and
    it was a second description of something the payloads could simply say. Since every
    committed payload carries the command line that produced it (`cli._emit`), the table had
    become a thing to keep in sync for no gain, which is how a codebase accumulates.

    A payload without the record returns `None` rather than being guessed at: an unlabelled
    figure is exactly what this module refuses to serve.
    """
    run = payload.get(PROVENANCE_KEY)
    if not isinstance(run, Mapping):
        return None
    command = run.get("command")
    return str(command) if isinstance(command, str) else None


def catalogue(root: Path) -> list[Published]:
    """Every committed payload under ``root``, in a form a caller can match against.

    A file that is not a payload - the data manifest, most obviously - is skipped rather
    than raising: this directory is a research log, and new kinds of file arriving in it is
    the normal case.
    """
    directory = root / SIDECAR_DIRECTORY
    if not directory.is_dir():
        return []

    found: list[Published] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        command = command_of(payload)
        if command is None:
            continue
        target = payload.get("target") or payload.get("symbol")
        timeframe = payload.get("timeframe")
        if not target or not timeframe:
            continue
        found.append(
            Published(
                command=command,
                path=path,
                payload=payload,
                target=str(target),
                timeframe=str(timeframe),
                mode=payload.get("mode"),
                generated_at=payload.get("generated_at"),
            )
        )
    return found


def find(
    root: Path,
    command: str,
    *,
    target: str,
    timeframe: str,
    mode: str | None = None,
) -> Published | None:
    """The committed result for exactly this request, or ``None``.

    ``None`` rather than a near miss, and that is the whole design. Serving gold's verdict
    to someone who asked about the S&P, or a `focus` run to someone who chose `whole`,
    would put a number on screen under a caption that does not describe it - which is the
    failure this repository is a correction of, arriving through a convenience.

    Where several sidecars match - the canonical dataset and the reference exports both
    have an `explore` payload for gold - the most recently generated wins, and the panel
    names the file it came from.
    """
    matches = [
        entry
        for entry in catalogue(root)
        if entry.command == command
        and entry.target.upper() == target.upper()
        and entry.timeframe.upper() == timeframe.upper()
        and (mode is None or entry.mode is None or entry.mode == mode)
    ]
    if not matches:
        return None
    return max(matches, key=lambda entry: (entry.generated_at or "", entry.name))


def available_for(root: Path, *, target: str, timeframe: str) -> frozenset[str]:
    """Which commands have a committed result for this series.

    A panel uses this to offer the published option only where one exists, rather than
    offering it everywhere and failing on half the pages.
    """
    return frozenset(
        entry.command
        for entry in catalogue(root)
        if entry.target.upper() == target.upper()
        and entry.timeframe.upper() == timeframe.upper()
    )
