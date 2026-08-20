"""Provenance: tying a published number to the exact bytes that produced it.

The data directory is not versioned - the series is regenerable, so committing it would
be committing a cache. But "regenerable" is not "identical": a venue can revise a bar, a
download can truncate, a file can be edited by hand at two in the morning. Without a
record, a result and the data behind it drift apart with no symptom.

So the manifest is committed and the data is not. It holds, per file, a SHA-256 of the
bytes, the row count, and the time span. `verify` re-computes and compares.

It is a **command, not a test** (ADR-002 sec. 7). A pytest that reads `data/` would skip
itself in a clean clone and in CI, and a gate that skips is decoration - it reports green
while guaranteeing nothing. The gates stay hermetic on synthetic fixtures; provenance is
something the operator checks deliberately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from forecast_lab.contracts import ForecastLabError, SeriesMeta

#: Read in blocks so a large file never has to fit in memory at once.
_BLOCK: Final = 1 << 20

MANIFEST_VERSION: Final = 1


class ManifestError(ForecastLabError):
    """The data on disk does not match what the manifest records."""


@dataclass(frozen=True)
class Entry:
    """One file's provenance."""

    path: str  # relative to the data root, with forward slashes, so it is portable
    sha256: str
    size_bytes: int
    rows: int
    first: str | None
    last: str | None
    source: str

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "first": self.first,
            "last": self.last,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Entry:
        return cls(
            path=str(raw["path"]),
            sha256=str(raw["sha256"]),
            size_bytes=int(raw["size_bytes"]),
            rows=int(raw["rows"]),
            first=raw.get("first"),
            last=raw.get("last"),
            source=str(raw.get("source", "unknown")),
        )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def entry_for(path: Path, root: Path, meta: SeriesMeta, source: str) -> Entry:
    """Describe one file for the manifest."""
    return Entry(
        path=path.relative_to(root).as_posix(),
        sha256=sha256_of(path),
        size_bytes=path.stat().st_size,
        rows=meta.rows,
        first=meta.first.isoformat() if meta.first else None,
        last=meta.last.isoformat() if meta.last else None,
        source=source,
    )


def write(destination: Path, entries: list[Entry]) -> None:
    """Write the manifest, sorted, so identical data always produces an identical file.

    Two deliberate omissions make that true. Entries are **sorted**, because directory
    iteration order is not reproducible and an unsorted manifest would show a diff on
    every regeneration. And there is **no generation timestamp**: it would change on
    every write even when nothing else did, and git already records when a version of
    this file came into existence. A file that always looks changed is a file nobody
    reads, and this one has to be read to be worth committing.
    """
    payload = {
        "version": MANIFEST_VERSION,
        "entries": [e.to_json() for e in sorted(entries, key=lambda e: e.path)],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read(source: Path) -> list[Entry]:
    """Load a manifest, refusing a version this code does not understand."""
    if not source.is_file():
        raise ManifestError(f"no manifest at {source}; run `forecast-lab ingest` first")
    raw = json.loads(source.read_text(encoding="utf-8"))
    version = int(raw.get("version", 0))
    if version != MANIFEST_VERSION:
        raise ManifestError(
            f"{source.name} is manifest version {version}; this build reads "
            f"version {MANIFEST_VERSION}"
        )
    return [Entry.from_json(e) for e in raw["entries"]]


@dataclass(frozen=True)
class VerifyReport:
    """What `verify` found. Every discrepancy is reported, not just the first."""

    ok: tuple[str, ...]
    changed: tuple[tuple[str, str], ...]  # (path, what differs)
    missing: tuple[str, ...]
    untracked: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not (self.changed or self.missing)


def verify(root: Path, entries: list[Entry]) -> VerifyReport:
    """Re-hash what is on disk and compare it against the manifest.

    Untracked files are reported but do not fail the check: a data directory may hold
    something legitimately new. A file that *changed* or *vanished* does fail, because
    either one silently invalidates every result computed from it.
    """
    ok: list[str] = []
    changed: list[tuple[str, str]] = []
    missing: list[str] = []

    recorded = {e.path for e in entries}
    for entry in sorted(entries, key=lambda e: e.path):
        path = root / entry.path
        if not path.is_file():
            missing.append(entry.path)
            continue
        size = path.stat().st_size
        if size != entry.size_bytes:
            changed.append((entry.path, f"size {size} != {entry.size_bytes} recorded"))
            continue
        digest = sha256_of(path)
        if digest != entry.sha256:
            changed.append((entry.path, "same size, different content"))
            continue
        ok.append(entry.path)

    untracked = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).as_posix() not in recorded
    )
    return VerifyReport(
        ok=tuple(ok), changed=tuple(changed), missing=tuple(missing), untracked=tuple(untracked)
    )
