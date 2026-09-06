"""Tests for provenance.

The manifest exists to answer one question: is the data on disk still the data that
produced the published numbers? These tests pin that it answers honestly - especially in
the case that matters, where a file changed but its size did not.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from forecast_lab.contracts import SeriesMeta, SymbolSpec, Timeframe
from forecast_lab.ingest import manifest

XAU = SymbolSpec(name="XAUUSD")


def _meta(rows: int = 3) -> SeriesMeta:
    return SeriesMeta(
        symbol=XAU,
        timeframe=Timeframe.H1,
        rows=rows,
        first=datetime(2022, 1, 3, 0, tzinfo=UTC),
        last=datetime(2022, 1, 3, 2, tzinfo=UTC),
        anchors=frozenset({0}),
    )


def _series(root: Path, name: str = "XAUUSD_1H.csv", body: str = "a,b,c\n1,2,3\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(body, encoding="utf-8")
    return path


# --- hashing ------------------------------------------------------------------------


@pytest.mark.unit
def test_the_same_data_always_writes_the_same_file(tmp_path: Path) -> None:
    """No generation timestamp, so re-running `ingest` over unchanged data is a no-op.

    Otherwise the committed manifest would show a diff every time somebody checked that
    nothing had broken - and a file that always looks changed stops being read.
    """
    root = tmp_path / "data"
    entry = manifest.entry_for(_series(root), root, _meta(), source="reference")
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    manifest.write(first, [entry])
    manifest.write(second, [entry])
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.unit
def test_the_hash_is_stable_and_content_dependent(tmp_path: Path) -> None:
    a = _series(tmp_path, "a.csv", "same\n")
    b = _series(tmp_path, "b.csv", "same\n")
    c = _series(tmp_path, "c.csv", "different\n")

    assert manifest.sha256_of(a) == manifest.sha256_of(b)
    assert manifest.sha256_of(a) != manifest.sha256_of(c)


# --- round trip ---------------------------------------------------------------------


@pytest.mark.unit
def test_write_then_read_preserves_every_field(tmp_path: Path) -> None:
    root = tmp_path / "data"
    path = _series(root)
    entry = manifest.entry_for(path, root, _meta(), source="reference")

    destination = tmp_path / "docs" / "data-manifest.json"
    manifest.write(destination, [entry])

    assert manifest.read(destination) == [entry]


@pytest.mark.unit
def test_entries_are_sorted_on_disk(tmp_path: Path) -> None:
    """An unsorted manifest would show a diff on every regeneration.

    The order would follow directory iteration, which is not reproducible - and a file
    that always looks changed is a file nobody reads.
    """
    root = tmp_path / "data"
    entries = [
        manifest.entry_for(_series(root, name), root, _meta(), source="reference")
        for name in ("SPX_1H.csv", "BTCUSD_1D.csv", "XAUUSD_4H.csv")
    ]
    destination = tmp_path / "m.json"
    manifest.write(destination, entries)

    paths = [e.path for e in manifest.read(destination)]
    assert paths == sorted(paths)


@pytest.mark.unit
def test_paths_are_stored_with_forward_slashes(tmp_path: Path) -> None:
    """A manifest written on Windows must verify on Linux."""
    root = tmp_path / "data"
    nested = root / "sub"
    entry = manifest.entry_for(_series(nested), root, _meta(), source="reference")
    assert "\\" not in entry.path
    assert entry.path == "sub/XAUUSD_1H.csv"


@pytest.mark.unit
def test_a_future_manifest_version_is_refused(tmp_path: Path) -> None:
    """Reading a format this build does not understand would silently mis-verify."""
    destination = tmp_path / "m.json"
    destination.write_text('{"version": 999, "entries": []}', encoding="utf-8")
    with pytest.raises(manifest.ManifestError, match="version"):
        manifest.read(destination)


@pytest.mark.unit
def test_a_missing_manifest_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(manifest.ManifestError, match="ingest"):
        manifest.read(tmp_path / "absent.json")


# --- verification -------------------------------------------------------------------


@pytest.mark.unit
def test_untouched_data_verifies_clean(tmp_path: Path) -> None:
    root = tmp_path / "data"
    entry = manifest.entry_for(_series(root), root, _meta(), source="reference")

    report = manifest.verify(root, [entry])
    assert report.is_clean
    assert report.ok == ("XAUUSD_1H.csv",)


@pytest.mark.unit
def test_a_same_size_edit_is_caught(tmp_path: Path) -> None:
    """The case a size or row check would miss, and the reason the hash exists.

    Editing one digit of one price leaves the file byte-length identical. Every number
    computed from it changes; nothing about the file announces that.
    """
    root = tmp_path / "data"
    path = _series(root, body="a,b,c\n1,2,3\n")
    entry = manifest.entry_for(path, root, _meta(), source="reference")

    path.write_text("a,b,c\n1,2,9\n", encoding="utf-8")  # same length, different content
    report = manifest.verify(root, [entry])

    assert not report.is_clean
    assert report.revised[0][0] == "XAUUSD_1H.csv"
    assert "different content" in report.revised[0][1]


@pytest.mark.unit
def test_appending_bars_is_an_extension_not_a_revision(tmp_path: Path) -> None:
    """The case that made this distinction necessary.

    The series extends forward in time, so anyone who runs `fetch` after the manifest was
    cut has longer files than it records. Reported as "changed - no longer reproducible",
    that message arrives for every operator on every fetch, and a warning everyone sees
    every time is one nobody reads. Nothing published is invalidated: the recorded bytes
    are still there, at the front of the file.
    """
    root = tmp_path / "data"
    path = _series(root, body="a,b,c\n1,2,3\n")
    entry = manifest.entry_for(path, root, _meta(rows=1), source="dukascopy")

    path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")  # two more bars arrived
    report = manifest.verify(root, [entry])

    assert not report.is_clean
    assert report.is_only_extended
    assert report.revised == ()
    assert report.extended[0][0] == "XAUUSD_1H.csv"
    assert "appended" in report.extended[0][1]


@pytest.mark.unit
def test_a_file_that_grew_but_rewrote_its_past_is_revised(tmp_path: Path) -> None:
    """The one the distinction must not let through.

    A venue that appends two days and a venue that appends two days *and quietly restates
    a bar from last March* both produce a longer file. Only the second means a published
    number came from bytes that no longer exist anywhere, and length alone cannot tell
    them apart - which is why the check re-hashes the recorded prefix instead of trusting
    that bigger means append-only.
    """
    root = tmp_path / "data"
    path = _series(root, body="a,b,c\n1,2,3\n")
    entry = manifest.entry_for(path, root, _meta(rows=1), source="dukascopy")

    path.write_text("a,b,c\n1,2,9\n4,5,6\n", encoding="utf-8")  # longer, and the past moved
    report = manifest.verify(root, [entry])

    assert not report.is_clean
    assert not report.is_only_extended
    assert report.extended == ()
    assert report.revised[0][0] == "XAUUSD_1H.csv"


@pytest.mark.unit
def test_a_truncated_file_is_never_an_extension(tmp_path: Path) -> None:
    """Shorter is not longer, and a download that stopped early must not read as benign."""
    root = tmp_path / "data"
    path = _series(root, body="a,b,c\n1,2,3\n4,5,6\n")
    entry = manifest.entry_for(path, root, _meta(rows=2), source="dukascopy")

    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    report = manifest.verify(root, [entry])

    assert report.extended == ()
    assert report.revised[0][0] == "XAUUSD_1H.csv"


@pytest.mark.unit
def test_a_deleted_file_is_caught(tmp_path: Path) -> None:
    root = tmp_path / "data"
    path = _series(root)
    entry = manifest.entry_for(path, root, _meta(), source="reference")
    path.unlink()

    report = manifest.verify(root, [entry])
    assert report.missing == ("XAUUSD_1H.csv",)
    assert not report.is_clean


@pytest.mark.unit
def test_an_untracked_file_is_reported_but_does_not_fail(tmp_path: Path) -> None:
    """A new file may be legitimate; a changed or vanished one never is.

    Failing on anything unrecognised would make the check useless the first time
    somebody drops a note in the data directory.
    """
    root = tmp_path / "data"
    entry = manifest.entry_for(_series(root), root, _meta(), source="reference")
    _series(root, "SPX_1H.csv")

    report = manifest.verify(root, [entry])
    assert report.is_clean
    assert report.untracked == ("SPX_1H.csv",)


@pytest.mark.unit
def test_every_problem_is_reported_not_just_the_first(tmp_path: Path) -> None:
    """Three round trips to learn about three problems is three too many."""
    root = tmp_path / "data"
    entries = [
        manifest.entry_for(_series(root, name), root, _meta(), source="reference")
        for name in ("a.csv", "b.csv", "c.csv")
    ]
    (root / "a.csv").write_text("edited but same size\n", encoding="utf-8")
    (root / "b.csv").unlink()

    report = manifest.verify(root, entries)
    assert len(report.revised) == 1
    assert len(report.missing) == 1
    assert report.ok == ("c.csv",)
