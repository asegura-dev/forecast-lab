"""Tests for the record of results this repository has already computed.

The one that matters is `test_a_result_for_another_series_is_never_offered`. Serving gold's
verdict to a reader who asked about the S&P would put a number on screen under a caption
that does not describe it - which is the failure this whole repository is a correction of,
arriving through a convenience. `find` returns `None` rather than a near miss, and this
file is where that stays true.

The second is `test_every_committed_payload_is_recognised`: the signatures below are
matched against the **real** sidecars, so a renamed payload key fails here rather than
silently making a published result unfindable and every panel slow again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forecast_lab.interfaces.published import (
    SIGNATURES,
    available_for,
    catalogue,
    command_of,
    find,
)

ROOT = Path(__file__).resolve().parents[2]


def _write(directory: Path, name: str, payload: dict[str, object]) -> None:
    (directory / "docs" / "status").mkdir(parents=True, exist_ok=True)
    (directory / "docs" / "status" / name).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A tree with two committed results that differ only in what they describe."""
    _write(
        tmp_path,
        "gold.json",
        {
            "symbol": "XAUUSD", "timeframe": "1H", "mode": "focus",
            "generated_at": "2026-08-30T00:00:00+00:00",
            "skill": {}, "profit": {}, "multiplicity": {},
        },
    )
    _write(
        tmp_path,
        "gold-features.json",
        {
            "target": "XAUUSD", "timeframe": "1H", "mode": "whole",
            "policy_passes": True, "features": [],
        },
    )
    return tmp_path


# --- the safety property ------------------------------------------------------------------


@pytest.mark.unit
def test_a_result_for_another_series_is_never_offered(repository: Path) -> None:
    """`None`, not the nearest thing. A published figure for gold must not answer a
    question about anything else."""
    assert find(repository, "verdict", target="SPX", timeframe="1H") is None
    assert find(repository, "verdict", target="XAUUSD", timeframe="4H") is None


@pytest.mark.unit
def test_a_result_for_another_mode_is_never_offered(repository: Path) -> None:
    """`focus` and `whole` are different experiments, not different views of one."""
    assert find(repository, "features", target="XAUUSD", timeframe="1H", mode="whole")
    assert find(repository, "features", target="XAUUSD", timeframe="1H", mode="focus") is None


@pytest.mark.unit
def test_a_command_with_no_committed_result_returns_nothing(repository: Path) -> None:
    assert find(repository, "train", target="XAUUSD", timeframe="1H") is None


@pytest.mark.unit
def test_the_symbol_is_matched_without_regard_to_case(repository: Path) -> None:
    assert find(repository, "verdict", target="xauusd", timeframe="1h") is not None


# --- the contract against the real sidecars -------------------------------------------------


@pytest.mark.unit
def test_every_committed_payload_is_recognised() -> None:
    """The signatures are matched against what the repository actually ships.

    A renamed payload key would not raise anywhere - it would quietly make a published
    result unfindable, and every panel would go back to taking minutes with nobody knowing
    why. So the check is against the files, not against a fixture.
    """
    found = catalogue(ROOT)

    assert found, "no committed payload was recognised at all"
    assert {entry.command for entry in found} >= {
        "verdict", "validate", "train", "features", "explore", "baseline"
    }


@pytest.mark.unit
def test_the_headline_payloads_are_findable_for_the_series_findings_quotes() -> None:
    """FINDINGS quotes gold at one hour in focus mode; those must resolve."""
    for command in ("verdict", "validate", "train", "features"):
        entry = find(ROOT, command, target="XAUUSD", timeframe="1H", mode="focus")
        assert entry is not None, f"no committed {command} for the series FINDINGS quotes"
        assert entry.payload


@pytest.mark.unit
def test_no_two_commands_share_a_signature() -> None:
    """An ambiguous discriminator would label a panel with the wrong command's name."""
    for name, signature in SIGNATURES.items():
        others = [s for other, s in SIGNATURES.items() if other != name]
        assert not any(signature <= other for other in others), f"{name} is not distinctive"


# --- the mechanics ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_payload_that_belongs_to_no_command_is_skipped(tmp_path: Path) -> None:
    """The data manifest lives in the same directory and is not a result."""
    _write(tmp_path, "data-manifest.json", {"version": 1, "entries": []})

    assert catalogue(tmp_path) == []


@pytest.mark.unit
def test_an_unreadable_file_does_not_break_the_catalogue(tmp_path: Path) -> None:
    """A research log gains new kinds of file; one of them being unparseable is not a
    reason for every panel to lose its published results."""
    (tmp_path / "docs" / "status").mkdir(parents=True)
    (tmp_path / "docs" / "status" / "broken.json").write_text("{not json", encoding="utf-8")
    _write(tmp_path, "ok.json", {"symbol": "XAUUSD", "timeframe": "1H",
                                 "skill": {}, "profit": {}, "multiplicity": {}})

    assert [entry.command for entry in catalogue(tmp_path)] == ["verdict"]


@pytest.mark.unit
def test_a_tree_without_the_directory_yields_nothing(tmp_path: Path) -> None:
    assert catalogue(tmp_path) == []


@pytest.mark.unit
def test_available_for_lists_only_what_this_series_has(repository: Path) -> None:
    assert available_for(repository, target="XAUUSD", timeframe="1H") == {"verdict", "features"}
    assert available_for(repository, target="SPX", timeframe="1H") == frozenset()


@pytest.mark.unit
def test_the_provenance_says_it_is_committed_rather_than_fresh(repository: Path) -> None:
    """The hazard of offering both is a reader who cannot tell which they have."""
    entry = find(repository, "verdict", target="XAUUSD", timeframe="1H")

    assert entry is not None
    assert "Committed result" in entry.provenance
    assert "not recomputed now" in entry.provenance
    assert entry.name in entry.provenance


@pytest.mark.unit
@pytest.mark.parametrize("command", sorted(SIGNATURES))
def test_each_signature_identifies_its_own_command(command: str) -> None:
    payload: dict[str, object] = {key: {} for key in SIGNATURES[command]}

    assert command_of(payload) == command


@pytest.mark.unit
def test_a_payload_matching_nothing_is_unidentified() -> None:
    assert command_of({"something": 1}) is None
