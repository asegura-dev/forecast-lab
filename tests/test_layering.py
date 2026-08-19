"""An executable guard for the dependency rule (ADR-001 sec. 4).

The hexagon's one hard rule - dependencies point inward, and `contracts` imports
nothing of ours - is worth far more as a test than as prose in a guide. A reviewer
forgets on a Friday; a gate does not.

It walks every module under `src/forecast_lab`, reads its imports **statically** (no
importing, so it stays fast and free of side effects) and fails naming the exact file
and the illegal arrow.

Two holes that the obvious implementation leaves open are closed here:

1. **Relative imports escape.** ``from .. import ingest`` parses as an ``ImportFrom``
   with ``module=None``, so an implementation that skips those nodes never sees it.
2. **Root modules are unguarded.** A helper that derives the layer from the first path
   component returns nothing for a module sitting at the package root, so such a file
   is neither checked nor checkable - which is precisely where a leak would hide.

A third rule the import table cannot express is checked separately: `research` may not
read from disk or the network. The table constrains which *modules* talk to each other;
it does not stop a module from hardcoding a path. The property that keeps an experiment
reproducible is about I/O, not imports.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = "forecast_lab"
SRC = Path(__file__).resolve().parents[1] / "src" / PACKAGE

# What each layer may import *of ours*. Anything not listed is a leak.
# Read it as the arrow direction: ingest may look inward at contracts, never outward
# at research.
ALLOWED: dict[str, frozenset[str]] = {
    "contracts": frozenset(),
    "ingest": frozenset({"contracts"}),
    "research": frozenset({"contracts"}),
    "interfaces": frozenset({"contracts", "ingest", "research"}),
}

# Names that mean "this module reaches for data". Matched on the syntax tree rather
# than on raw text, so prose in a docstring describing the rule cannot trip it.
IO_CALLS = frozenset({"open", "urlopen"})
IO_ATTRS = frozenset({"read_csv", "read_parquet", "read_json", "glob", "rglob", "iterdir", "get"})
IO_MODULES = frozenset({"requests", "httpx", "urllib", "pathlib", "csv", "sqlite3"})


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _layer_of(path: Path) -> str | None:
    """The layer a module belongs to, or ``None`` for the package root."""
    parts = path.relative_to(SRC).parts
    return parts[0] if len(parts) > 1 else None


def _io_name(node: ast.AST) -> str | None:
    """The I/O construct this node performs, or ``None``.

    Deliberately narrow. It catches the ordinary ways a module starts reading the
    world - a bare ``open``, a pandas reader, a path walk, an HTTP client import - and
    makes no attempt to be a sandbox. ``importlib`` would slip past it. It is a guard
    against drift and honest mistakes, not against determined circumvention.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in IO_CALLS:
            return f"{func.id}()"
        if isinstance(func, ast.Attribute) and func.attr in IO_ATTRS:
            return f".{func.attr}()"
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] in IO_MODULES:
                return f"import {alias.name}"
    if (
        isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module
        and node.module.split(".")[0] in IO_MODULES
    ):
        return f"from {node.module} import ..."
    return None


def _imported_layers(path: Path) -> set[tuple[str, int]]:
    """Every layer of ours that ``path`` imports, with the line number of the import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    own_parts = path.relative_to(SRC).parts[:-1]  # package containing this module
    found: set[tuple[str, int]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                # Absolute: `from forecast_lab.ingest import scan`
                targets = [node.module or ""]
            else:
                # Relative. `node.level` counts the leading dots: 1 is the current
                # package, 2 its parent, and so on.
                base = list(own_parts[: len(own_parts) - (node.level - 1)])
                if node.module:
                    targets = [".".join([PACKAGE, *base, node.module])]
                else:
                    # HOLE 1: `from .. import ingest` has module=None, so the imported
                    # name lives in the aliases, not in `node.module`.
                    targets = [".".join([PACKAGE, *base, alias.name]) for alias in node.names]
        else:
            continue

        # `lineno` is read here, where the node is known to be an import statement.
        for target in targets:
            bits = target.split(".")
            if bits and bits[0] == PACKAGE and len(bits) > 1 and bits[1] in ALLOWED:
                found.add((bits[1], node.lineno))

    return found


@pytest.mark.unit
def test_no_python_modules_at_the_package_root() -> None:
    """HOLE 2: a module at the package root is invisible to the layer check.

    `_layer_of` returns None for it, so it is neither attributed to a layer nor
    checkable against one. Rather than special-case it, the layout forbids it.
    """
    stray = [p.name for p in SRC.glob("*.py") if p.name != "__init__.py"]
    assert not stray, (
        f"modules at the package root are outside the layering guard: {stray}. "
        "Move each into a layer (ADR-001 sec. 4)."
    )


@pytest.mark.unit
def test_every_directory_is_a_declared_layer() -> None:
    """No package may exist that the dependency table says nothing about."""
    undeclared = [
        d.name
        for d in SRC.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name not in ALLOWED
    ]
    assert not undeclared, f"packages missing from ALLOWED: {undeclared}"


@pytest.mark.unit
@pytest.mark.parametrize("module", _modules(), ids=lambda p: str(p.name))
def test_dependencies_point_inward(module: Path) -> None:
    """No module imports a layer its own layer is not allowed to see."""
    layer = _layer_of(module)
    if layer is None:  # the package root; covered by its own test above
        return

    allowed = ALLOWED[layer]
    for imported, lineno in sorted(_imported_layers(module)):
        if imported == layer:
            continue
        assert imported in allowed, (
            f"{module.relative_to(SRC.parent.parent)}:{lineno} - "
            f"'{layer}' imports '{imported}', which is not allowed. "
            f"'{layer}' may import: {sorted(allowed) or 'nothing'}."
        )


@pytest.mark.unit
def test_research_never_reaches_for_data() -> None:
    """The rule the import table cannot express (ADR-001 sec. 4).

    Research receives frames; it does not go and get them. A research layer that can
    quietly re-read the world produces experiments that cannot be reproduced from
    stored inputs, and a result that cannot be reproduced cannot be falsified.
    """
    research = SRC / "research"
    if not research.is_dir():
        pytest.skip("the research layer does not exist yet")

    offenders: list[str] = []
    for module in sorted(research.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        where = module.relative_to(SRC)
        for node in ast.walk(tree):
            name = _io_name(node)
            if name is not None and isinstance(node, ast.stmt | ast.expr):
                offenders.append(f"{where}:{node.lineno} uses {name}")

    assert not offenders, (
        "research must not read from disk or the network; the CLI loads the data and "
        f"passes frames in (ADR-001 sec. 4). Found: {offenders}"
    )
