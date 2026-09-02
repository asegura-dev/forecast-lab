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
# `.get` is deliberately absent: `dict.get` is everywhere, and flagging it would train
# everyone to ignore this guard. HTTP clients are caught by their import instead.
IO_ATTRS = frozenset(
    {
        "read_csv",
        "read_parquet",
        "read_json",
        "glob",
        "rglob",
        "iterdir",
        # `research/plots.py` builds figures and hands them back; writing one is the
        # CLI's job, so a stray savefig here would be the layer reaching for disk
        # under a name the original list did not anticipate.
        "savefig",
    }
)
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


#: The one module allowed to import `ta`. It ships no type information, so every symbol
#: crossing this boundary arrives as `Any` - and `Any` spreading through a codebase takes
#: the type checker's guarantees with it wherever it goes.
TA_QUARANTINE = "research/features/technical.py"

#: The untyped modelling stack, confined to one package for the same reason.
ESTIMATOR_PACKAGES = frozenset({"sklearn", "lightgbm", "xgboost"})

#: matplotlib is untyped too, and is confined to the module that builds figures.
PLOT_QUARANTINE = "research/plots.py"
ESTIMATOR_QUARANTINE = "research/models/"


@pytest.mark.unit
def test_the_untyped_indicator_library_stays_in_one_module() -> None:
    """`ta` is quarantined (ADR-006, Consequences).

    `mypy --strict` is one of the three gates, and it is only worth having while the
    untyped surface is small enough to audit. Confining `ta` to a single wrapper means
    the rest of `research` is checked for real rather than nominally.
    """
    offenders: list[str] = []
    for module in sorted(SRC.rglob("*.py")):
        where = module.relative_to(SRC).as_posix()
        if where.endswith(TA_QUARANTINE):
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Import) and any(
                a.name == "ta" or a.name.startswith("ta.") for a in node.names
            )) or (isinstance(node, ast.ImportFrom) and (
                node.module == "ta" or (node.module or "").startswith("ta.")
            )):
                offenders.append(f"{where}:{node.lineno}")

    assert not offenders, (
        f"`ta` may only be imported by {TA_QUARANTINE} (ADR-006, Consequences). "
        f"Found: {offenders}"
    )


@pytest.mark.unit
def test_the_untyped_estimator_stack_stays_in_one_package() -> None:
    """sklearn, lightgbm and xgboost are confined to `research/models/`.

    Same argument as `ta`: none ships type information, so every symbol crossing that
    boundary arrives as `Any`. Confining them keeps the rest of the codebase genuinely
    checked rather than nominally so.

    `metrics.py` imports `roc_auc_score` and `training.py` imports the Pipeline, so the
    quarantine is a package rather than a single module - but it is still one place.
    """
    offenders: list[str] = []
    for module in sorted(SRC.rglob("*.py")):
        where = module.relative_to(SRC).as_posix()
        if ESTIMATOR_QUARANTINE in where:
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in ESTIMATOR_PACKAGES:
                    offenders.append(f"{where}:{node.lineno} imports {name}")

    assert not offenders, (
        f"the estimator stack may only be imported under {ESTIMATOR_QUARANTINE} "
        f"(ADR-007). Found: {offenders}"
    )


@pytest.mark.unit
def test_matplotlib_stays_in_the_plotting_module() -> None:
    """Figures are built in one place (ADR-008).

    Same argument as `ta` and the estimator stack: matplotlib ships no type information.
    Confining it also keeps a second, quieter property true - only one module needs the
    `Agg` backend dance, so no other import can accidentally probe for a display.
    """
    offenders: list[str] = []
    for module in sorted(SRC.rglob("*.py")):
        where = module.relative_to(SRC).as_posix()
        if where.endswith(PLOT_QUARANTINE) or where.endswith("interfaces/cli.py"):
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] == "matplotlib" for n in names):
                offenders.append(f"{where}:{node.lineno}")

    assert not offenders, f"matplotlib belongs in {PLOT_QUARANTINE}. Found: {offenders}"


#: `SRC` already ends in the package name, so the interfaces package is one level below
#: it. Spelled out because getting this wrong made an earlier version of the guard scan
#: an empty directory and pass vacuously - caught only by deliberately planting a
#: violation and watching it stay green.
INTERFACES = SRC / "interfaces"

#: The command-line entry points: the composition root and the `python -m` shim that
#: forwards to it. These are the only modules under `interfaces/` allowed to reach the
#: analysis layer; everything else in the package belongs to the dashboard.
ENTRY_POINTS = ("interfaces/cli.py", "interfaces/__main__.py")

#: What a dashboard module may import from this package. `runner` and `presentation` import
#: nothing at all; `dashboard` may import those two.
DASHBOARD_IMPORTS: dict[str, set[str]] = {
    "interfaces/runner.py": set(),
    "interfaces/presentation.py": set(),
    "interfaces/dashboard.py": {
        "forecast_lab.interfaces.runner",
        "forecast_lab.interfaces.presentation",
    },
}


@pytest.mark.unit
def test_the_dashboard_cannot_reach_the_analysis_layer() -> None:
    """ADR-005 sec. 1, made executable - and scoped by **default deny**.

    The first version listed the two files to check by name. That is the wrong direction,
    and an audit found the walk-around: Streamlit discovers a `pages/` directory beside the
    entrypoint and renders each file in it as a page of this very app. A
    `interfaces/pages/verdict.py` importing `research` would need **no import statement in
    `dashboard.py`** to be reached, so a filename allow-list would never look at it and
    every gate would stay green while the decision was dead.

    So the rule is inverted: every module under `interfaces/` except the composition root
    is a dashboard module, and an unknown one may import nothing from the package until
    someone adds it to `DASHBOARD_IMPORTS` on purpose.

    **What this does not catch**, stated for the same reason `_io_name` states it:
    `importlib.import_module`, `__import__` and `sys.modules[...]` all slip past an
    import-statement check. It is a guard against drift and honest mistakes, not against
    determined circumvention.
    """
    offenders: list[str] = []
    for module in sorted(INTERFACES.rglob("*.py")):
        where = module.relative_to(SRC).as_posix()
        if where.endswith(ENTRY_POINTS) or module.name == "__init__.py":
            continue
        allowed = DASHBOARD_IMPORTS.get(
            next((k for k in DASHBOARD_IMPORTS if where.endswith(k)), ""), set()
        )
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif node.level:
                imported = ["a relative import, which escapes this check"]
            else:
                imported = [node.module or ""]
            for name in imported:
                reaches = name.startswith("forecast_lab") and name not in allowed
                if reaches or name.startswith("a relative"):
                    offenders.append(f"{where}:{node.lineno} -> {name}")

    assert not offenders, (
        "every module under interfaces/ except the composition root runs the CLI as a "
        f"subprocess and may not import the analysis layer (ADR-005 sec. 1). Found: {offenders}"
    )


@pytest.mark.unit
def test_a_new_module_under_interfaces_is_denied_by_default() -> None:
    """The property that makes the guard above worth having.

    A file nobody thought about - `pages/verdict.py`, most plausibly - gets an empty
    allow-list rather than being skipped, so its first import from the package fails the
    gate instead of passing unseen.
    """
    known = set(DASHBOARD_IMPORTS) | set(ENTRY_POINTS)
    present = {
        m.relative_to(SRC).as_posix()
        for m in INTERFACES.rglob("*.py")
        if m.name != "__init__.py"
    }
    unlisted = {p for p in present if not any(p.endswith(k) for k in known)}

    assert not unlisted, (
        f"{sorted(unlisted)} is under interfaces/ and not in DASHBOARD_IMPORTS. Add it "
        "with the imports it is allowed, rather than letting the guard skip it."
    )


@pytest.mark.unit
def test_the_dashboard_cannot_run_a_command_that_writes() -> None:
    """ADR-005 sec. 4: read-only commands only.

    Asserted against the allow-list itself rather than against behaviour, because the
    failure this guards against is someone adding `fetch` to it for a convenience button.
    """
    from forecast_lab.interfaces.runner import JSON_CAPABLE, READ_ONLY, WRITING

    assert {"fetch", "ingest"} == WRITING
    assert not (READ_ONLY & WRITING), "a writing command reached the allow-list"
    assert JSON_CAPABLE <= READ_ONLY, "a JSON-capable command is not in the allow-list"
