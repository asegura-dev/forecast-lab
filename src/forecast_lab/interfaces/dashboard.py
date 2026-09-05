"""The dashboard, which runs the command line rather than reimplementing it.

    uv sync --extra dashboard
    uv run forecast-lab dashboard

Every panel builds an argument list, runs `forecast-lab <command> --json` as a subprocess,
parses the payload, and renders it - with the command shown beside the result, so what is
on screen can be reproduced in a terminal by copying one line. That is ADR-005, decided in
Phase 1 precisely so that this file could not become a second implementation of the
analysis.

**This module holds layout and nothing else.** Running commands is `runner`; turning
numbers into strings is `presentation`. Both are Streamlit-free and therefore testable
without a browser, which is not a preference: the first version formatted inside
`st.markdown` calls, and a defect that printed four different p-values as `0.0000` hid
there for exactly as long as that was true.

**It imports nothing from `forecast_lab` except those two**, neither of which imports
anything from the package at all. The layering suite enforces it across the whole
`interfaces` package rather than by filename, so a `pages/` directory - Streamlit's own
idiomatic way to grow an app, and one that needs no import here to be discovered - cannot
walk around the rule.

**Results are cached on the manifest hash**, because ADR-002 already makes that hash the
identity of the inputs: when the data changes the manifest changes and every cached answer
expires. Failures are cached too, so a panel that cannot run does not re-pay its subprocess
on every widget click.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

from forecast_lab.interfaces.presentation import (
    Style,
    accuracy,
    bar_spec,
    cell,
    lines_spec,
    markdown_table,
    percent,
    rows_from,
    scatter_spec,
    shell_path,
    significance,
    unavailable_note,
)
from forecast_lab.interfaces.published import Published, available_for, find
from forecast_lab.interfaces.runner import Invocation, RunnerError, run

#: Repository root, four levels up from this file.
ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "docs" / "status" / "data-manifest.json"
FIGURES = ROOT / "docs" / "status" / "figures"
DATA_ROOTS = ("data/raw", "data/reference")
MODES = ("focus", "whole")

#: Lifted verbatim from FINDINGS.md rather than paraphrased, for the same reason the
#: Findings page renders that document instead of summarising it: a paraphrase is a second
#: copy of a number, and a second copy drifts.
GLOSSARY = {
    "independent": (
        "What accuracy the two marginals produce on their own, with no information passing "
        "between them. A predictor that always says UP on a series rising 52% of the time "
        "scores 52% and knows nothing - against this benchmark it scores exactly zero."
    ),
    "break-even": (
        "The accuracy this model needs to cover ITS OWN turnover. A persistent model trades "
        "less often and faces a lower bar."
    ),
    "flip rate": "The share of bars on which the position changes. Every change pays the spread.",
    "survives Holm": (
        "Whether the p-value survives correcting for having tried every configuration. The "
        "maximum of eighteen draws from noise is not centred on zero."
    ),
    "cumulative": (
        "What one unit of capital became over the scored period, reinvested each bar and "
        "net of costs. It cannot pass -100%: this entry used to explain why a figure did, "
        "which was a defect being described rather than fixed."
    ),
    "inflation": (
        "How much of a correlation is shared trend: the level correlation minus the return "
        "correlation, in absolute value."
    ),
}


# --- running commands ------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """What a panel got back. Carries the failure rather than raising it, so a failed
    command can be cached like a successful one and not re-run on every rerun."""

    payload: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        # `is None`, never truthiness: an empty dict or an empty string is a legitimate
        # answer from a command that ran correctly.
        return self.error is None and self.payload is not None


def manifest_fingerprint() -> str:
    """Identity of the inputs, as ADR-002 defines it - and the cache key for every panel."""
    if not MANIFEST.is_file():
        return "no-manifest"
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def _invoke(
    command: str,
    options: tuple[tuple[str, str], ...],
    flags: tuple[str, ...],
    fingerprint: str,
) -> Outcome:
    """Run one command, keyed by what it was asked and what data it read.

    The Invocation is rebuilt from its own constructor arguments, never reparsed from an
    argument list - an earlier version did that and silently re-derived `--json`, so the
    line displayed could differ from the line that ran.

    `cwd=ROOT` because every path here is relative: without it, launching the server from
    any other directory leaves the page looking healthy while every panel reports no data.
    """
    del fingerprint  # a cache key, not an input
    try:
        return Outcome(payload=run(Invocation(command, options, flags), cwd=str(ROOT)))
    except RunnerError as exc:
        return Outcome(error=str(exc))


def option_of(invocation: Invocation, name: str) -> str | None:
    """One option's value, for matching a request against a committed result."""
    return next((value for key, value in invocation.options if key == name), None)


def committed(invocation: Invocation) -> Published | None:
    """The published result for exactly this request, if the repository has one."""
    target = option_of(invocation, "--target")
    timeframe = option_of(invocation, "--timeframe")
    if not target or not timeframe:
        return None
    return find(
        ROOT,
        invocation.command,
        target=target,
        timeframe=timeframe,
        mode=option_of(invocation, "--mode"),
    )


#: The sidebar's answer to "published or fresh", read where it is needed rather than
#: threaded through five page signatures. Streamlit's session state is the framework's
#: own idiom for this, and passing a boolean down every call would be noise that hides
#: the one place it matters.
PUBLISHED_FIRST = "published_first"


def prefers_published() -> bool:
    return bool(st.session_state.get(PUBLISHED_FIRST, True))


def execute(
    invocation: Invocation, *, label: str | None = None, prefer_published: bool | None = None
) -> Any | None:
    """Render one command's result, from the record or from a fresh run.

    **The record is the nine payloads committed under `docs/status/`** - the ones the STATUS
    logs and FINDINGS quote. The dashboard had them on disk and re-ran every command anyway,
    which cost two minutes for a verdict page and four with every configuration, on numbers
    that were already there.

    Where a published result exists it is served instantly and **labelled as committed**,
    because the hazard of offering both is a reader who cannot tell which they have. Where
    none exists - another symbol, another mode - the panel runs the command and says so
    rather than serving a near miss.
    """
    if prefer_published is None:
        prefer_published = prefers_published()
    if prefer_published:
        entry = committed(invocation)
        if entry is not None:
            st.caption(entry.provenance)
            command_line(invocation)
            return entry.payload
    try:
        with st.spinner(label or f"Running `{invocation.display}`"):
            outcome = _invoke(
                invocation.command,
                invocation.options,
                invocation.flags,
                manifest_fingerprint(),
            )
    except RunnerError as exc:
        # Construction refusals - a writing command, a writing option - reach here rather
        # than becoming a Streamlit traceback, so the runner's worded refusal is what the
        # reader sees.
        st.error(str(exc))
        return None
    command_line(invocation)
    if not outcome.ok:
        st.error(outcome.error)
        return None
    return outcome.payload


def command_line(invocation: Invocation) -> None:
    """The line that produced the result beside it (ADR-005 sec. 2)."""
    st.caption("Produced by")
    st.code(shell_path(invocation.display), language="bash")


def text_panel(invocation: Invocation, *, success_marker: str | None = None) -> None:
    """A command that reports to a human rather than emitting a payload.

    `verify` and `symbols` have no `--json`; their answer *is* the rendered text. An earlier
    version called `execute` and discarded the return value, so both panels showed a
    command line and nothing else - and the `verify` panel, which the module docstring
    names as the safety net for the one gap caching cannot cover, showed nothing at all.
    """
    output = execute(invocation)
    if output is None:
        return
    body = str(output).strip()
    if success_marker and body.startswith(success_marker):
        st.success(body.splitlines()[0])
    else:
        st.code(body, language="text")


# --- rendering helpers -------------------------------------------------------------------


@lru_cache(maxsize=1)
def arrow_available() -> bool:
    """Whether `st.dataframe` can render on this machine.

    It could not, for two days: `pyarrow` ships a native library and Windows Smart App
    Control blocks an unsigned binary until its release has accrued reputation. An earlier
    version of this project concluded from five retries that it never clears; it cleared
    overnight, which is the same mechanism already recorded for LightGBM and XGBoost.

    So the capability is *probed* rather than assumed in either direction, once per process.
    """
    # `find_spec` rather than an import: it raises the same OSError when the policy
    # blocks the native library, and it does not drag an untyped module into a
    # `mypy --strict` run for a capability probe.
    try:
        return importlib.util.find_spec("pyarrow") is not None
    except (ImportError, OSError, ValueError):
        return False


def table(rows: list[dict[str, Any]], styles: dict[str, str] | None = None) -> None:
    """A sortable grid where the machine can render one, a Markdown table where it cannot.

    Both paths format through `presentation`, so the two cannot disagree about how a
    p-value is written - which is the failure that made this module split necessary.
    """
    if not rows:
        st.caption("Nothing to show.")
        return
    styles = styles or {}
    if not arrow_available():
        st.markdown(markdown_table(rows, styles))
        return
    # Pre-formatted strings rather than column_config: the formatting rules live in one
    # place, and a sortable grid over strings still sorts a percentage column correctly
    # because they share a width.
    st.dataframe(
        [{column: cell(row.get(column), styles.get(column, Style.TEXT)) for column in rows[0]}
         for row in rows],
        width="stretch",
        hide_index=True,
    )


def figure(name: str, caption: str) -> None:
    """A committed chart, labelled with the fact that it is pinned.

    The PNGs in `docs/status/figures/` were produced by `train --figures` and `explore
    --figures` against the 2026-08-26 snapshot, while every table around them is computed
    live. Showing a pinned image beside a fresh table without saying so is exactly the
    drift this project exists to correct, so the caption always says which it is.
    """
    path = FIGURES / name
    if not path.is_file():
        return
    st.image(str(path), caption=f"{caption} - committed figure, pinned to the 2026-08-26 snapshot")


def chart(spec: dict[str, Any], caption: str | None = None) -> None:
    """Draw a spec that `presentation` built.

    The spec is a dict, so building one is pure data and lives beside the formatting where
    a test can check it - including the property that keeps it off the Arrow path.
    """
    st.vega_lite_chart(spec, width="stretch")
    if caption:
        st.caption(caption)


def glossary(*terms: str) -> None:
    """What the columns mean, for a reader who has not read thirteen ADRs."""
    with st.expander("How to read this"):
        for term in terms:
            st.markdown(f"**{term}** - {GLOSSARY[term]}")


# --- sidebar -----------------------------------------------------------------------------


def dataset_choice() -> str | None:
    """Which dataset a page reads. Keyed, so the choice survives a visit to another page."""
    options = [d for d in DATA_ROOTS if (ROOT / d).is_dir()]
    if not options:
        st.sidebar.warning("No `data/` directory.")
        return None
    return st.sidebar.selectbox(
        "Dataset",
        options,
        key="dataset",
        help="`data/raw` is the canonical Dukascopy series; `data/reference` holds the "
        "original project's exports and carries no spread column, so break-even falls "
        "back to an assumed cost.",
    )


def series_choice(directory: str) -> tuple[str, str] | None:
    """Symbol and interval, both read off disk.

    The interval used to be a hard-coded `("1H","4H","1D")` while the symbol was globbed,
    so two of three choices produced a red error on a dataset holding only hourly bars.
    """
    files = sorted((ROOT / directory).glob("*_*.csv"))
    symbols = sorted({f.stem.rsplit("_", 1)[0] for f in files})
    if not symbols:
        st.sidebar.info(f"No series in `{directory}`.")
        return None
    # Seeded once rather than passed as `index=`: a keyed widget that also carries a
    # default is ambiguous to Streamlit, and `keep_controls` writes that key on every run.
    if st.session_state.get("symbol") not in symbols:
        st.session_state["symbol"] = "XAUUSD" if "XAUUSD" in symbols else symbols[0]
    symbol = st.sidebar.selectbox("Target", symbols, key="symbol")
    intervals = sorted({f.stem.rsplit("_", 1)[1] for f in files if f.stem.startswith(f"{symbol}_")})
    choices = intervals or ["1H"]
    if st.session_state.get("timeframe") not in choices:
        st.session_state["timeframe"] = choices[0]
    timeframe = st.sidebar.selectbox("Timeframe", choices, key="timeframe")
    return symbol, timeframe


def source_choice(target: str, timeframe: str) -> None:
    """Published results or a fresh run.

    Offered only where the repository actually has a committed result for this series -
    there are nine, all for gold at one hour, and offering the option on a symbol with none
    would promise something every panel then failed to deliver.
    """
    if not available_for(ROOT, target=target, timeframe=timeframe):
        st.session_state[PUBLISHED_FIRST] = False
        st.sidebar.caption(
            f"No committed result for {target} at {timeframe}, so every panel runs its "
            "command. The published ones cover gold at one hour."
        )
        return
    st.sidebar.selectbox(
        "Results",
        ("Published - instant", "Run now - live"),
        key="source",
        help="Published reads the payloads committed under `docs/status/` - the numbers the "
        "STATUS logs and FINDINGS quote, pinned to the snapshot the manifest describes. "
        "Run now recomputes, which takes two to four minutes on the Verdict page.",
    )
    st.session_state[PUBLISHED_FIRST] = st.session_state["source"].startswith("Published")


def breadth_choice() -> bool:
    """Eighteen configurations or six.

    `validate` defaults `--pca` off and `verdict` defaults it on, which put a six-row table
    above an eighteen-row one with no way to reconcile them. Both now take the same answer -
    and because the honest default is the published one, the cost is stated rather than
    hidden: eighteen configurations is three times the work, about four minutes cold.
    Cached on the manifest hash, so it is paid once per version of the data.
    """
    return (
        st.sidebar.selectbox(
            "Configurations",
            ("18 - as published", "6 - raw features only, faster"),
            key="breadth",
            help="PCA at 95% and 90% of training variance, beside the raw features. Every "
            "PCA row scores below its raw counterpart and trades more often.",
        )
        .startswith("18")
    )


def mode_choice() -> str:
    """Only shown by the pages that pass it. `align` and `explore` take no `--mode`, and a
    control that visibly does nothing costs trust."""
    return st.sidebar.selectbox(
        "Mode", MODES, key="mode", help="focus: the target only. whole: every symbol."
    )


def options_for(symbol: str, timeframe: str, directory: str) -> tuple[tuple[str, str], ...]:
    return (("--target", symbol), ("--timeframe", timeframe), ("--dir", directory))


# --- pages ---------------------------------------------------------------------------------


def page_findings() -> None:
    """The published document, verbatim.

    No paraphrase and no hard-coded figure: an earlier version opened with a sentence
    copied out of FINDINGS.md, which then disagreed with the Verdict page two clicks away
    once the data moved - nine of eighteen in prose against thirteen computed.
    """
    document = ROOT / "FINDINGS.md"
    if not document.is_file():
        st.info("FINDINGS.md is not in this checkout.")
        return
    st.markdown(document.read_text(encoding="utf-8"))
    st.caption(
        "Rendered verbatim from FINDINGS.md. Its figures are pinned to the 2026-08-26 "
        "snapshot; the analysis pages compute against whatever is on disk now, so the "
        "Data page's provenance panel is the place to check whether they still agree."
    )


def page_data(directory: str, symbol: str, timeframe: str) -> None:
    st.header("Data")
    st.markdown(
        "What is on disk, whether it still matches the manifest that published figures "
        "were computed from, and what had to be carried forward to put several symbols on "
        "one timeline."
    )

    st.subheader("Provenance")
    st.caption(
        f"Manifest fingerprint `{manifest_fingerprint()}` - the cache key for every panel. "
        "`verify` reads the whole `data/` tree, so it reports on both datasets regardless "
        "of the selector."
    )
    text_panel(Invocation("verify"), success_marker="OK")

    st.subheader("Series on disk")
    text_panel(Invocation("symbols", options=(("--dir", directory),)))

    st.subheader("One timeline, nothing invented")
    payload = execute(Invocation("align", options=options_for(symbol, timeframe, directory)))
    if payload is None:
        return
    left, right = st.columns(2)
    left.metric("Bars", cell(payload["rows"], Style.COUNT))
    right.metric("Columns", cell(payload["columns"], Style.COUNT))
    st.caption(
        f"{payload['first'][:10]} to {payload['last'][:10]}. No row exists that "
        f"{payload['target']} did not trade."
    )
    table(
        payload["coverage"],
        {
            "rows": Style.COUNT,
            "missing": Style.COUNT,
            "missing_fraction": Style.PERCENT,
            "stale": Style.COUNT,
            "stale_fraction": Style.PERCENT,
            "max_stale_seconds": Style.COUNT,
        },
    )


def page_exploration(directory: str, symbol: str, timeframe: str) -> None:
    st.header("Exploration")
    payload = execute(Invocation("explore", options=options_for(symbol, timeframe, directory)))
    if payload is None:
        return

    correlations = payload["correlations"]
    st.subheader("Correlation on levels is mostly shared trend")
    worst = max(correlations.items(), key=lambda kv: kv[1]["inflation"])
    columns = st.columns(3)
    columns[0].metric(f"{worst[0]} on levels", accuracy(worst[1]["on_levels"]))
    columns[1].metric("on returns", accuracy(worst[1]["on_returns"]))
    columns[2].metric("inflation", accuracy(worst[1]["inflation"]), help=GLOSSARY["inflation"])
    table(
        sorted(
            rows_from(correlations, key_column="symbol"),
            key=lambda r: -float(r["inflation"]),
        ),
        {"on_levels": Style.RATIO, "on_returns": Style.RATIO, "inflation": Style.RATIO},
    )
    figure("eda-correlations.png", "Levels against returns")

    st.subheader("The share of rising bars drifts across years")
    st.caption("The argument for walk-forward, stated without reference to any model.")
    yearly = sorted(rows_from(payload["by_year"], key_column="year"), key=lambda r: r["year"])
    table(
        yearly,
        {
            "bars": Style.COUNT,
            "mean_price": Style.RATIO,
            "min_price": Style.RATIO,
            "max_price": Style.RATIO,
            "range": Style.RATIO,
            "volatility": Style.RATIO,
            "up_share": Style.PERCENT,
        },
    )
    chart(
        bar_spec(
            [
                {"year": r["year"], "above a coin flip": float(r["up_share"]) - 0.5}
                for r in yearly
            ],
            category="year",
            value="above a coin flip",
            rule=0.0,
            title="share of rising bars, less one half",
        ),
        "Three points of drift across nine years - more than any effect being hunted.",
    )
    figure("eda-yearly.png", "The yearly breakdown")

    moves = payload["moves"]
    st.subheader("What a model has to beat before it starts")
    columns = st.columns(3)
    columns[0].metric("Always-UP baseline", accuracy(moves["up_share"]))
    columns[1].metric("Rising bars", cell(moves["up"], Style.COUNT))
    columns[2].metric("Falling bars", cell(moves["down"], Style.COUNT))
    normality = payload["normality"]
    on_price = significance(normality["price"]["p_value"])
    on_returns = significance(normality["returns"]["p_value"])
    st.caption(
        f"Normality is rejected on both the price (p = {on_price}) and the returns "
        f"(p = {on_returns}), which is why a "
        "textbook confidence interval on a Sharpe ratio is wrong here."
    )
    figure("eda-returns.png", "Returns and their distribution")
    with st.expander("Raw payload"):
        st.json(payload, expanded=False)


def page_features(directory: str, symbol: str, timeframe: str, mode: str) -> None:
    st.header("Features")
    payload = execute(
        Invocation(
            "features",
            options=(*options_for(symbol, timeframe, directory), ("--mode", mode)),
        )
    )
    if payload is None:
        return

    violations = payload["violations"]
    if payload["policy_passes"]:
        st.success(
            f"No column carries a price level - 0 violations across {payload['columns']} columns, "
            "checked by rebuilding on prices multiplied by ten rather than by trusting a name list."
        )
    else:
        st.error(f"{len(violations)} column(s) carry a price level: {violations}")

    columns = st.columns(3)
    columns[0].metric("Rows", cell(payload["rows"], Style.COUNT))
    columns[1].metric("Columns", cell(payload["columns"], Style.COUNT))
    columns[2].metric("Warm-up dropped", cell(payload["warmup_dropped"], Style.COUNT))
    st.caption(
        f"{payload['first'][:10]} to {payload['last'][:10]}, mode {payload['mode']}, "
        f"symbols {', '.join(payload['symbols'])}."
    )

    st.subheader("Every column, and how far it moved when prices were multiplied by ten")
    features = sorted(payload["features"], key=lambda f: -float(f["worst_change"]))
    table(
        [
            {
                "name": f["name"],
                "scale": f["scale"],
                "worst_change": f["worst_change"],
                "adf_pvalue": f["adf_pvalue"],
                "kpss_pvalue": f["kpss_pvalue"],
                "both tests agree": f["adf_pvalue"] < 0.05 and f["kpss_pvalue"] > 0.05,
            }
            for f in features
        ],
        {
            "worst_change": Style.SIGNIFICANCE,
            "adf_pvalue": Style.SIGNIFICANCE,
            "kpss_pvalue": Style.SIGNIFICANCE,
        },
    )
    disputed = [
        f["name"] for f in features if not (f["adf_pvalue"] < 0.05 and f["kpss_pvalue"] > 0.05)
    ]
    if disputed:
        st.caption(
            f"**The policy passing is not the whole story.** ADF and KPSS disagree on "
            f"{len(disputed)} column(s): {', '.join(disputed)}. Both are reported as "
            "diagnostics and neither decides - with this many rows ADF rejects almost "
            "anything, and both are invalid under changing variance."
        )
    left, right = st.columns(2)
    with left:
        figure("feature-correlations.png", "How the features correlate")
    with right:
        figure("feature-redundancy.png", "Redundancy among them")
    with st.expander("Raw payload"):
        st.json(payload, expanded=False)


def page_models(directory: str, symbol: str, timeframe: str, mode: str) -> None:
    st.header("Models")
    payload = execute(
        Invocation(
            "train",
            options=(*options_for(symbol, timeframe, directory), ("--mode", mode)),
        ),
        label="Fitting six estimators on three representations",
    )
    if payload is None:
        return

    if payload["models_unavailable"]:
        st.warning(
            f"Not run on this machine: {unavailable_note(payload['models_unavailable'])}"
        )

    columns = st.columns(2)
    columns[0].metric("Selected on validation", payload["selected_on_validation"])
    columns[1].metric(
        "Its edge on test",
        percent(payload["test_edge"]),
        help="Selection happens on validation and the test block is scored once, afterwards.",
    )

    st.subheader("Why every edge is small, before any model is fitted")
    blocks = rows_from(payload["blocks"], key_column="block")
    table(blocks, {"rows": Style.COUNT, "baseline_accuracy": Style.PERCENT})
    st.caption(
        "The always-UP baseline differs by block. A rule fitted on the block it is scored "
        "on would gain that difference for free, which is the trap the original analysis "
        "fell into."
    )

    st.subheader("Every configuration, on each block")
    st.caption(
        "Selection happens on the left and the right is scored once, afterwards. Ordering "
        "these tabs the other way round is what the original analysis did."
    )
    for block, tab in zip(("validation", "test"), st.tabs(["Validation", "Test"]), strict=True):
        rows = [s for s in payload["scores"] if s["block"] == block]
        with tab:
            table(
                [
                    {
                        "model": s["model"],
                        "repr": s["representation"],
                        "accuracy": s["accuracy"],
                        "baseline": s["baseline_accuracy"],
                        "edge": s["edge"],
                        "auc": s["auc"],
                        "brier": s["brier"],
                        "predicted UP": s.get("predicted_up_rate"),
                    }
                    for s in sorted(rows, key=lambda s: -s["edge"])
                ],
                {
                    "accuracy": Style.PERCENT,
                    "baseline": Style.PERCENT,
                    "edge": Style.PERCENT,
                    "auc": Style.RATIO,
                    "brier": Style.RATIO,
                    "predicted UP": Style.PERCENT,
                },
            )
            chart(
                bar_spec(
                    [
                        {
                            "configuration": f"{s['model']} [{s['representation']}]",
                            "edge": s["edge"],
                        }
                        for s in rows
                    ],
                    category="configuration",
                    value="edge",
                    rule=0.0,
                    title="edge over the constant predictor",
                )
            )
            figure(f"edge-{block}.png", f"The committed chart for {block}")

    labels = payload["labels"]
    st.caption(
        f"{labels['labelled']:,} bars labelled, {labels['gapped']:,} skipped because the "
        f"horizon spans a gap, {labels['flat']:,} exact ties kept out of the direction."
    )
    with st.expander("Raw payload"):
        st.json(payload, expanded=False)


def page_verdict(directory: str, symbol: str, timeframe: str, mode: str, pca: bool) -> None:
    st.header("The verdict")
    st.markdown(
        "Two questions, kept apart because they disagree: is there skill, and is it worth "
        "anything."
    )
    shared = (*options_for(symbol, timeframe, directory), ("--mode", mode))

    # The same answer to both commands, so the two tables cover the same configurations.
    # `validate` defaults `--pca` off and `verdict` defaults it on, which put a six-row
    # table above an eighteen-row one with no way for a reader to reconcile them.
    flags = ("--pca",) if pca else ("--no-pca",)
    count = 18 if pca else 6
    walk = execute(
        Invocation("validate", options=shared, flags=flags),
        label=f"Five expanding folds over {count} configurations, plus a stationary bootstrap",
    )
    verdict = execute(
        Invocation("verdict", options=shared, flags=flags),
        label=f"Four significance tests over {count} configurations",
    )

    if verdict is not None:
        _verdict_headline(verdict)
    if walk is not None:
        _walk_forward(walk)
    if verdict is not None:
        _significance(verdict)


def _verdict_headline(payload: dict[str, Any]) -> None:
    skill, profit, costs = payload["skill"], payload["profit"], payload["costs"]
    total = payload["configurations"]
    columns = st.columns(4)
    columns[0].metric(
        "Survive Holm", f"{skill['surviving_holm']} of {total}", help=GLOSSARY["survives Holm"]
    )
    columns[1].metric("Make money", f"{profit['profitable']} of {total}")
    columns[2].metric("Beat holding the asset", f"{profit['beating_benchmark']} of {total}")
    columns[3].metric("Buy and hold", percent(profit["benchmark_cumulative"]))
    st.caption(
        f"Costs: **{costs['round_trip_bps']:.2f} bps** round trip, {costs['source']}, positions "
        f"{costs['position']}. {payload['scored_bars']:,} bars over {payload['folds']} folds."
    )
    if "assumed" in costs["source"]:
        st.warning(
            "This threshold is **assumed, not measured** - this series carries no spread "
            "column. Every break-even below is optimistic."
        )


def _walk_forward(payload: dict[str, Any]) -> None:
    st.subheader("Scored across the whole history")
    table(
        [
            {
                "model": m["model"],
                "repr": m["representation"],
                "accuracy": m["accuracy"],
                "baseline": m["baseline_accuracy"],
                "flip rate": m["flip_rate"],
                "bars held": m["bars_held"],
                "break-even": m["break_even"],
                "short by": m["accuracy"] - m["break_even"],
            }
            for m in sorted(payload["models"], key=lambda m: m["break_even"] - m["accuracy"])
        ],
        {
            "accuracy": Style.PERCENT,
            "baseline": Style.PERCENT,
            "flip rate": Style.PERCENT,
            "bars held": Style.BARS,
            "break-even": Style.PERCENT,
            "short by": Style.PERCENT,
        },
    )
    glossary("break-even", "flip rate")

    st.subheader("Could this design have seen an edge worth having?")
    power, dependence = payload["power"], payload["dependence"]
    columns = st.columns(4)
    columns[0].metric("Detectable effect", percent(power["walk_forward_mde"]))
    columns[1].metric("On a single split", percent(power["single_split_mde"]))
    columns[2].metric("Power for a paying edge", accuracy(power["power_for_break_even"]))
    columns[3].metric("Bars it would need", cell(power["bars_required"], Style.COUNT))
    st.caption(
        "A negative result means nothing without this: an experiment that finds nothing "
        "either found nothing, or could not have seen it."
    )

    st.subheader("Are the standard errors honest?")
    columns = st.columns(3)
    columns[0].metric("Dependence inflation", f"{dependence['inflation']:.2f}x")
    columns[1].metric("Effective sample", cell(round(dependence["effective_sample"]), Style.COUNT))
    columns[2].metric(
        "Lag-1 of correctness", cell(dependence["lag_one_autocorrelation"], Style.RATIO)
    )
    contrast = payload["serial_dependence_contrast"]
    table(
        [
            {"series": name, "lag-1 autocorrelation": value}
            for name, value in list(contrast["most_autocorrelated_features"].items())[:3]
        ]
        + [
            {"series": "the label (direction)", "lag-1 autocorrelation": contrast["label"]},
            {
                "series": "model correctness",
                "lag-1 autocorrelation": dependence["lag_one_autocorrelation"],
            },
        ],
        {"lag-1 autocorrelation": Style.RATIO},
    )
    st.caption(
        "The features are strongly dependent; the thing being averaged is not. A "
        "near-coin-flip outcome inherits almost none of the dependence of its inputs, "
        "which is why the naive standard error was already honest."
    )

    st.subheader("The folds")
    table(
        payload["scheme"]["blocks"],
        {"train_rows": Style.COUNT, "test_rows": Style.COUNT},
    )
    chart(
        lines_spec(
            [
                {
                    "fold": fold["fold"],
                    "accuracy": fold["accuracy"],
                    "configuration": model["model"],
                }
                for model in payload["models"]
                for fold in model["folds"]
            ],
            x="fold",
            y="accuracy",
            series="configuration",
            rule=0.5,
            y_title="accuracy on the fold's test block",
        ),
        "How far the answer moves between regimes. The spread is a description, never an "
        "interval - the folds share training data, so they are not independent experiments.",
    )


def _significance(payload: dict[str, Any]) -> None:
    skill, profit = payload["skill"], payload["profit"]
    st.subheader("Is there skill, and is it worth anything?")
    rows = [
        {
            "configuration": name,
            "accuracy": row["accuracy"],
            "independent": row["independent_accuracy"],
            "excess": row["excess"],
            "z": row["statistic"],
            "p": row["p_value"],
            "survives Holm": row["survives_holm"],
            "cumulative": profit["per_configuration"][name]["cumulative"],
        }
        for name, row in sorted(
            skill["per_configuration"].items(), key=lambda kv: kv[1]["p_value"]
        )
    ]
    table(
        rows,
        {
            "accuracy": Style.PERCENT,
            "independent": Style.PERCENT,
            "excess": Style.PERCENT,
            "z": Style.RATIO,
            "p": Style.SIGNIFICANCE,
            "cumulative": Style.PERCENT,
        },
    )
    glossary("independent", "survives Holm", "cumulative")
    chart(
        scatter_spec(
            [
                {
                    "configuration": r["configuration"],
                    "skill": float(r["excess"]),
                    "money": float(r["cumulative"]),
                    "survives": bool(r["survives Holm"]),
                }
                for r in rows
            ],
            x="skill",
            y="money",
            label="configuration",
            highlight="survives",
            x_title="directional edge over independence",
            y_title="cumulative return, net of costs",
        ),
        "The finding in one image: every configuration is to the **right** of zero on skill "
        "and **below** zero on money. Green survives the correction for having tried them all.",
    )
    chart(
        bar_spec(
            [{"configuration": r["configuration"], "edge": float(r["excess"])} for r in rows],
            category="configuration",
            value="edge",
            rule=0.0,
            title="directional edge over independence",
        )
    )

    multiplicity = payload["multiplicity"]
    sharpe = multiplicity["deflated_sharpe"]
    st.subheader("Does the best survive having been the best?")
    columns = st.columns(3)
    columns[0].metric("Hansen SPA", significance(multiplicity["spa_p_consistent"]))
    columns[1].metric("StepM rejects", ", ".join(multiplicity["stepm_rejected"]) or "nothing")
    columns[2].metric("Deflated Sharpe", significance(sharpe["deflated"]))
    st.caption(
        f"On {sharpe['configuration']}: Sharpe {sharpe['sharpe_per_bar']:+.5f} per bar, skew "
        f"{sharpe['skew']:+.2f}, kurtosis {sharpe['kurtosis']:.1f}, against an expected "
        f"maximum of {sharpe['expected_maximum']:+.5f} for {sharpe['trials']} trials. "
        f"SPA brackets: {significance(multiplicity['spa_p_lower'])} to "
        f"{significance(multiplicity['spa_p_upper'])}."
    )
    with st.expander("Raw payload"):
        st.json(payload, expanded=False)


def page_reasoning() -> None:
    st.header("The reasoning")
    st.markdown(
        "The **book** explains the whole apparatus as it now stands, formula by formula. The "
        "**ADRs** record why each decision was made at the moment it was made, and the "
        "**STATUS logs** what each experiment measured - both written as the work happened "
        "rather than assembled afterwards."
    )
    # The book leads because it is the way in; the ADRs and logs are the record behind it.
    documents = (
        sorted((ROOT / "docs" / "book").glob("*.md"))
        + sorted((ROOT / "docs" / "adr").glob("*.md"))
        + sorted((ROOT / "docs" / "status").glob("*.md"))
    )
    if not documents:
        st.info("No documents in this checkout.")
        return
    # One document at a time. Rendering all twenty-one on every rerun pushed 200 KB of
    # Markdown through the page, and their relative image and cross-document links are all
    # dead in a browser.
    # Twenty-eight stems in one flat list is a scroll, not a menu. The parent directory is
    # the only grouping that matters here and it is already the reading order.
    labels = {
        "book": "The book", "adr": "Decisions (ADR)", "status": "Experiments (STATUS)"
    }
    names = [f"{labels[d.parent.name]} - {d.stem}" for d in documents]
    chosen = st.selectbox("Document", names, key="document")
    body = documents[names.index(chosen)].read_text(encoding="utf-8")
    st.markdown(body)


# --- layout ---------------------------------------------------------------------------------


#: Every keyed control. Streamlit discards the state of a widget that a run does not
#: instantiate - a key is not enough on its own - so the document pages, which build no
#: controls, would silently reset the reader's dataset and symbol. Reassigning each key
#: to itself on every run marks it as still wanted.
CONTROLS = ("page", "dataset", "symbol", "timeframe", "source", "mode", "breadth", "document")

PAGES = ("Findings", "Data", "Exploration", "Features", "Models", "Verdict", "Reasoning")
#: Pages that read a dataset; the rest render documents.
ANALYSIS = {"Data", "Exploration", "Features", "Models", "Verdict"}
#: Pages whose command takes `--mode`. `align` and `explore` do not.
TAKES_MODE = {"Features", "Models", "Verdict"}


def keep_controls() -> None:
    """Stop Streamlit garbage-collecting the sidebar while a document page is open.

    Found by a test rather than by reading: keying the widgets was not enough. A reader who
    chose `data/reference`, opened Findings and came back was silently returned to
    `data/raw`, and the only thing on screen that said so was the command line.
    """
    for key in CONTROLS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def main() -> None:
    st.set_page_config(page_title="forecast-lab", layout="wide")
    keep_controls()
    st.title("A real edge, worth less than nothing")
    st.caption(
        "Short-horizon direction on gold. Every panel runs `forecast-lab <command> --json` "
        "as a subprocess and shows the command beside the result, so the page is the "
        "commands rather than a second implementation of them."
    )

    # Keyed so a visit to Findings or Reasoning does not garbage-collect the choices: the
    # widgets are not created on those pages, and unkeyed state does not survive that.
    choice = st.sidebar.radio("Page", PAGES, key="page")
    st.sidebar.divider()

    if choice not in ANALYSIS:
        (page_findings if choice == "Findings" else page_reasoning)()
        return

    directory = dataset_choice()
    if directory is None:
        return
    series = series_choice(directory)
    if series is None:
        return
    symbol, timeframe = series
    source_choice(symbol, timeframe)
    mode = mode_choice() if choice in TAKES_MODE else "focus"
    pca = breadth_choice() if choice == "Verdict" else True
    st.sidebar.divider()
    st.sidebar.caption(
        "Read-only: `fetch` and `ingest` are refused by the runner, and so is any option "
        "that writes (ADR-005 sec. 4)."
    )

    match choice:
        case "Data":
            page_data(directory, symbol, timeframe)
        case "Exploration":
            page_exploration(directory, symbol, timeframe)
        case "Features":
            page_features(directory, symbol, timeframe, mode)
        case "Models":
            page_models(directory, symbol, timeframe, mode)
        case "Verdict":
            page_verdict(directory, symbol, timeframe, mode, pca)


# Streamlit executes this file as a script with `__name__ == "__main__"`, so one guard
# serves both `streamlit run` and a direct invocation - and importing the module, which the
# tests do, renders nothing.
if __name__ == "__main__":
    main()
