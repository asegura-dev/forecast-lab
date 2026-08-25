"""Figures, built in memory and handed back for someone else to write.

This module returns `Figure` objects and never touches the filesystem - the same rule
that governs the rest of `research` (ADR-001 sec. 1), and the layering guard enforces it
by treating `savefig` as I/O. The CLI decides where a figure goes.

**What is plotted, and why these four.** The original project drew nineteen charts across
its two notebooks and none of them could have shown its mistake, because every one plots
a model against itself. The figures here are chosen to make the comparison the original
lacked impossible to miss:

- `edge_chart` - every model against the constant predictor and the break-even line. The
  chart the original needed and did not have. Its two headline numbers, 51.53% and
  51.86%, appear in its own output two rows apart; on this chart they would have been a
  bar and a line, and the bar would have been under the line.
- `roc_chart` - the original drew this one, but from hard labels, which collapses every
  curve into a single kink. From probabilities the curves are real, and their closeness
  to the diagonal is the finding.
- `confusion_grid` - also in the original. Kept because a confusion matrix is where
  "100% recall, 0% specificity" stops being a pair of numbers and becomes a picture of a
  model that only ever says UP.
- `calibration_chart` - not in the original at all. A model can rank well and still be
  systematically overconfident, and at these effect sizes that is the difference between
  a usable probability and a decorative one.

Style is deliberately plain: no seaborn, no colour cycling beyond what distinguishes the
lines, and a light grid. A chart in a research log is an argument, not decoration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from forecast_lab.contracts import ForecastLabError

if TYPE_CHECKING:  # pragma: no cover - import cost only matters at runtime
    from matplotlib.figure import Figure

#: Colour for anything that represents a floor a model has to clear.
BASELINE_COLOUR = "#c0392b"
#: Colour for the model that was actually selected.
SELECTED_COLOUR = "#2c3e50"
NEUTRAL_COLOUR = "#7f8c8d"


class PlotError(ForecastLabError):
    """A figure cannot be built from what was provided."""


def _pyplot() -> Any:
    """Import pyplot with a non-interactive backend.

    `Agg` is set before pyplot is imported because the default backend probes for a
    display, which on a headless machine either warns or blocks. A figure that is going
    to be written to a file never needs a window.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def edge_chart(scores: list[Any], *, baseline: float, break_even: float | None = None,
               block: str = "test", selected: str | None = None) -> Figure:
    """Every model's accuracy against the two lines that decide whether it means anything.

    The horizontal rules are the point. `baseline` is what predicting the training
    majority every time scores on this block; `break_even` is the accuracy at which a
    strategy would pay for its own transaction costs. A bar below the first is a model
    that lost to a rule with no parameters. A bar below the second is a model that could
    not be traded even if it were real.
    """
    if not scores:
        raise PlotError("no scores to plot")
    plt = _pyplot()

    # Horizontal bars, because eighteen configurations with names like
    # "HistGradientBoosting pca-90" are unreadable on a vertical axis - they overlap into
    # a smear at any font size that still fits the page.
    ordered = sorted(scores, key=lambda s: s.accuracy)
    labels = [f"{s.model}  [{s.representation}]" for s in ordered]
    values = [s.accuracy * 100 for s in ordered]
    colours = [
        SELECTED_COLOUR if selected and s.key == selected else NEUTRAL_COLOUR for s in ordered
    ]

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(ordered) + 2.0))
    positions = np.arange(len(ordered))
    ax.barh(positions, values, color=colours, height=0.7)

    ax.axvline(
        baseline * 100, color=BASELINE_COLOUR, linewidth=1.7, zorder=3,
        label=f"always-UP  {baseline:.2%}",
    )
    if break_even is not None:
        ax.axvline(
            break_even * 100, color="#8e44ad", linewidth=1.5, linestyle="--", zorder=3,
            label=f"break-even  {break_even:.2%}",
        )

    # An axis starting at zero would compress every difference in this project into
    # invisibility; one starting at the data would exaggerate them. Anchoring on the
    # baseline and padding symmetrically keeps the comparison honest.
    reference = [*values, break_even * 100] if break_even is not None else values
    span = max(abs(v - baseline * 100) for v in reference)
    left = baseline * 100 - span * 1.35
    ax.set_xlim(left, baseline * 100 + span * 1.45)

    for position, s in zip(positions, ordered, strict=True):
        edge = s.edge * 100
        ax.text(
            s.accuracy * 100 + span * 0.05, position, f"{edge:+.2f}",
            va="center", ha="left", fontsize=8.5,
            color="#27ae60" if edge > 0 else BASELINE_COLOUR,
        )

    ax.set_yticks(positions, labels, fontsize=8.5)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(
        f"Accuracy against the rules it has to beat - {block} block",
        fontsize=12, fontweight="bold", loc="left",
    )
    # Legend, not inline text: with the lines this close together, labels drawn on the
    # axes collide with each other and with the bar annotations.
    ax.legend(fontsize=8.5, loc="lower right", frameon=True, framealpha=0.95)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)


def roc_chart(
    curves: dict[str, Any], *, block: str = "test", selected: str | None = None
) -> Figure:
    """ROC curves from probabilities, with the diagonal for reference.

    ``curves`` maps a label to a `RocPoints` - coordinates computed by the metrics
    module, so this one needs no estimator library to draw them.

    The original drew this panel from hard predictions, which gives every curve a single
    interior point and makes the whole thing meaningless: its Naive Bayes came out at
    exactly 0.5000.
    """
    if not curves:
        raise PlotError("no curves to plot")
    plt = _pyplot()

    # Wide enough to hold the legend outside the axes. With eighteen configurations an
    # inline legend covers the very region the curves occupy, which on this chart is the
    # whole argument - the curves sit on the diagonal and you have to be able to see it.
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.plot([0, 1], [0, 1], color="#95a5a6", linewidth=1.2, linestyle=":", label="chance")

    for name, points in sorted(
        curves.items(), key=lambda kv: -(kv[1].auc if kv[1] is not None else 0.0)
    ):
        if points is None:
            continue
        is_selected = selected is not None and name == selected
        ax.plot(
            points.false_positive_rate, points.true_positive_rate,
            linewidth=2.2 if is_selected else 0.9,
            color=SELECTED_COLOUR if is_selected else None,
            alpha=1.0 if is_selected else 0.55,
            label=f"{name}  ({points.auc:.4f})" + ("  <- selected" if is_selected else ""),
        )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC from probabilities - {block} block", fontsize=12, fontweight="bold",
                 loc="left")
    ax.legend(
        fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
        title="sorted by AUC", title_fontsize=8,
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)


def confusion_grid(
    matrices: dict[str, pd.DataFrame], *, block: str = "test", columns: int = 3
) -> Figure:
    """Confusion matrices side by side, as row-normalised heatmaps.

    Normalised by row - by what actually happened - because that is what makes a model
    that only ever says UP unmistakable: its DOWN row is entirely in the UP column.
    Raw counts would hide it behind the class imbalance.
    """
    if not matrices:
        raise PlotError("no matrices to plot")
    plt = _pyplot()

    rows = (len(matrices) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.5 * rows), squeeze=False)

    for axis, (name, table) in zip(axes.flat, matrices.items(), strict=False):
        totals = table.to_numpy(dtype=float).sum(axis=1, keepdims=True)
        shares = np.divide(
            table.to_numpy(dtype=float), totals, out=np.zeros_like(totals * table.to_numpy()),
            where=totals > 0,
        )
        axis.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(2):
            for j in range(2):
                axis.text(
                    j, i, f"{shares[i, j]:.0%}\n{int(table.to_numpy()[i, j]):,}",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if shares[i, j] > 0.55 else "#2c3e50",
                )
        axis.set_xticks([0, 1], ["DOWN", "UP"], fontsize=8)
        axis.set_yticks([0, 1], ["DOWN", "UP"], fontsize=8)
        axis.set_xlabel("predicted", fontsize=8)
        axis.set_ylabel("actual", fontsize=8)
        axis.set_title(name, fontsize=9, fontweight="bold")

    for axis in list(axes.flat)[len(matrices):]:
        axis.axis("off")

    fig.suptitle(f"Confusion, normalised by row - {block} block", fontsize=12,
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return cast("Figure", fig)


def calibration_chart(
    curves: dict[str, tuple[pd.Series, pd.Series]], *, bins: int = 10, block: str = "test"
) -> Figure:
    """Predicted probability against observed frequency.

    Not in the original at all. A model can rank well - a respectable AUC - and still be
    systematically overconfident, and at these effect sizes that is the difference
    between a probability worth acting on and a decorative one. Points below the diagonal
    are promises the model did not keep.
    """
    if not curves:
        raise PlotError("no curves to plot")
    plt = _pyplot()

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.plot([0, 1], [0, 1], color="#95a5a6", linewidth=1.2, linestyle=":", label="perfect")

    for name, (proba, truth) in curves.items():
        aligned = truth.reindex(proba.index)
        keep = aligned.notna()
        y, p = aligned[keep].to_numpy(dtype=float), proba[keep].to_numpy(dtype=float)
        if y.size == 0:
            continue
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if edges.size < 3:
            continue
        which = np.clip(np.digitize(p, edges[1:-1]), 0, edges.size - 2)
        xs, ys = [], []
        for b in range(edges.size - 1):
            inside = which == b
            if inside.sum() >= 20:
                xs.append(float(p[inside].mean()))
                ys.append(float(y[inside].mean()))
        ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.1, alpha=0.85, label=name)

    ax.set_xlabel("Predicted probability of UP")
    ax.set_ylabel("Observed frequency of UP")
    ax.set_title(f"Calibration - {block} block", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)


def correlation_comparison(prices: pd.DataFrame, *, target: str) -> Figure:
    """The same correlations computed on levels and on returns, side by side.

    The original's EDA reports a +0.923 correlation between gold and the S&P and reads it
    as an economic finding. It is an artefact: two series that both trend upward correlate
    on levels regardless of any relationship between them. On returns - which is what a
    model at this horizon actually sees - the same pair is near zero.

    This is the feature-scale argument of ADR-006 applied to the exploratory analysis,
    and it is far more convincing as a picture than as a paragraph.
    """
    if target not in prices.columns:
        raise PlotError(f"{target} is not among the columns supplied")
    plt = _pyplot()

    level_corr = cast(pd.Series, prices.corr().loc[target]).drop(target)
    levels = level_corr.sort_values(ascending=False)
    return_corr = cast(pd.Series, prices.pct_change().corr().loc[target]).drop(target)
    returns = return_corr.reindex(levels.index)

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(levels))
    ax.barh(positions + 0.2, levels.to_numpy(), height=0.38, color="#e67e22",
            label="on price levels (spurious)")
    ax.barh(positions - 0.2, returns.to_numpy(), height=0.38, color=SELECTED_COLOUR,
            label="on returns (what a model sees)")

    ax.axvline(0, color="#2c3e50", linewidth=0.8)
    ax.set_yticks(positions, [str(i) for i in levels.index], fontsize=8.5)
    ax.set_xlabel(f"Correlation with {target}")
    ax.set_xlim(-1, 1)
    ax.set_title(
        "The same correlations, computed two ways",
        fontsize=12, fontweight="bold", loc="left",
    )
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)
