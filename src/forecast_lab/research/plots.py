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

def price_overview(close: pd.Series, qq: tuple[Any, Any], *, symbol: str = "") -> Figure:
    """Four panels on the price itself: path, distribution, spread, and normality.

    The Q-Q panel is the one worth reading, and it is where the original's normality test
    went wrong. It ran D'Agostino-Pearson on the price and reported "not normal", which
    is true of any trending series and licenses nothing. The panel shows *how* it fails:
    the price's tails are **shorter** than a normal's, because a price is bounded below
    and was in an uptrend. That is a fact about the shape of the sample, not a property
    a model could use.
    """
    plt = _pyplot()
    values = close.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    axes[0, 0].plot(values.index, values.to_numpy(), linewidth=0.7, color=SELECTED_COLOUR)
    axes[0, 0].set_title(f"{symbol} close".strip(), fontsize=11, fontweight="bold", loc="left")
    axes[0, 0].set_ylabel("Price")

    axes[0, 1].hist(values.to_numpy(), bins=80, color=SELECTED_COLOUR, alpha=0.85)
    axes[0, 1].axvline(float(values.mean()), color=BASELINE_COLOUR, linewidth=1.4, label="mean")
    axes[0, 1].axvline(
        float(values.median()), color="#8e44ad", linewidth=1.4, linestyle="--", label="median"
    )
    axes[0, 1].set_title("Distribution", fontsize=11, fontweight="bold", loc="left")
    axes[0, 1].legend(fontsize=8, frameon=False)

    axes[1, 0].boxplot(
        values.to_numpy(), vert=False, widths=0.6,
        patch_artist=True, boxprops={"facecolor": "#dfe6e9"},
        medianprops={"color": BASELINE_COLOUR, "linewidth": 1.6},
    )
    axes[1, 0].set_title("Spread and outliers", fontsize=11, fontweight="bold", loc="left")
    axes[1, 0].set_yticks([])

    theoretical, observed = qq
    axes[1, 1].scatter(theoretical, observed, s=3, alpha=0.4, color=SELECTED_COLOUR)
    limit = float(max(abs(theoretical).max(), abs(observed).max()))
    axes[1, 1].plot([-limit, limit], [-limit, limit], color=BASELINE_COLOUR, linewidth=1.2)
    axes[1, 1].set_title(
        "Q-Q against a normal - tails are shorter, not fatter",
        fontsize=11, fontweight="bold", loc="left",
    )
    axes[1, 1].set_xlabel("Theoretical quantiles")
    axes[1, 1].set_ylabel("Observed (standardised)")

    for axis in axes.flat:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)


def returns_overview(
    returns: pd.Series, qq: tuple[Any, Any], *, symbol: str = "", window: int = 168
) -> Figure:
    """Four panels on returns: the series, its distribution, its sum, and its volatility.

    This is the variable a model at this horizon actually consumes, and its Q-Q panel is
    the one that carries a consequence: the tails run to roughly **ten** standard
    deviations where a normal reaches 3.7. Fat tails are why a Sharpe ratio's textbook
    confidence interval is wrong on this data, and why the evaluation layer will need a
    bootstrap rather than a t-statistic.
    """
    plt = _pyplot()
    values = returns.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    axes[0, 0].plot(values.index, values.to_numpy() * 100, linewidth=0.4, color=SELECTED_COLOUR)
    axes[0, 0].set_title(
        f"{symbol} hourly returns".strip(), fontsize=11, fontweight="bold", loc="left"
    )
    axes[0, 0].set_ylabel("%")

    axes[0, 1].hist(values.to_numpy() * 100, bins=140, color=SELECTED_COLOUR, alpha=0.85)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title(
        "Distribution (log count - the tails are the point)",
        fontsize=11, fontweight="bold", loc="left",
    )
    axes[0, 1].set_xlabel("%")

    cumulative = (1.0 + values).cumprod()
    axes[1, 0].plot(cumulative.index, cumulative.to_numpy(), linewidth=0.9, color=SELECTED_COLOUR)
    axes[1, 0].axhline(1.0, color=BASELINE_COLOUR, linewidth=1.0, linestyle=":")
    axes[1, 0].set_title(
        f"Compounded return: {(float(cumulative.iloc[-1]) - 1) * 100:+.1f}%",
        fontsize=11, fontweight="bold", loc="left",
    )

    rolling = values.rolling(window=window, min_periods=window).std() * np.sqrt(24 * 365)
    axes[1, 1].plot(rolling.index, rolling.to_numpy(), linewidth=0.8, color=SELECTED_COLOUR)
    axes[1, 1].set_title(
        f"Realised volatility, {window}-bar window, annualised",
        fontsize=11, fontweight="bold", loc="left",
    )

    theoretical, observed = qq
    del theoretical, observed  # drawn on the price panel; kept in the signature for symmetry

    for axis in axes.flat:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)


def yearly_overview(yearly: pd.DataFrame, *, symbol: str = "") -> Figure:
    """Four panels per year: price, volatility, direction share, and range.

    The third panel is the one that matters for everything downstream. If the share of
    rising bars moves year to year - and it does, from 49.6% to 52.5% - then which years
    land in the test block decides a meaningful part of any accuracy reported on it. That
    is the `oracle_gap` of ADR-004 sec. 5, seen from the data's side rather than the
    model's.
    """
    if yearly.empty:
        raise PlotError("no yearly rows to plot")
    plt = _pyplot()

    years = [str(y) for y in yearly.index]
    positions = np.arange(len(years))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].bar(positions, yearly["mean_price"], color=SELECTED_COLOUR, width=0.6)
    axes[0, 0].set_title("Mean price", fontsize=11, fontweight="bold", loc="left")

    axes[0, 1].bar(positions, yearly["volatility"] * 100, color="#8e44ad", width=0.6)
    axes[0, 1].set_title(
        "Annualised volatility (%)", fontsize=11, fontweight="bold", loc="left"
    )

    shares = yearly["up_share"] * 100
    axes[1, 0].bar(positions, shares, color=SELECTED_COLOUR, width=0.6)
    axes[1, 0].axhline(50.0, color=BASELINE_COLOUR, linewidth=1.4, label="50%")
    axes[1, 0].set_ylim(min(45.0, float(shares.min()) - 2), max(55.0, float(shares.max()) + 2))
    axes[1, 0].set_title(
        "Share of rising bars - the class balance drifts",
        fontsize=11, fontweight="bold", loc="left",
    )
    axes[1, 0].legend(fontsize=8, frameon=False)
    for position, share in zip(positions, shares, strict=True):
        axes[1, 0].text(position, share, f"{share:.1f}", ha="center", va="bottom", fontsize=8.5)

    axes[1, 1].bar(positions, yearly["range"], color="#7f8c8d", width=0.6)
    axes[1, 1].set_title("Price range", fontsize=11, fontweight="bold", loc="left")

    for axis in axes.flat:
        axis.set_xticks(positions, years, fontsize=9)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    fig.suptitle(
        f"{symbol} by year".strip(), fontsize=12, fontweight="bold", x=0.02, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return cast("Figure", fig)

def feature_correlation_chart(
    correlations: pd.Series, *, top: int = 20, target: str = "direction"
) -> Figure:
    """Each feature's correlation with the answer, ranked, beside their distribution.

    The original ranked these and printed a top twenty. The ranking is the lesser half.
    The number that decides what follows is the **largest** one: measured on this data
    the strongest single feature correlates with the direction at **0.019**, and no
    combination of nineteen such columns is going to produce a large edge. That is
    knowable before a model is fitted, which is why this figure belongs beside the
    exploratory work rather than after the results.
    """
    if correlations.empty:
        raise PlotError("no correlations to plot")
    plt = _pyplot()

    ranked = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    head = ranked.head(top)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, 0.32 * len(head) + 1.8)))

    positions = np.arange(len(head))
    colours = [SELECTED_COLOUR if v >= 0 else BASELINE_COLOUR for v in head]
    axes[0].barh(positions, head.to_numpy(), color=colours, height=0.7)
    axes[0].axvline(0, color="#2c3e50", linewidth=0.8)
    axes[0].set_yticks(positions, [str(i) for i in head.index], fontsize=8)
    axes[0].set_xlabel(f"Correlation with {target}")
    axes[0].set_title(
        f"Top {len(head)} features - the strongest is {ranked.abs().max():.4f}",
        fontsize=11, fontweight="bold", loc="left",
    )

    axes[1].hist(ranked.to_numpy(), bins=30, color=SELECTED_COLOUR, alpha=0.85)
    axes[1].axvline(0, color=BASELINE_COLOUR, linewidth=1.2)
    axes[1].set_xlabel(f"Correlation with {target}")
    axes[1].set_title(
        "All features - centred on zero", fontsize=11, fontweight="bold", loc="left"
    )

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)


def redundancy_chart(pairs: pd.DataFrame, *, threshold: float = 0.9, total: int = 0) -> Figure:
    """Feature pairs that are near-duplicates of one another.

    This is the measurement behind a claim the project had been making without evidence:
    that PCA collapses the matrix because indicators computed from one price series are
    variations of each other. Measured here, **ten of 171 pairs** exceed 0.9 - starting
    with `return` against `log_return` at exactly **1.0000**, since log(1+r) is r for
    small moves.
    """
    plt = _pyplot()

    if pairs.empty:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.text(
            0.5, 0.5, f"No pair exceeds |{threshold:.2f}|",
            ha="center", va="center", fontsize=12, color=SELECTED_COLOUR,
        )
        ax.axis("off")
        fig.tight_layout()
        return cast("Figure", fig)

    labels = [f"{a}  /  {b}" for a, b in zip(pairs["left"], pairs["right"], strict=True)][::-1]
    values = pairs["correlation"].to_numpy()[::-1]
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 0.36 * len(labels) + 2.0))
    ax.barh(positions, np.abs(values), color=SELECTED_COLOUR, height=0.68)
    ax.axvline(threshold, color=BASELINE_COLOUR, linewidth=1.5, label=f"threshold {threshold:.2f}")
    ax.set_xlim(min(threshold - 0.02, float(np.abs(values).min()) - 0.01), 1.005)
    ax.set_yticks(positions, labels, fontsize=8.5)
    ax.set_xlabel("Absolute correlation")

    counted = f" of {total:,} pairs" if total else ""
    ax.set_title(
        f"{len(pairs)} near-duplicate feature pairs{counted}",
        fontsize=12, fontweight="bold", loc="left",
    )
    for position, value in zip(positions, values, strict=True):
        ax.text(abs(value) - 0.002, position, f"{value:+.4f}", va="center", ha="right",
                fontsize=8, color="white")

    ax.legend(fontsize=8.5, loc="lower left", frameon=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)


def variance_chart(explained: dict[str, tuple[float, ...]], *, columns: int = 0) -> Figure:
    """Variance explained per component, and cumulatively, for each PCA setting.

    The original plotted this and read the component count off it. The curve says more
    than the count does: measured here the **first component alone carries 44.3%** of the
    variance of nineteen features, and the first three carry 73%. That is the redundancy
    of `redundancy_chart` seen from the other side - a matrix of near-duplicates has one
    dominant direction and a tail of small corrections.
    """
    if not explained:
        raise PlotError("no variance ratios to plot")
    plt = _pyplot()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for name, ratios in explained.items():
        if not ratios:
            continue
        positions = np.arange(1, len(ratios) + 1)
        axes[0].bar(
            positions, np.array(ratios) * 100, alpha=0.75, width=0.65, label=name
        )
        cumulative = np.cumsum(ratios) * 100
        axes[1].plot(positions, cumulative, marker="o", markersize=4, linewidth=1.4, label=name)

    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Variance explained (%)")
    title = f"Per component (from {columns} features)" if columns else "Per component"
    axes[0].set_title(title, fontsize=11, fontweight="bold", loc="left")
    axes[0].legend(fontsize=8.5, frameon=False)

    for level, style in ((90.0, "--"), (95.0, ":")):
        axes[1].axhline(level, color=BASELINE_COLOUR, linewidth=1.1, linestyle=style)
        axes[1].text(0.6, level + 0.6, f"{level:.0f}%", color=BASELINE_COLOUR, fontsize=8)
    axes[1].set_xlabel("Components retained")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative", fontsize=11, fontweight="bold", loc="left")
    axes[1].legend(fontsize=8.5, loc="lower right", frameon=False)

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)


def block_chart(blocks: dict[str, int], balance: dict[str, float]) -> Figure:
    """How the rows divide between blocks, and how the class balance differs in each.

    The original drew the sample split and the class balance as two panels and read them
    as a sanity check. The second panel is not a formality here: the share of rising bars
    differs between blocks by more than the effect being hunted, so the right-hand chart
    is a picture of how much any test accuracy owes to which stretch of history it landed
    on (ADR-004 sec. 5).
    """
    if not blocks:
        raise PlotError("no blocks to plot")
    plt = _pyplot()

    names = list(blocks)
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    counts = [blocks[n] for n in names]
    axes[0].bar(positions, counts, color=SELECTED_COLOUR, width=0.6)
    total = sum(counts) or 1
    for position, count in zip(positions, counts, strict=True):
        axes[0].text(
            position, count, f"{count:,}\n{count / total:.0%}",
            ha="center", va="bottom", fontsize=8.5,
        )
    axes[0].set_title("Rows per block", fontsize=11, fontweight="bold", loc="left")
    axes[0].set_ylim(0, max(counts) * 1.18)

    shares = [balance.get(n, 0.0) * 100 for n in names]
    axes[1].bar(positions, shares, color=NEUTRAL_COLOUR, width=0.6)
    axes[1].axhline(50.0, color=BASELINE_COLOUR, linewidth=1.4, label="50%")
    for position, share in zip(positions, shares, strict=True):
        axes[1].text(position, share, f"{share:.2f}%", ha="center", va="bottom", fontsize=8.5)
    spread = max(shares) - min(shares) if shares else 0.0
    axes[1].set_ylim(min(45.0, min(shares) - 2), max(55.0, max(shares) + 2))
    axes[1].set_title(
        f"Share rising - {spread:.2f} points apart",
        fontsize=11, fontweight="bold", loc="left",
    )
    axes[1].legend(fontsize=8.5, frameon=False)

    for axis in axes:
        axis.set_xticks(positions, names, fontsize=9)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    return cast("Figure", fig)
