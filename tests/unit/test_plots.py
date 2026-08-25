"""Tests for the figures.

A chart cannot be asserted pixel by pixel without becoming a test of matplotlib, so
these check the two things that actually matter here: that the module builds a figure
rather than writing one, and that it refuses input it cannot honestly draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_lab.research import (
    ModelScore,
    PlotError,
    calibration_chart,
    confusion_grid,
    correlation_comparison,
    edge_chart,
    roc_chart,
    roc_points,
)


def _score(name: str, accuracy: float, baseline: float = 0.51) -> ModelScore:
    return ModelScore(
        model=name, representation="raw", block="test", n=1000,
        accuracy=accuracy, precision=0.5, recall=0.5, specificity=0.5,
        auc=0.51, brier=0.25, baseline_accuracy=baseline, predicted_up_rate=0.5,
    )


def _series(n: int = 400, seed: int = 2) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-03", periods=n, freq="h", tz="UTC")
    truth = pd.Series((rng.normal(size=n) > 0).astype(float), index=index)
    proba = pd.Series(rng.uniform(0.3, 0.7, n), index=index)
    return proba, truth


# --- the architectural property -----------------------------------------------------


@pytest.mark.unit
def test_a_figure_is_returned_rather_than_written(tmp_path: object) -> None:
    """`research` never touches disk (ADR-001 sec. 1), and the layering guard enforces it
    by treating `savefig` as I/O. This asserts the other half: that something usable
    comes back, so the CLI has a figure to write.
    """
    figure = edge_chart([_score("A", 0.52), _score("B", 0.50)], baseline=0.51)
    assert hasattr(figure, "savefig")
    assert figure.axes, "the figure has no axes to draw on"


@pytest.mark.unit
def test_the_baseline_and_break_even_lines_are_drawn() -> None:
    """The two rules are the whole argument of this chart, so their absence is a bug."""
    figure = edge_chart(
        [_score("A", 0.52), _score("B", 0.50)], baseline=0.5131, break_even=0.5192
    )
    legend = figure.axes[0].get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert any("always-UP" in label for label in labels)
    assert any("break-even" in label for label in labels)


# --- refusals -----------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "call",
    [
        lambda: edge_chart([], baseline=0.5),
        lambda: roc_chart({}),
        lambda: calibration_chart({}),
        lambda: confusion_grid({}),
    ],
)
def test_an_empty_chart_is_refused(call: object) -> None:
    """A figure of nothing is worse than no figure: it looks like a measurement."""
    with pytest.raises(PlotError):
        call()  # type: ignore[operator]


@pytest.mark.unit
def test_a_missing_target_column_is_refused() -> None:
    frame = pd.DataFrame({"SPX": [1.0, 2.0, 3.0], "DXY": [3.0, 2.0, 1.0]})
    with pytest.raises(PlotError, match="not among the columns"):
        correlation_comparison(frame, target="XAUUSD")


# --- what the charts do with real shapes ---------------------------------------------


@pytest.mark.unit
def test_roc_survives_a_single_class_block() -> None:
    """`roc_points` returns None there, and the chart has to skip it rather than crash."""
    proba, _ = _series()
    one_class = pd.Series(1.0, index=proba.index)
    figure = roc_chart({"A": roc_points(proba, one_class), "B": roc_points(*_series())})
    assert figure.axes


@pytest.mark.unit
def test_confusion_grid_handles_a_number_of_models_that_does_not_fill_the_grid() -> None:
    """Four models in a three-column layout leaves two empty cells."""
    from forecast_lab.research import confusion

    proba, truth = _series()
    matrices = {f"model {i}": confusion(proba, truth) for i in range(4)}
    figure = confusion_grid(matrices, columns=3)
    # Six axes, of which two are switched off rather than left with empty ticks.
    assert len(figure.axes) == 6
    assert sum(not axis.axison for axis in figure.axes) == 2


@pytest.mark.unit
def test_calibration_skips_bins_with_too_few_points() -> None:
    """A bin holding three rows reports a frequency of 0, 0.33, 0.67 or 1, and plotting
    that as calibration would be drawing noise as if it were a measurement."""
    proba, truth = _series(n=60)
    figure = calibration_chart({"A": (proba, truth)}, bins=20)
    assert figure.axes


@pytest.mark.unit
def test_the_correlation_chart_shows_both_computations() -> None:
    """The point of the chart is the contrast, so both series must be present."""
    rng = np.random.default_rng(5)
    index = pd.date_range("2022-01-03", periods=300, freq="h", tz="UTC")
    trend = np.cumsum(rng.normal(0.1, 1, 300))
    frame = pd.DataFrame(
        {
            "XAUUSD": 1800 + trend,
            # Same trend, independent noise: correlated on levels, uncorrelated on
            # returns. Exactly the artefact the original's EDA read as a finding.
            "SPX": 4000 + trend * 2 + rng.normal(0, 1, 300),
        },
        index=index,
    )
    figure = correlation_comparison(frame, target="XAUUSD")
    legend = figure.axes[0].get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert any("levels" in label for label in labels)
    assert any("returns" in label for label in labels)
