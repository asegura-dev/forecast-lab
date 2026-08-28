"""Scoring a model across walk-forward folds, and the trap that lives in the result.

`walkforward` cuts the index; this fits and scores on the cuts. Two decisions make the
number mean what it appears to mean.

**The folds are pooled, not averaged.** Every prediction from every fold goes into one
accuracy over one row count. Averaging the five fold accuracies and quoting a t-statistic
over them is the obvious alternative and it is wrong: the folds share training data - fold
5 trains on everything fold 1 trained on - so they are not five independent experiments
and their spread is not a sampling distribution. The spread is still reported, as a
description of how much the answer moves across regimes, never as an interval.

**The baseline is refitted in every fold.** The constant predictor picks the majority
class of *its own* training block, exactly as the model sees only its own training block.
Carrying one global majority across all five folds would score the baseline with
information the model was denied, which is the same leak this project exists to refuse,
pointed the other way.

**And here is what that produced.** Measured on the canonical series, every model's edge
over the baseline turns positive under walk-forward, after being negative or negligible
on a single split. It reads like the models improving. They do not:

| | model accuracy | baseline | edge |
|---|---:|---:|---:|
| Single split, HistGradientBoosting | 51.44% | 50.93% | +0.51% |
| Walk-forward, HistGradientBoosting | 51.23% | **50.26%** | +0.97% |

The accuracy falls by 0.21 points. The baseline falls by 0.67. The single split lands on
one rising stretch where always-UP scores 50.93%; averaged over five stretches the
majority class sits nearer a half, and the same model looks better against it. **The edge
moved because the thing it is measured against moved.** Reporting the edge alone would
have made a measurement artefact look like a finding - which is the failure this whole
repository is a correction of - so both terms are carried on every row.

None of it changes the verdict, because the verdict is arithmetic rather than statistical:
the best edge is 0.97 points and the venue's measured costs demand 3.49.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from forecast_lab.contracts import ForecastLabError
from forecast_lab.research.models.catalogue import ModelSpec
from forecast_lab.research.models.training import fit_and_predict
from forecast_lab.research.walkforward import WalkForward


class ValidationError(ForecastLabError):
    """A model cannot be scored across the folds it was given."""


@dataclass(frozen=True)
class FoldScore:
    """What one fold scored, with the baseline that fold's own training block implies."""

    number: int
    n: int
    accuracy: float
    baseline_accuracy: float

    @property
    def edge(self) -> float:
        return self.accuracy - self.baseline_accuracy


@dataclass(frozen=True)
class PooledScore:
    """One model over every fold at once."""

    model: str
    representation: str
    folds: tuple[FoldScore, ...]
    #: Folds that could not be fitted, with the reason. Never silent: a model scored on
    #: four folds is not comparable with one scored on five, and the row has to say so.
    skipped: tuple[tuple[int, str], ...] = ()

    @property
    def key(self) -> str:
        return f"{self.model} [{self.representation}]"

    @property
    def n(self) -> int:
        return sum(fold.n for fold in self.folds)

    @property
    def accuracy(self) -> float:
        """Pooled, not averaged - see the module docstring."""
        return sum(f.accuracy * f.n for f in self.folds) / self.n

    @property
    def baseline_accuracy(self) -> float:
        return sum(f.baseline_accuracy * f.n for f in self.folds) / self.n

    @property
    def edge(self) -> float:
        return self.accuracy - self.baseline_accuracy

    @property
    def worst_fold(self) -> float:
        return min(f.accuracy for f in self.folds)

    @property
    def best_fold(self) -> float:
        return max(f.accuracy for f in self.folds)


def score_walk_forward(
    spec: ModelSpec,
    features: pd.DataFrame,
    labels: pd.Series,
    scheme: WalkForward,
    *,
    variance: float | None = None,
) -> PooledScore:
    """Fit ``spec`` once per fold and pool what it scored on the blocks that follow.

    A fold that cannot be fitted - a training block with one class, a native library the
    machine refuses to load - is recorded in ``skipped`` rather than dropped quietly.
    """
    scored: list[FoldScore] = []
    skipped: list[tuple[int, str]] = []
    representation = "raw" if variance is None else f"pca-{int(variance * 100)}"

    for fold in scheme:
        blocks = {"train": fold.train, "test": fold.test}
        try:
            fitted = fit_and_predict(spec, features, labels, blocks, variance=variance)
        except ForecastLabError as exc:
            skipped.append((fold.number, str(exc)))
            continue

        probabilities = fitted.probabilities.get("test")
        if probabilities is None or probabilities.empty:
            skipped.append((fold.number, "the test block held no usable rows"))
            continue

        truth = labels.reindex(probabilities.index)
        keep = truth.notna()
        actual = truth[keep].to_numpy(dtype=int)
        predicted = (probabilities[keep].to_numpy() >= 0.5).astype(int)

        # The constant predictor, given exactly what the model was given: this fold's
        # training block and nothing after it.
        train_labels = labels.reindex(fold.train).dropna()
        if train_labels.empty:
            skipped.append((fold.number, "the training block held no labelled rows"))
            continue
        majority = 1 if float((train_labels == 1.0).mean()) > 0.5 else 0

        scored.append(
            FoldScore(
                number=fold.number,
                n=int(actual.size),
                accuracy=float((predicted == actual).mean()),
                baseline_accuracy=float((actual == majority).mean()),
            )
        )

    if not scored:
        raise ValidationError(f"{spec.name} could not be scored on any fold")
    return PooledScore(
        model=spec.name,
        representation=representation,
        folds=tuple(scored),
        skipped=tuple(skipped),
    )
