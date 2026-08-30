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

None of it changes the verdict. What *does* change its margin is turnover: each model's
break-even depends on how often it actually changes position, and these models are
persistent rather than the coin flips `costs` had assumed. `flip_rate` measures it, so
a pooled score carries the threshold it personally has to clear - 51.23% for Naive
Bayes holding 5.7 bars, 52.67% for the trees - instead of one global 53.49% that fits
none of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from forecast_lab.contracts import ForecastLabError
from forecast_lab.research.costs import flip_rate
from forecast_lab.research.dependence import lag_one_autocorrelation
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
    #: Predictions and hits for this fold, in bar order. Kept rather than reduced to a
    #: mean because turnover and serial dependence are properties of the *sequence*: a
    #: flip rate cannot be recovered from an accuracy, and averaging first would have
    #: hidden the fact that these models hold positions for several bars.
    predictions: tuple[float, ...] = ()
    correct: tuple[float, ...] = ()

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

    @property
    def prediction_segments(self) -> tuple[tuple[float, ...], ...]:
        """One contiguous stretch per fold - never concatenated across a boundary."""
        return tuple(fold.predictions for fold in self.folds)

    @property
    def correct_segments(self) -> tuple[tuple[float, ...], ...]:
        return tuple(fold.correct for fold in self.folds)

    @property
    def flip_rate(self) -> float:
        """How often this model changes position, measured inside folds.

        The parameter every break-even in this project had been assuming at 0.5. These
        models are persistent, so their real thresholds are lower and the verdict's
        margin is thinner than what was published.
        """
        return flip_rate(self.prediction_segments)

    @property
    def persistence(self) -> float:
        """Lag-one autocorrelation of the predictions - the flip rate's other face.

        ADR-010 claimed this was "indistinguishable from zero" for every model here
        without measuring it. It runs +0.22 to +0.61, which is why the flip rates are
        nowhere near a half and why every break-even quoted before ADR-012 was too high.
        """
        return lag_one_autocorrelation(self.prediction_segments)


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
                predictions=tuple(predicted.astype(float)),
                correct=tuple((predicted == actual).astype(float)),
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
