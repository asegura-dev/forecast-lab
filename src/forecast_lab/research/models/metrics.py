"""Scoring a model, against the rules it has to beat.

Two things this module refuses to do, both of which the original analysis did.

**AUC is computed from probabilities.** The notebook that the technical report documents
passed hard predictions to `roc_auc_score`. With binary input the ROC curve has a single
interior point, so the statistic degenerates into a rescaled balanced accuracy - which is
why its Naive Bayes reports an AUC of exactly 0.5000 and why nothing in that column
carries the information it appears to. The other notebook used `predict_proba` correctly;
the report documents the one that did not.

**No score is reported without the baseline beside it.** An accuracy of 51.5% is not a
result, it is half of one. The other half is that predicting UP every time scores 51.46%
on the same block, and the pair is what the original never put on the same line - despite
both numbers appearing in its own output, two rows apart.

`edge` is the difference, and it is the only column worth reading first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_lab.contracts import ForecastLabError


class MetricError(ForecastLabError):
    """A score cannot be computed from what was provided."""


@dataclass(frozen=True)
class ModelScore:
    """One model on one block, with the baseline it has to clear."""

    model: str
    representation: str
    block: str
    n: int

    accuracy: float
    precision: float
    recall: float
    specificity: float
    #: Negative predictive value - precision's mirror on the DOWN class. Reported because
    #: the original's `classification_report` reported it, and because a model that is
    #: precise on UP and worthless on DOWN looks fine on precision alone.
    negative_predictive_value: float
    #: Harmonic mean of precision and recall. The original quoted it as its headline
    #: alongside accuracy, and an F1 of 0.68 beside an accuracy of 52% is a signature
    #: worth being able to reproduce: it is what a near-constant UP predictor scores.
    f1: float
    auc: float
    brier: float

    #: What a constant predictor of the training majority scores on this same block.
    baseline_accuracy: float
    #: How often the model says UP. A model that always says UP is a constant with extra
    #: steps, and this column is where that becomes obvious.
    predicted_up_rate: float
    components: int | None = None

    @property
    def key(self) -> str:
        return f"{self.model} [{self.representation}]"

    @property
    def edge(self) -> float:
        """Accuracy minus the baseline. The first number anyone should read.

        Negative means the model lost to a rule with no parameters. The original
        analysis's selected model scored -0.33 points here and was published as evidence
        that machine learning beats chance.
        """
        return self.accuracy - self.baseline_accuracy

    @property
    def beats_baseline(self) -> bool:
        return self.edge > 0.0


def score_model(
    *,
    model: str,
    representation: str,
    block: str,
    probabilities: pd.Series,
    labels: pd.Series,
    baseline_accuracy: float,
    threshold: float = 0.5,
    components: int | None = None,
) -> ModelScore:
    """Score one model's probabilities against the truth on one block."""
    truth = labels.reindex(probabilities.index)
    keep = truth.notna()
    y = truth[keep].to_numpy(dtype="int64")
    p = probabilities[keep].to_numpy(dtype="float64")
    if y.size == 0:
        raise MetricError(f"{model} on {block}: nothing to score")

    predicted = (p >= threshold).astype("int64")
    pred_up, act_up = predicted == 1, y == 1

    true_up = int((pred_up & act_up).sum())
    false_up = int((pred_up & ~act_up).sum())
    true_down = int((~pred_up & ~act_up).sum())
    false_down = int((~pred_up & act_up).sum())

    return ModelScore(
        model=model,
        representation=representation,
        block=block,
        n=int(y.size),
        accuracy=float((predicted == y).mean()),
        precision=_ratio(true_up, true_up + false_up),
        recall=_ratio(true_up, true_up + false_down),
        specificity=_ratio(true_down, true_down + false_up),
        negative_predictive_value=_ratio(true_down, true_down + false_down),
        f1=_harmonic(
            _ratio(true_up, true_up + false_up), _ratio(true_up, true_up + false_down)
        ),
        auc=_auc(y, p),
        # Brier is the mean squared error of the probability itself, so a model that is
        # right for the wrong reasons - confidently, and often - is penalised where
        # accuracy would not notice.
        brier=float(np.mean((p - y) ** 2)),
        baseline_accuracy=baseline_accuracy,
        predicted_up_rate=float(pred_up.mean()),
        components=components,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _harmonic(precision: float, recall: float) -> float:
    """F1: the harmonic mean, which is zero when either term is.

    The harmonic mean rather than the arithmetic one because it refuses to be rescued by
    one good half. A predictor that always says UP has recall 1.0 and precision equal to
    the class balance - an arithmetic mean would report about 0.76 for it, and F1 reports
    0.68, which is exactly the figure the original published beside its 52% accuracy.
    """
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC from probabilities, or 0.5 when the block has one class only.

    `roc_auc_score` raises on a single-class block rather than returning a value, and a
    metric that crashes the report it belongs to is worse than one that abstains - 0.5 is
    the honest answer when there is nothing to discriminate.
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


@dataclass(frozen=True)
class RocPoints:
    """A ROC curve as coordinates, so plotting needs no estimator library."""

    false_positive_rate: np.ndarray
    true_positive_rate: np.ndarray
    auc: float


def roc_points(probabilities: pd.Series, labels: pd.Series) -> RocPoints | None:
    """The curve's coordinates, or ``None`` when the block has a single class.

    Computed here rather than in the plotting module so that `sklearn` stays confined to
    this package: a figure is a drawing of numbers, and the numbers are this layer's job.
    """
    from sklearn.metrics import auc, roc_curve

    truth = labels.reindex(probabilities.index)
    keep = truth.notna()
    y = truth[keep].to_numpy(dtype="int64")
    p = probabilities[keep].to_numpy(dtype="float64")
    if y.size == 0 or len(np.unique(y)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y, p)
    return RocPoints(false_positive_rate=fpr, true_positive_rate=tpr, auc=float(auc(fpr, tpr)))


def confusion(probabilities: pd.Series, labels: pd.Series, threshold: float = 0.5) -> pd.DataFrame:
    """The 2x2 table, as a frame - rows are truth, columns are the prediction."""
    truth = labels.reindex(probabilities.index)
    keep = truth.notna()
    y = truth[keep].to_numpy(dtype="int64")
    predicted = (probabilities[keep].to_numpy(dtype="float64") >= threshold).astype("int64")

    table = np.zeros((2, 2), dtype="int64")
    for actual in (0, 1):
        for guess in (0, 1):
            table[actual, guess] = int(((y == actual) & (predicted == guess)).sum())
    return pd.DataFrame(table, index=["DOWN", "UP"], columns=["DOWN", "UP"])
