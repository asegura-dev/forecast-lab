"""Fitting a model, and the two places a temporal experiment leaks.

**Every transform is fitted on train alone.** The scaler's mean and variance, and the
PCA's components, are learned from the training block and then applied unchanged to
validation and test. Fitting a scaler on the whole matrix first is the commonest leak in
tabular work and the hardest to see, because nothing crashes and the number that comes
out is merely optimistic. `sklearn`'s `Pipeline` makes the correct version the easy one:
`fit` on train propagates to every step at once, and a step cannot be fitted twice by
accident.

**Selection happens on validation, never on test.** The original project ordered its
results by `Test_Accuracy` in one notebook and picked the winner with
`results_df['Test_AUC'].idxmax()` in the other. That is choosing a model by the number
that is supposed to be measuring it - and it is why the reported 51.53% is not an
out-of-sample figure at all, but the maximum of twelve in-sample draws - that notebook
compares six models on two representations. (An earlier version of this docstring said
thirty, which is the *other* notebook's count of five estimators by three representations
by two modes. The two were conflated; the argument holds and the effect is weaker than
claimed.) Here the test block is scored once, for every model, and the selection is
already made.

**Probabilities, not hard labels.** AUC is computed from `predict_proba`. The notebook
that the technical report documents passed hard predictions to `roc_auc_score`, which is
why its Naive Bayes reports an AUC of exactly 0.5000 and why that whole column carries no
information. The other notebook did it correctly; the report documents the one that did
not.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from forecast_lab.contracts import ForecastLabError
from forecast_lab.research.models.catalogue import RANDOM_STATE, ModelSpec

#: Variance retained by the two PCA settings the original project used.
PCA_VARIANCE = (0.95, 0.90)


class TrainingError(ForecastLabError):
    """A model cannot be fitted on what was provided."""


@dataclass(frozen=True)
class Fitted:
    """One model, on one representation, with its predictions on each block."""

    model: str
    representation: str  # "raw", "pca-95" or "pca-90"
    #: Probability of UP per block, aligned with the block's own index.
    probabilities: dict[str, pd.Series]
    #: Components retained, when the representation is a PCA.
    components: int | None = None
    #: Share of training variance each retained component explains. The curve behind the
    #: component count: reporting "6 components" without it hides whether the sixth was
    #: carrying anything or was rounding error.
    explained_variance: tuple[float, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.model} [{self.representation}]"


def build_pipeline(spec: ModelSpec, *, variance: float | None = None) -> Any:
    """Scaler, optional PCA, estimator - as one object that is fitted exactly once.

    ``variance`` selects a PCA retaining that fraction of the training set's variance;
    ``None`` skips the reduction. The scaler is included even when the estimator does not
    need it, because PCA does: it maximises variance, so a column measured in larger
    units would dominate the components purely by being larger.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[tuple[str, Any]] = []
    if spec.needs_scaling or variance is not None:
        steps.append(("scaler", StandardScaler()))
    if variance is not None:
        steps.append(("pca", PCA(n_components=variance, random_state=RANDOM_STATE)))
    steps.append(("model", spec.build()))
    return Pipeline(steps)


def fit_and_predict(
    spec: ModelSpec,
    features: pd.DataFrame,
    labels: pd.Series,
    blocks: dict[str, pd.DatetimeIndex],
    *,
    variance: float | None = None,
) -> Fitted:
    """Fit on the train block and produce UP probabilities for every block.

    ``blocks`` maps a name to the rows belonging to it and must contain ``train``. Rows
    whose label is missing - the 4.37% whose horizon spans a gap, and the exact ties -
    are dropped per block rather than globally, so a block's score is computed over
    exactly the rows it could have been scored over.
    """
    if "train" not in blocks:
        raise TrainingError("a train block is required; there is nothing to fit on")

    prepared = {name: _usable(features, labels, index) for name, index in blocks.items()}
    x_train, y_train = prepared["train"]
    if len(x_train) == 0:
        raise TrainingError(f"{spec.name}: the train block has no usable rows")
    if y_train.nunique() < 2:
        raise TrainingError(f"{spec.name}: the train block has only one class")

    pipeline = build_pipeline(spec, variance=variance)
    try:
        pipeline.fit(x_train, y_train)
    except (ValueError, MemoryError) as exc:
        raise TrainingError(f"{spec.name}: {exc}") from exc

    probabilities: dict[str, pd.Series] = {}
    for name, (x_block, _) in prepared.items():
        if len(x_block) == 0:
            continue
        # Every block is sliced from the same frame, so the columns are identical by
        # construction - but asserting it is cheap and the failure it prevents is a
        # model silently reading the wrong feature into the wrong coefficient.
        if not x_block.columns.equals(x_train.columns):
            raise TrainingError(f"{spec.name}: block {name} has different columns from train")
        # Column 1 is the probability of the positive class. Taking column 0 by mistake
        # yields a perfectly inverted model that still scores near 50%, so it looks like
        # noise rather than like a bug.
        proba = _predict_up(pipeline, x_block)
        probabilities[name] = pd.Series(proba, index=x_block.index, name="p_up")

    components = None
    explained: tuple[float, ...] = ()
    if variance is not None:
        pca = pipeline.named_steps["pca"]
        components = int(pca.n_components_)
        explained = tuple(float(v) for v in pca.explained_variance_ratio_)

    return Fitted(
        model=spec.name,
        representation="raw" if variance is None else f"pca-{int(variance * 100)}",
        probabilities=probabilities,
        components=components,
        explained_variance=explained,
    )


def _predict_up(pipeline: Any, block: pd.DataFrame) -> np.ndarray:
    """Probability of UP, without the feature-name warning LightGBM provokes.

    LightGBM records column names at fit time and hands sklearn a bare array back at
    predict time, so sklearn warns that the names are missing. It is cosmetic here - the
    caller has just asserted the columns are identical, and the ordering is positional
    either way - but it fires once per block per model, which on a six-model run buries
    the table underneath it and, worse, corrupts `--json` output when a caller redirects
    stderr into the same stream.

    Suppressed narrowly, by category and message, so a *different* validation warning
    still gets through.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*valid feature names.*", category=UserWarning)
        proba: np.ndarray = pipeline.predict_proba(block)[:, 1]
    return proba


def _usable(
    features: pd.DataFrame, labels: pd.Series, index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.Series]:
    """The rows of one block that carry both a complete feature vector and a label.

    Two separate reasons a row is unusable, and both matter downstream. A missing label
    means the horizon spanned a gap or the move was an exact tie. A missing feature means
    an auxiliary symbol had not started trading yet. Dropping them here, per block, keeps
    a block's score honest about its own denominator.
    """
    x = features.reindex(index)
    y = labels.reindex(index)
    keep = y.notna() & x.notna().all(axis=1)
    return x.loc[keep], y.loc[keep].astype(int)


def positive_rate(labels: pd.Series) -> float:
    """The share of UP among decided labels - the number a constant predictor scores."""
    decided = labels.dropna()
    return float((decided == 1).mean()) if len(decided) else 0.0


def hard(probabilities: pd.Series, threshold: float = 0.5) -> np.ndarray:
    """Turn probabilities into decisions.

    The threshold is 0.5 here and that is a placeholder rather than a choice: the
    decision threshold that matters is derived from transaction costs, not from the
    midpoint of a probability. It arrives with the cost model.
    """
    return (probabilities.to_numpy() >= threshold).astype(int)
