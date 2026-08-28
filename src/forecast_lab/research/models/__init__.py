"""Fitting models, and refusing the two shortcuts that invalidated the original result.

**Every transform is fitted on train alone** (`training`). The scaler's statistics and
the PCA's components come from the training block and are applied unchanged elsewhere.

**Selection happens on validation** (`training`). The original ordered its results by
`Test_Accuracy` in one notebook and picked its winner with `Test_AUC.idxmax()` in the
other, which turns the test set into a hyperparameter.

And one thing this layer adds that the original had nowhere: **no score is reported
without its baseline on the same line** (`metrics`). The pair is what makes 51.53%
readable as -0.33 rather than as a result.

Which estimators exist is discovered rather than assumed (`catalogue`), because two of
them load a native DLL that an application control policy can refuse - and a comparison
that quietly omits a model is a comparison of a different experiment.
"""

from forecast_lab.research.models.catalogue import (
    CATALOGUE,
    RANDOM_STATE,
    Availability,
    ModelSpec,
    availability,
)
from forecast_lab.research.models.metrics import (
    MetricError,
    ModelScore,
    RocPoints,
    confusion,
    roc_points,
    score_model,
)
from forecast_lab.research.models.training import (
    PCA_VARIANCE,
    Fitted,
    TrainingError,
    build_pipeline,
    fit_and_predict,
    hard,
    positive_rate,
)
from forecast_lab.research.models.validation import (
    FoldScore,
    PooledScore,
    ValidationError,
    score_walk_forward,
)

__all__ = [
    "CATALOGUE",
    "PCA_VARIANCE",
    "RANDOM_STATE",
    "Availability",
    "Fitted",
    "FoldScore",
    "MetricError",
    "ModelScore",
    "ModelSpec",
    "PooledScore",
    "RocPoints",
    "TrainingError",
    "ValidationError",
    "availability",
    "build_pipeline",
    "confusion",
    "fit_and_predict",
    "hard",
    "positive_rate",
    "roc_points",
    "score_model",
    "score_walk_forward",
]
