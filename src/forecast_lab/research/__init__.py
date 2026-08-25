"""Computation over data that has already been handed in.

This layer never reaches for data: no `open`, no reader, no HTTP client. It receives
frames and does not know where they came from (ADR-001 sec. 1), which is what lets every
function here be exercised from a test with synthetic input and no filesystem - and what
makes an experiment reproducible from stored inputs rather than from whatever the world
happened to look like when it ran.

A test enforces it rather than a convention.
"""

from forecast_lab.research.align import (
    STALENESS_SUFFIX,
    AlignedPanel,
    AlignmentError,
    SymbolAlignment,
    align_to_target,
)
from forecast_lab.research.baselines import (
    DEFAULT_SEED,
    BaselineError,
    BaselineReport,
    Score,
    evaluate_baselines,
)
from forecast_lab.research.features import (
    LONGEST_WINDOW,
    ColumnReport,
    FeatureError,
    FeatureMatrix,
    Mode,
    PolicyReport,
    Scale,
    ScaleVerdict,
    StationarityError,
    build_features,
    evaluate_stationarity,
    indicators,
    probe_scale,
)
from forecast_lab.research.labeling import (
    Direction,
    LabelingError,
    LabelReport,
    label_direction,
    rolling_dead_band,
)
from forecast_lab.research.models import (
    CATALOGUE,
    PCA_VARIANCE,
    Availability,
    Fitted,
    MetricError,
    ModelScore,
    ModelSpec,
    RocPoints,
    TrainingError,
    availability,
    confusion,
    fit_and_predict,
    positive_rate,
    roc_points,
    score_model,
)
from forecast_lab.research.plots import (
    PlotError,
    calibration_chart,
    confusion_grid,
    correlation_comparison,
    edge_chart,
    roc_chart,
)
from forecast_lab.research.splitting import Block, SplitError, TemporalSplit, temporal_split

__all__ = [
    "CATALOGUE",
    "DEFAULT_SEED",
    "LONGEST_WINDOW",
    "PCA_VARIANCE",
    "STALENESS_SUFFIX",
    "AlignedPanel",
    "AlignmentError",
    "Availability",
    "BaselineError",
    "BaselineReport",
    "Block",
    "ColumnReport",
    "Direction",
    "FeatureError",
    "FeatureMatrix",
    "Fitted",
    "LabelReport",
    "LabelingError",
    "MetricError",
    "Mode",
    "ModelScore",
    "ModelSpec",
    "PlotError",
    "PolicyReport",
    "RocPoints",
    "Scale",
    "ScaleVerdict",
    "Score",
    "SplitError",
    "StationarityError",
    "SymbolAlignment",
    "TemporalSplit",
    "TrainingError",
    "align_to_target",
    "availability",
    "build_features",
    "calibration_chart",
    "confusion",
    "confusion_grid",
    "correlation_comparison",
    "edge_chart",
    "evaluate_baselines",
    "evaluate_stationarity",
    "fit_and_predict",
    "indicators",
    "label_direction",
    "positive_rate",
    "probe_scale",
    "roc_chart",
    "roc_points",
    "rolling_dead_band",
    "score_model",
    "temporal_split",
]
