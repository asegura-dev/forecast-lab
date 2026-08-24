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
from forecast_lab.research.splitting import Block, SplitError, TemporalSplit, temporal_split

__all__ = [
    "DEFAULT_SEED",
    "LONGEST_WINDOW",
    "STALENESS_SUFFIX",
    "AlignedPanel",
    "AlignmentError",
    "BaselineError",
    "BaselineReport",
    "Block",
    "ColumnReport",
    "Direction",
    "FeatureError",
    "FeatureMatrix",
    "LabelReport",
    "LabelingError",
    "Mode",
    "PolicyReport",
    "Scale",
    "ScaleVerdict",
    "Score",
    "SplitError",
    "StationarityError",
    "SymbolAlignment",
    "TemporalSplit",
    "align_to_target",
    "build_features",
    "evaluate_baselines",
    "evaluate_stationarity",
    "indicators",
    "label_direction",
    "probe_scale",
    "rolling_dead_band",
    "temporal_split",
]
