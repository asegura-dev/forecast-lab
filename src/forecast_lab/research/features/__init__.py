"""Turning bars into the matrix a model is fitted on.

Two rules govern everything here, and both exist because breaking them once produced a
number that looked right.

**Indicators are computed before alignment, never after** (`build`). Computing them on
carried-forward prices manufactures zero returns on exactly the rows where a venue is
shut - a session clock that a tree will happily learn and a report will call a macro
signal.

**No column may be a price level** (`stationarity`). Gold runs 1,616 to 4,378 across this
sample, so a feature carrying that level puts the test block outside the training data's
support. The rule is enforced by rescaling the input prices and observing which columns
move, rather than by trusting a list of names.
"""

from forecast_lab.research.features.build import (
    LATE_STARTERS,
    FeatureMatrix,
    Mode,
    build_features,
)
from forecast_lab.research.features.stationarity import (
    SCALE_PERCENTILE,
    SCALE_PROBE,
    SCALE_TOLERANCE,
    ColumnReport,
    PolicyReport,
    Scale,
    ScaleVerdict,
    StationarityError,
    evaluate_stationarity,
    probe_scale,
)
from forecast_lab.research.features.technical import (
    ADX_WINDOW,
    ATR_WINDOW,
    BOLLINGER_WINDOW,
    EMA_WINDOWS,
    LONGEST_WINDOW,
    MACD_WINDOW,
    REALISED_VOL_WINDOWS,
    ROC_WINDOW,
    RSI_WINDOW,
    SMA_WINDOWS,
    FeatureError,
    indicators,
)

__all__ = [
    "ADX_WINDOW",
    "ATR_WINDOW",
    "BOLLINGER_WINDOW",
    "EMA_WINDOWS",
    "LATE_STARTERS",
    "LONGEST_WINDOW",
    "MACD_WINDOW",
    "REALISED_VOL_WINDOWS",
    "ROC_WINDOW",
    "RSI_WINDOW",
    "SCALE_PERCENTILE",
    "SCALE_PROBE",
    "SCALE_TOLERANCE",
    "SMA_WINDOWS",
    "ColumnReport",
    "FeatureError",
    "FeatureMatrix",
    "Mode",
    "PolicyReport",
    "Scale",
    "ScaleVerdict",
    "StationarityError",
    "build_features",
    "evaluate_stationarity",
    "indicators",
    "probe_scale",
]
