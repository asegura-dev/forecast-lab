"""Which models are tried, and which ones this machine can actually load.

The five estimators come from the original project, with its hyperparameters unchanged.
Changing them would count as a fresh trial in the deflated-Sharpe accounting, and the
object here is to isolate the effect of the corrections rather than of a sweep.

**Availability is discovered rather than assumed**, which is the unusual part of this
module. Two of the five load a native DLL through `ctypes`, and Windows Smart App Control
refuses any binary without a known publisher's signature or an established reputation -
including, on a clean Windows 11 install, the wheels PyPI ships. So an import that works
on most machines fails outright on some, and a pipeline that assumed otherwise would
either crash at the worst moment or - far worse - silently report a comparison over four
models while its output implied five.

Every estimator therefore reports whether it is available and, if not, why. The command
prints what it could not run. A result that quietly omits a model is a result about a
different experiment than the one it claims.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: Fixed across every estimator that takes one, so a run is a fact about the data rather
#: than about the day it happened.
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    """One candidate: how to build it, and whether it needs standardised inputs."""

    name: str
    #: Deferred so that importing this module never imports an estimator that might be
    #: blocked. The failure has to be catchable per model, not fatal at import time.
    build: Callable[[], Any]
    #: Distance-based and linear models need standardised inputs; trees do not care.
    needs_scaling: bool
    #: What the original project called it, when the name differs.
    origin: str = ""
    #: Present in this project but not in the original.
    added_here: bool = False
    notes: str = field(default="")


@dataclass(frozen=True)
class Availability:
    """Whether a model can be built on this machine."""

    spec: ModelSpec
    available: bool
    reason: str = ""

    @property
    def name(self) -> str:
        return self.spec.name


def _logistic_regression() -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


def _naive_bayes() -> Any:
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB()


def _random_forest() -> Any:
    from sklearn.ensemble import RandomForestClassifier

    # `n_jobs=1`, where the original used -1, and this is the one place a setting is
    # deliberately changed. It is not a hyperparameter: the forest is identical either
    # way, because `random_state` fixes every tree. What changes is the order in which
    # 100 tree votes are summed - and floating-point addition is not associative, so a
    # parallel reduction lands on a different last bit from run to run.
    #
    # Measured: with -1 the probabilities differ by up to 3.3e-16 between runs and the
    # command's JSON output is not byte-reproducible; with 1 it is exact. The cost is
    # 0.29s against 3.04s per fit, taking the whole command from about six seconds to
    # fourteen. A repository whose thesis is that a published number must recompute
    # identically does not get to trade that away for eight seconds.
    return RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=1
    )


def _hist_gradient_boosting() -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_depth=5, random_state=RANDOM_STATE)


def _xgboost() -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )


def _lightgbm() -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbose=-1
    )


#: The candidates, in the order the original listed them. Order is fixed rather than
#: taken from a set, because a table that reorders itself between runs is not a table
#: anyone can compare against a previous one.
CATALOGUE: tuple[ModelSpec, ...] = (
    ModelSpec("Logistic Regression", _logistic_regression, needs_scaling=True),
    ModelSpec("Naive Bayes", _naive_bayes, needs_scaling=True),
    ModelSpec("Random Forest", _random_forest, needs_scaling=False),
    ModelSpec("XGBoost", _xgboost, needs_scaling=False),
    ModelSpec(
        "LightGBM",
        _lightgbm,
        needs_scaling=False,
        notes="the model the original analysis selected, at 51.53% accuracy",
    ),
    ModelSpec(
        "HistGradientBoosting",
        _hist_gradient_boosting,
        needs_scaling=False,
        added_here=True,
        notes="gradient boosting with no native DLL, so it runs where the other two cannot",
    ),
)


def availability(catalogue: tuple[ModelSpec, ...] = CATALOGUE) -> tuple[Availability, ...]:
    """Try to build each model, and record what happened.

    Building is the only honest test. An estimator's package can import while the shared
    library it wraps is refused - which is exactly how LightGBM and XGBoost fail under
    Smart App Control - so checking for the module would report available and then crash.

    `OSError` is caught by name because that is what a blocked `ctypes.CDLL` raises;
    `ImportError` covers a package that simply is not installed. Nothing broader is
    caught, because a genuine bug inside an estimator's constructor should surface as a
    bug rather than as "unavailable".
    """
    results: list[Availability] = []
    for spec in catalogue:
        try:
            spec.build()
        except (ImportError, OSError) as exc:
            results.append(Availability(spec, available=False, reason=_reason(exc)))
        else:
            results.append(Availability(spec, available=True))
    return tuple(results)


def _reason(exc: BaseException) -> str:
    """A short, readable cause - the raw message is a wall of DLL paths."""
    text = str(exc)
    probe = text.lower()
    # Windows error 4551 is what an application control policy raises. The message
    # itself is localised, so the numeric code is the reliable half of the match.
    if "4551" in probe or "application control" in probe or "control de aplic" in probe:
        return "blocked by an application control policy (Windows Smart App Control)"
    if isinstance(exc, ImportError):
        return "not installed"
    return text.splitlines()[0][:120]
