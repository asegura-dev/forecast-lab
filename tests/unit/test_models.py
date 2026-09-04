"""Tests for fitting models without leaking the answer into them.

The ones that matter are the leakage tests. A model that has seen the test set's
distribution scores better and nothing crashes, so the only way to know is to arrange a
case where the difference is visible and assert it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_lab.research import (
    CATALOGUE,
    Availability,
    TrainingError,
    availability,
    confusion,
    fit_and_predict,
    positive_rate,
    score_model,
)
from forecast_lab.research.models import build_pipeline

N = 600


def _data(seed: int = 3, *, signal: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    """Features and labels with a controllable amount of real signal."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-03", periods=N, freq="h", tz="UTC")
    x = rng.normal(size=(N, 5))
    logit = signal * x[:, 0] + rng.normal(0, 1, N)
    y = (logit > 0).astype("float64")
    return (
        pd.DataFrame(x, index=index, columns=[f"f{i}" for i in range(5)]),
        pd.Series(y, index=index, name="label"),
    )


def _blocks(index: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    cut, second = int(len(index) * 0.7), int(len(index) * 0.85)
    return {
        "train": index[:cut],
        "validation": index[cut:second],
        "test": index[second:],
    }


def _spec(name: str) -> object:
    return next(s for s in CATALOGUE if s.name == name)


# --- the leak that does not crash ---------------------------------------------------


@pytest.mark.unit
def test_the_scaler_is_fitted_on_train_only() -> None:
    """The commonest leak in tabular work, and the quietest.

    Test features are shifted by a large constant. A scaler fitted on train alone leaves
    that shift visible in the transformed test data; one fitted on everything would
    absorb it and quietly hand the model a distribution it should never have seen.
    """
    x, y = _data()
    blocks = _blocks(pd.DatetimeIndex(x.index))
    x.loc[blocks["test"], :] += 50.0

    pipeline = build_pipeline(_spec("Logistic Regression"))  # type: ignore[arg-type]
    pipeline.fit(x.loc[blocks["train"]], y.loc[blocks["train"]].astype(int))

    scaled_test = pipeline.named_steps["scaler"].transform(x.loc[blocks["test"]])
    # Untouched by test's own statistics, so the shift survives transformation.
    assert float(np.mean(scaled_test)) > 10.0


@pytest.mark.unit
def test_pca_components_come_from_train_only() -> None:
    x, y = _data()
    blocks = _blocks(pd.DatetimeIndex(x.index))

    pipeline = build_pipeline(_spec("Random Forest"), variance=0.95)  # type: ignore[arg-type]
    pipeline.fit(x.loc[blocks["train"]], y.loc[blocks["train"]].astype(int))
    fitted_on = int(pipeline.named_steps["pca"].n_samples_)

    assert fitted_on == len(blocks["train"])


# --- what fit_and_predict guarantees ------------------------------------------------


@pytest.mark.unit
def test_probabilities_are_returned_for_every_block_and_aligned() -> None:
    x, y = _data()
    blocks = _blocks(pd.DatetimeIndex(x.index))

    fitted = fit_and_predict(_spec("Naive Bayes"), x, y, blocks)  # type: ignore[arg-type]

    assert set(fitted.probabilities) == {"train", "validation", "test"}
    for name, proba in fitted.probabilities.items():
        assert proba.index.equals(blocks[name])
        assert ((proba >= 0.0) & (proba <= 1.0)).all()


@pytest.mark.unit
def test_a_model_learns_a_signal_that_is_really_there() -> None:
    """A sanity check on the harness itself.

    Without it, every "no edge" result in this project could equally be a broken
    pipeline. Given a genuine signal, the AUC has to rise well above 0.5.
    """
    x, y = _data(signal=2.0)
    blocks = _blocks(pd.DatetimeIndex(x.index))

    fitted = fit_and_predict(_spec("Logistic Regression"), x, y, blocks)  # type: ignore[arg-type]
    score = score_model(
        model="LogReg",
        representation="raw",
        block="test",
        probabilities=fitted.probabilities["test"],
        labels=y,
        baseline_accuracy=positive_rate(y.reindex(blocks["train"])),
    )
    assert score.auc > 0.8, "the harness cannot detect a signal that is plainly there"


@pytest.mark.unit
def test_rows_without_a_label_are_dropped_per_block() -> None:
    """The 4.37% whose horizon spans a gap, and the exact ties."""
    x, y = _data()
    blocks = _blocks(pd.DatetimeIndex(x.index))
    y.iloc[:100] = float("nan")

    fitted = fit_and_predict(_spec("Naive Bayes"), x, y, blocks)  # type: ignore[arg-type]
    assert len(fitted.probabilities["train"]) == len(blocks["train"]) - 100


@pytest.mark.unit
def test_pca_reports_how_many_components_it_kept() -> None:
    x, y = _data()
    fitted = fit_and_predict(
        _spec("Random Forest"), x, y, _blocks(pd.DatetimeIndex(x.index)), variance=0.95  # type: ignore[arg-type]
    )
    assert fitted.components is not None
    assert 0 < fitted.components <= len(x.columns)
    assert fitted.representation == "pca-95"


@pytest.mark.unit
def test_a_missing_train_block_is_refused() -> None:
    x, y = _data()
    index = pd.DatetimeIndex(x.index)
    with pytest.raises(TrainingError, match="train block is required"):
        fit_and_predict(_spec("Naive Bayes"), x, y, {"test": index})  # type: ignore[arg-type]


@pytest.mark.unit
def test_a_single_class_train_block_is_refused() -> None:
    """Fitting on one class produces a constant that reports a confident accuracy."""
    x, y = _data()
    blocks = _blocks(pd.DatetimeIndex(x.index))
    y.loc[blocks["train"]] = 1.0
    with pytest.raises(TrainingError, match="only one class"):
        fit_and_predict(_spec("Naive Bayes"), x, y, blocks)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("name", ["Random Forest", "Logistic Regression", "Naive Bayes"])
def test_fitting_twice_gives_bit_identical_probabilities(name: str) -> None:
    """A published number has to recompute to the same bits, not merely to the same story.

    This caught a real defect. `RandomForestClassifier(n_jobs=-1)` - the original's
    setting - sums 100 tree votes in whatever order the workers finish, and
    floating-point addition is not associative, so probabilities moved by up to 3.3e-16
    between runs and the command's JSON output was not byte-reproducible. Nothing about
    any conclusion changed; the point is that it could not be *shown* not to have.

    Asserted with `array_equal` rather than `allclose` on purpose: approximate equality
    is what let the problem exist unnoticed.
    """
    x, y = _data()
    blocks = _blocks(pd.DatetimeIndex(x.index))

    first = fit_and_predict(_spec(name), x, y, blocks)  # type: ignore[arg-type]
    second = fit_and_predict(_spec(name), x, y, blocks)  # type: ignore[arg-type]

    for block in first.probabilities:
        assert np.array_equal(
            first.probabilities[block].to_numpy(), second.probabilities[block].to_numpy()
        ), f"{name} is not reproducible on the {block} block"


# --- scoring ------------------------------------------------------------------------


@pytest.mark.unit
def test_the_edge_is_accuracy_minus_the_baseline() -> None:
    """The one column worth reading first, and the one the original never printed."""
    index = pd.date_range("2022-01-03", periods=100, freq="h", tz="UTC")
    labels = pd.Series([1.0] * 60 + [0.0] * 40, index=index)
    always_up = pd.Series([0.9] * 100, index=index)

    score = score_model(
        model="constant",
        representation="raw",
        block="test",
        probabilities=always_up,
        labels=labels,
        baseline_accuracy=0.60,
    )
    assert score.accuracy == pytest.approx(0.60)
    assert score.edge == pytest.approx(0.0)
    assert not score.beats_baseline
    # The signature: perfect recall, no specificity.
    assert score.recall == pytest.approx(1.0)
    assert score.specificity == pytest.approx(0.0)


@pytest.mark.unit
def test_auc_uses_probabilities_and_not_hard_labels() -> None:
    """Defect D7, stated as a test.

    The notebook the technical report documents passed hard predictions to
    `roc_auc_score`. With binary input the ROC curve has one interior point, so the
    statistic collapses. Here a well-ranked but badly-thresholded model must still show
    a high AUC - which is exactly the information hard labels destroy.
    """
    index = pd.date_range("2022-01-03", periods=100, freq="h", tz="UTC")
    labels = pd.Series([0.0] * 50 + [1.0] * 50, index=index)
    # Perfectly ranked, but every probability sits below 0.5.
    ranked = pd.Series(np.linspace(0.01, 0.49, 100), index=index)

    score = score_model(
        model="ranked",
        representation="raw",
        block="test",
        probabilities=ranked,
        labels=labels,
        baseline_accuracy=0.5,
    )
    assert score.auc == pytest.approx(1.0)
    assert score.accuracy == pytest.approx(0.5)  # every prediction is DOWN


@pytest.mark.unit
def test_a_single_class_block_abstains_instead_of_crashing() -> None:
    index = pd.date_range("2022-01-03", periods=50, freq="h", tz="UTC")
    score = score_model(
        model="x",
        representation="raw",
        block="test",
        probabilities=pd.Series([0.6] * 50, index=index),
        labels=pd.Series([1.0] * 50, index=index),
        baseline_accuracy=1.0,
    )
    assert score.auc == 0.5


@pytest.mark.unit
def test_the_confusion_table_is_truth_by_prediction() -> None:
    index = pd.date_range("2022-01-03", periods=4, freq="h", tz="UTC")
    labels = pd.Series([1.0, 1.0, 0.0, 0.0], index=index)
    proba = pd.Series([0.9, 0.1, 0.9, 0.1], index=index)

    table = confusion(proba, labels)
    assert table.loc["UP", "UP"] == 1
    assert table.loc["UP", "DOWN"] == 1
    assert table.loc["DOWN", "UP"] == 1
    assert table.loc["DOWN", "DOWN"] == 1


# --- availability -------------------------------------------------------------------


@pytest.mark.unit
def test_availability_is_discovered_by_building_each_model() -> None:
    """Checking the module would report available and then crash.

    Two estimators load a native DLL, and an application control policy can refuse it
    while the Python package imports perfectly well.
    """
    results = availability()
    assert len(results) == len(CATALOGUE)
    assert all(isinstance(r, Availability) for r in results)
    # sklearn ships no native DLL, so these must always be available.
    by_name = {r.name: r for r in results}
    for name in ("Logistic Regression", "Naive Bayes", "Random Forest"):
        assert by_name[name].available, by_name[name].reason


@pytest.mark.unit
def test_an_unavailable_model_carries_a_reason() -> None:
    from forecast_lab.research.models.catalogue import ModelSpec

    def blocked() -> object:
        raise OSError("[WinError 4551] an application control policy blocked this file")

    results = availability((ModelSpec("Blocked", blocked, needs_scaling=False),))
    assert not results[0].available
    assert "application control" in results[0].reason


# --- the two metrics the original published and this project did not compute -------------


@pytest.mark.unit
def test_a_constant_up_predictor_scores_the_f1_the_original_published() -> None:
    """The number that makes F1 worth carrying at all.

    `Proyecto_Final_Completo` reported **F1 0.68** beside 52% accuracy, as a headline. A
    predictor that always says UP has recall 1.0 and precision equal to the class balance,
    so its F1 is `2p/(1+p)` - and at the 51.86% balance of that block, that is **0.6830**.
    The figure published as performance is what a rule with no parameters scores.
    """
    balance = 0.5186
    index = pd.date_range("2022-01-03", periods=10_000, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    truth = pd.Series((rng.random(10_000) < balance).astype(float), index=index)
    always_up = pd.Series(1.0, index=index)

    score = score_model(
        model="always-UP",
        representation="raw",
        block="test",
        probabilities=always_up,
        labels=truth,
        baseline_accuracy=balance,
    )

    assert score.recall == pytest.approx(1.0)
    assert score.f1 == pytest.approx(2 * balance / (1 + balance), abs=0.01)
    assert score.f1 > 0.67
    # And the thing F1 hides, which is why this project never leads with it.
    assert score.specificity == 0.0


@pytest.mark.unit
def test_f1_is_the_harmonic_mean_and_refuses_to_be_rescued_by_one_half() -> None:
    """The arithmetic mean would report 0.5 for a predictor that never finds a positive."""
    index = pd.date_range("2022-01-03", periods=100, freq="h", tz="UTC")
    truth = pd.Series([1.0] * 50 + [0.0] * 50, index=index)
    always_down = pd.Series(0.0, index=index)

    score = score_model(
        model="always-DOWN",
        representation="raw",
        block="test",
        probabilities=always_down,
        labels=truth,
        baseline_accuracy=0.5,
    )

    assert score.recall == 0.0
    assert score.f1 == 0.0


@pytest.mark.unit
def test_the_negative_predictive_value_mirrors_precision_on_the_other_class() -> None:
    """A model precise on UP and worthless on DOWN looks fine on precision alone.

    Here every UP call is right and every DOWN call is wrong, so precision is 1.0 and NPV
    is 0.0 - the asymmetry the original's classification report showed and this project
    could not, until now.
    """
    index = pd.date_range("2022-01-03", periods=100, freq="h", tz="UTC")
    truth = pd.Series([1.0] * 50 + [1.0] * 50, index=index)
    half_up = pd.Series([0.9] * 50 + [0.1] * 50, index=index)

    score = score_model(
        model="half", representation="raw", block="test",
        probabilities=half_up, labels=truth, baseline_accuracy=1.0,
    )

    assert score.precision == pytest.approx(1.0)
    assert score.negative_predictive_value == pytest.approx(0.0)
