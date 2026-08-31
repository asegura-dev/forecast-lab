"""Tests for the significance battery.

The one that carries the argument is `test_a_constant_predictor_has_accuracy_and_no_skill`.
Everything this module exists for is in that case: a predictor that always says UP scores
52% on a series that rises 52% of the time, beats a coin flip, beats a naive binomial test
- and knows nothing. Pesaran-Timmermann is the test that says so, and if it ever stops
saying so this file fails.

The rest come in pairs. Each test is shown a case where the effect is real and a case
where it is not, because a battery that only ever returns "not significant" on this
project's data would be indistinguishable from one that is broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecast_lab.research import (
    SignificanceError,
    deflated_sharpe,
    holm,
    pesaran_timmermann,
    superior_predictive_ability,
)

#: Small enough to keep the suite fast; the bootstrap is the expensive part.
REPS = 200


def _informative(n: int, *, accuracy: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A predictor that is right ``accuracy`` of the time, by construction."""
    rng = np.random.default_rng(seed)
    actual = (rng.random(n) < 0.5).astype(float)
    correct = rng.random(n) < accuracy
    return np.where(correct, actual, 1 - actual), actual


# --- the case the module exists for ----------------------------------------------------


@pytest.mark.unit
def test_a_constant_predictor_has_accuracy_and_no_skill() -> None:
    """52% accuracy, zero information, and the test has to tell them apart.

    A series that rises 52% of the time hands an always-UP predictor 52% accuracy for
    free. Comparing that against a half calls it a win by two points; comparing it against
    what the two marginals imply on their own calls it exactly nothing, which is correct.
    """
    rng = np.random.default_rng(0)
    actual = (rng.random(30_000) < 0.52).astype(float)
    predicted = np.ones_like(actual)

    result = pesaran_timmermann(predicted, actual)

    assert result.accuracy > 0.51, "the constant predictor does score well"
    assert result.independent_accuracy == pytest.approx(result.accuracy, abs=1e-9)
    assert result.excess == pytest.approx(0.0, abs=1e-9)
    assert not result.significant()


@pytest.mark.unit
def test_a_predictor_with_real_information_is_detected() -> None:
    """The other half: a genuine 53% predictor must be found, or the test is useless."""
    predicted, actual = _informative(30_000, accuracy=0.53)

    result = pesaran_timmermann(predicted, actual)

    assert result.excess > 0.02
    assert result.statistic > 4
    assert result.significant()


@pytest.mark.unit
def test_a_coin_flip_predictor_is_not_detected() -> None:
    """And no false alarm on a predictor that knows nothing and is not constant."""
    rng = np.random.default_rng(3)
    actual = (rng.random(30_000) < 0.5).astype(float)
    predicted = (rng.random(30_000) < 0.5).astype(float)

    assert not pesaran_timmermann(predicted, actual).significant()


# --- Holm -------------------------------------------------------------------------------


@pytest.mark.unit
def test_holm_steps_down_and_stops_at_the_first_failure() -> None:
    """Once a p-value fails its threshold, every larger one fails too - by construction."""
    verdict = holm({"a": 0.001, "b": 0.02, "c": 0.3, "d": 0.9})

    assert verdict == {"a": True, "b": False, "c": False, "d": False}


@pytest.mark.unit
def test_holm_is_never_weaker_than_bonferroni() -> None:
    """The reason for choosing it: uniformly more powerful, no less valid.

    p = 0.02 with four tests fails Bonferroni (0.0125) and passes Holm at rank two
    (0.05/3), once the smaller p-value has already been rejected.
    """
    p_values = {"a": 0.001, "b": 0.014, "c": 0.4, "d": 0.6}
    bonferroni = {k: v < 0.05 / len(p_values) for k, v in p_values.items()}
    stepwise = holm(p_values)

    assert stepwise["b"] and not bonferroni["b"]
    assert all(stepwise[k] or not bonferroni[k] for k in p_values)


@pytest.mark.unit
def test_holm_rejects_nothing_when_nothing_deserves_it() -> None:
    assert not any(holm({"a": 0.2, "b": 0.5, "c": 0.9}).values())


# --- SPA and StepM ----------------------------------------------------------------------


@pytest.mark.unit
def test_spa_finds_a_model_that_genuinely_beats_the_benchmark() -> None:
    """One model with lower loss, four without. It must name the one."""
    rng = np.random.default_rng(1)
    n = 3_000
    benchmark = rng.normal(0, 1, n)
    models = {f"noise-{i}": benchmark + rng.normal(0, 1, n) for i in range(4)}
    models["real"] = benchmark + rng.normal(0, 1, n) - 0.20

    result = superior_predictive_ability(benchmark, models, reps=REPS, block_size=2)

    assert result.any_survives
    assert "real" in result.better
    assert "real" in result.stepwise
    assert not any(name.startswith("noise") for name in result.stepwise)


@pytest.mark.unit
def test_spa_finds_nothing_when_every_model_is_noise() -> None:
    """The case this project's own data produces, and the one it must not fake."""
    rng = np.random.default_rng(2)
    n = 3_000
    benchmark = rng.normal(0, 1, n)
    models = {f"noise-{i}": benchmark + rng.normal(0, 1, n) for i in range(5)}

    result = superior_predictive_ability(benchmark, models, reps=REPS, block_size=2)

    assert not result.any_survives
    assert result.stepwise == ()


@pytest.mark.unit
def test_the_names_come_back_rather_than_positions() -> None:
    """Pins a defect this test found: the two `arch` APIs are not symmetric.

    Given the same DataFrame, `StepM.superior_models` returns column *names* while
    `SPA.better_models` returns positional *indices*. The first version of this module
    passed both through `str()`, so a superior model at position 4 was reported as the
    label `'4'` - valid-looking and belonging to nothing. Nothing else would have caught
    it, because the count was right and only the identity was wrong.
    """
    rng = np.random.default_rng(4)
    n = 2_000
    benchmark = rng.normal(0, 1, n)
    models = {
        "alpha": benchmark + rng.normal(0, 1, n) - 0.3,
        "beta": benchmark + rng.normal(0, 1, n),
    }

    result = superior_predictive_ability(benchmark, models, reps=REPS, block_size=2)

    assert result.models == ("alpha", "beta")
    assert all(isinstance(name, str) for name in result.better)
    assert "alpha" in result.better


# --- the Deflated Sharpe Ratio ----------------------------------------------------------


@pytest.mark.unit
def test_more_trials_demand_a_higher_sharpe() -> None:
    """The whole point of deflating: a Sharpe is less impressive as the best of many."""
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0004, 0.01, 5_000)

    one = deflated_sharpe(returns, trials=1)
    many = deflated_sharpe(returns, trials=100)

    assert many.expected_maximum > one.expected_maximum
    assert many.deflated < one.deflated
    # The undeflated probability does not depend on how many were tried.
    assert one.probabilistic == pytest.approx(many.probabilistic)


@pytest.mark.unit
def test_a_strong_sharpe_survives_a_modest_search() -> None:
    rng = np.random.default_rng(6)
    returns = rng.normal(0.003, 0.01, 5_000)  # Sharpe about 0.3 per bar

    assert deflated_sharpe(returns, trials=18).survives()


@pytest.mark.unit
def test_a_negative_sharpe_never_survives() -> None:
    """The case measured here: the best configuration loses money."""
    rng = np.random.default_rng(7)
    returns = rng.normal(-0.0002, 0.01, 5_000)

    verdict = deflated_sharpe(returns, trials=18)

    assert verdict.sharpe < 0
    assert verdict.deflated < 0.5
    assert not verdict.survives()


@pytest.mark.unit
def test_the_shape_gold_actually_has_widens_the_interval() -> None:
    """Negative skew and fat tails together, which is what hourly gold returns look like.

    Two earlier versions of this test were wrong, and both errors are worth keeping:

    1. The first compared two independently drawn samples, so their Sharpe ratios differed
       (0.061 against 0.055) and the higher one swamped the shape penalty. Comparing two
       things that differ in two ways measures neither.
    2. The second held the Sharpe fixed but asked whether *fat tails alone* reduce
       certainty. They do not. In `1 - skew*SR + (kurtosis-1)/4 * SR^2`, **positive skew
       reduces the variance** - a series whose surprises are to the upside has a more
       reliable Sharpe, and the heavy-tailed draw happened to have skew +6.9, which more
       than cancelled a kurtosis of 434.

    What actually costs certainty is negative skew *with* fat tails: rare large losses.
    Measured on the best configuration here, skew is -0.75 and kurtosis 28.3.
    """
    rng = np.random.default_rng(8)
    n = 20_000

    def _at(sample: np.ndarray, *, sharpe: float) -> np.ndarray:
        return (sample - sample.mean()) / sample.std() + sharpe

    gaussian = _at(rng.normal(size=n), sharpe=0.05)
    # Ninety-nine quiet bars to every crash, which is the shape of a real return series.
    jumps = rng.normal(size=n) - 8.0 * (rng.random(n) < 0.01)
    crash_prone = _at(jumps, sharpe=0.05)

    quiet = deflated_sharpe(gaussian, trials=1)
    violent = deflated_sharpe(crash_prone, trials=1)

    assert quiet.sharpe == pytest.approx(violent.sharpe, rel=1e-6)
    assert violent.skew < -1.0, "the point of the fixture is the left tail"
    assert violent.kurtosis > quiet.kurtosis
    assert violent.probabilistic < quiet.probabilistic


# --- refusals ----------------------------------------------------------------------------


@pytest.mark.unit
def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(SignificanceError, match="predictions against"):
        pesaran_timmermann([1.0, 0.0, 1.0], [1.0, 0.0])


@pytest.mark.unit
def test_a_benchmark_with_no_models_is_refused() -> None:
    with pytest.raises(SignificanceError, match="no models"):
        superior_predictive_ability(np.zeros(100), {})


@pytest.mark.unit
def test_a_model_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(SignificanceError, match="observations"):
        superior_predictive_ability(np.zeros(100), {"a": np.zeros(50)}, reps=REPS)


@pytest.mark.unit
def test_a_constant_return_series_has_no_sharpe() -> None:
    with pytest.raises(SignificanceError, match="constant return series"):
        deflated_sharpe(np.ones(100), trials=1)


@pytest.mark.unit
def test_too_few_returns_are_refused() -> None:
    with pytest.raises(SignificanceError, match="at least four"):
        deflated_sharpe([0.1, 0.2], trials=1)


@pytest.mark.unit
def test_correcting_an_empty_family_is_refused() -> None:
    with pytest.raises(SignificanceError, match="no p-values"):
        holm({})
