"""Tests for the power calculation.

The one that matters is `test_the_measured_break_even_is_easy_to_detect`. It pins the
moment this project's own conclusion turned over: against an assumed cost the design
could not resolve the effect, and against the measured cost it resolves it with power to
spare - which is the difference between "we could not see" and "there was nothing".
"""

from __future__ import annotations

import pytest

from forecast_lab.research import PowerError, design, required_sample

# The two break-evens, as points above chance.
ASSUMED_EFFECT = 0.0192  # a 1 bp round trip
MEASURED_EFFECT = 0.0349  # the venue's median spread, 1.86 bps


# --- the arithmetic -------------------------------------------------------------------


@pytest.mark.unit
def test_more_observations_resolve_smaller_effects() -> None:
    small = design(1_000).minimum_detectable_effect
    large = design(16_000).minimum_detectable_effect
    assert large < small
    # Four times the sample halves the detectable effect - the square-root law.
    assert small == pytest.approx(large * 4, rel=0.01)


@pytest.mark.unit
def test_the_standard_error_uses_the_widest_variance() -> None:
    """p(1-p) peaks at one half, so 0.5 is the conservative choice and costs nothing
    on a question whose answers all sit within two points of it."""
    assert design(10_000).standard_error == pytest.approx(0.005)


@pytest.mark.unit
def test_a_two_sided_test_demands_a_larger_effect() -> None:
    """Asking "is it different" instead of "is it better" costs resolution.

    One-sided is the default here because the question is directional. Using two-sided
    while asking a one-sided question is quiet conservatism that makes a negative result
    harder to falsify.
    """
    one = design(5_000, one_sided=True).minimum_detectable_effect
    two = design(5_000, one_sided=False).minimum_detectable_effect
    assert two > one


# --- the finding this module produced -------------------------------------------------


@pytest.mark.unit
def test_the_measured_break_even_is_easy_to_detect() -> None:
    """The turn in the project's argument, pinned.

    For a week the claim was that a single split "cannot resolve the effect it exists to
    test" - true against an assumed 1 bp cost, which implies a 1.92-point effect against
    a 2.17-point MDE. The venue's measured spread puts break-even at 53.49%, so the
    effect worth finding is **3.49 points**, and the same design sees it at 99% power.

    The negative result therefore means something: nothing was found by an experiment
    that would almost certainly have found it.
    """
    reference = design(3_294, label="reference split")
    canonical = design(7_306, label="canonical split")

    # Against the assumed effect the reference design genuinely falls short.
    assert not reference.resolves(ASSUMED_EFFECT)

    # Against the measured one, both designs resolve it comfortably.
    assert reference.resolves(MEASURED_EFFECT)
    assert canonical.resolves(MEASURED_EFFECT)
    assert reference.power_for(MEASURED_EFFECT) > 0.98
    assert canonical.power_for(MEASURED_EFFECT) > 0.99


@pytest.mark.unit
def test_a_profitable_edge_needs_far_fewer_bars_than_are_available() -> None:
    """1,269 bars would do at the rounded 3.49 points, 1,268 at the measured 3.4927.

    The canonical test block holds 7,306 and the walk-forward design scores 40,587.
    """
    assert required_sample(MEASURED_EFFECT) < 1_500
    assert required_sample(ASSUMED_EFFECT) < 4_500
    # A half-point edge is a different matter, and is not what the question was about.
    assert required_sample(0.005) > 50_000


@pytest.mark.unit
def test_required_sample_and_the_detectable_effect_agree() -> None:
    """The two directions of the same calculation must not drift apart."""
    n = required_sample(0.02)
    assert design(n).minimum_detectable_effect <= 0.02


@pytest.mark.unit
def test_power_rises_with_the_effect_and_with_the_sample() -> None:
    small = design(2_000)
    assert small.power_for(0.01) < small.power_for(0.03)
    assert small.power_for(0.02) < design(8_000).power_for(0.02)


@pytest.mark.unit
def test_a_non_existent_effect_is_detected_at_the_false_positive_rate() -> None:
    """With no effect, "detection" happens exactly alpha of the time - by definition."""
    assert design(5_000, alpha=0.05).power_for(0.0) == pytest.approx(0.05)


# --- refusals -------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("n", [0, 1, -10])
def test_an_impossible_sample_is_refused(n: int) -> None:
    with pytest.raises(PowerError, match="at least two"):
        design(n)


@pytest.mark.unit
@pytest.mark.parametrize(("power", "alpha"), [(0.0, 0.05), (1.0, 0.05), (0.8, 0.0), (0.8, 1.0)])
def test_impossible_power_or_alpha_is_refused(power: float, alpha: float) -> None:
    with pytest.raises(PowerError):
        design(1_000, power=power, alpha=alpha)


@pytest.mark.unit
def test_a_non_positive_effect_is_refused() -> None:
    with pytest.raises(PowerError, match="must be positive"):
        required_sample(0.0)
