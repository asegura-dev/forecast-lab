"""Tests for the serial-dependence measurement.

The load-bearing one is `test_the_instrument_finds_dependence_that_is_really_there`. This
module's published result is that dependence is **absent** - a negative finding about the
project's own caveat - and a negative finding is only worth anything if the instrument
could have returned the opposite. So the same code is pointed at a series built to be
strongly dependent, and has to say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecast_lab.research import Dependence, DependenceError, measure_dependence

#: Small enough to keep the suite fast, large enough for a stable standard error.
REPLICATIONS = 200


def _independent(n: int, *, seed: int = 0, rate: float = 0.51) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) < rate).astype(float)


def _dependent(n: int, *, seed: int = 0, rho: float = 0.9) -> np.ndarray:
    """A correctness series with strong persistence, via a thresholded AR(1)."""
    rng = np.random.default_rng(seed)
    latent = np.zeros(n)
    for i in range(1, n):
        latent[i] = rho * latent[i - 1] + rng.normal()
    return (latent > np.quantile(latent, 0.49)).astype(float)


# --- the one that makes the negative result credible ----------------------------------


@pytest.mark.unit
def test_the_instrument_finds_dependence_that_is_really_there() -> None:
    """On a strongly persistent series the standard error must inflate sharply.

    Measured on an AR(1) at rho = 0.9: an inflation above 3 and an effective sample an
    order of magnitude below the row count. Without this, "we measured 0.99" would be
    indistinguishable from "we measured nothing".
    """
    result = measure_dependence([_dependent(6_000)], replications=REPLICATIONS)

    assert result.inflation > 2.0
    assert result.effective_sample < result.n / 2
    assert result.lag_one > 0.5
    assert max(result.blocks) > 10.0
    assert result.material


@pytest.mark.unit
def test_independent_draws_are_reported_as_independent() -> None:
    """The other half of the same claim: no false alarm on a series with no dependence.

    This is the shape the real correctness series turned out to have - inflation 0.97,
    lag-one -0.020, blocks around 3 - which is why the caveat ADR-011 carried dissolved.
    """
    result = measure_dependence([_independent(6_000)], replications=REPLICATIONS)

    assert result.inflation == pytest.approx(1.0, abs=0.15)
    assert not result.material
    assert abs(result.lag_one) < 0.05


# --- the mechanics --------------------------------------------------------------------


@pytest.mark.unit
def test_segments_are_never_joined_across_a_boundary() -> None:
    """Two folds whose join would fabricate a jump must not produce one.

    Each segment is constant, so within-segment dependence is nil and the lag-one
    autocorrelation must be zero. Concatenating first would see one enormous step at the
    seam and report dependence that no fold contains.
    """
    result = measure_dependence(
        [np.zeros(500), np.ones(500)], replications=REPLICATIONS
    )

    assert result.lag_one == pytest.approx(0.0, abs=1e-12)
    assert result.segments == (500, 500)
    assert result.n == 1_000


@pytest.mark.unit
def test_the_block_length_never_drops_below_one() -> None:
    """`1/b` is a probability, so a Politis-White selection of 0.1 has to be floored.

    Measured on fold 4 of the real series, which returned 0.1 before the floor existed.
    """
    result = measure_dependence([_independent(2_000, seed=3)], replications=REPLICATIONS)

    assert all(block >= 1.0 for block in result.blocks)


@pytest.mark.unit
def test_the_result_is_byte_reproducible() -> None:
    """A published bootstrap figure that moves between runs is not a measurement."""
    series = _independent(2_000)
    first = measure_dependence([series], replications=REPLICATIONS)
    second = measure_dependence([series], replications=REPLICATIONS)

    assert first.bootstrap_standard_error == second.bootstrap_standard_error


@pytest.mark.unit
def test_a_different_seed_moves_the_answer_only_slightly() -> None:
    """Reproducible is not the same as arbitrary: the estimate has to be stable."""
    series = _independent(4_000)
    a = measure_dependence([series], replications=REPLICATIONS, seed=1)
    b = measure_dependence([series], replications=REPLICATIONS, seed=99)

    assert a.bootstrap_standard_error == pytest.approx(b.bootstrap_standard_error, rel=0.15)


@pytest.mark.unit
def test_a_constant_segment_is_handled_rather_than_crashing() -> None:
    """A model that never changes its mind has no dependence to find, and no variance."""
    result = measure_dependence([np.ones(300)], replications=REPLICATIONS)

    assert result.blocks == (1.0,)
    assert result.bootstrap_standard_error == 0.0


@pytest.mark.unit
def test_the_naive_standard_error_is_the_familiar_one() -> None:
    result = measure_dependence([_independent(10_000)], replications=REPLICATIONS)

    assert result.naive_standard_error == pytest.approx(0.005)


@pytest.mark.unit
def test_material_is_a_stated_threshold_not_a_vibe() -> None:
    """Ten percent either way; the boundary is asserted so it cannot drift silently."""
    def at(bootstrap: float) -> Dependence:
        return Dependence(
            n=1_000,
            segments=(1_000,),
            blocks=(1.0,),
            lag_one=0.0,
            naive_standard_error=0.10,
            bootstrap_standard_error=bootstrap,
        )

    assert not at(0.109).material
    assert at(0.111).material
    # And it is symmetric: a deflated standard error is just as much a correction.
    assert at(0.089).material


# --- refusals -------------------------------------------------------------------------


@pytest.mark.unit
def test_an_empty_series_is_refused() -> None:
    with pytest.raises(DependenceError, match="no observations"):
        measure_dependence([])


@pytest.mark.unit
def test_a_single_observation_is_refused() -> None:
    with pytest.raises(DependenceError, match="at least two observations"):
        measure_dependence([[1.0]])


@pytest.mark.unit
def test_too_few_replications_are_refused() -> None:
    with pytest.raises(DependenceError, match="at least two replications"):
        measure_dependence([_independent(100)], replications=1)
