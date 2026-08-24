"""Tests for the stationarity policy.

The policy is only worth having if it fails on the thing it exists to catch, so most of
these feed it a deliberate violation rather than a clean matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_lab.research import Scale, evaluate_stationarity, indicators, probe_scale
from forecast_lab.research.features import SCALE_TOLERANCE


def _bars(n: int = 300, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-03", periods=n, freq="h", tz="UTC")
    close = pd.Series(1800 + np.cumsum(rng.normal(0, 3, n)), index=index)
    return pd.DataFrame(
        {"open": close, "high": close + 2.0, "low": close - 2.0, "close": close}, index=index
    )


# --- layer 1, the only gate ---------------------------------------------------------


@pytest.mark.unit
def test_a_clean_matrix_passes() -> None:
    bars = _bars()
    report = evaluate_stationarity(indicators(bars), probe_scale(indicators, bars))
    assert report.passes
    assert report.violations == ()


@pytest.mark.unit
def test_a_price_level_fails_the_gate() -> None:
    """Without this the policy is decoration."""
    bars = _bars()

    def leaky(frame: pd.DataFrame) -> pd.DataFrame:
        out = indicators(frame)
        out["sma_20_raw"] = frame["close"].rolling(20).mean()
        return out

    report = evaluate_stationarity(leaky(bars), probe_scale(leaky, bars))
    assert not report.passes
    assert [c.name for c in report.violations] == ["sma_20_raw"]


@pytest.mark.unit
def test_the_probe_reports_how_far_a_column_moved() -> None:
    """A verdict without a magnitude cannot be argued with."""
    bars = _bars()

    def leaky(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"level": frame["close"]}, index=frame.index)

    verdict = probe_scale(leaky, bars)["level"]
    assert verdict.scale is Scale.DEPENDENT
    # Multiplied by ten, so it moved by nine times its own magnitude.
    assert verdict.change == pytest.approx(9.0, rel=0.01)
    assert verdict.change > SCALE_TOLERANCE


@pytest.mark.unit
def test_a_dimensionless_ratio_with_an_unstable_denominator_is_not_a_price_level() -> None:
    """Pins the threshold against the false positive that set it.

    `bb_pband` is (close - lower) / (upper - lower) - dimensionless by construction. But
    the denominator is a Bollinger band's width, which goes small in quiet stretches, so
    the division amplifies floating-point error. Measured on the real gold series it
    moves by 1.3e-09 under rescaling, and the first threshold was 1e-9: the line sat
    inside the noise and called a ratio a price level.

    The separation this asserts is the reason the threshold can be loosened safely: a
    genuine price level moves by nine, not by a billionth.
    """
    bars = _bars(800, seed=21)

    def ratio_with_a_small_denominator(frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["close"]
        band = close.rolling(20).std() * 2.0
        return pd.DataFrame(
            {
                "pband": (close - (close.rolling(20).mean() - band)) / (2 * band),
                "level": close,
            },
            index=frame.index,
        )

    scales = probe_scale(ratio_with_a_small_denominator, bars)
    assert scales["pband"].scale is Scale.FREE
    assert scales["level"].scale is Scale.DEPENDENT
    # Orders of magnitude between the noise and the real thing, which is what makes the
    # threshold safe rather than lucky.
    assert scales["level"].change / max(scales["pband"].change, 1e-18) > 1e6


@pytest.mark.unit
def test_a_handful_of_deviating_rows_does_not_condemn_a_column() -> None:
    """Why the verdict is a percentile rather than the maximum.

    `ta`'s ADX is a Wilder recursion over comparisons of consecutive highs and lows.
    Where two of those are equal to within floating-point error, rescaling can break the
    tie the other way, and the recursion carries that flipped comparison forward through
    a long run of rows. On the real gold series: **134 deviating rows out of 23,181
    (0.58%), at consecutive positions**, pushing the maximum to 7.1e-03 while the 99th
    percentile stays at 6.9e-10.

    ADX is dimensionless by construction, so the maximum was measuring float64 rather
    than the feature. This simulates the same shape - a clean column with a contaminated
    run - and asserts the verdict survives it while the maximum still reports it.
    """
    bars = _bars(1000, seed=31)

    def mostly_clean(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame["close"].pct_change().to_frame("ret")
        # A run of 5 rows out of 1,000 - 0.5%, matching the 0.58% measured on the real
        # series - perturbed only when the prices are large, which is exactly how a
        # rescaling-sensitive numerical artefact behaves. It has to stay under the 1%
        # the percentile discards, or the percentile would see it too.
        if frame["close"].iloc[0] > 5000:
            out.iloc[400:405] += 0.5
        return out

    verdict = probe_scale(mostly_clean, bars)["ret"]
    assert verdict.scale is Scale.FREE  # the percentile is unmoved
    assert verdict.worst > SCALE_TOLERANCE  # and the maximum still says so, visibly
    assert verdict.worst > verdict.change * 1000


@pytest.mark.unit
def test_a_price_level_moves_on_every_row_so_no_percentile_hides_it() -> None:
    """The other half of the percentile argument, and the one that makes it safe."""
    bars = _bars()

    def level(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"close_level": frame["close"]}, index=frame.index)

    verdict = probe_scale(level, bars)["close_level"]
    assert verdict.scale is Scale.DEPENDENT
    # The percentile and the maximum agree, because every single row moved.
    assert verdict.change == pytest.approx(verdict.worst, rel=0.2)


@pytest.mark.unit
def test_an_all_nan_column_is_undefined_rather_than_free() -> None:
    """Absence of evidence is reported as such, not as a pass."""
    bars = _bars()

    def empty(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"nothing": [float("nan")] * len(frame)}, index=frame.index)

    assert probe_scale(empty, bars)["nothing"].scale is Scale.UNDEFINED


# --- layer 2, reported and never decisive -------------------------------------------


@pytest.mark.unit
def test_the_tests_are_reported_but_do_not_decide() -> None:
    """A column can fail ADF and still be allowed. That is the design, not a bug.

    With n around 23,000 the ADF rejects almost anything, and both tests are invalid
    under the heteroskedasticity and regime change that characterise this data. Treating
    either as pass/fail would be a false guarantee.
    """
    bars = _bars()
    report = evaluate_stationarity(indicators(bars), probe_scale(indicators, bars))

    scored = [c for c in report.columns if c.adf_pvalue is not None]
    assert scored, "no column produced a p-value at all"
    assert all(c.allowed for c in scored if c.scale is not Scale.DEPENDENT)


@pytest.mark.unit
def test_a_degenerate_column_abstains_instead_of_crashing() -> None:
    """A diagnostic that crashes the report it belongs to is worse than one that says
    nothing."""
    frame = pd.DataFrame({"constant": [1.0] * 200, "tiny": [1.0, 2.0] + [float("nan")] * 198})
    report = evaluate_stationarity(frame, {})

    for column in report.columns:
        assert column.adf_pvalue is None
        assert column.kpss_pvalue is None
        assert column.adf_rejects_unit_root is None


@pytest.mark.unit
def test_the_row_cap_bounds_the_work() -> None:
    """ADF and KPSS on the full series, per column, would dominate the command."""
    frame = pd.DataFrame({"x": np.random.default_rng(1).normal(size=8000)})
    report = evaluate_stationarity(frame, {}, max_rows=1000)
    assert report.columns[0].n_finite == 1000
