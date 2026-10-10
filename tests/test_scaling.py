import numpy as np
import pytest

from ucsa.training import scaling

LADDER = np.array([200.0, 400.0, 800.0, 1600.0])


def curve(floor, amp, alpha):
    return lambda t: floor + amp * np.asarray(t, dtype=float) ** -alpha


# ------------------------------------------------------------------ fitting


def test_exact_data_recovers_the_curve_parameters():
    truth = curve(2.0, 30.0, 0.5)
    fitted = scaling.fit(LADDER, truth(LADDER))
    assert fitted.floor == pytest.approx(2.0, abs=0.05)
    assert fitted.alpha == pytest.approx(0.5, abs=0.03)
    assert fitted.amplitude == pytest.approx(30.0, rel=0.1)
    assert fitted.residual < 1e-3


def test_extrapolation_to_the_full_run_is_accurate_on_clean_data():
    truth = curve(1.5, 25.0, 0.4)
    fitted = scaling.fit(LADDER, truth(LADDER))
    assert fitted.predict(12000) == pytest.approx(truth(12000), abs=0.02)


def test_noisy_data_still_extrapolates_within_the_forecast_interval():
    truth = curve(2.0, 20.0, 0.45)
    rng = np.random.default_rng(0)
    hits = 0
    for trial in range(40):
        y = truth(LADDER) + rng.normal(0, 0.03, 4)
        fc = scaling.forecast(
            LADDER, y, 12000, noise=0.03, seed=trial, resamples=150
        )
        hits += fc.low <= truth(12000) <= fc.high
    assert hits >= 30  # the stated 95% interval is honest enough


def test_loss_never_forecast_to_rise_with_more_training():
    y = np.array([5.0, 4.0, 4.2, 4.1])  # a noisy, non-monotone measurement
    fitted = scaling.fit(LADDER, y)
    assert fitted.amplitude >= 0
    assert fitted.predict(12000) <= fitted.predict(200) + 1e-9


def test_fewer_than_three_points_are_rejected():
    with pytest.raises(ValueError, match="three"):
        scaling.fit(np.array([1.0, 2.0]), np.array([3.0, 2.0]))
    with pytest.raises(ValueError):
        scaling.fit(LADDER, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------- uncertainty and reach


def test_uncertainty_grows_with_measurement_noise():
    y = curve(2.0, 20.0, 0.45)(LADDER)
    quiet = scaling.forecast(LADDER, y, 12000, noise=0.005, resamples=200)
    noisy = scaling.forecast(LADDER, y, 12000, noise=0.08, resamples=200)
    assert (noisy.high - noisy.low) > 3 * (quiet.high - quiet.low)


def test_uncertainty_grows_the_further_the_extrapolation_reaches():
    y = curve(2.0, 20.0, 0.45)(LADDER)
    near = scaling.forecast(LADDER, y, 3200, noise=0.03, resamples=200)
    far = scaling.forecast(LADDER, y, 120000, noise=0.03, resamples=200)
    assert (far.high - far.low) > (near.high - near.low)
    assert far.reach > near.reach > 1


def test_reach_is_measured_in_multiples_of_the_longest_run():
    y = curve(2.0, 20.0, 0.45)(LADDER)
    assert scaling.forecast(LADDER, y, 12000, noise=0.01).reach == (
        pytest.approx(12000 / 1600)
    )


# ------------------------------------------- the failure mode this exists for


def test_a_crossover_is_predicted_from_short_runs_only():
    """B leads early (fast, low ceiling); A improves slower but goes lower.
    A one-budget comparison at 400 steps picks B; the curves say A wins."""
    a, b = curve(1.0, 30.0, 0.35), curve(2.5, 12.0, 0.6)
    ya, yb = a(LADDER), b(LADDER)
    assert yb[1] < ya[1]  # B is ahead at every measured budget...
    assert yb[-1] < ya[-1]
    fa, fb = scaling.fit(LADDER, ya), scaling.fit(LADDER, yb)
    flip = scaling.crossing(fa, fb, horizon=12000)
    assert flip is not None  # ...but the fits foresee the swap
    assert 1600 < flip < 12000
    assert fa.predict(12000) < fb.predict(12000)  # A is better at 12k
    assert a(12000) < b(12000)  # and the truth agrees


def test_curves_that_never_cross_report_no_crossing():
    fa = scaling.fit(LADDER, curve(1.0, 10.0, 0.5)(LADDER))
    fb = scaling.fit(LADDER, curve(1.5, 10.0, 0.5)(LADDER))
    assert scaling.crossing(fa, fb, horizon=12000) is None


def test_a_real_gap_survives_extrapolation_and_noise_does_not():
    good = curve(1.8, 20.0, 0.45)(LADDER)
    base = curve(2.0, 20.0, 0.45)(LADDER)
    real = scaling.paired_forecast_gap(
        (LADDER, good), (LADDER, base), 12000, noise=0.01, resamples=200
    )
    assert real.high < 0  # clearly better, interval excludes zero
    same = scaling.paired_forecast_gap(
        (LADDER, base), (LADDER, base), 12000, noise=0.05, resamples=200
    )
    assert same.low < 0 < same.high  # indistinguishable
