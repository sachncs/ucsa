"""Learning-curve fits that carry short-run results to the full run.

A design change that helps in the first minutes of training may not help
after twenty times more data. Judging it from one short run is a gamble; this
module replaces the gamble with arithmetic. Each candidate is trained for
several budgets in a geometric ladder (for example 200, 400, 800, 1600 steps,
each with its own annealed schedule), and its loss is fitted to the standard
learning-curve form

    L(t) = floor + A * t ** (-alpha)

whose three numbers say where the curve is heading (`floor`), how large the
reducible loss is now (`A`) and how fast it closes (`alpha`). Candidates are
then compared at the length that matters, by extrapolation, with an
uncertainty that comes from the measured run-to-run noise.

Two failure modes of short-run comparison are detected explicitly: curves
that cross inside the horizon (an early winner that loses later), and
extrapolations that reach so far beyond the data that no conclusion is
justified.
"""

import dataclasses
import math

import numpy as np

from ucsa.utils import precision

ALPHA_GRID = np.exp(np.linspace(math.log(0.02), math.log(3.0), 400))


@dataclasses.dataclass(frozen=True)
class Curve:
    """A fitted learning curve `L(t) = floor + amplitude * t ** -alpha`.

    Attributes:
      floor: Loss the curve approaches as training continues.
      amplitude: Reducible loss at `t = 1`.
      alpha: Decay exponent; larger means faster improvement.
      residual: Root-mean-square error of the fit on its own data.
      last_budget: Largest training length the fit saw.
    """

    floor: float
    amplitude: float
    alpha: float
    residual: float
    last_budget: float

    def predict(self, budget: float) -> float:
        """Returns the fitted loss after `budget` training units."""
        return self.floor + self.amplitude * budget**-self.alpha

    def reach(self, budget: float) -> float:
        """Returns how many times beyond the data `budget` extrapolates."""
        return budget / self.last_budget


def fit(budgets: np.ndarray, losses: np.ndarray) -> Curve:
    """Fits `floor + amplitude * t ** -alpha` by exact least squares.

    For a fixed `alpha` the model is linear in `(floor, amplitude)`, so each
    candidate exponent on a fine grid is solved exactly and the best is kept.
    Amplitude is constrained to be non-negative (loss does not rise with
    more training) and the floor to be non-negative.

    Args:
      budgets: Training lengths, at least three, distinct.
      losses: Held-out loss at each length.

    Returns:
      The best-fitting `Curve`.

    Raises:
      ValueError: If there are fewer than three points or shapes differ.
    """
    t = np.asarray(budgets, dtype=precision.NUMPY_DTYPE)
    y = np.asarray(losses, dtype=precision.NUMPY_DTYPE)
    if t.shape != y.shape or t.size < 3:
        raise ValueError("need at least three (budget, loss) points")
    best: Curve | None = None
    for alpha in ALPHA_GRID:
        x = t**-alpha
        design = np.stack([np.ones_like(x), x], axis=1)
        (floor, amp), *_ = np.linalg.lstsq(design, y, rcond=None)
        if amp < 0 or floor < 0:
            amp = max(amp, 0.0)
            floor = max(float(np.mean(y - amp * x)), 0.0)
        err = float(np.sqrt(np.mean((floor + amp * x - y) ** 2)))
        if best is None or err < best.residual:
            best = Curve(
                float(floor), float(amp), float(alpha), err, float(t.max())
            )
    assert best is not None
    return best


@dataclasses.dataclass(frozen=True)
class Forecast:
    """A predicted loss with an uncertainty interval.

    Attributes:
      mean: Median predicted loss.
      low: Lower end of the interval.
      high: Upper end of the interval.
      reach: How many times beyond the data the prediction extrapolates.
    """

    mean: float
    low: float
    high: float
    reach: float


def forecast(
    budgets: np.ndarray,
    losses: np.ndarray,
    target: float,
    noise: float,
    resamples: int = 400,
    seed: int = 0,
) -> Forecast:
    """Predicts the loss at `target` with noise-aware uncertainty.

    Each resample perturbs the measured losses by the run-to-run noise
    `noise` (the standard deviation seen when only the seed changes), refits
    the curve, and extrapolates. The spread of those predictions is the
    uncertainty, so noisier measurements and longer extrapolations both widen
    the interval.

    Args:
      budgets: Training lengths of the measurements.
      losses: Measured held-out losses.
      target: Training length to predict.
      noise: Standard deviation of a single measurement, in loss units.
      resamples: Number of perturbed refits.
      seed: Seed for the perturbations.

    Returns:
      The `Forecast` at `target`.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(budgets, dtype=precision.NUMPY_DTYPE)
    y = np.asarray(losses, dtype=precision.NUMPY_DTYPE)
    preds = [
        fit(t, y + rng.normal(0.0, noise, y.shape)).predict(target)
        for _ in range(resamples)
    ]
    low, mid, high = np.quantile(preds, [0.025, 0.5, 0.975])
    return Forecast(
        float(mid), float(low), float(high), target / float(t.max())
    )


def crossing(a: Curve, b: Curve, horizon: float) -> float | None:
    """Finds where two curves swap order, if they do before `horizon`.

    Args:
      a: First fitted curve.
      b: Second fitted curve.
      horizon: Largest training length to consider.

    Returns:
      The first budget in `(0, horizon]` where `a - b` changes sign, or None
      if one curve stays ahead for the whole range.
    """
    grid = np.exp(np.linspace(0.0, math.log(horizon), 4000))
    gap = np.array([a.predict(g) - b.predict(g) for g in grid])
    flips = np.nonzero(np.sign(gap[:-1]) * np.sign(gap[1:]) < 0)[0]
    return float(grid[flips[0]]) if flips.size else None


def paired_forecast_gap(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
    target: float,
    noise: float,
    resamples: int = 400,
    seed: int = 0,
) -> Forecast:
    """Forecasts `loss_a(target) - loss_b(target)` with its uncertainty.

    Args:
      a: `(budgets, losses)` of the first candidate.
      b: `(budgets, losses)` of the second candidate.
      target: Training length to compare at.
      noise: Run-to-run standard deviation of one measurement.
      resamples: Number of perturbed refits.
      seed: Seed for the perturbations.

    Returns:
      The gap forecast; negative means `a` is better. Its interval excluding
      zero is the criterion for a real difference at `target`.
    """
    rng = np.random.default_rng(seed)

    def refit(pair: tuple[np.ndarray, np.ndarray]) -> float:
        t = np.asarray(pair[0], dtype=precision.NUMPY_DTYPE)
        y = np.asarray(pair[1], dtype=precision.NUMPY_DTYPE)
        return fit(t, y + rng.normal(0.0, noise, y.shape)).predict(target)

    gaps = [refit(a) - refit(b) for _ in range(resamples)]
    low, mid, high = np.quantile(gaps, [0.025, 0.5, 0.975])
    reach = target / float(max(np.max(a[0]), np.max(b[0])))
    return Forecast(float(mid), float(low), float(high), reach)
