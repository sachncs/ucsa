"""Paired comparison of models on identical held-out windows.

Two models scored on the same windows differ by a per-window loss gap whose
mean is far better determined than either model's mean alone, because the
window-to-window difficulty cancels. A bootstrap over windows turns that gap
into a confidence interval, so a design change is judged by whether the
interval excludes zero, not by whether one number is smaller.
"""

import dataclasses
import math

import numpy as np
import torch
from torch.nn import functional

from ucsa.models import recurrent
from ucsa.training import engine


@torch.no_grad()
def window_nll(
    model: recurrent.Model,
    batches: engine.BatchIterator,
    count: int,
    tail: int | None = None,
) -> np.ndarray:
    """Returns the mean per-token NLL of each window, in nats.

    Args:
      model: Model to score.
      batches: Held-out batches in a fixed order.
      count: Maximum number of batches.
      tail: If given, score only the last `tail` positions of each window.

    Returns:
      A float array with one entry per window, in batch order.
    """
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for n, (x, y) in enumerate(batches):
        if n >= count:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)["logits"].float()
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        ).view(y.shape)
        if tail is not None:
            loss = loss[:, -tail:]
        rows.append(loss.mean(dim=1).cpu())
    model.train()
    return torch.cat(rows).numpy() if rows else np.zeros(0)


@dataclasses.dataclass(frozen=True)
class PairedResult:
    """Outcome of comparing two models on the same windows.

    Attributes:
      mean_gap: Mean of `a - b` per-window NLL; negative means `a` is better.
      low: Lower end of the confidence interval of the gap.
      high: Upper end of the confidence interval of the gap.
      significant: Whether the interval excludes zero.
      windows: Number of paired windows.
    """

    mean_gap: float
    low: float
    high: float
    significant: bool
    windows: int


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedResult:
    """Bootstraps the mean per-window gap between two models.

    Args:
      a: Per-window NLL of the first model.
      b: Per-window NLL of the second model, same windows and order.
      resamples: Number of bootstrap resamples.
      confidence: Confidence level of the interval.
      seed: Seed for the resampling.

    Returns:
      The gap `a - b` with its interval.

    Raises:
      ValueError: If the arrays differ in length or are empty.
    """
    if a.shape != b.shape or a.size == 0:
        raise ValueError("need two equal-length, non-empty arrays")
    gap = a - b
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, gap.size, size=(resamples, gap.size))
    means = gap[picks].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return PairedResult(
        mean_gap=float(gap.mean()),
        low=float(low),
        high=float(high),
        significant=bool(low > 0 or high < 0),
        windows=int(gap.size),
    )


def bits_gap(result: PairedResult) -> float:
    """Converts a nats-per-token gap into bits per token."""
    return result.mean_gap / math.log(2.0)
