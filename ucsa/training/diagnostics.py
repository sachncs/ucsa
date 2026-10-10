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
from ucsa.utils import precision


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
        logits = model(x)["logits"]
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        ).view(y.shape)
        if tail is not None:
            loss = loss[:, -tail:]
        rows.append(loss.mean(dim=1).cpu())
    model.train()
    return (
        torch.cat(rows).numpy()
        if rows
        else torch.zeros(0, dtype=precision.DTYPE).numpy()
    )


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


@torch.no_grad()
def chunk_profile(
    model: recurrent.Model,
    batches: engine.BatchIterator,
    count: int,
    active_slots: int | None = None,
) -> np.ndarray:
    """Returns the mean NLL of each chunk position across windows.

    Entry `k` averages every token of the `k`-th chunk of every window. A
    persistent state that compresses the past makes later chunks cheaper to
    predict, so the profile falls; a model without state is flat, because
    each chunk sees only itself.

    Args:
      model: Model to profile.
      batches: Held-out batches in a fixed order.
      count: Maximum number of batches.
      active_slots: Read only this many state slots (None reads all).

    Returns:
      A float array of length `ceil(seq / chunk_size)`; empty without data.
    """
    model.eval()
    device = next(model.parameters()).device
    size = model.config.chunk_size
    total = None
    seen = 0
    for n, (x, y) in enumerate(batches):
        if n >= count:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x, active_slots=active_slots)["logits"]
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        ).view(y.shape)
        chunks = -(-loss.shape[1] // size)
        padded = functional.pad(loss, (0, chunks * size - loss.shape[1]))
        per_chunk = padded.view(loss.shape[0], chunks, size).sum(dim=(0, 2))
        total = per_chunk if total is None else total + per_chunk
        seen += loss.shape[0] * size
    model.train()
    return np.zeros(0) if total is None else (total / seen).cpu().numpy()


def state_rate_distortion(
    model: recurrent.Model,
    make_batches: engine.Batches,
    count: int,
    slot_counts: list[int],
    byte_lengths: torch.Tensor,
) -> list[dict[str, float]]:
    """Measures bits per byte against how much state the decoder may read.

    Each row is one point of a memory rate-distortion curve: the state a
    decoder is allowed to read (the rate, in stored bits) against the
    compression it achieves (the distortion, in bits per byte).

    Args:
      model: A model trained with `slot_dropout` so prefixes are meaningful.
      make_batches: Maps batches consumed to a held-out iterator.
      count: Batches per point.
      slot_counts: Numbers of leading slots to read.
      byte_lengths: Raw bytes per token id.

    Returns:
      One dict per slot count with `slots`, `state_bits` and `bpb`.
    """
    dim = model.config.hidden
    bits_per_value = torch.finfo(precision.DTYPE).bits
    rows = []
    for slots in slot_counts:
        metrics = engine.evaluate(
            model, make_batches(0), count, byte_lengths, active_slots=slots
        )
        rows.append(
            {
                "slots": float(slots),
                "state_bits": float(slots * dim * bits_per_value),
                "bpb": metrics["bpb_all"],
            }
        )
    return rows
