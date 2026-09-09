"""Deterministic seeding utilities.

Seeds Python's :mod:`random`, NumPy, and PyTorch (CPU + CUDA) so that
training runs are reproducible. Also exposes a context manager that
saves and restores RNG state around stochastic operations that should
not affect reproducibility.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG.

    Args:
        seed: Seed value.
        deterministic: If ``True``, enables PyTorch deterministic algorithms.
            May degrade performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_seed() -> int:
    """Return the current torch seed."""
    return int(torch.initial_seed())


@contextmanager
def seed_context(seed: int) -> Iterator[None]:
    """Temporarily seed every RNG.

    Args:
        seed: Seed value for the context.
    """
    state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)


__all__ = ["get_seed", "seed_context", "set_seed"]


def environment_fingerprint() -> dict[str, str]:
    """internal: collect environment info for logging."""
    return {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
