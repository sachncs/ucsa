"""Hyperparameter search for UCSA-R: successive halving over config knobs.

Successive halving trains many candidates for a short budget, keeps the best
fraction, and re-trains the survivors with a larger budget. It spends most
compute on the configurations that are already winning, which is what makes
tuning a 60M-parameter model affordable on a single laptop GPU.

The search logic is pure (a scoring callable in, a ranking out) so it can be
tested without training anything.
"""

import dataclasses
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: Default search space: each entry is the list of values to sample from.
DEFAULT_SPACE: dict[str, list[Any]] = {
    "train.lr": [3e-4, 6e-4, 1e-3],
    "model.chunk_size": [64, 128, 256],
    "model.state_iterations": [1, 2, 3],
    "model.write_top_k": [0, 8, 16],
    "model.jepa_weight": [0.0, 0.1, 0.3],
    "model.bptt_chunks": [0, 2, 4],
}

Candidate = dict[str, Any]
Scorer = Callable[[Candidate, int], float]


@dataclasses.dataclass
class Trial:
    """One scored candidate.

    Attributes:
      candidate: The configuration that was scored.
      rung: Index of the budget rung.
      steps: Training steps used.
      score: Validation perplexity (lower is better); infinite on failure.
      error: Failure description, if any.
    """

    candidate: Candidate
    rung: int
    steps: int
    score: float
    error: str | None = None


@dataclasses.dataclass
class SearchResult:
    """Outcome of a search.

    Attributes:
      best: The winning configuration.
      best_score: Its final-rung score.
      trials: Every trial, in order.
    """

    best: Candidate
    best_score: float
    trials: list[Trial] = dataclasses.field(default_factory=list)


def sample_candidates(
    space: Mapping[str, Sequence[Any]], n: int, seed: int
) -> list[Candidate]:
    """Sample up to ``n`` distinct candidates (the full grid if smaller)."""
    keys = sorted(space)
    total = math.prod(len(space[k]) for k in keys)
    rng = random.Random(seed)
    if n >= total:
        grid: list[Candidate] = [{}]
        for k in keys:
            grid = [{**g, k: v} for g in grid for v in space[k]]
        return grid
    seen: set[str] = set()
    out: list[Candidate] = []
    while len(out) < n:
        cand = {k: rng.choice(list(space[k])) for k in keys}
        key = json.dumps(cand, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def successive_halving(
    candidates: Sequence[Candidate],
    score: Scorer,
    budgets: Sequence[int],
    keep: float = 1 / 3,
    log: Callable[[str], None] = print,
    cache_path: str | None = None,
) -> SearchResult:
    """Lower score is better (validation perplexity).

    Args:
        candidates: Configurations to try.
        score: ``(candidate, steps) -> score``; may raise, in which case the
            trial is recorded as failed and ranks last.
        budgets: Training steps per rung, increasing.
        keep: Fraction of candidates surviving each rung (at least one).
        log: Sink for one progress line per trial.
        cache_path: JSON file of finished trials; completed ones are reused
            so an interrupted search resumes without repeating work.

    Returns:
        The best candidate, its score and every trial.

    Raises:
        ValueError: If there are no candidates or budgets, or `keep` is not
            in (0, 1).
    """
    if not candidates or not budgets:
        raise ValueError("need at least one candidate and one budget")
    if not 0.0 < keep < 1.0:
        raise ValueError("keep must be in (0, 1)")
    cache: dict[str, float] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    trials: list[Trial] = []
    alive = list(candidates)
    ranked: list[tuple[float, Candidate]] = []
    for rung, steps in enumerate(budgets):
        ranked = []
        for cand in alive:
            key = f"{steps}|{json.dumps(cand, sort_keys=True)}"
            err = None
            if key in cache:
                val = cache[key]
            else:
                try:
                    val = float(score(cand, steps))
                    if not math.isfinite(val):
                        raise FloatingPointError("non-finite score")
                except (
                    Exception
                ) as exc:  # a bad config must not kill the search
                    val, err = math.inf, f"{type(exc).__name__}: {exc}"
                cache[key] = val
                if cache_path:
                    with open(cache_path, "w") as f:
                        json.dump(cache, f, indent=1)
            trials.append(Trial(cand, rung, steps, val, err))
            ranked.append((val, cand))
            log(f"  rung={rung} steps={steps} score={val:.2f} {cand}")
        ranked.sort(key=lambda t: t[0])
        survivors = max(1, int(len(alive) * keep))
        alive = [c for _, c in ranked[:survivors]]
        if len(alive) == 1 and rung < len(budgets) - 1:
            # One survivor: spend the remaining budget confirming it.
            continue
    best_score, best = ranked[0]
    return SearchResult(best, best_score, trials)
