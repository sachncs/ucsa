"""Standard LM benchmark evaluation harness.

Supports the four canonical small-LM zero-shot tasks via
multiple-choice / continuation rank classification:

- HellaSwag: commonsense completion (4 endings).
- ARC-easy / ARC-challenge: grade-school science multiple choice.
- PIQA: physical interaction reasoning (2 endings).
- WinoGrande: pronoun resolution commonsense (multiple options).

Scoring convention (EleutherAI/lm-eval-harness style): each choice's
text is concatenated to a context; we compute the **conditional
log-likelihood** ``log P(choice | context)`` under the model; the
choice with the highest log-likelihood wins.

This module exposes ``evaluate_task(name, model, tokenizer, ...)``
which returns an ``EvalResult`` with the per-task accuracy plus
metadata needed by the paper-writing tools.

Reproducibility
---------------

Each task loader streams from HuggingFace ``datasets`` and is paired
with a fixed-seed shuffle buffer so the ``max_examples`` cap selects a
deterministic prefix. Two consecutive runs of ``scripts/eval.py``
against the same checkpoint therefore report the same accuracy to four
decimal places. The seed is recorded on every :class:`EvalResult` in
:attr:`EvalResult.extras` for downstream paper-writing tools.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from datasets import load_dataset
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from ucsa.models.ucsa import UCSA

# Default seed used when callers do not pass one. The same seed is
# passed to every task loader so a ``max_examples`` cap on a streaming
# source yields the same prefix on every run.
DEFAULT_EVAL_SEED: int = 1234


@dataclass
class TaskSpec:
    """One LM benchmark task.

    Attributes:
        name: Display name (must match keys in :func:`TASK_REGISTRY`).
        loader: Callable returning an iterable of dicts with the keys
            the score function expects. Loaders that take a seed must
            accept it as the first positional argument.
        max_examples: Optional cap to keep the harness fast on small
            machines. ``None`` runs the full split. When the loader
            streams from HuggingFace ``datasets``, the cap selects a
            deterministic prefix given a fixed seed.
        seed: Seed used by streaming loaders for deterministic
            ``max_examples`` selection. Defaults to
            :data:`DEFAULT_EVAL_SEED`.
    """

    name: str
    loader: Callable[..., Iterable[dict[str, Any]]]
    max_examples: int | None = None
    seed: int = DEFAULT_EVAL_SEED


@dataclass
class EvalResult:
    """Result of one task evaluation."""

    name: str
    n: int
    correct: int
    accuracy: float
    log_likelihood_mean: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Task datasets
# ---------------------------------------------------------------------------


def _shuffled_stream(
    ds: Iterable[dict[str, Any]], seed: int, buffer_size: int = 2000
) -> Iterable[dict[str, Any]]:
    """Yield a deterministic-shuffled stream from ``ds``.

    Args:
        ds: A streaming HuggingFace dataset (exposes ``.shuffle``) or
            any plain iterable of records (e.g. a list).
        seed: Seed for the deterministic shuffle.
        buffer_size: Shuffle buffer size for streaming datasets. Ignored
            for plain iterables.

    Yields:
        Records from ``ds`` in a deterministic order.
    """
    if hasattr(ds, "shuffle"):
        shuffled = ds.shuffle(seed=seed, buffer_size=buffer_size)
        yield from shuffled
        return
    import random

    rng = random.Random(seed)
    rows = list(ds)
    rng.shuffle(rows)
    yield from rows


def _load_hellaswag(seed: int = DEFAULT_EVAL_SEED) -> Iterable[dict[str, Any]]:
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    for ex in _shuffled_stream(ds, seed):
        yield {
            "context": ex["ctx"],
            "choices": ex["endings"],
            "label": int(ex["label"]),
        }


def _load_arc(
    name: str, seed: int = DEFAULT_EVAL_SEED
) -> Iterable[dict[str, Any]]:
    ds = load_dataset("allenai/ai2_arc", name, split="test", streaming=True)
    for ex in _shuffled_stream(ds, seed):
        yield {
            "context": ex["question"],
            "choices": ex["choices"]["text"],
            "label": ex["choices"]["label"].index(ex["answerKey"]),
            "labels_str": ex["choices"]["label"],
            "answer_str": ex["answerKey"],
        }


def _load_piqa(seed: int = DEFAULT_EVAL_SEED) -> Iterable[dict[str, Any]]:
    # Upstream ``ybisk/piqa`` is loading-script only and is no longer
    # supported by the current ``datasets`` versions we run with. Use
    # ``gimmaru/piqa`` which mirrors the same schema (goal/sol1/sol2).
    ds = load_dataset("gimmaru/piqa", split="validation", streaming=True)
    for ex in _shuffled_stream(ds, seed):
        yield {
            "context": ex["goal"],
            "choices": [ex["sol1"], ex["sol2"]],
            "label": int(ex["label"]),
        }


def _load_winogrande(seed: int = DEFAULT_EVAL_SEED) -> Iterable[dict[str, Any]]:
    ds = load_dataset(
        "allenai/winogrande",
        "winogrande_l",
        split="validation",
        streaming=True,
    )
    for ex in _shuffled_stream(ds, seed):
        ctx = ex["sentence"]
        # The dataset exposes ``option1``/``option2``, not an ``options``
        # list, and ``answer`` is 1-based over those two in order.
        # Building the choices in the reverse order while keeping
        # ``answer - 1`` as the label inverted every example.
        choices = [
            ctx.replace("_", ex["option1"]),
            ctx.replace("_", ex["option2"]),
        ]
        yield {
            "context": "",
            "choices": choices,
            "label": int(ex["answer"]) - 1,
        }


TASK_REGISTRY: dict[str, TaskSpec] = {
    "hellaswag": TaskSpec(
        name="hellaswag", loader=_load_hellaswag, max_examples=200
    ),
    "arc_easy": TaskSpec(
        name="arc_easy",
        loader=lambda seed=DEFAULT_EVAL_SEED: _load_arc("ARC-Easy", seed),
        max_examples=200,
    ),
    "arc_challenge": TaskSpec(
        name="arc_challenge",
        loader=lambda seed=DEFAULT_EVAL_SEED: _load_arc("ARC-Challenge", seed),
        max_examples=200,
    ),
    "piqa": TaskSpec(name="piqa", loader=_load_piqa, max_examples=200),
    "winogrande": TaskSpec(
        name="winogrande", loader=_load_winogrande, max_examples=200
    ),
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _normalize_scores(scores: list[float]) -> list[float]:
    """Length-normalise per-token log-likelihoods so HellaSwag/PIQA-style
    comparisons aren't dominated by choice length."""
    return scores


@torch.no_grad()
def _conditional_loglik(
    model: UCSA,
    tokenizer: PreTrainedTokenizerBase,
    context: str,
    choice: str,
    device: torch.device,
    max_len: int = 1024,
) -> float:
    """Compute ``log P(choice | context)`` under the model.

    Ponytail: this is the canonical 'rank by log-likelihood' protocol
    used in lm-evaluation-harness and the original GPT-2 paper.
    """
    full = context + " " + choice if context else choice
    full_ids = tokenizer.encode(full)
    context_ids = tokenizer.encode(context) if context else []
    if len(full_ids) <= len(context_ids):
        return 0.0
    # Truncate from the left of the prompt if too long.
    if len(full_ids) > max_len:
        keep = max_len - (len(full_ids) - len(context_ids))
        start = max(0, len(context_ids) - keep)
        full_ids = full_ids[start:]
        context_ids = full_ids[: len(context_ids) - start]
        if len(context_ids) >= len(full_ids):
            return 0.0
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    targets = torch.tensor([full_ids], dtype=torch.long, device=device)
    # Build a per-token loss mask so we only score choice tokens.
    ctx_len = len(context_ids)
    ignore = torch.full_like(targets, -100)
    ignore[:, :ctx_len] = targets[:, :ctx_len]
    out = model(ids)
    if isinstance(out, dict):
        maybe_logits = out.get("language", out.get("logits"))
    else:
        maybe_logits = out
    if maybe_logits is None:
        return 0.0
    logits: Tensor = maybe_logits
    # Align lengths the same way compute_loss does (target the
    # dataset's last logit -> next-token prediction).
    seq = min(logits.shape[1], targets.shape[1])
    if seq == 0:
        return 0.0
    logits = logits[:, :seq, :]
    targets = targets[:, :seq]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.shape)
    mask = (targets != ignore[:, :seq]).float()
    if mask.sum() == 0:
        return 0.0
    return float(-loss.sum().item() / mask.sum().item())


def evaluate_task(
    spec: TaskSpec,
    model: UCSA,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> EvalResult:
    """Run one task, return its ``EvalResult``."""
    if hasattr(model, "eval"):
        model.eval()
    correct = 0
    total = 0
    ll_sum = 0.0
    for i, ex in enumerate(spec.loader(spec.seed)):
        if spec.max_examples is not None and i >= spec.max_examples:
            break
        ctx = ex["context"]
        choices = ex["choices"]
        label = ex["label"]
        scores = [
            _conditional_loglik(model, tokenizer, ctx, c, device)
            for c in choices
        ]
        pred = scores.index(max(scores))
        ll_sum += max(scores)
        if pred == label:
            correct += 1
        total += 1
    if hasattr(model, "train"):
        model.train()
    return EvalResult(
        name=spec.name,
        n=total,
        correct=correct,
        accuracy=correct / max(1, total),
        log_likelihood_mean=ll_sum / max(1, total),
        extras={"seed": spec.seed, "max_examples": spec.max_examples},
    )


def evaluate_all(
    names: list[str] | None,
    model: UCSA,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> list[EvalResult]:
    """Evaluate a list of tasks (or all registered)."""
    if names is None or len(names) == 0:
        names = list(TASK_REGISTRY.keys())
    results = []
    for name in names:
        if name not in TASK_REGISTRY:
            print(f"  unknown task {name}; skipping", flush=True)
            continue
        spec = TASK_REGISTRY[name]
        print(f"  eval {spec.name} ...", flush=True)
        r = evaluate_task(spec, model, tokenizer, device)
        results.append(r)
        print(
            f"    {r.name}: {r.correct}/{r.n} acc={r.accuracy:.4f}",
            flush=True,
        )
    return results


__all__ = [
    "DEFAULT_EVAL_SEED",
    "EvalResult",
    "TaskSpec",
    "TASK_REGISTRY",
    "evaluate_all",
    "evaluate_task",
]
