"""Scoring of next-token predictions: tail positions and continuations.

Every logit of a causal model is an exact next-token prediction, so a model is
scored on the tokens that follow what it has read and never on tokens it has
seen. Two readouts are used throughout the repository:

* the loss of the last `DEFAULT_NUM_TARGETS` positions of a window, so every
  model is compared at the same context length, and
* the log-probability of a continuation given a context, for multiple-choice
  benchmarks.

History: the original slot-based UCSA decoded its logits from slots that had
read the whole input, so a slot could copy the token it was scored on
(validation perplexity 220 with the targets visible against 13,611 with them
hidden; see `runs/old-leaky`). The model was retired; this module keeps only
the scoring that is valid for a causal model.
"""

import math

import torch
from torch.nn import functional

# Positions scored by the headline perplexity, and the most choice tokens
# scored per benchmark option.
DEFAULT_NUM_TARGETS = 64


def continuation_logprobs(
    logits: torch.Tensor, ids: torch.Tensor, n_cont: int
) -> torch.Tensor:
    """Scores the last `n_cont` tokens of a causal model's input.

    Args:
      logits: Logits of shape `(batch, seq, vocab)` for input `ids`.
      ids: Unshifted token ids of shape `(batch, seq)`.
      n_cont: Number of trailing tokens to score.

    Returns:
      Per-token log-probabilities of shape `(batch * n_cont,)`.
    """
    pred = logits[:, -n_cont - 1 : -1, :]
    tgt = ids[:, -n_cont:]
    return -functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), reduction="none"
    )


def tail_nll(
    logits: torch.Tensor, targets: torch.Tensor, k: int
) -> tuple[float, int]:
    """Sums the negative log-likelihood of the last `k` positions.

    `targets` is the already-shifted batch target (`targets[i]` is the token
    that `logits[i]` predicts), so the last `k` logits pair with the last `k`
    targets with no further shift.

    Args:
      logits: Logits of shape `(batch, seq, vocab)`.
      targets: Shifted target ids of shape `(batch, seq)`.
      k: Number of trailing positions to score.

    Returns:
      `(summed_nll, token_count)`.
    """
    pred, tgt = logits[:, -k:, :], targets[:, -k:]
    nll = functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), reduction="sum"
    )
    return float(nll.item()), tgt.numel()


def perplexity(total_nll: float, total_tokens: int) -> float:
    """Returns `exp(total_nll / total_tokens)`, or infinity for no tokens."""
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)
