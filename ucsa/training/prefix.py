"""Prefix-only evaluation protocol.

The original UCSA writes its whole input into the persistent cognitive state
and decodes its working-bank slots, so a slot can read any token in the
input. If the tokens a slot is scored on are also in the input, the model is
rewarded for copying them rather than predicting them (measured: validation
perplexity 220 with the targets visible against 13,611 with them hidden).

The protocol here removes that route. The model is given only a prefix and
its `k` slots must predict the `k` tokens that follow it. A causal model sees
the same prefix and is scored on the same `k` positions, so the numbers
measure the same quantity.
"""

import math
from collections.abc import Iterable, Iterator

import torch
from torch.nn import functional

# Number of tokens a slot model predicts: one per working-bank slot.
DEFAULT_NUM_TARGETS = 64


def split_prefix_targets(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    k: int = DEFAULT_NUM_TARGETS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turns a shifted language-model batch into a prefix and its successors.

    `inputs` is `tokens[:L]` and `targets` is `tokens[1:L + 1]`. The prefix
    is `tokens[:L - k]` and the new targets are `tokens[L - k:L]`, the `k`
    tokens that immediately follow it. No target appears in the prefix.

    Args:
      inputs: Input ids of shape `(batch, L)`.
      targets: Shifted target ids of shape `(batch, L)`.
      k: Number of tokens to predict.

    Returns:
      `(prefix, next_k_tokens)` of shapes `(batch, L - k)` and `(batch, k)`.

    Raises:
      ValueError: If `k` is not in `(0, L)`.
    """
    length = inputs.shape[1]
    if not 0 < k < length:
        raise ValueError(f"k must be in (0, {length}), got {k}.")
    return inputs[:, : length - k], targets[:, length - k - 1 : length - 1]


class PrefixBatches:
    """Iterable view of a batch source that yields prefix/target splits."""

    def __init__(
        self,
        source: Iterable[tuple[torch.Tensor, torch.Tensor]],
        k: int = DEFAULT_NUM_TARGETS,
    ) -> None:
        """Wraps a batch source.

        Args:
          source: Iterable of `(inputs, targets)` batches.
          k: Number of tokens to predict after each prefix.
        """
        self.source = source
        self.k = k

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yields `split_prefix_targets` of every source batch."""
        for inputs, targets in self.source:
            yield split_prefix_targets(inputs, targets, self.k)


def causal_continuation_logprobs(
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


def slot_continuation_logprobs(
    logits: torch.Tensor, cont: torch.Tensor
) -> torch.Tensor:
    """Scores a continuation with one logit per slot.

    Args:
      logits: Slot logits of shape `(batch, slots, vocab)`; slot `j`
        predicts continuation token `j`.
      cont: Continuation ids of shape `(batch, n)`.

    Returns:
      Log-probabilities of the first `min(n, slots)` tokens.
    """
    n = min(cont.shape[-1], logits.shape[1])
    pred = logits[:, :n, :]
    return -functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]),
        cont[:, :n].reshape(-1),
        reduction="none",
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
