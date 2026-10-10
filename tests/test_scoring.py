"""Scoring never rewards a model for tokens it has read."""

import math

import pytest
import torch

from ucsa.training import scoring


def one_hot_logits(targets, vocab=64):
    logits = torch.full((*targets.shape, vocab), -30.0)
    logits.scatter_(-1, targets.unsqueeze(-1), 30.0)
    return logits


def test_tail_nll_is_aligned_with_shifted_targets():
    """A model that puts all mass on the right next token has ~0 NLL.

    Regression: scoring logits[-k-1:-1] against already-shifted targets is
    off by one and gives near-random perplexity for a perfect model.
    """
    targets = torch.arange(1, 33).unsqueeze(0) % 64
    nll, n = scoring.tail_nll(one_hot_logits(targets), targets, 8)
    assert n == 8
    assert nll < 1e-3
    wrong, _ = scoring.tail_nll(one_hot_logits(targets).roll(1, 1), targets, 8)
    assert wrong > 100


def test_tail_nll_counts_only_the_last_k_positions():
    targets = torch.zeros(2, 10, dtype=torch.long)
    logits = torch.zeros(2, 10, 5)
    nll, n = scoring.tail_nll(logits, targets, 3)
    assert n == 6
    assert nll == pytest.approx(6 * math.log(5), rel=1e-5)


def test_continuation_logprobs_score_exactly_the_trailing_tokens():
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    logits = one_hot_logits(torch.tensor([[6, 7, 8, 9, 0]]))
    out = scoring.continuation_logprobs(logits, ids, 2)
    assert out.shape == (2,)
    assert torch.all(out > -1e-3)
    wrong = scoring.continuation_logprobs(one_hot_logits(ids), ids, 2)
    assert torch.all(wrong < -50)


def test_perplexity_of_a_uniform_model_is_the_vocabulary_size():
    assert scoring.perplexity(3 * math.log(7), 3) == pytest.approx(7.0)
    assert scoring.perplexity(0.0, 0) == math.inf
