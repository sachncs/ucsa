"""Scoring never rewards a model for tokens it has read."""

import math

import pytest
import torch

from tests.helpers import Tokenizer, tiny_model
from ucsa.training import eval_harness, scoring


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


def test_a_choice_token_is_scored_only_from_what_precedes_it():
    """Changing a later choice token never changes an earlier token's score."""
    model, tok, device = tiny_model(), Tokenizer(), torch.device("cpu")
    ctx = "the cat sat on the mat"
    base = eval_harness.choice_loglik(model, tok, ctx, "abcd", device)
    ids = torch.tensor([tok.encode(ctx) + tok.encode(" abcd")])
    other = torch.tensor([tok.encode(ctx) + tok.encode(" abce")])
    n = len(tok.encode(" abcd"))
    with torch.no_grad():
        a = scoring.continuation_logprobs(model(ids)["logits"], ids, n)
        b = scoring.continuation_logprobs(model(other)["logits"], other, n)
    assert torch.allclose(a[:-1], b[:-1], atol=1e-5)
    assert not torch.allclose(a[-1:], b[-1:])
    assert base[1] == n
    assert base[0] == pytest.approx(float(a.sum()), abs=1e-4)


def test_an_empty_choice_without_context_scores_nothing():
    model, tok = tiny_model(), Tokenizer()
    out = eval_harness.choice_loglik(model, tok, "", "", torch.device("cpu"))
    assert out == (0.0, 0)


def test_an_empty_context_is_conditioned_on_the_end_token():
    model, tok = tiny_model(), Tokenizer()
    total, count = eval_harness.choice_loglik(
        model, tok, "", "xy", torch.device("cpu")
    )
    assert count == 2
    assert total < 0


def test_a_long_context_is_truncated_from_the_left():
    model, tok = tiny_model(), Tokenizer()
    total, count = eval_harness.choice_loglik(
        model, tok, "a" * 500, "bc", torch.device("cpu"), max_len=32
    )
    assert count == len(tok.encode(" bc"))
    assert math.isfinite(total)


def test_unknown_tasks_are_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="unknown tasks"):
        eval_harness.evaluate_all(
            ["nope"], tiny_model(), Tokenizer(), torch.device("cpu")
        )
