"""The prefix-only protocol never shows a model the tokens it is scored on."""

import torch

from ucsa.training import eval_harness
from ucsa.training.prefix import PrefixBatches, split_prefix_targets


def make_batch(length: int):
    tokens = torch.arange(length + 1).unsqueeze(0)
    return tokens[:, :length], tokens[:, 1 : length + 1]


def test_targets_are_the_k_tokens_after_the_prefix():
    x, y = make_batch(100)
    prefix, targets = split_prefix_targets(x, y, k=10)
    assert prefix.shape == (1, 90)
    assert targets.shape == (1, 10)
    assert targets[0].tolist() == list(range(90, 100))
    assert prefix[0, -1] + 1 == targets[0, 0]


def test_no_target_token_appears_in_the_prefix():
    x, y = make_batch(1024)
    prefix, targets = split_prefix_targets(x, y, k=64)
    assert not set(prefix[0].tolist()) & set(targets[0].tolist())


def test_prefixmake_batches_wraps_a_source():
    out = list(PrefixBatches([make_batch(50), make_batch(50)], k=5))
    assert len(out) == 2
    assert out[0][0].shape[1] == 45


def test_ucsa_scorer_never_feeds_the_choice_to_the_model(monkeypatch):
    seen = []

    class Spy:
        def __call__(self, ids):
            seen.append(ids.clone())
            return {"language": torch.zeros(1, 64, 256)}

    class Tok:
        def encode(self, text):
            return [ord(c) for c in text]

    monkeypatch.setattr(eval_harness.ucsa, "UCSA", Spy)
    eval_harness.choice_loglik(Spy(), Tok(), "ab", "cd", torch.device("cpu"))
    assert seen[0].tolist() == [[ord("a"), ord("b")]]


def test_tail_nll_is_aligned_with_shifted_targets():
    """A model that puts all mass on the right next token has ~0 NLL.

    Regression: scoring logits[-k-1:-1] against already-shifted targets is
    off by one and gives near-random perplexity for a perfect model.
    """
    from ucsa.training.prefix import tail_nll

    x, y = make_batch(32)
    logits = torch.full((1, 32, 64), -30.0)
    logits[0, torch.arange(32), y[0] % 64] = 30.0
    nll, n = tail_nll(logits, y % 64, 8)
    assert n == 8
    assert nll < 1e-3
    wrong, _ = tail_nll(logits.roll(1, 1), y % 64, 8)
    assert wrong > 100
