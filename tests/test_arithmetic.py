import math
import random

import pytest
import torch

from ucsa import arithmetic
from ucsa.models import recurrent

VOCAB = 64
BOS = 63


class FixedPredictor:
    """A context-free source with the given logits (a unigram model)."""

    def __init__(self, logits):
        self.fixed = logits

    def logits(self):
        return self.fixed

    def update(self, token):
        pass


class HistoryPredictor:
    """Next-token distribution depends on the previous token (order 1)."""

    def __init__(self, table):
        self.table, self.previous = table, 0

    def logits(self):
        return self.table[self.previous]

    def update(self, token):
        self.previous = token


def tiny_model(**kw):
    torch.manual_seed(0)
    cfg = recurrent.Config(
        vocab_size=VOCAB,
        hidden=32,
        layers=2,
        heads=2,
        ffn_dim=64,
        chunk_size=8,
        banks=(("working", 4), ("long_term", 2)),
        bank_write_bias=(("working", 0.0), ("long_term", -2.0)),
        **kw,
    )
    return recurrent.Model(cfg).eval()


def roundtrip(make_predictor, tokens):
    data = arithmetic.compress(make_predictor(), tokens)
    return data, arithmetic.decompress(make_predictor(), data)


# ------------------------------------------------------------- the coder itself


@pytest.mark.parametrize("n", [0, 1, 2, 17, 500])
def test_round_trip_is_exact_for_many_lengths(n):
    rng = random.Random(n)
    logits = torch.randn(VOCAB)
    tokens = [rng.randrange(VOCAB) for _ in range(n)]
    _, out = roundtrip(lambda: FixedPredictor(logits), tokens)
    assert out == tokens


def test_compressed_size_matches_the_shannon_bound():
    rng = random.Random(1)
    logits = torch.randn(VOCAB) * 2
    probs = torch.softmax(logits, -1)
    tokens = rng.choices(range(VOCAB), weights=probs.tolist(), k=4000)
    data, out = roundtrip(lambda: FixedPredictor(logits), tokens)
    ideal = arithmetic.ideal_bits(FixedPredictor(logits), tokens)
    assert out == tokens
    payload_bits = 8 * (len(data) - arithmetic.HEADER.size)
    assert ideal <= payload_bits <= ideal + 16  # a few bits of flush overhead


def test_skewed_distributions_do_not_break_the_coder():
    logits = torch.full((VOCAB,), -30.0)
    logits[5] = 30.0  # probability of everything else is ~1e-26
    tokens = [5] * 200 + [7] + [5] * 200 + [0, 63]
    _, out = roundtrip(lambda: FixedPredictor(logits), tokens)
    assert out == tokens


def test_rare_symbols_cost_their_information_content():
    logits = torch.zeros(VOCAB)
    uniform = arithmetic.ideal_bits(FixedPredictor(logits), [3] * 100)
    assert uniform == pytest.approx(100 * math.log2(VOCAB), rel=1e-4)


def test_context_dependent_models_round_trip():
    gen = torch.Generator().manual_seed(2)
    table = torch.randn(VOCAB, VOCAB, generator=gen) * 3
    rng = random.Random(3)
    tokens = [rng.randrange(VOCAB) for _ in range(800)]
    _, out = roundtrip(lambda: HistoryPredictor(table), tokens)
    assert out == tokens


def test_better_models_give_smaller_files():
    gen = torch.Generator().manual_seed(4)
    table = torch.randn(VOCAB, VOCAB, generator=gen) * 4
    # Sample text from the true order-1 source, then code it with the true
    # model and with a uniform model that ignores the context.
    rng = random.Random(5)
    previous, tokens = 0, []
    for _ in range(3000):
        p = torch.softmax(table[previous], -1).tolist()
        previous = rng.choices(range(VOCAB), weights=p)[0]
        tokens.append(previous)
    true_size = len(arithmetic.compress(HistoryPredictor(table), tokens))
    blind_size = len(
        arithmetic.compress(FixedPredictor(torch.zeros(VOCAB)), tokens)
    )
    assert true_size < 0.8 * blind_size


def test_short_or_empty_data_is_rejected():
    with pytest.raises(ValueError, match="header"):
        arithmetic.decompress(FixedPredictor(torch.zeros(VOCAB)), b"abc")


def test_a_corrupted_stream_decodes_to_different_tokens_not_a_crash():
    rng = random.Random(6)
    logits = torch.randn(VOCAB)
    tokens = [rng.randrange(VOCAB) for _ in range(300)]
    data = bytearray(arithmetic.compress(FixedPredictor(logits), tokens))
    data[arithmetic.HEADER.size + 5] ^= 0xFF
    out = arithmetic.decompress(FixedPredictor(logits), bytes(data))
    assert len(out) == len(tokens)
    assert out != tokens


# -------------------------------------------------- the model as a compressor


@pytest.mark.parametrize("use_state", [True, False])
@pytest.mark.parametrize("surprise", [True, False])
def test_model_round_trip_is_exact_across_chunk_boundaries(use_state, surprise):
    model = tiny_model(use_state=use_state, surprise_gate=surprise)
    with torch.no_grad():
        if use_state and surprise:
            model.updater.surprise_gain.fill_(1.0)
    rng = random.Random(7)
    tokens = [rng.randrange(VOCAB - 1) for _ in range(40)]  # 5 chunks of 8
    data, out = roundtrip(lambda: arithmetic.ModelPredictor(model, BOS), tokens)
    assert out == tokens
    ideal = arithmetic.ideal_bits(arithmetic.ModelPredictor(model, BOS), tokens)
    payload_bits = 8 * (len(data) - arithmetic.HEADER.size)
    assert ideal <= payload_bits <= ideal + 16


def test_coded_size_equals_the_models_cross_entropy():
    """Bits actually stored equal the teacher-forced NLL: the compressor and
    the training loss measure the same thing."""
    model = tiny_model()
    rng = random.Random(8)
    tokens = [rng.randrange(VOCAB - 1) for _ in range(24)]
    ids = torch.tensor([[BOS, *tokens]])
    with torch.no_grad():
        logits = model(ids[:, :-1])["logits"][0]
    nll = torch.nn.functional.cross_entropy(
        logits, torch.tensor(tokens), reduction="sum"
    ).item()
    ideal = arithmetic.ideal_bits(arithmetic.ModelPredictor(model, BOS), tokens)
    assert ideal == pytest.approx(nll / math.log(2), rel=2e-3)


class PeekingPredictor:
    """A leaky model: it is told the next token, so it puts all its mass
    there. Perfect when encoding, impossible to reproduce when decoding."""

    def __init__(self, future=None):
        self.future, self.i = future, 0

    def logits(self):
        logits = torch.zeros(VOCAB)
        if self.future is not None:
            logits[self.future[self.i]] = 20.0
        return logits

    def update(self, token):
        self.i += 1


def test_a_leaking_model_cannot_be_decoded():
    """The property that makes compression a causality test."""
    rng = random.Random(9)
    tokens = [rng.randrange(VOCAB) for _ in range(300)]
    data = arithmetic.compress(PeekingPredictor(future=tokens), tokens)
    assert len(data) < 40  # looks spectacular: ~0 bits per token
    honest = arithmetic.decompress(PeekingPredictor(future=None), data)
    assert honest != tokens  # the decoder has no future to peek at
