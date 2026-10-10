"""Random configurations must keep the model's three core invariants.

Hand-written tests cover chosen settings; this draws many combinations of
chunk size, window, read frequency, slot layout, gates and sequence length,
and checks on each that outputs are finite, the model is strictly causal,
and streaming reproduces teacher forcing.
"""

import random

import pytest
import torch

from ucsa.models import recurrent
from ucsa.utils import precision

SEEDS = list(range(24))


def random_config(rng: random.Random) -> recurrent.Config:
    heads = rng.choice([1, 2, 4])
    chunk = rng.choice([2, 3, 4, 8])
    banks = tuple(
        (name, rng.randint(1, 4))
        for name in rng.sample(
            ["working", "long_term", "intent"], rng.randint(1, 3)
        )
    )
    slots = sum(n for _, n in banks)
    return recurrent.Config(
        vocab_size=32,
        hidden=16 * heads * rng.choice([1, 2]),
        layers=rng.randint(1, 3),
        heads=heads,
        ffn_dim=32,
        chunk_size=chunk,
        encoder_layers=rng.randint(1, 2),
        banks=banks,
        bank_write_bias=tuple((n, rng.uniform(-2, 0)) for n, _ in banks),
        read_every=rng.randint(1, 2),
        write_top_k=rng.randint(0, slots),
        window=rng.choice([None, 0, 1, chunk // 2, chunk]),
        use_state=rng.random() < 0.8,
        read_gate=rng.random() < 0.5,
        jepa_weight=rng.choice([0.0, 0.1]),
    )


def build(seed):
    rng = random.Random(seed)
    precision.configure()
    torch.manual_seed(seed)
    cfg = random_config(rng)
    model = recurrent.Model(cfg).eval()
    with torch.no_grad():  # open the gates so the state actually matters
        for block in model.blocks:
            if block.slot_scale is not None:
                block.slot_scale.fill_(rng.uniform(0.2, 1.0))
    return model, cfg, rng.randint(1, 5 * cfg.chunk_size)


@pytest.mark.parametrize("seed", SEEDS)
def test_outputs_are_finite_for_any_valid_configuration(seed):
    model, cfg, length = build(seed)
    ids = torch.randint(0, 32, (2, length))
    out = model(ids)
    assert out["logits"].shape == (2, length, 32)
    assert torch.isfinite(out["logits"]).all(), cfg
    loss, _ = model.train().compute_loss(ids, torch.roll(ids, -1, 1))
    assert torch.isfinite(loss), cfg


@pytest.mark.parametrize("seed", SEEDS)
def test_causality_holds_for_any_valid_configuration(seed):
    model, cfg, length = build(seed)
    ids = torch.randint(0, 32, (1, length))
    base = model(ids)["logits"]
    for pos in sorted({0, length // 2, length - 1}):
        changed = ids.clone()
        changed[0, pos] = (changed[0, pos] + 1) % 32
        diff = (model(changed)["logits"] - base).abs().amax(-1)[0]
        assert diff[:pos].numel() == 0 or diff[:pos].max() < 1e-5, (cfg, pos)


@pytest.mark.parametrize("seed", SEEDS)
def test_streaming_equals_teacher_forcing_for_any_valid_configuration(seed):
    model, cfg, length = build(seed)
    ids = torch.randint(0, 32, (1, length))
    full = model(ids)["logits"][0]
    size = cfg.chunk_size
    state = model.initial_state(1)
    previous = cache = None
    for pos in range(length):
        start = (pos // size) * size
        if pos and pos % size == 0:
            state = model.advance(state, ids[:, pos - size : pos])
            previous = cache
        logits, cache = model.next_logits(
            state, ids[:, start : pos + 1], previous
        )
        assert torch.allclose(logits[0], full[pos], atol=1e-4), (cfg, pos)
