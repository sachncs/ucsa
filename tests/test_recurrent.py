import pytest
import torch

from ucsa.models.recurrent import (
    RecurrentConfig,
    RecurrentUCSA,
    config_for_params,
    count_parameters,
)


def tiny(**kw):
    base = dict(  # noqa: C408
        vocab_size=64,
        hidden=64,
        layers=2,
        heads=4,
        ffn_dim=128,
        chunk_size=8,
        banks=(("working", 4), ("long_term", 2)),
        bank_write_bias=(("working", 0.0), ("long_term", -2.0)),
    )
    base.update(kw)
    return RecurrentConfig(**base)


def make(**kw):
    torch.manual_seed(0)
    return RecurrentUCSA(tiny(**kw)).eval()


def test_logits_cover_every_position():
    m = make()
    ids = torch.randint(0, 64, (2, 40))
    assert m(ids)["logits"].shape == (2, 40, 64)


def test_strictly_causal_across_and_within_chunks():
    """Changing token p must not change any logit at a position <= p - 1...
    precisely: logits at positions < p are untouched."""
    m = make()
    x = torch.randint(0, 64, (1, 40))
    base = m(x)["logits"]
    for p in (3, 8, 17, 39):
        y = x.clone()
        y[0, p] = (y[0, p] + 1) % 64
        diff = (m(y)["logits"] - base).abs().amax(-1)[0]
        assert diff[:p].max().item() < 1e-5, f"leak at p={p}"
        assert diff[p:].max().item() > 0


def test_state_carries_information_across_chunks():
    """With the state on, an early chunk changes later chunks' logits; with
    use_state=False it provably cannot."""
    x = torch.randint(0, 64, (1, 24))
    y = x.clone()
    y[0, :8] = (y[0, :8] + 1) % 64
    on, off = make(), make(use_state=False)
    d_on = (on(x)["logits"] - on(y)["logits"]).abs()[0, 8:].max()
    d_off = (off(x)["logits"] - off(y)["logits"]).abs()[0, 8:].max()
    assert d_on > 1e-6
    assert d_off < 1e-6


def test_loss_backprops_into_state_path():
    m = make().train()
    x = torch.randint(0, 64, (2, 32))
    loss, metrics = m.compute_loss(x, torch.roll(x, -1, 1))
    loss.backward()
    assert "ce" in metrics
    assert "jepa" in metrics
    assert m.state0.grad is not None
    assert m.state0.grad.abs().sum() > 0
    assert m.updater.slot_bias.grad is not None
    assert all(p.grad is None for p in m.target_summary.parameters())


def test_top_k_write_updates_at_most_k_slots():
    m = make(write_top_k=2)
    s = m.initial_state(1)
    new, _ = m.updater(s, torch.randn(1, 8, 64))
    changed = (new - s).abs().amax(-1)[0] > 0
    assert int(changed.sum()) <= 2


def test_bptt_detach_cuts_gradient_to_early_chunks():
    full = make(bptt_chunks=0).train()
    cut = make(bptt_chunks=1).train()
    cut.load_state_dict(full.state_dict())
    x = torch.randint(0, 64, (1, 24))
    g = []
    for m in (full, cut):
        m.zero_grad()
        loss, _ = m.compute_loss(x, torch.roll(x, -1, 1))
        loss.backward()
        g.append(m.state0.grad.clone())
    assert not torch.allclose(g[0], g[1])


def test_ema_target_moves_toward_online_encoder_only_after_update():
    m = make().train()
    before = [p.clone() for p in m.target_summary.parameters()]
    with torch.no_grad():
        for p in m.summary.parameters():
            p.add_(1.0)
    m.update_ema()
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, m.target_summary.parameters(), strict=True)
    )


def test_generate_streams_past_a_chunk_boundary():
    m = make()
    out = m.generate(torch.randint(0, 64, (1, 5)), 20, temperature=0.0)
    assert out.shape == (1, 25)
    # Greedy decode is deterministic.
    again = m.generate(out[:, :5], 20, temperature=0.0)
    assert torch.equal(out, again)


def test_config_rejects_bad_values_and_unknown_keys():
    with pytest.raises(ValueError):
        tiny(heads=3)
    with pytest.raises(ValueError):
        tiny(write_top_k=99)
    with pytest.raises(ValueError):
        RecurrentConfig.from_dict({"hiden": 4})
    cfg = tiny()
    assert RecurrentConfig.from_dict(cfg.to_dict()) == cfg


def test_config_for_params_hits_target():
    cfg = config_for_params(63_000_000)
    n = count_parameters(RecurrentUCSA(cfg))
    assert abs(n - 63_000_000) / 63_000_000 < 0.15
