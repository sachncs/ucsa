import pytest
import torch

from ucsa.models import recurrent
from ucsa.models.recurrent import config_for_params, count_parameters


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
    return recurrent.Config(**base)


def make(**kw):
    torch.manual_seed(0)
    return recurrent.Model(tiny(**kw)).eval()


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
    """With chunk-local attention (window 0) an early chunk can reach later
    chunks only through the state: it does with the state on and provably
    cannot with it off."""
    x = torch.randint(0, 64, (1, 24))
    y = x.clone()
    y[0, :8] = (y[0, :8] + 1) % 64
    on, off = make(window=0), make(window=0, use_state=False)
    d_on = (on(x)["logits"] - on(y)["logits"]).abs()[0, 8:].max()
    d_off = (off(x)["logits"] - off(y)["logits"]).abs()[0, 8:].max()
    assert d_on > 1e-6
    assert d_off < 1e-6


def test_the_window_lets_the_previous_chunk_reach_the_next_without_a_state():
    x = torch.randint(0, 64, (1, 24))
    y = x.clone()
    y[0, :8] = (y[0, :8] + 1) % 64
    m = make(use_state=False)  # default window is one chunk
    diff = (m(x)["logits"] - m(y)["logits"]).abs()[0]
    assert diff[8:16].max() > 1e-6  # the next chunk sees the changed tokens


def test_window_visibility_is_exactly_the_last_window_tokens():
    """One layer, no state: position i of chunk 1 depends on position j of
    chunk 0 if and only if j > i (a window of one chunk of 8 tokens)."""
    m = make(layers=1, use_state=False, read_gate=True, jepa_weight=0.0)
    m.eval()
    x = torch.randint(0, 64, (1, 16))
    base = m(x)["logits"][0]
    for j in range(8):
        y = x.clone()
        y[0, j] = (y[0, j] + 1) % 64
        changed = (m(y)["logits"][0] - base).abs().amax(-1) > 1e-6
        for i in range(8):
            assert bool(changed[8 + i]) == (j > i), (j, i)


def test_a_narrower_window_sees_less_of_the_previous_chunk():
    base = make(layers=1, use_state=False, read_gate=True, jepa_weight=0.0)
    narrow = make(
        layers=1, use_state=False, read_gate=True, jepa_weight=0.0, window=2
    )
    narrow.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(0, 64, (1, 16))
    y = x.clone()
    y[0, 5] = (y[0, 5] + 1) % 64

    def reach(model):
        diff = (model(y)["logits"] - model(x)["logits"])[0, 8:]
        return int((diff.abs().amax(-1) > 1e-6).sum())

    assert reach(narrow) < reach(base)


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
        recurrent.Config.from_dict({"hiden": 4})
    cfg = tiny()
    assert recurrent.Config.from_dict(cfg.to_dict()) == cfg


def test_config_for_params_hits_target():
    cfg = config_for_params(63_000_000)
    n = count_parameters(recurrent.Model(cfg))
    assert abs(n - 63_000_000) / 63_000_000 < 0.15


def test_active_slots_changes_reads_and_full_slots_matches_default():
    m = make()
    x = torch.randint(0, 64, (1, 24))
    full = m(x)["logits"]
    assert torch.allclose(m(x, active_slots=6)["logits"], full, atol=1e-6)
    assert not torch.allclose(m(x, active_slots=2)["logits"], full, atol=1e-6)


def test_slot_dropout_reads_random_prefixes_only_while_training():
    m = make(slot_dropout=1.0, min_slots=2).train()
    seen = []
    hook = m.blocks[0].slot_norm.register_forward_hook(
        lambda mod, args, out: seen.append(args[0].shape[1])
    )
    x = torch.randint(0, 64, (1, 24))
    for _ in range(40):
        m.hidden_states(x)
    assert set(seen) <= set(range(2, 7))
    assert len(set(seen)) > 1  # genuinely random, not a constant
    seen.clear()
    m.eval()
    m.hidden_states(x)
    assert set(seen) == {6}  # all slots at evaluation time
    hook.remove()


def test_truncated_reads_stay_causal():
    m = make().eval()
    x = torch.randint(0, 64, (1, 24))
    y = x.clone()
    y[0, 20] = (y[0, 20] + 1) % 64
    diff = m(y, active_slots=3)["logits"] - m(x, active_slots=3)["logits"]
    assert diff.abs().amax(-1)[0, :20].max().item() < 1e-5


def test_slot_dropout_config_is_validated():
    with pytest.raises(ValueError):
        tiny(slot_dropout=1.5)
    with pytest.raises(ValueError):
        tiny(min_slots=99)


def gated(**kw):
    return make(surprise_gate=True, **kw)


def test_surprise_gate_starts_identical_to_the_baseline():
    base, with_gate = make(), gated()
    with_gate.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(0, 64, (2, 40))
    assert torch.equal(with_gate.updater.surprise_gain, torch.zeros(6, 1))
    assert torch.allclose(base(x)["logits"], with_gate(x)["logits"], atol=1e-6)


def test_higher_surprise_writes_more_once_gain_is_positive():
    m = gated().eval()
    with torch.no_grad():
        m.updater.surprise_gain.fill_(2.0)
    state = m.initial_state(1)
    chunk = torch.randn(1, 8, 64)
    _, calm = m.updater(state, chunk, torch.tensor([0.0]))
    _, shocked = m.updater(state, chunk, torch.tensor([1.5]))
    assert shocked.mean() > calm.mean()


def test_zero_gain_ignores_surprise_entirely():
    m = gated().eval()
    state = m.initial_state(1)
    chunk = torch.randn(1, 8, 64)
    a, _ = m.updater(state, chunk, torch.tensor([0.0]))
    b, _ = m.updater(state, chunk, torch.tensor([1.9]))
    assert torch.equal(a, b)


def test_surprise_gain_receives_gradient_from_the_language_loss():
    m = gated().train()
    with torch.no_grad():
        m.updater.surprise_gain.fill_(0.5)
    x = torch.randint(0, 64, (2, 32))
    loss, _ = m.compute_loss(x, torch.roll(x, -1, 1))
    loss.backward()
    assert m.updater.surprise_gain.grad is not None
    assert m.updater.surprise_gain.grad.abs().sum() > 0


def test_surprise_gated_model_is_still_strictly_causal():
    m = gated().eval()
    with torch.no_grad():
        m.updater.surprise_gain.fill_(1.0)
    x = torch.randint(0, 64, (1, 40))
    base = m(x)["logits"]
    for p in (3, 9, 25):
        y = x.clone()
        y[0, p] = (y[0, p] + 1) % 64
        diff = (m(y)["logits"] - base).abs().amax(-1)[0]
        assert diff[:p].max().item() < 1e-5, f"leak at p={p}"


def test_surprise_gate_requires_the_predictor_to_be_trained():
    with pytest.raises(ValueError, match="jepa_weight"):
        tiny(surprise_gate=True, jepa_weight=0.0)


def test_generation_with_surprise_gate_streams_across_chunks():
    m = gated()
    with torch.no_grad():
        m.updater.surprise_gain.fill_(1.0)
    prompt = torch.randint(0, 64, (1, 5))
    out = m.generate(prompt, 20, temperature=0.0)
    assert out.shape == (1, 25)
    assert torch.equal(out, m.generate(prompt, 20, temperature=0.0))


@pytest.mark.parametrize("use_state", [True, False])
@pytest.mark.parametrize("surprise", [True, False])
def test_streaming_logits_match_teacher_forced_logits(use_state, surprise):
    """The streaming API must agree with `forward` at every position, across
    chunk boundaries; this is what lets a decoder reproduce an encoder."""
    m = make(use_state=use_state, surprise_gate=surprise)
    with torch.no_grad():
        if surprise and use_state:
            m.updater.surprise_gain.fill_(1.0)
    x = torch.randint(0, 64, (1, 29))  # chunk_size 8: three boundaries
    full = m(x)["logits"][0]
    size = m.config.chunk_size
    state = m.initial_state(1)
    previous = cache = None
    for pos in range(x.shape[1]):
        start = (pos // size) * size
        if pos and pos % size == 0:
            state = m.advance(state, x[:, pos - size : pos])
            previous = cache  # the chunk that just completed
        logits, cache = m.next_logits(state, x[:, start : pos + 1], previous)
        assert torch.allclose(logits[0], full[pos], atol=1e-4), pos


def test_closed_read_gate_makes_the_logits_independent_of_the_state():
    """With every read scale at zero the state cannot influence a logit, so
    the model equals its own state-disabled twin sharing the same weights."""
    gated_model = make(read_gate=True).eval()
    twin = make(read_gate=True, use_state=False).eval()
    twin.load_state_dict(gated_model.state_dict(), strict=False)
    x = torch.randint(0, 64, (2, 40))
    assert torch.allclose(
        gated_model(x)["logits"], twin(x)["logits"], atol=1e-5
    )


def test_read_scales_start_at_zero_and_learn():
    m = make(read_gate=True).train()
    scales = [b.slot_scale for b in m.blocks if b.slot_scale is not None]
    assert scales
    assert all(float(s.detach()) == 0.0 for s in scales)
    x = torch.randint(0, 64, (2, 32))
    loss, _ = m.compute_loss(x, torch.roll(x, -1, 1))
    loss.backward()
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in scales)


def test_once_the_scale_opens_the_state_matters():
    m = make(read_gate=True, window=0).eval()
    x = torch.randint(0, 64, (1, 24))
    y = x.clone()
    y[0, :8] = (y[0, :8] + 1) % 64
    before = (m(x)["logits"] - m(y)["logits"]).abs()[0, 8:].max()
    with torch.no_grad():
        for b in m.blocks:
            b.slot_scale.fill_(1.0)
    after = (m(x)["logits"] - m(y)["logits"]).abs()[0, 8:].max()
    assert before < 1e-6  # closed gate and no window: chunk 0 is invisible
    assert after > 1e-6  # open gate: it influences later chunks
