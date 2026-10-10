import math

import pytest
import torch

from ucsa.models import recurrent
from ucsa.training import engine
from ucsa.training.engine import WeightEma, evaluate, fit, load_model, lr_at


def tiny_model():
    torch.manual_seed(0)
    return recurrent.Model(
        recurrent.Config(
            vocab_size=32,
            hidden=32,
            layers=1,
            heads=2,
            ffn_dim=64,
            chunk_size=8,
            banks=(("working", 4),),
            bank_write_bias=(("working", 0.0),),
        )
    )


def batches(skip=0):
    g = torch.Generator().manual_seed(1)
    i = 0
    while True:
        x = torch.randint(0, 32, (2, 33), generator=g)
        if i >= skip:
            yield x[:, :32], x[:, 1:33]
        i += 1


def cfg(tmp_path, **kw):
    base = dict(  # noqa: C408
        steps=6,
        seq_len=32,
        batch_size=2,
        warmup_steps=2,
        eval_every=3,
        eval_batches=2,
        ckpt_every=3,
        log_every=3,
        out_dir=str(tmp_path),
    )
    base.update(kw)
    return engine.Config(**base)


def test_lr_schedule_warms_up_then_decays_to_floor():
    c = engine.Config(steps=100, warmup_steps=10, lr=1.0, min_lr_ratio=0.1)
    assert lr_at(0, c) < lr_at(9, c) <= 1.0
    assert lr_at(10, c) == pytest.approx(1.0)
    assert lr_at(99, c) == pytest.approx(0.1, abs=1e-2)


def test_fit_trains_and_writes_record(tmp_path):
    rec = fit(tiny_model(), cfg(tmp_path), batches, batches, log=lambda s: None)
    assert math.isfinite(rec["final"]["ppl_last64"])
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "record.json").exists()


def test_resume_matches_an_uninterrupted_run(tmp_path):
    full = tmp_path / "full"
    fit(tiny_model(), cfg(full, steps=6), batches, None, log=lambda s: None)
    part = tmp_path / "part"
    fit(tiny_model(), cfg(part, steps=3), batches, None, log=lambda s: None)
    # Continue the 3-step run to 6 with the same schedule.
    fit(
        tiny_model(),
        cfg(part, steps=6),
        batches,
        None,
        resume=True,
        log=lambda s: None,
    )
    a = torch.load(full / "final.pt", weights_only=False)["model"]
    b = torch.load(part / "final.pt", weights_only=False)["model"]
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), k


def test_nonfinite_batches_are_skipped_not_applied(tmp_path):
    def poisoned(skip=0):
        for i, (x, y) in enumerate(batches(skip)):
            yield (x, y)
            if i > 100:
                return

    model = tiny_model()
    with torch.no_grad():
        model.state0[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        fit(
            model,
            cfg(tmp_path, max_bad_steps=3),
            poisoned,
            None,
            log=lambda s: None,
        )


def test_load_model_round_trips(tmp_path):
    m = tiny_model()
    fit(m, cfg(tmp_path), batches, None, log=lambda s: None)
    loaded = load_model(str(tmp_path / "final.pt"), torch.device("cpu"))
    x, _ = next(batches())
    assert torch.allclose(
        m.cpu().eval()(x)["logits"], loaded(x)["logits"], atol=1e-5
    )


def test_evaluate_reports_tail_and_all(tmp_path):
    m = tiny_model().eval()
    r = evaluate(m, batches(), 2)
    assert r["ppl_all"] > 1
    assert r["ppl_last64"] > 1


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        engine.Config(prefetch=0)
    with pytest.raises(ValueError):
        engine.Config.from_dict({"stepz": 1})


def test_weight_ema_swap_round_trips_and_tracks_the_average(tmp_path):
    model = tiny_model()
    ema = WeightEma(model, decay=0.5, every=1)
    live = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, step=1)
    shifted = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema.swap(model)  # now holds the average: halfway between old and new
    for name, p in model.named_parameters():
        if name in live and p.requires_grad:
            expected = 0.5 * live[name] + 0.5 * shifted[name]
            assert torch.allclose(p, expected, atol=1e-5), name
    ema.swap(model)  # restores the live weights exactly
    for name, p in model.named_parameters():
        assert torch.equal(p, shifted[name]), name


def test_ema_is_skipped_between_update_steps():
    model = tiny_model()
    ema = WeightEma(model, decay=0.9, every=4)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, step=1)  # not a multiple of 4: no change
    assert all(torch.equal(before[k], ema.shadow[k]) for k in before)
    ema.update(model, step=4)
    assert any(not torch.equal(before[k], ema.shadow[k]) for k in before)


def test_ema_settings_are_validated():
    with pytest.raises(ValueError):
        engine.Config(weight_ema=1.0)


def test_evaluate_reports_bits_per_byte_consistent_with_perplexity():
    model = tiny_model().eval()
    lengths = torch.full((32,), 2, dtype=torch.long)  # every token = 2 bytes
    r = evaluate(model, batches(), 2, lengths)
    # bpb = log2(ppl) / bytes_per_token when every token has 2 bytes.
    assert r["bpb_all"] == pytest.approx(math.log2(r["ppl_all"]) / 2, rel=1e-5)
    assert r["bpb_last64"] == pytest.approx(
        math.log2(r["ppl_last64"]) / 2, rel=1e-5
    )


def test_evaluate_omits_bits_per_byte_without_byte_lengths():
    r = evaluate(tiny_model().eval(), batches(), 1)
    assert "bpb_all" not in r


# ---------------------------------------------- findings from the code review


def test_retention_biases_and_gates_are_never_weight_decayed():
    """Regression: `slot_bias` and `surprise_gain` are 2-D (slots, 1), so a
    rule based on ndim alone decayed the per-bank retention biases."""
    model = tiny_model()
    model.updater.surprise_gain = torch.nn.Parameter(torch.zeros(4, 1))
    optimizer = engine.build_optimizer(model, engine.Config())
    decayed = {
        id(p)
        for g in optimizer.param_groups
        if g["weight_decay"] > 0
        for p in g["params"]
    }
    for name, p in model.named_parameters():
        if any(k in name for k in ("bias", "gain", "scale", "state0", "embed")):
            assert id(p) not in decayed, name
    matrices = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and p.ndim >= 2
        and "embed" not in n
        and "state0" not in n
        and "bias" not in n
        and "gain" not in n
    ]
    assert all(id(p) in decayed for p in matrices)


def test_resuming_with_ema_from_a_checkpoint_saved_without_it(tmp_path):
    fit(tiny_model(), cfg(tmp_path, steps=3), batches, None, log=lambda s: None)
    rec = fit(
        tiny_model(),
        cfg(tmp_path, steps=5, weight_ema=0.9),
        batches,
        None,
        resume=True,
        log=lambda s: None,
    )
    assert rec["params"] > 0  # did not crash on the missing EMA state


@pytest.mark.parametrize(
    "bad",
    [
        {"seq_len": 0},
        {"lr": 0.0},
        {"lr": -1.0},
        {"warmup_steps": -1},
        {"log_every": 0},
        {"eval_every": -1},
        {"ckpt_every": -5},
        {"grad_clip": 0.0},
        {"weight_decay": -0.1},
        {"beta1": 1.0},
        {"beta2": -0.1},
        {"eval_batches": 0},
        {"keep_ckpts": -1},
    ],
)
def test_every_numeric_setting_is_validated(bad):
    with pytest.raises(ValueError):
        engine.Config(**bad)


def test_evaluate_restores_the_mode_it_was_called_in():
    model = tiny_model().eval()
    evaluate(model, batches(), 1)
    assert not model.training
    model.train()
    evaluate(model, batches(), 1)
    assert model.training
