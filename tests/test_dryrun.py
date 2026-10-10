import pytest
import torch
from torch import nn

from ucsa import dryrun
from ucsa.models import recurrent
from ucsa.training import engine
from ucsa.utils import precision


def other_dtype():
    """A floating dtype different from the library's (MPS has no float64)."""
    return torch.float16 if torch.float32 == precision.DTYPE else torch.float32


def tiny_model():
    precision.configure()
    torch.manual_seed(0)
    return recurrent.Model(
        recurrent.Config(
            vocab_size=64,
            hidden=32,
            layers=2,
            heads=2,
            ffn_dim=64,
            chunk_size=8,
            banks=(("working", 4), ("long_term", 2)),
            bank_write_bias=(("working", 0.0), ("long_term", -2.0)),
        )
    )


def token_batch(batch_size=2, seq_len=32, vocab=64):
    """A fixed token batch: a fixture, not data the results depend on."""
    ids = torch.arange(batch_size * (seq_len + 1)) % vocab
    ids = ids.view(batch_size, seq_len + 1)
    return ids[:, :-1], ids[:, 1:]


def test_a_uniform_module_has_no_offenders():
    precision.configure()
    assert dryrun.off_dtype_parameters(nn.Linear(3, 3)) == []


def test_a_mixed_precision_module_is_reported():
    other = other_dtype()
    module = nn.Linear(3, 3).to(other)
    bad = dryrun.off_dtype_parameters(module)
    assert bad
    assert all(str(other) in line for line in bad)


def test_torch_itself_refuses_a_gradient_in_another_dtype():
    """Gradients are tied to their parameter's dtype, so they cannot drift."""
    precision.configure()
    module = nn.Linear(2, 2)
    other = other_dtype()
    with pytest.raises(RuntimeError, match="dtype"):
        module.weight.grad = torch.zeros(2, 2, dtype=other)


def test_recorder_sees_only_the_library_dtype_in_clean_code():
    precision.configure()
    layer = nn.Linear(4, 4)
    with dryrun.Recorder() as rec:
        layer(torch.randn(2, 4)).sum().backward()
    assert rec.seen == {precision.DTYPE}
    assert rec.offenders == []


def test_recorder_catches_a_hidden_upcast():
    """The failure mode this exists for: a stray `.double()` in the middle."""
    precision.configure()
    x = torch.randn(3)
    other = other_dtype()
    with dryrun.Recorder() as rec:
        _ = (x.to(other) * 2).sum()
    assert other in rec.seen
    assert rec.offenders


def test_dry_run_reports_a_clean_healthy_model():
    cfg = engine.Config(steps=1, batch_size=2, seq_len=32)
    report = dryrun.run(tiny_model(), cfg, token_batch(), steps=2)
    assert report.clean
    assert report.offenders == []
    assert report.dtypes_seen == [str(precision.DTYPE)]
    assert report.loss_finite
    assert report.grad_norm_finite
    assert report.tokens_per_second > 0
    assert report.step_seconds == pytest.approx(
        report.forward_seconds
        + report.backward_seconds
        + report.optimizer_seconds
    )
    assert report.parameters > 0


def test_dry_run_rejects_a_mixed_dtype_model_without_running_it(monkeypatch):
    """Mixed-dtype kernels can abort the GPU driver, so the model is judged
    from its storage and no step is launched."""
    model = tiny_model()
    model.state0.data = model.state0.data.to(other_dtype())
    launched = []
    monkeypatch.setattr(
        dryrun.Recorder, "__enter__", lambda self: launched.append(1)
    )
    report = dryrun.run(
        model,
        engine.Config(steps=1, batch_size=2, seq_len=32),
        token_batch(),
        steps=1,
    )
    assert not report.clean
    assert any("state0" in line for line in report.off_dtype_tensors)
    assert launched == []
    assert report.tokens_per_second == 0.0


def test_the_timed_passes_run_without_the_hook(monkeypatch):
    """The auditing hook is installed exactly once, for the audit step."""
    installs = []
    real_enter = dryrun.Recorder.__enter__

    def counting_enter(self):
        installs.append(1)
        return real_enter(self)

    monkeypatch.setattr(dryrun.Recorder, "__enter__", counting_enter)
    dryrun.run(
        tiny_model(),
        engine.Config(steps=1, batch_size=2, seq_len=32),
        token_batch(),
        steps=3,
    )
    assert installs == [1]
