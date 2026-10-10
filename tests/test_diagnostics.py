import numpy as np
import pytest
import torch

from ucsa.models import recurrent
from ucsa.training import diagnostics
from ucsa.utils import precision


def test_identical_models_have_an_interval_around_zero():
    a = np.random.default_rng(0).normal(5.0, 1.0, 300)
    r = diagnostics.paired_bootstrap(a, a.copy())
    assert r.mean_gap == 0.0
    assert r.low <= 0.0 <= r.high
    assert not r.significant


def test_a_consistent_improvement_is_detected_despite_noisy_windows():
    rng = np.random.default_rng(1)
    difficulty = rng.normal(5.0, 2.0, 400)  # large shared window noise
    better = difficulty - 0.05 + rng.normal(0, 0.02, 400)
    r = diagnostics.paired_bootstrap(better, difficulty)
    assert r.mean_gap == pytest.approx(-0.05, abs=0.01)
    assert r.significant
    assert r.high < 0  # a is better, with confidence


def test_pairing_beats_comparing_means_when_difficulty_varies():
    rng = np.random.default_rng(2)
    difficulty = rng.normal(5.0, 2.0, 400)
    b = difficulty
    a = difficulty - 0.05 + rng.normal(0, 0.02, 400)
    # Unpaired spread (~2/sqrt(n)) dwarfs the gain; the paired gap is exact.
    assert abs(a.mean() - b.mean()) < 3 * (a.std() / np.sqrt(len(a)))
    assert diagnostics.paired_bootstrap(a, b).significant


def test_interval_covers_the_true_gap_at_its_nominal_rate():
    rng = np.random.default_rng(3)
    covered = 0
    for trial in range(100):
        base = rng.normal(0, 1, 100)
        a = base + 0.3 + rng.normal(0, 0.5, 100)
        r = diagnostics.paired_bootstrap(a, base, resamples=300, seed=trial)
        covered += r.low <= 0.3 <= r.high
    assert covered >= 88  # nominal 95%, allow sampling slack


def test_result_is_reproducible_for_a_seed():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
    assert diagnostics.paired_bootstrap(a, b, seed=7) == (
        diagnostics.paired_bootstrap(a, b, seed=7)
    )


@pytest.mark.parametrize(
    ("a", "b"), [(np.zeros(3), np.zeros(4)), (np.zeros(0), np.zeros(0))]
)
def test_bad_shapes_are_rejected(a, b):
    with pytest.raises(ValueError):
        diagnostics.paired_bootstrap(a, b)


def test_bits_gap_converts_nats_to_bits():
    r = diagnostics.PairedResult(-0.693147, -1.0, 0.0, False, 10)
    assert diagnostics.bits_gap(r) == pytest.approx(-1.0, abs=1e-4)


def test_window_nll_is_per_window_and_deterministic():
    torch.manual_seed(0)
    model = recurrent.Model(
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
    g = torch.Generator().manual_seed(1)

    def batches():
        for _ in range(5):
            x = torch.randint(0, 32, (2, 33), generator=g)
            yield x[:, :32], x[:, 1:33]

    out = diagnostics.window_nll(model, batches(), count=3)
    assert out.shape == (6,)  # 3 batches x 2 windows
    tail = diagnostics.window_nll(model, batches(), count=1, tail=8)
    assert tail.shape == (2,)
    assert np.isfinite(out).all()


def small_model(**kw):
    precision.configure()
    torch.manual_seed(0)
    return recurrent.Model(
        recurrent.Config(
            vocab_size=32,
            hidden=32,
            layers=1,
            heads=2,
            ffn_dim=64,
            chunk_size=8,
            banks=(("working", 4), ("long_term", 2)),
            bank_write_bias=(("working", 0.0), ("long_term", -2.0)),
            **kw,
        )
    )


def repeated_batch(first_chunk_seed):
    """Chunk 0 differs between calls; every later token (and target) is the
    same, so only the model's use of earlier chunks can change later losses."""
    g = torch.Generator().manual_seed(1)
    tail = torch.randint(0, 32, (1, 25), generator=g)
    h = torch.Generator().manual_seed(first_chunk_seed)
    head = torch.randint(0, 32, (1, 8), generator=h)
    tokens = torch.cat([head, tail], dim=1)  # 33 tokens
    return [(tokens[:, :32], tokens[:, 1:33])]


def test_chunk_profile_has_one_entry_per_chunk():
    profile = diagnostics.chunk_profile(
        small_model(), iter(repeated_batch(0)), count=1
    )
    assert profile.shape == (4,)
    assert np.isfinite(profile).all()


def test_without_state_or_window_later_chunks_ignore_earlier_ones():
    """The chunk-local control's profile for chunks 1+ cannot depend on
    chunk 0: it has neither the state nor a window into the previous chunk."""
    model = small_model(use_state=False, window=0)
    a = diagnostics.chunk_profile(model, iter(repeated_batch(0)), 1)
    b = diagnostics.chunk_profile(model, iter(repeated_batch(5)), 1)
    assert not np.allclose(a[0], b[0])  # chunk 0 itself differs
    assert np.allclose(a[1:], b[1:], atol=1e-5)  # the rest cannot notice


def test_with_state_later_chunks_do_depend_on_earlier_ones():
    model = small_model(window=0)
    a = diagnostics.chunk_profile(model, iter(repeated_batch(0)), 1)
    b = diagnostics.chunk_profile(model, iter(repeated_batch(5)), 1)
    assert not np.allclose(a[1:], b[1:], atol=1e-6)


def test_rate_distortion_rows_scale_with_the_slots_read():
    model = small_model()
    lengths = torch.full((32,), 2, dtype=torch.long)
    rows = diagnostics.state_rate_distortion(
        model, lambda skip: iter(repeated_batch(0)), 1, [1, 3, 6], lengths
    )
    assert [r["slots"] for r in rows] == [1.0, 3.0, 6.0]
    bits = torch.finfo(precision.DTYPE).bits
    assert rows[1]["state_bits"] == 3 * 32 * bits
    assert all(np.isfinite(r["bpb"]) for r in rows)
    assert len({r["bpb"] for r in rows}) > 1  # reading less state changes it
