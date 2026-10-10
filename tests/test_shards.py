import itertools

import numpy as np
import pytest
import torch

from ucsa.training import shards


def make_shard(tmp_path, n_tokens=1000):
    path = str(tmp_path / "s.bin")
    shards.write_shard([list(range(n_tokens - 1))], path)
    return shards.Shard(path)


def take_inputs(batches, count):
    return [x for x, _ in itertools.islice(batches, count)]


def test_write_shard_appends_eos_and_stops_after_the_limit(tmp_path):
    path = str(tmp_path / "a.bin")
    written = shards.write_shard([[1, 2], [3], [4, 5, 6]], path, limit=4)
    tokens = np.fromfile(path, dtype=shards.DTYPE).tolist()
    eos = shards.EOS_ID
    # Documents are written whole: [1,2,EOS] (3 tokens) then [3,EOS] reaches 5.
    assert tokens == [1, 2, eos, 3, eos]
    assert written == 5


def test_targets_are_inputs_shifted_by_one(tmp_path):
    shard = make_shard(tmp_path)
    x, y = next(shard.batches(2, 16, seed=None))
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x.dtype == torch.int64


def test_batches_are_a_pure_function_of_seed_and_step(tmp_path):
    shard = make_shard(tmp_path)
    first = take_inputs(shard.batches(2, 16, seed=7), 6)
    again = take_inputs(shard.batches(2, 16, seed=7), 6)
    other = take_inputs(shard.batches(2, 16, seed=8), 6)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))


def test_skip_resumes_the_exact_stream(tmp_path):
    shard = make_shard(tmp_path)
    full = take_inputs(shard.batches(2, 16, seed=3), 9)
    resumed = take_inputs(shard.batches(2, 16, skip=5, seed=3), 4)
    assert all(
        torch.equal(a, b) for a, b in zip(full[5:], resumed, strict=True)
    )


def test_an_epoch_visits_every_window_once(tmp_path):
    shard = make_shard(tmp_path, n_tokens=1 + 16 * 20)
    seen = []
    for x, _ in shard.batches(4, 16, seed=1, loop=False):
        seen += x[:, 0].tolist()
    assert sorted(seen) == [16 * i for i in range(20)]


def test_missing_or_short_shards_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        shards.Shard(str(tmp_path / "nope.bin"))
    shard = make_shard(tmp_path, n_tokens=8)
    with pytest.raises(ValueError):
        next(shard.batches(1, 16))


def test_rewriting_a_shard_never_disturbs_a_reader_that_has_it_mapped(tmp_path):
    """Regression: an in-place rewrite truncated the file under a mapped
    reader, which can crash it. The reader must keep its intact copy."""
    path = str(tmp_path / "s.bin")
    shards.write_shard([list(range(1000))], path)
    reader = shards.Shard(path)
    before = np.array(reader.tokens[:50])
    shards.write_shard([list(range(500, 700))], path)  # a different shard
    assert np.array_equal(np.array(reader.tokens[:50]), before)
    assert len(reader) == 1001  # the old mapping is unchanged
    assert len(shards.Shard(path)) == 201  # new readers see the new file


def test_a_failed_write_leaves_the_previous_shard_intact(tmp_path):
    path = str(tmp_path / "s.bin")
    shards.write_shard([[1, 2, 3]], path)
    with open(path, "rb") as f:
        original = f.read()

    def exploding():
        yield [4, 5]
        raise RuntimeError("source failed mid-write")

    with pytest.raises(RuntimeError):
        shards.write_shard(exploding(), path)
    with open(path, "rb") as f:
        assert f.read() == original
    assert not (tmp_path / "s.bin.tmp").exists()
