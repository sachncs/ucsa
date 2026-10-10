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


def test_filter_drops_short_and_duplicate_documents_and_counts_them():
    docs = [
        list(range(200)),
        list(range(200)) + [7],  # shares the first 128 tokens: duplicate
        [1, 2, 3],  # too short
        list(range(500, 700)),
    ]
    stats = {}
    kept = list(shards.filter_documents(docs, min_tokens=10, stats=stats))
    assert kept == [docs[0], docs[3]]
    assert stats == {"seen": 4, "kept": 2, "short": 1, "duplicate": 1}


def test_filter_keeps_documents_that_only_share_a_short_start():
    a = [9] * 5 + list(range(100))
    b = [9] * 5 + list(range(200, 300))
    assert len(list(shards.filter_documents([a, b], min_tokens=10))) == 2


def test_compression_ratio_orders_repetitive_prose_and_noise():
    import random

    rng = random.Random(0)
    repetitive = "click here to read more. " * 200
    prose = " ".join(
        rng.choice(
            [
                "the",
                "model",
                "learns",
                "language",
                "from",
                "text",
                "and",
                "predicts",
                "next",
                "words",
                "well",
                "enough",
            ]
        )
        for _ in range(800)
    )
    noise = "".join(chr(rng.randrange(33, 126)) for _ in range(5000))
    r_rep = shards.compression_ratio(repetitive)
    r_prose = shards.compression_ratio(prose)
    r_noise = shards.compression_ratio(noise)
    assert r_rep < r_prose < r_noise
    assert r_rep < 0.1
    assert shards.compression_ratio("") == 1.0


def test_compressibility_filter_drops_both_tails_and_counts_them():
    import random

    rng = random.Random(1)
    repetitive = "buy now " * 300
    noise = "".join(chr(rng.randrange(33, 126)) for _ in range(3000))
    prose = " ".join(
        rng.choice(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
        for _ in range(600)
    )
    stats = {}
    kept = list(
        shards.filter_by_compressibility(
            [repetitive, prose, noise], low=0.05, high=0.6, stats=stats
        )
    )
    assert kept == [prose]
    assert stats == {
        "seen": 3,
        "kept": 1,
        "too_repetitive": 1,
        "too_random": 1,
    }


def test_a_shared_seen_set_keeps_a_document_out_of_a_second_shard():
    docs = [list(range(200)), list(range(300, 500))]
    seen = set()
    first = list(shards.filter_documents(docs[:1], seen=seen))
    second = list(shards.filter_documents(docs, seen=seen))
    assert first == docs[:1]
    assert second == docs[1:]  # the repeated document is dropped
