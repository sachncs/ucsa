import math

import pytest

from ucsa.training.tuning import sample_candidates, successive_halving

SPACE = {"a": [1, 2, 3], "b": [10, 20]}


def test_sampling_is_deterministic_and_distinct():
    a = sample_candidates(SPACE, 4, seed=0)
    assert a == sample_candidates(SPACE, 4, seed=0)
    assert len({tuple(sorted(c.items())) for c in a}) == 4


def test_sampling_returns_full_grid_when_n_is_large():
    assert len(sample_candidates(SPACE, 100, seed=0)) == 6


def test_halving_finds_the_optimum_and_shrinks_each_rung():
    cands = sample_candidates(SPACE, 6, 0)
    seen = []

    def score(c, steps):
        seen.append((steps, c["a"]))
        return abs(c["a"] - 2) + c["b"] / 100 + 1 / steps

    res = successive_halving(
        cands, score, [10, 20, 40], keep=0.5, log=lambda s: None
    )
    assert res.best == {"a": 2, "b": 10}
    per_rung = [sum(1 for s, _ in seen if s == b) for b in (10, 20, 40)]
    assert per_rung == [6, 3, 1]


def test_failures_rank_last_and_do_not_abort():
    def score(c, steps):
        if c["a"] == 1:
            raise RuntimeError("OOM")
        return float(c["a"])

    res = successive_halving(
        sample_candidates(SPACE, 6, 0), score, [5, 10], log=lambda s: None
    )
    assert res.best["a"] == 2
    assert any(t.error and "OOM" in t.error for t in res.trials)


def test_cache_resumes_without_rescoring(tmp_path):
    calls = []

    def score(c, steps):
        calls.append(1)
        return float(c["a"])

    cache = str(tmp_path / "c.json")
    cands = sample_candidates(SPACE, 6, 0)
    successive_halving(cands, score, [5], log=lambda s: None, cache_path=cache)
    n = len(calls)
    successive_halving(cands, score, [5], log=lambda s: None, cache_path=cache)
    assert len(calls) == n


def test_nonfinite_score_is_a_failure():
    res = successive_halving(
        [{"a": 1}, {"a": 2}],
        lambda c, s: math.nan if c["a"] == 1 else 5.0,
        [3],
        log=lambda s: None,
    )
    assert res.best == {"a": 2}


def test_bad_arguments():
    with pytest.raises(ValueError):
        successive_halving([], lambda c, s: 0.0, [1])
    with pytest.raises(ValueError):
        successive_halving([{"a": 1}], lambda c, s: 0.0, [1], keep=1.5)
