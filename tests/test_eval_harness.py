"""Tests for the standard-LM eval harness.

Kept small and offline-friendly: only smoke tests the loader
definitions, registry, and dataclass shape. The full task data
needs network access; that's exercised by ``scripts/eval.py``.
"""
from __future__ import annotations

from ucsa.training.eval_harness import (
    TASK_REGISTRY,
    EvalResult,
    evaluate_all,
    evaluate_task,
)


class DummyModel:
    """Minimal model stand-in: returns constant logits favoring choice 0."""

    def eval(self):
        pass

    def train(self):
        pass


class DummyTokenizer:
    """Maps every string to a list of token ids equal to the character codes."""

    def encode(self, text: str) -> list[int]:
        return [min(255, ord(c)) for c in text][:64]


class _ConstantLogitsModel:
    """Returns logits so that token 0 is always the most likely."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, ids):
        # ponytail: not used by rank-by-loglik — kept only for shape parity
        raise NotImplementedError


def test_eval_result_dataclass():
    r = EvalResult(name="hellaswag", n=10, correct=4, accuracy=0.4)
    d = r.to_dict()
    assert d["name"] == "hellaswag"
    assert d["n"] == 10
    assert d["accuracy"] == 0.4


def test_task_registry_has_five_tasks():
    expected = {"hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"}
    assert expected.issubset(set(TASK_REGISTRY.keys()))
    for spec in TASK_REGISTRY.values():
        assert spec.name in TASK_REGISTRY
        assert spec.max_examples is None or spec.max_examples > 0


def test_evaluate_all_smoke_with_fake_examples(monkeypatch):
    """Make each loader return 2 trivial examples; rank-by-loglik
    should produce a finite accuracy number without touching network."""
    import torch

    from ucsa.training import eval_harness

    def fake_loader():
        yield {"context": "x", "choices": ["a", "b"], "label": 0}
        yield {"context": "x", "choices": ["a", "b"], "label": 0}

    class FakeModel:
        """Constant-logits model: returns a (1, S, V) tensor with
        uniform logits so all choices score equally."""

        def eval(self):
            pass

        def train(self):
            pass

        def __call__(self, ids):
            # (B, S, V) zeros — all choices equally likely
            return torch.zeros(
                ids.shape[0], ids.shape[1], 256,
                dtype=torch.float32,
            )

    # Replace each loader with our trivial one.
    saved = {n: s.loader for n, s in eval_harness.TASK_REGISTRY.items()}
    for n, _s in eval_harness.TASK_REGISTRY.items():
        eval_harness.TASK_REGISTRY[n] = eval_harness.TaskSpec(
            name=n, loader=fake_loader, max_examples=2
        )
    try:
        results = evaluate_all(
            None,
            model=FakeModel(),
            tokenizer=DummyTokenizer(),
            device=None,
        )
        assert len(results) == 5
        for r in results:
            assert r.n == 2
            assert 0.0 <= r.accuracy <= 1.0
            # Log-likelihood is finite (uniform = -log(V) per token).
            assert -100.0 < r.log_likelihood_mean < 0.0
    finally:
        for n, s in saved.items():
            eval_harness.TASK_REGISTRY[n] = eval_harness.TaskSpec(
                name=n, loader=s, max_examples=eval_harness.TASK_REGISTRY[n].max_examples
            )


def test_evaluate_task_returns_finite_accuracy_when_loader_is_empty():
    spec = TASK_REGISTRY["hellaswag"]
    saved_loader = spec.loader
    spec.loader = lambda: iter([])
    try:
        r = evaluate_task(spec, _ConstantLogitsModel(), DummyTokenizer(), None)
        assert r.n == 0
        assert r.accuracy == 0.0
        assert r.log_likelihood_mean == 0.0
    finally:
        spec.loader = saved_loader
