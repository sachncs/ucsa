"""The benchmark harness: scoring rule, ties, grouping and determinism."""

import math

import pytest
import torch

from tests.helpers import Tokenizer, tiny_model
from ucsa.training import eval_harness

CPU = torch.device("cpu")


def pairs(seed: int = 0):
    yield {"context": "", "choices": ["abcd", "efgh"], "label": 0, "group": "a"}
    yield {"context": "", "choices": ["ijkl", "mnop"], "label": 0, "group": "a"}
    yield {"context": "", "choices": ["qrst", "uvwx"], "label": 0, "group": "b"}


TASK = eval_harness.Task("pairs", pairs)


def test_the_task_registry_is_blimp():
    assert list(eval_harness.TASKS) == ["blimp"]


def test_accuracy_is_the_mean_over_groups_not_over_examples():
    """Group a has two examples and b one; groups weigh equally."""
    result = eval_harness.evaluate_task(TASK, tiny_model(), Tokenizer(), CPU)
    assert result.n == 3
    assert set(result.groups) == {"a", "b"}
    expected = sum(result.groups.values()) / 2
    assert result.accuracy == pytest.approx(expected)
    assert 0.0 <= result.stderr <= 0.5


def test_a_cap_limits_examples():
    result = eval_harness.evaluate_task(
        TASK, tiny_model(), Tokenizer(), CPU, max_examples=2
    )
    assert result.n == 2
    assert result.max_examples == 2


def test_the_right_choice_must_score_strictly_higher():
    """Identical sentences tie, and a tie is not a win."""

    def same(seed: int = 0):
        yield {"context": "", "choices": ["abcd", "abcd"], "label": 0}

    result = eval_harness.evaluate_task(
        eval_harness.Task("same", same), tiny_model(), Tokenizer(), CPU
    )
    assert result.accuracy == 0.0


def test_a_task_with_no_examples_reports_zero_not_a_crash():
    empty = eval_harness.Task("empty", lambda seed: iter([]))
    result = eval_harness.evaluate_task(empty, tiny_model(), Tokenizer(), CPU)
    assert (result.n, result.accuracy, result.groups) == (0, 0.0, {})


def test_results_are_deterministic():
    model = tiny_model()
    a = eval_harness.evaluate_task(TASK, model, Tokenizer(), CPU)
    b = eval_harness.evaluate_task(TASK, model, Tokenizer(), CPU)
    assert a == b


def test_scoring_a_batch_equals_scoring_each_choice_alone():
    """Padding rows to a common width never changes a score."""
    model, tok = tiny_model(), Tokenizer()
    together = eval_harness.choice_logprobs(
        model, tok, "", ["ab", "cdefghij"], CPU
    )
    for i, choice in enumerate(["ab", "cdefghij"]):
        alone = eval_harness.choice_logprobs(model, tok, "", [choice], CPU)[0]
        assert together[i][1] == alone[1]
        assert together[i][0] == pytest.approx(alone[0], abs=1e-4)


def test_a_choice_token_is_scored_only_from_what_precedes_it():
    """Changing a later token never changes an earlier token's score."""
    model, tok = tiny_model(), Tokenizer()
    short = eval_harness.choice_logprobs(model, tok, "", ["abc"], CPU)[0]
    longer = eval_harness.choice_logprobs(model, tok, "", ["abcd"], CPU)[0]
    ids = torch.tensor([[tok.eos_token_id, *tok.encode("abcd")]])
    with torch.no_grad():
        logits = model(ids)["logits"][0]
    first3 = -torch.nn.functional.cross_entropy(
        logits[:3], ids[0, 1:4], reduction="sum"
    )
    assert short[0] == pytest.approx(float(first3), abs=1e-4)
    assert longer[1] == short[1] + 1


def test_an_empty_choice_scores_nothing():
    model, tok = tiny_model(), Tokenizer()
    assert eval_harness.choice_logprobs(model, tok, "", [""], CPU) == [(0.0, 0)]


def test_a_context_is_read_before_the_choice():
    model, tok = tiny_model(), Tokenizer()
    with_ctx = eval_harness.choice_logprobs(model, tok, "hello", ["ab"], CPU)
    without = eval_harness.choice_logprobs(model, tok, "", ["ab"], CPU)
    assert with_ctx[0][0] != without[0][0]
    assert math.isfinite(with_ctx[0][0])


def test_a_long_context_is_truncated_from_the_left():
    model, tok = tiny_model(), Tokenizer()
    total, count = eval_harness.choice_logprobs(
        model, tok, "a" * 500, ["bc"], CPU, max_len=32
    )[0]
    assert count == len(tok.encode(" bc"))
    assert math.isfinite(total)


def test_unknown_tasks_are_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="unknown tasks"):
        eval_harness.evaluate_all(["nope"], tiny_model(), Tokenizer(), CPU)


def test_blimp_loader_yields_the_grammatical_sentence_first(monkeypatch):
    class Rows(list):
        pass

    data = Rows(
        [
            {"sentence_good": "good one", "sentence_bad": "bad one"},
            {"sentence_good": "good two", "sentence_bad": "bad two"},
        ]
    )
    monkeypatch.setattr(
        eval_harness.datasets, "get_dataset_config_names", lambda name: ["p1"]
    )
    monkeypatch.setattr(
        eval_harness.datasets, "load_dataset", lambda *a, **k: data
    )
    examples = list(eval_harness.load_blimp(seed=0))
    assert len(examples) == 2
    for ex in examples:
        assert ex["label"] == 0
        assert ex["group"] == "p1"
        assert ex["choices"][0].startswith("good")
        assert ex["choices"][1].startswith("bad")
    assert [e["choices"] for e in examples] == [
        e["choices"] for e in eval_harness.load_blimp(seed=0)
    ]
