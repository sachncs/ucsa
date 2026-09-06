"""Tests for the origination probe script."""

from __future__ import annotations

import pytest
import torch
from scripts.probe_origination import build_probe_inputs, descent_sweep

from ucsa.models.ucsa import UCSA, UCSAConfig


def tiny_model() -> UCSA:
    """Return a tiny UCSA with the origination path active."""
    torch.manual_seed(0)
    return UCSA(
        UCSAConfig(
            hidden_size=32,
            num_layers=2,
            vocab_size=100,
            reasoning_iterations=3,
            observation_mix=0.5,
        )
    )


def test_probe_inputs_repeat_their_first_half() -> None:
    """The probe task has structure, so there is something to improve."""
    pairs = build_probe_inputs(100, count=3, seq_len=8, seed=1)
    assert len(pairs) == 3
    for inputs, targets in pairs:
        assert inputs.shape == (1, 7)
        assert targets.shape == (1, 7)
        # targets are the inputs shifted by one
        assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_probe_inputs_differ_from_each_other() -> None:
    """Collapse metrics need differing inputs."""
    pairs = build_probe_inputs(100, count=2, seq_len=8, seed=1)
    assert not torch.equal(pairs[0][0], pairs[1][0])


def test_descent_sweep_reports_cost_with_quality() -> None:
    """Every row carries its forward-pass count.

    A quality claim without the cost is not a result, so the two are
    reported together.
    """
    model = tiny_model()
    pairs = build_probe_inputs(100, count=2, seq_len=8, seed=1)
    rows = descent_sweep(model, pairs, [0, 2], learning_rate=0.05)
    assert [row["intent_steps"] for row in rows] == [0, 2]
    for row in rows:
        assert row["num_inputs"] == 2
        assert isinstance(row["forward_passes_per_input"], int)
        assert row["forward_passes_per_input"] >= 2
        assert -1.0 <= float(row["outcome_correlation"]) <= 1.0


def test_descent_sweep_zero_steps_does_not_move_the_bank() -> None:
    """``K=0`` is inert."""
    model = tiny_model()
    pairs = build_probe_inputs(100, count=2, seq_len=8, seed=1)
    rows = descent_sweep(model, pairs, [0], learning_rate=0.05)
    assert float(rows[0]["mean_intent_shift"]) == pytest.approx(0.0)
