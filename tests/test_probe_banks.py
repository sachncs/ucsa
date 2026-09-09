"""Tests for the memory-bank probe utility."""

from __future__ import annotations

from scripts.probe_banks import probe_banks

from ucsa.models.state import BANK_NAMES
from ucsa.models.ucsa import UCSA, UCSAConfig


def test_probe_banks_summarises_every_bank():
    cfg = UCSAConfig(
        hidden_size=32,
        vocab_size=200,
        num_layers=2,
        reasoning_iterations=2,
    )
    model = UCSA(cfg)
    report = probe_banks(model, top_k=3)
    assert "banks" in report
    assert "centroid_cosine_sim" in report
    assert len(report["banks"]) == len(BANK_NAMES)
    bank_names = {s["name"] for s in report["banks"]}
    assert bank_names == set(BANK_NAMES)
    assert "intent" in bank_names


def test_probe_banks_centroid_matrix_is_symmetric_self_one():
    cfg = UCSAConfig(
        hidden_size=32,
        vocab_size=200,
        num_layers=2,
        reasoning_iterations=2,
    )
    model = UCSA(cfg)
    report = probe_banks(model, top_k=3)
    sim = report["centroid_cosine_sim"]
    matrix = sim["matrix"]
    n = len(matrix)
    assert n == len(sim["names"])
    # Self-similarity should be ~1.0 (centroid dot itself normalised).
    for i in range(n):
        assert abs(matrix[i][i] - 1.0) < 1e-4
    # Symmetric.
    for i in range(n):
        for j in range(n):
            assert abs(matrix[i][j] - matrix[j][i]) < 1e-4


def test_probe_banks_top_tokens_are_strings():
    cfg = UCSAConfig(
        hidden_size=32,
        vocab_size=200,
        num_layers=2,
        reasoning_iterations=2,
    )
    model = UCSA(cfg)
    report = probe_banks(model, top_k=3)
    for s in report["banks"]:
        # Top-tokens is a list of strings of length <= 10.
        assert isinstance(s["top_tokens"], list)
        assert all(isinstance(t, str) for t in s["top_tokens"])
        assert len(s["top_tokens"]) <= 10


def test_probe_banks_norm_stats_are_finite():
    cfg = UCSAConfig(
        hidden_size=32,
        vocab_size=200,
        num_layers=2,
        reasoning_iterations=2,
    )
    model = UCSA(cfg)
    report = probe_banks(model, top_k=3)
    for s in report["banks"]:
        for k in (
            "norm_mean",
            "norm_std",
            "norm_min",
            "norm_max",
            "retention_mean",
            "retention_min",
            "retention_max",
        ):
            assert k in s
            v = s[k]
            assert isinstance(v, (int, float))
            assert v == v  # not NaN
