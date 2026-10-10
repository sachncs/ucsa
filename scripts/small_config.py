"""Shared UCSA-small architecture overrides for train.py and eval.py.

Both scripts must build the exact same architecture, otherwise a
checkpoint trained by one cannot be loaded by the other.
"""

from __future__ import annotations

from typing import Any

SMALL_MODEL_OVERRIDES: dict[str, Any] = {
    "hidden_size": 384,
    "num_layers": 6,
    "num_q_heads": 8,
    "num_kv_heads": 4,
    "intermediate_size": 1024,
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "num_concepts": 16,
    "attention_dropout": 0.1,
    "residual_dropout": 0.1,
    "ffn_dropout": 0.1,
}


def apply_small_overrides(cfg: dict[str, Any]) -> None:
    """Mutate ``cfg`` in place to the UCSA-small architecture."""
    cfg["reasoning_iterations"] = 4
    cfg["model"].update(SMALL_MODEL_OVERRIDES)
