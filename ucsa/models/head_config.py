"""Head configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ucsa.models.projection_heads import HeadConfig


@dataclass(frozen=True)
class HeadSpec:
    """Specification for the projection heads."""

    vocab_size: int = 50257
    num_plan_tokens: int = 64
    num_tools: int = 32
    memory_query_dim: int = 64


def build_head_config(
    hidden_size: int,
    spec: HeadSpec | None = None,
) -> HeadConfig:
    """Build a :class:`HeadConfig` from a hidden size and optional spec."""
    if spec is None:
        spec = HeadSpec()
    return HeadConfig(
        hidden_size=hidden_size,
        vocab_size=spec.vocab_size,
        num_plan_tokens=spec.num_plan_tokens,
        num_tools=spec.num_tools,
        memory_query_dim=spec.memory_query_dim,
    )


def build_head_config_from_cfg(cfg: Mapping[str, object] | object) -> HeadConfig:
    """Build a :class:`HeadConfig` from a UCSAConfig-like object."""
    hidden_size = int(getattr(cfg, "hidden_size", 128))
    vocab_size = int(getattr(cfg, "vocab_size", 50257))
    head_section = getattr(cfg, "heads", None)
    if head_section is None and isinstance(cfg, Mapping):
        head_section = cfg.get("heads")  # type: ignore[union-attr]
    if head_section is None:
        return HeadConfig(hidden_size=hidden_size, vocab_size=vocab_size)
    spec = HeadSpec(
        vocab_size=int(getattr(head_section, "vocab_size", vocab_size)),
        num_plan_tokens=int(getattr(head_section, "num_plan_tokens", 64)),
        num_tools=int(getattr(head_section, "num_tools", 32)),
        memory_query_dim=int(getattr(head_section, "memory_query_dim", 64)),
    )
    return build_head_config(hidden_size, spec)


__all__ = ["HeadConfig", "HeadSpec", "build_head_config", "build_head_config_from_cfg"]