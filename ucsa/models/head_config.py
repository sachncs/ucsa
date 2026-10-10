"""Head configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ucsa.models import projection


@dataclass(frozen=True)
class Spec:
    """Specification for the projection heads."""

    vocab_size: int = 50257
    num_plan_tokens: int = 64
    num_tools: int = 32
    memory_query_dim: int = 64
    origination_top_k: int = 2
    origination_aux_loss_weight: float = 0.01
    intent_update_scale: float = 0.1


def build(
    hidden_size: int,
    spec: Spec | None = None,
) -> projection.Config:
    """Build a :class:`projection.Config` from a hidden size and optional spec."""
    if spec is None:
        spec = Spec()
    return projection.Config(
        hidden_size=hidden_size,
        vocab_size=spec.vocab_size,
        num_plan_tokens=spec.num_plan_tokens,
        num_tools=spec.num_tools,
        memory_query_dim=spec.memory_query_dim,
        origination_top_k=spec.origination_top_k,
        origination_aux_loss_weight=spec.origination_aux_loss_weight,
        intent_update_scale=spec.intent_update_scale,
    )


def from_cfg(
    cfg: Mapping[str, object] | object,
) -> projection.Config:
    """Build a :class:`projection.Config` from a architecture.Config-like object."""
    hidden_size = int(getattr(cfg, "hidden_size", 128))
    vocab_size = int(getattr(cfg, "vocab_size", 50257))
    top_k = int(getattr(cfg, "origination_top_k", 2))
    aux_weight = float(getattr(cfg, "origination_aux_loss_weight", 0.01))
    update_scale = float(getattr(cfg, "intent_update_scale", 0.1))
    head_section = getattr(cfg, "heads", None)
    if head_section is None and isinstance(cfg, Mapping):
        head_section = cfg.get("heads")
    if head_section is None:
        return projection.Config(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            origination_top_k=top_k,
            origination_aux_loss_weight=aux_weight,
            intent_update_scale=update_scale,
        )
    spec = Spec(
        vocab_size=int(getattr(head_section, "vocab_size", vocab_size)),
        num_plan_tokens=int(getattr(head_section, "num_plan_tokens", 64)),
        num_tools=int(getattr(head_section, "num_tools", 32)),
        memory_query_dim=int(getattr(head_section, "memory_query_dim", 64)),
        origination_top_k=top_k,
        origination_aux_loss_weight=aux_weight,
        intent_update_scale=update_scale,
    )
    return build(hidden_size, spec)


__all__ = [
    "Spec",
    "build",
    "from_cfg",
]
