"""Mixture of Experts block.

Implements a top-k router and a bank of expert FFNs. Used by the
:class:`ucsa.models.transformer_operator.TransformerBlock` on the upper
half of layers when Mixture of Experts is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MoEConfig:
    """Configuration for :class:`MixtureOfExperts`.

    Attributes:
        num_experts: Number of experts in the bank.
        top_k: Number of experts activated per token.
        capacity_factor: Multiplier on the average tokens-per-expert
            allocation, controlling the expert capacity.
        aux_loss_weight: Weight applied to the load-balancing auxiliary
            loss returned alongside the main loss.
    """

    num_experts: int = 4
    top_k: int = 2
    capacity_factor: float = 1.25
    aux_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError(
                f"num_experts must be positive, got {self.num_experts}."
            )
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError(
                f"top_k must be in [1, {self.num_experts}], got {self.top_k}."
            )
        if self.capacity_factor <= 0.0:
            raise ValueError(
                f"capacity_factor must be positive, got {self.capacity_factor}."
            )
        if self.aux_loss_weight < 0.0:
            raise ValueError(
                f"aux_loss_weight must be non-negative, got {self.aux_loss_weight}."
            )


class Expert(nn.Module):
    """A single gated FFN expert.

    Uses the same gated linear unit as
    :class:`ucsa.models.transformer_operator.FeedForward`.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Initialise the expert.

        Args:
            hidden_size: Hidden dimensionality.
            intermediate_size: Intermediate dimensionality.
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the expert.

        Args:
            x: Tensor of shape ``(..., hidden_size)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MixtureOfExperts(nn.Module):
    """Top-k Mixture of Experts block.

    Routes each token to the ``top_k`` highest-scoring experts. The output is
    the weighted combination of expert outputs. Returns a tuple
    ``(output, aux_loss)`` so the block can be a drop-in replacement for a
    dense FFN.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: MoEConfig,
    ) -> None:
        """Initialise the MoE block.

        Args:
            hidden_size: Hidden dimensionality.
            intermediate_size: Expert intermediate size.
            config: MoE configuration.
        """
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.router = nn.Linear(hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(hidden_size, intermediate_size)
            for _ in range(config.num_experts)
        )
        self.last_router_logits: Tensor | None = None  # ponytail: stash for aux loss wiring; overwritten each forward

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the MoE block.

        Args:
            x: Tensor of shape ``(batch, seq, hidden_size)``.

        Returns:
            Tuple ``(output, aux_loss)`` where ``output`` has the same shape
            as ``x`` and ``aux_loss`` is a scalar tensor suitable for adding
            to the main loss.
        """
        batch, seq, hidden = x.shape
        flat_x = x.reshape(-1, hidden)
        num_tokens = flat_x.shape[0]

        router_logits = self.router(flat_x)
        self.last_router_logits = router_logits
        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.config.top_k, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        expert_capacity = int(
            (num_tokens * self.config.top_k / self.config.num_experts)
            * self.config.capacity_factor
        ) + 1
        output = torch.zeros_like(flat_x)
        router_prob_per_expert = routing_weights.sum(dim=0) / num_tokens

        for expert_index, expert in enumerate(self.experts):
            mask = topk_indices == expert_index
            token_indices, slot_indices = torch.where(mask)
            if token_indices.numel() == 0:
                continue
            if token_indices.numel() > expert_capacity:
                token_indices = token_indices[:expert_capacity]
                slot_indices = slot_indices[:expert_capacity]
            expert_input = flat_x[token_indices]
            expert_output = expert(expert_input)
            weights = topk_weights[token_indices, slot_indices].unsqueeze(-1)
            output.index_add_(
                0,
                token_indices,
                expert_output * weights,
            )

        output = output.reshape(batch, seq, hidden)
        aux_loss = self._load_balancing_loss(router_logits, router_prob_per_expert)
        return output, aux_loss

    def _load_balancing_loss(
        self,
        router_logits: Tensor,
        router_prob_per_expert: Tensor,
    ) -> Tensor:
        """Compute the Switch Transformer load-balancing auxiliary loss.

        Args:
            router_logits: Tensor of shape ``(num_tokens, num_experts)``.
            router_prob_per_expert: Per-expert average routing probability
                of shape ``(num_experts,)``.

        Returns:
            Scalar tensor with the load-balancing loss.
        """
        num_tokens = router_logits.shape[0]
        router_prob_per_expert = router_prob_per_expert * num_tokens
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_logits.argmax(dim=-1), num_classes=self.config.num_experts
            ).float()
        tokens_per_expert = expert_mask.sum(dim=0)
        loss = (tokens_per_expert * router_prob_per_expert).sum()
        loss = loss / (num_tokens * self.config.num_experts)
        return self.config.aux_loss_weight * loss


__all__ = ["Expert", "MixtureOfExperts", "MoEConfig"]
