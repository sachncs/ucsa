"""Training losses.

UCSA combines four losses:

- ``L_AR`` -- autoregressive cross-entropy on language head output.
- ``L_JEPA`` -- auxiliary latent-prediction loss with block-level and
  iteration-level heads.
- ``L_MEMORY`` -- memory stability regulariser.
- ``L_ROUTER`` -- MoE router load-balancing auxiliary loss.

The total loss is::

    L = L_AR + lambda1 * L_JEPA + lambda2 * L_MEMORY + lambda3 * L_ROUTER
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LossWeights:
    """Weights for the four loss components.

    Attributes:
        jepa: Coefficient for the JEPA auxiliary loss.
        memory: Coefficient for the memory stability regulariser.
        router: Coefficient for the MoE load-balancing loss.
    """

    jepa: float = 0.1
    memory: float = 0.01
    router: float = 0.01

    def __post_init__(self) -> None:
        if self.jepa < 0.0:
            raise ValueError(f"jepa must be non-negative, got {self.jepa}.")
        if self.memory < 0.0:
            raise ValueError(f"memory must be non-negative, got {self.memory}.")
        if self.router < 0.0:
            raise ValueError(f"router must be non-negative, got {self.router}.")


class AutoregressiveLoss(nn.Module):
    """Cross-entropy loss for the language head."""

    def __init__(self, ignore_index: int = -100) -> None:
        """Initialise the loss.

        Args:
            ignore_index: Target index to ignore.
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Tensor of shape ``(batch, seq, vocab)``.
            targets: Tensor of shape ``(batch, seq)``.

        Returns:
            Scalar loss tensor.
        """
        batch, seq, vocab = logits.shape
        flat_logits = logits.reshape(batch * seq, vocab)
        flat_targets = targets.reshape(batch * seq)
        return torch.nn.functional.cross_entropy(
            flat_logits, flat_targets, ignore_index=self.ignore_index
        )


class JEPALoss(nn.Module):
    """Latent-prediction loss combining cosine similarity and SmoothL1.

    The loss computes:

    .. math::

        L = \\alpha * (1 - \\cos(pred, target)).mean()
          + (1 - \\alpha) * smooth_l1(pred, target).mean()

    where :math:`\\alpha` defaults to ``0.5``.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """Initialise the loss.

        Args:
            alpha: Weight of the cosine term.
        """
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        self.alpha = alpha

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        """Compute the JEPA loss.

        Args:
            predicted: Predicted embeddings of shape
                ``(batch, seq, hidden)``.
            target: Target embeddings of the same shape.

        Returns:
            Scalar loss tensor.
        """
        if predicted.shape != target.shape:
            raise ValueError(
                f"predicted and target must share shape; got "
                f"{tuple(predicted.shape)} and {tuple(target.shape)}."
            )
        cosine = torch.nn.functional.cosine_similarity(
            predicted, target, dim=-1
        )
        cosine_term = (1.0 - cosine).mean()
        l1_term = torch.nn.functional.smooth_l1_loss(predicted, target)
        return self.alpha * cosine_term + (1.0 - self.alpha) * l1_term


class MemoryStabilityLoss(nn.Module):
    """Penalise drift of the long-term memory bank.

    Encourages the long-term bank to remain close to a learned baseline.
    """

    def forward(self, long_term: Tensor, baseline: Tensor | None = None) -> Tensor:
        """Compute memory stability loss.

        Args:
            long_term: Long-term memory bank of shape
                ``(num_tokens, hidden_size)``.
            baseline: Optional reference. Defaults to a detached copy of
                ``long_term``.

        Returns:
            Scalar loss tensor.
        """
        if baseline is None:
            baseline = long_term.detach()
        return torch.nn.functional.mse_loss(long_term, baseline)


class RouterLoadBalancingLoss(nn.Module):
    """Auxiliary load-balancing loss for MoE routers.

    The loss penalises imbalanced routing by minimising the product of
    per-expert token count and per-expert routing probability. The result
    is averaged over the number of tokens and experts.
    """

    def forward(self, router_logits: Tensor) -> Tensor:
        """Compute the load-balancing loss.

        Args:
            router_logits: Tensor of shape ``(num_tokens, num_experts)``.

        Returns:
            Scalar loss tensor.
        """
        num_tokens, num_experts = router_logits.shape
        if num_tokens == 0:
            return torch.zeros(())
        routing_weights = torch.softmax(router_logits, dim=-1)
        avg_routing = routing_weights.mean(dim=0)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_logits.argmax(dim=-1), num_classes=num_experts
            ).float()
        tokens_per_expert = expert_mask.mean(dim=0)
        loss = (tokens_per_expert * avg_routing).sum() / num_experts
        return loss


class UCSACombinedLoss(nn.Module):
    """Combine the four UCSA loss components with configurable weights."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        """Initialise the combined loss.

        Args:
            weights: Optional :class:`LossWeights`.
        """
        super().__init__()
        if weights is None:
            weights = LossWeights()
        self.weights = weights
        self.ar = AutoregressiveLoss()
        self.jepa = JEPALoss()
        self.memory = MemoryStabilityLoss()
        self.router = RouterLoadBalancingLoss()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        jepa_predicted: Tensor | None = None,
        jepa_target: Tensor | None = None,
        long_term: Tensor | None = None,
        router_logits: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the combined loss.

        Args:
            logits: Language head logits.
            targets: Target token ids.
            jepa_predicted: Optional predicted embeddings for JEPA.
            jepa_target: Optional target embeddings for JEPA.
            long_term: Optional long-term memory tensor for stability loss.
            router_logits: Optional MoE router logits for load balancing.

        Returns:
            Tuple ``(total_loss, components)`` where ``components`` is a
            dict mapping each loss name to its float value.
        """
        total = self.ar(logits, targets)
        components = {"ar": float(total.item())}

        if jepa_predicted is not None and jepa_target is not None:
            jepa_loss = self.jepa(jepa_predicted, jepa_target)
            total = total + self.weights.jepa * jepa_loss
            components["jepa"] = float(jepa_loss.item())
        if long_term is not None:
            memory_loss = self.memory(long_term)
            total = total + self.weights.memory * memory_loss
            components["memory"] = float(memory_loss.item())
        if router_logits is not None:
            router_loss = self.router(router_logits)
            total = total + self.weights.router * router_loss
            components["router"] = float(router_loss.item())
        return total, components


__all__ = [
    "AutoregressiveLoss",
    "JEPALoss",
    "LossWeights",
    "MemoryStabilityLoss",
    "RouterLoadBalancingLoss",
    "UCSACombinedLoss",
]