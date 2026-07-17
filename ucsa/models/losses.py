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
        reconstruction: Coefficient for the input-reconstruction loss
            (LeWM-style capacity bottleneck).
    """

    jepa: float = 0.1
    memory: float = 0.01
    router: float = 0.01
    reconstruction: float = 0.1

    def __post_init__(self) -> None:
        if self.jepa < 0.0:
            raise ValueError(f"jepa must be non-negative, got {self.jepa}.")
        if self.memory < 0.0:
            raise ValueError(f"memory must be non-negative, got {self.memory}.")
        if self.router < 0.0:
            raise ValueError(f"router must be non-negative, got {self.router}.")
        if self.reconstruction < 0.0:
            raise ValueError(
                f"reconstruction must be non-negative, got {self.reconstruction}."
            )


class InputReconstructionLoss(nn.Module):
    """Predict input-token embeddings from working memory (LeWM-style).

    The JEPA latent must retain enough information that a small
    projection head can recover the input-token embeddings. Used
    alongside the JEPA prediction loss to enforce a "capacity
    bottleneck" — without it the latent can collapse to a constant
    and still satisfy the JEPA loss.
    """

    def forward(
        self, reconstructed: Tensor, target_embeddings: Tensor
    ) -> Tensor:
        return torch.nn.functional.smooth_l1_loss(
            reconstructed, target_embeddings
        )


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
    """Latent-prediction loss.

    Two modes:

    - ``"ijepa"`` (default; backwards-compatible): convex combination of
      cosine distance and SmoothL1, with weight ``alpha``.
      ``L = α (1 - cos(pred, target)).mean() + (1 - α) smooth_l1(pred, target)``
    - ``"lewm"`` (LeWorldModel, arXiv 2603.19312, Jun 2026, Maes et al.):
      a single SmoothL1 prediction term plus a per-batch Gaussian
      regulariser on the latent embeddings. The Gaussian reg enforces
      the predicted latents to match N(0, 1) — preventing collapse
      without needing stop-grad, EMA target networks, or multi-term
      hyperparameter tweaking. The paper drops the tuning budget from
      six loss terms to one weight.

    When ``mode="lewm"``, ``forward`` returns a tuple
    ``(loss, components)`` so the components dict can list the
    prediction and the regulariser separately.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        mode: str = "ijepa",
        gaussian_reg_weight: float = 0.1,
    ) -> None:
        """Initialise the loss.

        Args:
            alpha: Weight of the cosine term in ``"ijepa"`` mode.
            mode: ``"ijepa"`` or ``"lewm"``.
            gaussian_reg_weight: Weight for the latent Gaussian KL term
                in ``"lewm"`` mode.
        """
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        if mode not in {"ijepa", "lewm"}:
            raise ValueError(f"mode must be 'ijepa' or 'lewm', got {mode}.")
        self.alpha = alpha
        self.mode = mode
        self.gaussian_reg_weight = gaussian_reg_weight

    def _gaussian_kl(
        self, predicted: Tensor, latent_for_reg: Tensor | None
    ) -> Tensor:
        """KL( predicted_latent_distribution || N(0, I) ).

        Matches LeWM §3.4: enforce the latent to be unit-variance,
        zero-mean by penalising ``0.5 * (var + mu^2 - 1 - log var)``.
        """
        # The paper regularises the predicted embedding directly;
        # `latent_for_reg` lets callers pass a pre-conditioning latent
        # so the regulariser measures the underlying distribution
        # rather than any additive offset.
        target = latent_for_reg if latent_for_reg is not None else predicted
        mu = target.mean(dim=(0, 1))
        var = target.var(dim=(0, 1), unbiased=False) + 1e-6
        return 0.5 * (var + mu.pow(2) - 1.0 - var.log()).mean()

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        latent_for_reg: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        """Compute the JEPA loss.

        Args:
            predicted: Predicted embeddings of shape
                ``(batch, seq, hidden)``.
            target: Target embeddings of the same shape.
            latent_for_reg: Optional latent to regularise in LeWM mode
                (defaults to ``predicted``).

        Returns:
            Scalar tensor (``"ijepa"``) or ``(loss, components)`` tuple
            (``"lewm"``).
        """
        if predicted.shape != target.shape:
            raise ValueError(
                f"predicted and target must share shape; got "
                f"{tuple(predicted.shape)} and {tuple(target.shape)}."
            )
        if self.mode == "ijepa":
            cosine = torch.nn.functional.cosine_similarity(
                predicted, target, dim=-1
            )
            cosine_term = (1.0 - cosine).mean()
            l1_term = torch.nn.functional.smooth_l1_loss(predicted, target)
            return self.alpha * cosine_term + (1.0 - self.alpha) * l1_term
        # LeWM mode: single SmoothL1 prediction + Gaussian regulariser.
        pred_loss = torch.nn.functional.smooth_l1_loss(predicted, target)
        reg = self._gaussian_kl(predicted, latent_for_reg)
        loss = pred_loss + self.gaussian_reg_weight * reg
        return loss, {
            "jepa_pred": float(pred_loss.item()),
            "jepa_gaussian_reg": float(reg.item()),
        }


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

    Implements the Switch Transformer load-balancing loss:

    .. math::

        L = N \\cdot \\sum_i f_i \\cdot P_i

    where :math:`N` is the number of experts, :math:`f_i` is the fraction
    of tokens routed to expert :math:`i`, and :math:`P_i` is the average
    routing probability for expert :math:`i`. Perfect balance yields
    ``L = 1``.
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
        return num_experts * (tokens_per_expert * avg_routing).sum()


class UCSACombinedLoss(nn.Module):
    """Combine the four UCSA loss components with configurable weights."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        jepa_mode: str = "ijepa",
        jepa_alpha: float = 0.5,
        gaussian_reg_weight: float = 0.1,
    ) -> None:
        """Initialise the combined loss.

        Args:
            weights: Optional :class:`LossWeights`.
            jepa_mode: ``"ijepa"`` (default) or ``"lewm"`` (LeWorldModel).
            jepa_alpha: Cosine blend weight for ``"ijepa"`` mode.
            gaussian_reg_weight: Gaussian regulariser weight for
                ``"lewm"`` mode.
        """
        super().__init__()
        if weights is None:
            weights = LossWeights()
        self.weights = weights
        self.ar = AutoregressiveLoss()
        self.jepa = JEPALoss(
            alpha=jepa_alpha,
            mode=jepa_mode,
            gaussian_reg_weight=gaussian_reg_weight,
        )
        self.memory = MemoryStabilityLoss()
        self.router = RouterLoadBalancingLoss()
        self.reconstruction = InputReconstructionLoss()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        jepa_predicted: Tensor | None = None,
        jepa_target: Tensor | None = None,
        long_term: Tensor | None = None,
        router_logits: Tensor | None = None,
        reconstructed: Tensor | None = None,
        target_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the combined loss.

        Args:
            logits: Language head logits.
            targets: Target token ids.
            jepa_predicted: Optional predicted embeddings for JEPA.
            jepa_target: Optional target embeddings for JEPA.
            long_term: Optional long-term memory tensor for stability loss.
            router_logits: Optional MoE router logits for load balancing.
            reconstructed: Optional input-reconstruction projection.
            target_embeddings: Original input-token embeddings (target for
                the reconstruction loss).
        """
        total = self.ar(logits, targets)
        components = {"ar": float(total.item())}

        if jepa_predicted is not None and jepa_target is not None:
            jepa_out = self.jepa(jepa_predicted, jepa_target)
            if self.jepa.mode == "lewm":
                jepa_loss, jepa_components = jepa_out
                components.update(
                    {f"jepa_{k}": v for k, v in jepa_components.items()}
                )
            else:
                jepa_loss = jepa_out
                components["jepa"] = float(jepa_loss.item())
            total = total + self.weights.jepa * jepa_loss
        if long_term is not None:
            memory_loss = self.memory(long_term)
            total = total + self.weights.memory * memory_loss
            components["memory"] = float(memory_loss.item())
        if router_logits is not None:
            router_loss = self.router(router_logits)
            total = total + self.weights.router * router_loss
            components["router"] = float(router_loss.item())
        if reconstructed is not None and target_embeddings is not None:
            rec_loss = self.reconstruction(reconstructed, target_embeddings)
            total = total + self.weights.reconstruction * rec_loss
            components["reconstruction"] = float(rec_loss.item())
        return total, components


__all__ = [
    "AutoregressiveLoss",
    "InputReconstructionLoss",
    "JEPALoss",
    "LossWeights",
    "MemoryStabilityLoss",
    "RouterLoadBalancingLoss",
    "UCSACombinedLoss",
]
