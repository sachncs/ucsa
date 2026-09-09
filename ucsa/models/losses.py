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
        origination: Coefficient for the origination gate's load-balancing
            loss. Without it the gate collapses onto a fixed pair of intent
            slots and stops conditioning on the input, which the
            ``intent_gate_mutual_info`` diagnostic reports as zero.
    """

    jepa: float = 0.1
    memory: float = 0.01
    router: float = 0.01
    reconstruction: float = 0.1
    origination: float = 0.01

    def __post_init__(self) -> None:
        if self.jepa < 0.0:
            raise ValueError(f"jepa must be non-negative, got {self.jepa}.")
        if self.origination < 0.0:
            raise ValueError(
                f"origination must be non-negative, got {self.origination}."
            )
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
    - ``"lewm"`` (LeWorldModel, arXiv 2603.19312, Mar 2026, Maes et al.):
      a single SmoothL1 prediction term plus a per-batch Gaussian
      regulariser on the latent embeddings. The Gaussian reg enforces
      the predicted latents to match N(0, 1) — preventing collapse
      without needing stop-grad, EMA target networks, or multi-term
      hyperparameter tweaking. The paper drops the tuning budget from
      six loss terms to one weight.

    Optional multi-step prediction: when callers pass
    ``multi_step_pairs`` (a list of ``(predicted_k, target_k)``
    tensors), the loss is averaged across steps. This implements
    the autoregressive-latent prediction LeWM does for its "48x
    faster planning" claim — at step k we predict target k+1, and
    average across k.

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
        target = latent_for_reg if latent_for_reg is not None else predicted
        mu = target.mean(dim=(0, 1))
        var = target.var(dim=(0, 1), unbiased=False) + 1e-6
        return 0.5 * (var + mu.pow(2) - 1.0 - var.log()).mean()

    def _single_pair_loss(
        self,
        predicted: Tensor,
        target: Tensor,
        for_reg: Tensor | None = None,
    ) -> Tensor:
        """Compute the per-step prediction loss (no aggregation)."""
        if self.mode == "ijepa":
            cosine = torch.nn.functional.cosine_similarity(
                predicted, target, dim=-1
            )
            cosine_term = (1.0 - cosine).mean()
            l1_term = torch.nn.functional.smooth_l1_loss(predicted, target)
            return self.alpha * cosine_term + (1.0 - self.alpha) * l1_term
        # LeWM mode
        pred_loss = torch.nn.functional.smooth_l1_loss(predicted, target)
        reg = self._gaussian_kl(predicted, for_reg)
        return pred_loss + self.gaussian_reg_weight * reg

    def forward(
        self,
        predicted: Tensor | None = None,
        target: Tensor | None = None,
        latent_for_reg: Tensor | None = None,
        multi_step_pairs: list[tuple[Tensor, Tensor]] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        """Compute the JEPA loss.

        Args:
            predicted: Single-step predicted embedding ``(B, S, H)``.
            target: Single-step target embedding ``(B, S, H)``.
            latent_for_reg: Latent to regularise in LeWM mode.
            multi_step_pairs: Optional list of
                ``(predicted_k, target_k)`` tensors. The loss is the
                per-step mean; the Gaussian regulariser is applied
                to each predicted independently.

        Returns:
            Scalar tensor (``"ijepa"``) or ``(loss, components)`` tuple
            (``"lewm"``).
        """
        if multi_step_pairs is not None and len(multi_step_pairs) > 0:
            per_step = [
                self._single_pair_loss(p, t, for_reg=p)
                for p, t in multi_step_pairs
            ]
            stack = torch.stack(per_step)
            loss = stack.mean()
            if self.mode == "lewm":
                return loss, {
                    "jepa_pred": float(stack.mean().item()),
                    "jepa_gaussian_reg": float(
                        self._gaussian_kl(
                            torch.cat([p for p, _ in multi_step_pairs], dim=0),
                            None,
                        ).item()
                    ),
                    "jepa_steps": float(len(multi_step_pairs)),
                }
            return loss
        if predicted is None or target is None:
            reference = predicted if predicted is not None else target
            device = reference.device if reference is not None else None
            return torch.zeros((), device=device)
        if predicted.shape != target.shape:
            raise ValueError(
                f"predicted and target must share shape; got "
                f"{tuple(predicted.shape)} and {tuple(target.shape)}."
            )
        loss = self._single_pair_loss(predicted, target, for_reg=latent_for_reg)
        if self.mode == "lewm":
            return loss, {
                "jepa_pred": (
                    float(loss.item())
                    if self.gaussian_reg_weight == 0
                    else float(
                        self._single_pair_loss(
                            predicted, target, for_reg=None
                        ).item()
                    )
                ),
                "jepa_gaussian_reg": float(
                    self._gaussian_kl(predicted, latent_for_reg).item()
                ),
            }
        return loss


class MemoryStabilityLoss(nn.Module):
    """Penalise drift of the long-term memory bank.

    Encourages the long-term bank to remain close to a learned baseline.
    """

    def forward(
        self, long_term: Tensor, baseline: Tensor | None = None
    ) -> Tensor:
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
        jepa_multi_step: list[tuple[Tensor, Tensor]] | None = None,
        origination_aux_loss: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the combined loss.

        Args:
            logits: Language head logits.
            targets: Target token ids.
            jepa_predicted: Optional single-step predicted embedding.
            jepa_target: Optional single-step target embedding.
            long_term: Optional long-term memory tensor for stability loss.
            router_logits: Optional MoE router logits for load balancing.
            reconstructed: Optional input-reconstruction projection.
            target_embeddings: Original input-token embeddings (target for
                the reconstruction loss).
            jepa_multi_step: Optional list of multi-step JEPA pairs.
                When provided, this takes precedence over the
                single-step ``jepa_predicted``/``jepa_target`` pair.
            origination_aux_loss: Optional pre-weighted load-balancing loss
                from the origination gate.
        """
        total = self.ar(logits, targets)
        components = {"ar": float(total.item())}

        # Multi-step JEPA takes precedence over single-step. When the
        # EMA target encoder is active, the trainer swaps the targets in
        # the multi-step list for the EMA-tracked latents.
        active_jepa_pairs: list[tuple[Tensor, Tensor]] | None = None
        if jepa_multi_step is not None and len(jepa_multi_step) > 0:
            active_jepa_pairs = list(jepa_multi_step)
        elif jepa_predicted is not None and jepa_target is not None:
            active_jepa_pairs = [(jepa_predicted, jepa_target)]

        if active_jepa_pairs is not None:
            jepa_out = self.jepa(multi_step_pairs=active_jepa_pairs)
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
        if origination_aux_loss is not None:
            total = total + self.weights.origination * origination_aux_loss
            components["origination"] = float(origination_aux_loss.item())
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
