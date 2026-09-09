"""Exponential-Moving-Average target encoder.

Used to stabilise joint-embedding predictive training (I-JEPA, LeWM,
DINO). The target encoder is a deep-copy of the predictor whose
weights are blended toward the predictor's after each step::

    target_param = momentum * target_param + (1 - momentum) * predictor_param

During the JEPA loss we run the same input through both modules and
ask the predictor to match the (no-grad) target. The EMA update keeps
the target stable across steps without needing stop-grad on the
predictor or multi-term auxiliary losses.

References:
  - Assran et al., "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture", I-JEPA, 2023
  - Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding
    Predictive Architecture from Pixels", arXiv 2603.19312, Mar 2026
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn


class EMATargetEncoder(nn.Module):
    """Hard EMA target encoder. Buffers contain the EMA-updated copy.

    Args:
        teacher: The model to track. Weights are deep-copied at
            construction; the target is frozen for inference.
        momentum: EMA decay. ``0.996`` is a common default; for very
            small batches / short schedules, ``0.99`` is more responsive.
    """

    def __init__(self, teacher: nn.Module, momentum: float = 0.996) -> None:
        super().__init__()
        self.momentum = momentum
        self.target = deepcopy(teacher)
        for p in self.target.parameters():
            p.requires_grad = False
        # Buffers (e.g., register_buffer entries) need to be deep-copied
        # too — deepcopy already handles them, but make sure the target
        # sits on the same device as the teacher.
        self.target.eval()

    @torch.no_grad()
    def update(self, teacher: nn.Module) -> None:
        """Blend target weights toward teacher's weights (in place)."""
        m = self.momentum
        for tp, sp in zip(
            self.target.parameters(),
            teacher.parameters(),
            strict=True,
        ):
            tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)
        # Buffers (running stats) — LeWM does not update them; keep them
        # at their initial-deep-copied values to avoid drift. EMA is
        # applied to parameters only, by design.

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """Run the target encoder; no grads."""
        outputs: dict[str, Tensor] = self.target(*args, **kwargs)
        return outputs


__all__ = ["EMATargetEncoder"]
