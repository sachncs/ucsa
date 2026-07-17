"""Evaluation loop.

Runs the model in ``eval`` mode with no gradient, computing average loss
and perplexity over a dataloader. Supports periodic triggering from the
:class:`ucsa.training.trainer.Trainer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ucsa.training.metrics import (
    MetricsRegistry,
    perplexity_from_loss,
)


@dataclass
class EvaluationState:
    """internal: mutable state for the evaluation loop."""

    steps: int = 0
    running_loss: float = 0.0
    running_perplexity: float = 0.0
    last_loss: float = 0.0
    last_perplexity: float = 0.0


class EvaluationLoop:
    """Run periodic evaluation over a dataloader."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[Tensor, Tensor], tuple[Tensor, dict[str, float]]],
        device: torch.device | None = None,
    ) -> None:
        """Initialise the evaluation loop.

        Args:
            model: The model to evaluate.
            loss_fn: A loss function taking ``(inputs, targets)`` and
                returning ``(loss, components)``. May wrap the model's
                forward call internally.
            device: Optional device override.
        """
        self.model = model
        self.loss_fn = loss_fn
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.device = device
        self.state = EvaluationState()

    def move_batch(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Move a batch to the configured device."""
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)

    def evaluate(
        self, dataloader: DataLoader, max_batches: int | None = None
    ) -> dict[str, float]:
        """Run evaluation.

        Args:
            dataloader: Iterable yielding ``(inputs, targets)`` batches.
            max_batches: Optional cap on the number of batches.

        Returns:
            Dict with ``"loss"`` and ``"perplexity"`` keys.
        """
        self.model.eval()
        total_loss = 0.0
        total_perplexity = 0.0
        count = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                inputs, targets = self.move_batch(batch)
                loss, _ = self.loss_fn(inputs, targets)
                loss_value = float(loss.item())
                total_loss += loss_value
                total_perplexity += perplexity_from_loss(loss_value)
                count += 1
        self.state.steps += count
        if count > 0:
            self.state.running_loss += total_loss
            self.state.running_perplexity += total_perplexity
            self.state.last_loss = total_loss / count
            self.state.last_perplexity = total_perplexity / count
        self.model.train()
        return {
            "loss": self.state.last_loss,
            "perplexity": self.state.last_perplexity,
        }

    def should_evaluate(self, step: int, every_n_steps: int) -> bool:
        """Return ``True`` if evaluation should run at ``step``."""
        if every_n_steps <= 0:
            return False
        return step > 0 and step % every_n_steps == 0


__all__ = ["EvaluationLoop", "EvaluationState"]


def make_default_evaluation_loop(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], tuple[Tensor, dict[str, float]]],
) -> EvaluationLoop:
    """internal: build an :class:`EvaluationLoop` with default settings."""
    return EvaluationLoop(model=model, loss_fn=loss_fn)