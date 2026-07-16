"""Reasoning loop.

Each forward pass injects the new observation into the working memory and
then runs the state transition operator ``num_iterations`` times. The loop
is the *only* place where reasoning happens; the operator is the *only*
computation engine. Everything else (memory, retrieval, projection) reads
or writes the PCS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from ucsa.models.state import PersistentCognitiveState
from ucsa.models.transition_operator import StateTransitionOperator


@dataclass(frozen=True)
class ReasoningLoopConfig:
    """Configuration for :class:`ReasoningLoop`.

    Attributes:
        num_iterations: Number of operator calls per forward pass.
        capture_intermediates: If ``True`` the loop stores a copy of the
            working-memory bank after each iteration. Required for the
            iteration-level JEPA loss.
        working_token_fraction: Fraction of the observation tokens written
            into the working memory bank before the first iteration. The
            remaining observation tokens stay as context. ``1.0`` writes
            every observation token into working memory.
    """

    num_iterations: int = 4
    capture_intermediates: bool = False
    working_token_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.num_iterations <= 0:
            raise ValueError(
                f"num_iterations must be positive, got {self.num_iterations}."
            )
        if not 0.0 < self.working_token_fraction <= 1.0:
            raise ValueError(
                f"working_token_fraction must be in (0, 1], "
                f"got {self.working_token_fraction}."
            )


class ReasoningLoop(nn.Module):
    """Run ``num_iterations`` operator calls per forward pass.

    The loop is intentionally thin: it does not mutate the PCS directly
    beyond what the operator returns, and it does not duplicate the
    operator's own state-management logic. It exists to formalise the
    "N iterations of F" pattern described in the architecture.
    """

    def __init__(
        self,
        operator: StateTransitionOperator,
        config: ReasoningLoopConfig | None = None,
    ) -> None:
        """Initialise the reasoning loop.

        Args:
            operator: The state transition operator to run.
            config: Optional reasoning-loop configuration.
        """
        super().__init__()
        if config is None:
            config = ReasoningLoopConfig()
        self.config = config
        self.operator = operator
        self.iteration_count: int = 0
        self.last_intermediates: list[Tensor] = []

    def reset(self) -> None:
        """Reset per-request state (KV cache) and clear intermediates."""
        self.operator.reset()
        self.iteration_count = 0
        self.last_intermediates = []

    def inject_observation(
        self,
        cstate: PersistentCognitiveState,
        observation: Tensor,
    ) -> PersistentCognitiveState:
        """Write observation tokens into the working memory bank.

        Args:
            cstate: The current PCS.
            observation: Tensor of shape
                ``(batch, observation_tokens, hidden_size)``.

        Returns:
            The PCS with the observation copied into working memory.
        """
        if self.config.working_token_fraction >= 1.0:
            tokens_to_inject = observation[0]
        else:
            cutoff = max(
                1,
                int(observation.shape[1] * self.config.working_token_fraction),
            )
            tokens_to_inject = observation[0, :cutoff, :]
        working = cstate.get_bank("working")
        if tokens_to_inject.shape[0] >= working.shape[0]:
            replacement = tokens_to_inject[: working.shape[0]]
        else:
            pad = torch.zeros(
                working.shape[0] - tokens_to_inject.shape[0],
                working.shape[1],
                device=working.device,
                dtype=working.dtype,
            )
            replacement = torch.cat([tokens_to_inject, pad], dim=0)
        cstate.set_bank("working", replacement)
        return cstate

    def forward(
        self,
        cstate: PersistentCognitiveState,
        observation: Tensor,
    ) -> PersistentCognitiveState:
        """Run the reasoning loop.

        Args:
            cstate: The current PCS.
            observation: Tensor of shape
                ``(batch, observation_tokens, hidden_size)``.

        Returns:
            The PCS after ``num_iterations`` operator calls.
        """
        if observation.dim() != 3:
            raise ValueError(
                f"observation must be 3D (batch, seq, hidden), got "
                f"{tuple(observation.shape)}."
            )
        self.operator.reset()
        self.iteration_count = 0
        self.last_intermediates = []
        cstate = self.inject_observation(cstate, observation)
        for _ in range(self.config.num_iterations):
            cstate = self.operator(cstate, observation)
            self.iteration_count += 1
            if self.config.capture_intermediates:
                self.last_intermediates.append(
                    cstate.get_bank("working").detach().clone()
                )
        return cstate

    def get_intermediates(self) -> Sequence[Tensor]:
        """Return the per-iteration working-memory snapshots."""
        return list(self.last_intermediates)


__all__ = ["ReasoningLoop", "ReasoningLoopConfig"]