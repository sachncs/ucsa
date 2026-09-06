"""Reasoning loop.

Each forward pass injects the new observation into the working memory and
then runs the state transition operator ``num_iterations`` times. The loop
is the *only* place where reasoning happens; the operator is the *only*
computation engine. Everything else (memory, retrieval, projection) reads
or writes the PCS.

Endogenous origination
----------------------

Iteration ``k + 1``'s input need not be the same exogenous observation.
When an origination generator ``G`` is attached, the loop feeds

.. math::

    O_{k+1} = (1 - \\alpha_k) \\, G(\\text{intent}, \\text{working}_k)
              + \\alpha_k \\, O_0

so the input to the next state is partly *generated* from the ``intent``
bank -- the analogue of the signal that precedes a reach. ``alpha_k``
decays geometrically from ``observation_mix`` and controls how much of the
real observation survives, which is the brake on the loop running away on
its own output. With the defaults (``observation_mix=1.0``,
``observation_mix_decay=1.0``) every ``alpha_k`` is ``1.0`` and the loop
reduces exactly to feeding ``O_0`` every iteration, so the generator is a
strict generalisation of the previous behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
        observation_mix: ``alpha_0``, the weight on the real observation
            when generating the second iteration's input. ``1.0`` (the
            default) keeps the exogenous observation and never calls the
            origination generator.
        observation_mix_decay: Geometric decay applied to ``alpha`` per
            iteration, so ``alpha_k = observation_mix *
            observation_mix_decay ** k``. ``1.0`` (the default) holds
            ``alpha`` constant.
    """

    num_iterations: int = 4
    capture_intermediates: bool = False
    working_token_fraction: float = 1.0
    observation_mix: float = 1.0
    observation_mix_decay: float = 1.0

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
        if not 0.0 <= self.observation_mix <= 1.0:
            raise ValueError(
                f"observation_mix must be in [0, 1], "
                f"got {self.observation_mix}."
            )
        if not 0.0 <= self.observation_mix_decay <= 1.0:
            raise ValueError(
                f"observation_mix_decay must be in [0, 1], "
                f"got {self.observation_mix_decay}."
            )

    def observation_weight(self, iteration: int) -> float:
        """Return ``alpha_k`` for the given zero-based iteration.

        Args:
            iteration: Zero-based index of the iteration whose *output*
                input stream is being generated.

        Returns:
            The weight on the real observation, in ``[0, 1]``.

        Raises:
            ValueError: If ``iteration`` is negative.
        """
        if iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {iteration}.")
        return self.observation_mix * self.observation_mix_decay**iteration


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
        origination: nn.Module | None = None,
    ) -> None:
        """Initialise the reasoning loop.

        Args:
            operator: The state transition operator to run.
            config: Optional reasoning-loop configuration.
            origination: Optional origination generator ``G``, called as
                ``G(intent, working, observation)``. When ``None`` the loop
                feeds the exogenous observation to every iteration.
        """
        super().__init__()
        if config is None:
            config = ReasoningLoopConfig()
        self.config = config
        self.operator = operator
        self.origination = origination
        self.iteration_count: int = 0
        self.last_intermediates: list[Tensor] = []
        self.last_bank_tensors: dict[str, Tensor] | None = None
        self.last_observation_weights: list[float] = []
        self.last_generated_inputs: list[Tensor] = []

    def reset(self) -> None:
        """Reset per-request state (KV cache) and clear intermediates."""
        self.operator.reset()
        self.iteration_count = 0
        self.last_intermediates = []
        self.last_bank_tensors = None
        self.last_observation_weights = []
        self.last_generated_inputs = []

    def differentiable_bank(self, name: str) -> Tensor | None:
        """Return the differentiable post-loop tensor for one bank.

        The PCS write-back copies under ``torch.no_grad``, so reading a
        bank off the PCS after the loop yields a value with no autograd
        history. Operators that stash their pre-write-back tensors expose
        them here so downstream consumers (projection heads, JEPA losses)
        can keep the graph intact.

        Args:
            name: Bank identifier.

        Returns:
            The differentiable bank tensor of shape
            ``(num_tokens, hidden_size)``, or ``None`` when the operator
            does not expose one.
        """
        if self.last_bank_tensors is None:
            return None
        return self.last_bank_tensors.get(name)

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
        self.last_bank_tensors = None
        self.last_observation_weights = []
        self.last_generated_inputs = []
        cstate = self.inject_observation(cstate, observation)
        current = observation
        for iteration in range(self.config.num_iterations):
            cstate = self.operator(cstate, current)
            self.iteration_count += 1
            self.last_bank_tensors = getattr(
                self.operator, "last_bank_tensors", None
            )
            if self.config.capture_intermediates:
                self.last_intermediates.append(self.capture_working(cstate))
            if iteration + 1 < self.config.num_iterations:
                current = self.next_observation(
                    cstate, observation, current, iteration
                )
        return cstate

    def next_observation(
        self,
        cstate: PersistentCognitiveState,
        observation: Tensor,
        current: Tensor,
        iteration: int,
    ) -> Tensor:
        """Build the input stream for the next iteration.

        Implements ``O_{k+1} = (1 - alpha_k) * G(intent, working_k) +
        alpha_k * O_0``. The real observation is mixed against ``O_0``, not
        against ``current``, so the exogenous signal cannot decay away
        faster than ``alpha_k`` says it should.

        Args:
            cstate: The PCS after the current iteration.
            observation: The original exogenous observation ``O_0``.
            current: The input stream fed to the current iteration.
            iteration: Zero-based index of the current iteration.

        Returns:
            Tensor with ``observation``'s shape.
        """
        weight = self.config.observation_weight(iteration)
        self.last_observation_weights.append(weight)
        if self.origination is None or weight >= 1.0:
            return observation
        intent = self.read_bank(cstate, "intent")
        working = self.read_bank(cstate, "working")
        generated: Tensor = self.origination(intent, working, current)
        self.last_generated_inputs.append(generated)
        if weight <= 0.0:
            return generated
        return weight * observation + (1.0 - weight) * generated

    def read_bank(self, cstate: PersistentCognitiveState, name: str) -> Tensor:
        """Read a bank for the generator, preferring the differentiable one.

        A clone is returned when falling back to the PCS parameter: the
        write-back mutates that parameter in place, which would invalidate
        tensors autograd saved for the generator's weight gradients.

        Args:
            cstate: The current PCS.
            name: Bank identifier.

        Returns:
            Tensor of shape ``(num_tokens, hidden_size)``.
        """
        bank = self.differentiable_bank(name)
        if bank is not None:
            return bank
        return cstate.get_bank(name).clone()

    def capture_working(self, cstate: PersistentCognitiveState) -> Tensor:
        """Snapshot the working bank for the JEPA chain.

        Prefers the operator's differentiable pre-write-back tensor so the
        JEPA prediction chain carries gradients. Falls back to a detached
        clone of the PCS bank for operators that do not expose one.

        Args:
            cstate: The PCS after the current iteration.

        Returns:
            Tensor of shape ``(working_tokens, hidden_size)``.
        """
        working = self.differentiable_bank("working")
        if working is not None:
            return working
        return cstate.get_bank("working").detach().clone()

    def get_intermediates(self) -> Sequence[Tensor]:
        """Return the per-iteration working-memory snapshots."""
        return list(self.last_intermediates)


__all__ = ["ReasoningLoop", "ReasoningLoopConfig"]
