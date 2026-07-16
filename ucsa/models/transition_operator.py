"""State Transition Operator interface.

The state transition operator :math:`F` is the *only* computation engine in
UCSA. It maps the current cognitive state and a new observation to a new
cognitive state:

.. math::

    C_{t+1} = F(C_t, O_t)

The reference implementation is the
:class:`ucsa.models.transformer_operator.TransformerOperator`. Future
implementations (Mamba, RWKV, Hyena, SSMs) plug in by satisfying this
interface. The reasoning loop, memory pipeline, and projection heads all
work against the interface and require no changes when the operator is
swapped.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from ucsa.models.state import PersistentCognitiveState


class StateTransitionOperator(nn.Module, abc.ABC):
    """Abstract base class for UCSA state transition operators.

    Subclasses implement :meth:`forward`, which receives the current PCS and
    the observation tokens and returns the updated PCS (and optionally an
    auxiliary payload, e.g. JEPA predictions).

    Operators also implement :meth:`initialize` for parameter construction
    (called once by the reasoning loop) and :meth:`reset` for per-request
    state (e.g. KV cache) clearing.
    """

    def __init__(self) -> None:
        """Initialise the operator base."""
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        cstate: "PersistentCognitiveState",
        observation: Tensor,
    ) -> "PersistentCognitiveState":
        """Compute :math:`C_{t+1} = F(C_t, O_t)`.

        Args:
            cstate: The current PCS.
            observation: Tensor of shape
                ``(batch, observation_tokens, hidden_size)``.

        Returns:
            A new :class:`PersistentCognitiveState` representing the
            updated cognitive state. Implementations may mutate ``cstate``
            in place when the surrounding pipeline allows, but the default
            contract is to return a fresh state.
        """

    @abc.abstractmethod
    def initialize(self) -> None:
        """Build parameters and any lazy buffers.

        Called once before the first forward pass. Subclasses that build
        parameters eagerly in ``__init__`` may treat this as a no-op.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear per-request state such as the KV cache.

        Called between unrelated requests. Implementations that do not
        maintain per-request state may treat this as a no-op.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the operator's registration name."""


__all__ = ["StateTransitionOperator"]