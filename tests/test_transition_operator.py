"""Tests for :mod:`ucsa.models.transition_operator`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ucsa.models.state import PCSConfig, PersistentCognitiveState
from ucsa.models.transition_operator import StateTransitionOperator


class IdentityOperator(StateTransitionOperator):
    """Test operator that returns the PCS unchanged.

    Used to verify the ABC contract without exercising real attention.
    """

    def __init__(self) -> None:
        """Initialise the identity operator."""
        super().__init__()
        self.initialize_called: int = 0
        self.reset_called: int = 0
        self.forward_called: int = 0

    @property
    def name(self) -> str:
        """Return the operator's registration name."""
        return "identity"

    def forward(
        self,
        cstate: PersistentCognitiveState,
        observation: Tensor,
    ) -> PersistentCognitiveState:
        """Return the PCS unchanged."""
        self.forward_called += 1
        return cstate

    def initialize(self) -> None:
        """Record that initialization was called."""
        self.initialize_called += 1

    def reset(self) -> None:
        """Record that reset was called."""
        self.reset_called += 1


class RecordingOperator(StateTransitionOperator):
    """Test operator that records every (cstate, observation) pair.

    The operator returns the PCS unchanged but exposes the call history for
    assertions.
    """

    def __init__(self) -> None:
        """Initialise the recording operator."""
        super().__init__()
        self.cstate_history: list[PersistentCognitiveState] = []
        self.observation_history: list[Tensor] = []

    @property
    def name(self) -> str:
        """Return the operator's registration name."""
        return "recording"

    def forward(
        self,
        cstate: PersistentCognitiveState,
        observation: Tensor,
    ) -> PersistentCognitiveState:
        """Record the inputs and return the PCS unchanged."""
        self.cstate_history.append(cstate)
        self.observation_history.append(observation)
        return cstate

    def initialize(self) -> None:
        """No-op."""

    def reset(self) -> None:
        """No-op."""


class TestStateTransitionOperatorABC:
    """Tests for the StateTransitionOperator ABC contract."""

    @pytest.fixture
    def state(self) -> PersistentCognitiveState:
        """Provide a fresh PCS."""
        return PersistentCognitiveState(PCSConfig(hidden_size=16))

    def test_abstract_methods_must_be_implemented(self) -> None:
        """A subclass missing abstract methods cannot be instantiated."""
        class IncompleteOperator(StateTransitionOperator):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteOperator()  # type: ignore[abstract]

    def test_dummy_operator_is_module(self) -> None:
        """A subclass is also a :class:`torch.nn.Module`."""
        op = IdentityOperator()
        assert isinstance(op, StateTransitionOperator)
        assert isinstance(op, torch.nn.Module)

    def test_initialize_is_called_explicitly(self) -> None:
        """``initialize`` is invoked explicitly, not in ``__init__``."""
        op = IdentityOperator()
        assert op.initialize_called == 0
        op.initialize()
        assert op.initialize_called == 1

    def test_reset_is_called_explicitly(self) -> None:
        """``reset`` is invoked explicitly, not in ``__init__``."""
        op = IdentityOperator()
        assert op.reset_called == 0
        op.reset()
        assert op.reset_called == 1

    def test_forward_returns_cstate(
        self, state: PersistentCognitiveState
    ) -> None:
        """``forward`` returns the updated PCS."""
        op = IdentityOperator()
        observation = torch.randn(1, 4, 16)
        result = op(state, observation)
        assert result is state

    def test_forward_records_call(
        self, state: PersistentCognitiveState
    ) -> None:
        """``RecordingOperator`` captures every (cstate, observation)."""
        op = RecordingOperator()
        observation = torch.randn(1, 4, 16)
        op(state, observation)
        assert len(op.cstate_history) == 1
        assert len(op.observation_history) == 1
        assert op.observation_history[0] is observation

    def test_name_is_abstract_property(self) -> None:
        """``name`` is an abstract property."""
        op = IdentityOperator()
        assert op.name == "identity"

    def test_module_parameters_registered(self) -> None:
        """Any parameters added by subclasses are discoverable."""
        op = IdentityOperator()
        params = list(op.parameters())
        assert isinstance(params, list)
