"""Tests for :mod:`ucsa.models.reasoning_loop`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ucsa.models.projection_heads import OriginationHead
from ucsa.models.reasoning_loop import ReasoningLoop, ReasoningLoopConfig
from ucsa.models.state import PCSConfig, PersistentCognitiveState
from ucsa.models.transformer_operator import (
    TransformerOperator,
    TransformerOperatorConfig,
)
from ucsa.models.transition_operator import StateTransitionOperator


def tiny_operator() -> TransformerOperator:
    """Return a tiny transformer operator for tests."""
    return TransformerOperator(
        TransformerOperatorConfig(
            hidden_size=32,
            num_layers=2,
            num_q_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            sliding_window=4096,
            max_position=8192,
        )
    )


def tiny_pcs() -> PersistentCognitiveState:
    """Return a fresh PCS sized for tests."""
    return PersistentCognitiveState(PCSConfig(hidden_size=32))


class TestReasoningLoopConfig:
    """Tests for :class:`ReasoningLoopConfig`."""

    def test_default_config_valid(self) -> None:
        """Defaults construct without error."""
        config = ReasoningLoopConfig()
        assert config.num_iterations == 4
        assert config.capture_intermediates is False

    def test_zero_iterations_rejected(self) -> None:
        """``num_iterations`` of zero or less raises."""
        with pytest.raises(ValueError):
            ReasoningLoopConfig(num_iterations=0)

    def test_invalid_fraction_rejected(self) -> None:
        """``working_token_fraction`` outside (0, 1] raises."""
        with pytest.raises(ValueError):
            ReasoningLoopConfig(working_token_fraction=0.0)
        with pytest.raises(ValueError):
            ReasoningLoopConfig(working_token_fraction=1.5)


class TestReasoningLoop:
    """Tests for :class:`ReasoningLoop`."""

    def test_default_iteration_count_is_four(self) -> None:
        """Default loop runs 4 iterations per forward pass."""
        loop = ReasoningLoop(tiny_operator())
        assert loop.config.num_iterations == 4

    def test_forward_runs_configured_iterations(self) -> None:
        """``iteration_count`` matches ``num_iterations`` after forward."""
        loop = ReasoningLoop(tiny_operator(), ReasoningLoopConfig(num_iterations=3))
        pcs = tiny_pcs()
        obs = torch.randn(1, 4, 32)
        loop(pcs, obs)
        assert loop.iteration_count == 3

    def test_forward_observation_shape_validation(self) -> None:
        """Non-3D observation raises ``ValueError``."""
        loop = ReasoningLoop(tiny_operator())
        with pytest.raises(ValueError):
            loop(tiny_pcs(), torch.randn(4, 32))

    def test_inject_observation_writes_to_working(self) -> None:
        """``inject_observation`` overwrites the working bank."""
        loop = ReasoningLoop(tiny_operator())
        pcs = tiny_pcs()
        before = pcs.get_bank("working").clone()
        obs = torch.zeros(1, 6, 32)
        loop.inject_observation(pcs, obs)
        after = pcs.get_bank("working")
        # First six tokens of working now equal the observation tokens.
        assert torch.allclose(after[:6], obs[0])
        # Tail is zero-padded.
        assert torch.all(after[6:] == 0)
        # Working memory changed (at least at the head).
        assert not torch.allclose(before[:6], after[:6])

    def test_inject_with_partial_fraction(self) -> None:
        """Partial fraction truncates the observation."""
        loop = ReasoningLoop(
            tiny_operator(),
            ReasoningLoopConfig(num_iterations=1, working_token_fraction=0.5),
        )
        pcs = tiny_pcs()
        obs = torch.ones(1, 10, 32)
        loop.inject_observation(pcs, obs)
        # working bank has 64 slots. With fraction=0.5, 5 obs tokens are
        # written, the rest zero-padded.
        after = pcs.get_bank("working")
        assert torch.all(after[:5] == 1.0)
        assert torch.all(after[5:] == 0.0)

    def test_inject_with_more_tokens_than_working(self) -> None:
        """An observation longer than the working bank is truncated."""
        loop = ReasoningLoop(tiny_operator())
        pcs = tiny_pcs()
        obs = torch.ones(1, 100, 32)
        loop.inject_observation(pcs, obs)
        after = pcs.get_bank("working")
        assert after.shape == (64, 32)

    def test_intermediates_captured_when_enabled(self) -> None:
        """With ``capture_intermediates=True``, per-iteration snapshots exist."""
        loop = ReasoningLoop(
            tiny_operator(),
            ReasoningLoopConfig(num_iterations=3, capture_intermediates=True),
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        assert len(loop.last_intermediates) == 3
        for snapshot in loop.last_intermediates:
            assert snapshot.shape == (64, 32)

    def test_intermediates_empty_when_disabled(self) -> None:
        """With ``capture_intermediates=False``, no snapshots are stored."""
        loop = ReasoningLoop(
            tiny_operator(), ReasoningLoopConfig(num_iterations=2)
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        assert loop.last_intermediates == []

    def test_reset_clears_state(self) -> None:
        """``reset`` clears iteration count and intermediates."""
        loop = ReasoningLoop(
            tiny_operator(),
            ReasoningLoopConfig(num_iterations=2, capture_intermediates=True),
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        loop.reset()
        assert loop.iteration_count == 0
        assert loop.last_intermediates == []

    def test_reset_clears_operator_kv_cache(self) -> None:
        """``reset`` clears the operator's KV cache."""
        op = tiny_operator()
        loop = ReasoningLoop(op, ReasoningLoopConfig(num_iterations=1))
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        loop.reset()
        for block in op.blocks:
            assert block.self_attn.kv_cache["length"] == 0

    def test_state_isolation_between_calls(self) -> None:
        """PCS state from one call is not leaked into the next call's inputs."""
        loop = ReasoningLoop(
            tiny_operator(),
            ReasoningLoopConfig(num_iterations=1, capture_intermediates=True),
        )
        pcs = tiny_pcs()
        # Run two calls with different observations.
        loop(pcs, torch.zeros(1, 4, 32))
        first_intermediate = loop.last_intermediates[0].clone()
        loop(pcs, torch.ones(1, 4, 32))
        second_intermediate = loop.last_intermediates[0]
        # The working memory state differs after different observations.
        assert not torch.allclose(first_intermediate, second_intermediate)

    def test_forward_returns_pcs(self) -> None:
        """The forward pass returns a :class:`PersistentCognitiveState`."""
        loop = ReasoningLoop(
            tiny_operator(), ReasoningLoopConfig(num_iterations=1)
        )
        out = loop(tiny_pcs(), torch.randn(1, 4, 32))
        assert isinstance(out, PersistentCognitiveState)

    def test_forward_preserves_pcs_structure(self) -> None:
        """After a forward pass the PCS still has all six banks."""
        loop = ReasoningLoop(
            tiny_operator(), ReasoningLoopConfig(num_iterations=1)
        )
        out = loop(tiny_pcs(), torch.randn(1, 4, 32))
        for name in ("working", "long_term", "goal", "episode", "task", "memory_index"):
            assert name in out.bank_specs

    def test_get_intermediates_returns_copy(self) -> None:
        """``get_intermediates`` returns a fresh list each call."""
        loop = ReasoningLoop(
            tiny_operator(),
            ReasoningLoopConfig(num_iterations=2, capture_intermediates=True),
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        copy1 = loop.get_intermediates()
        copy2 = loop.get_intermediates()
        assert copy1 == copy2
        assert copy1 is not copy2


class TestObservationMixConfig:
    """Tests for the ``alpha_k`` schedule."""

    def test_defaults_keep_the_real_observation(self) -> None:
        """Defaults hold ``alpha_k`` at 1.0 for every iteration."""
        config = ReasoningLoopConfig(num_iterations=4)
        assert config.observation_mix == 1.0
        assert config.observation_mix_decay == 1.0
        assert [config.observation_weight(k) for k in range(4)] == [1.0] * 4

    def test_decay_is_geometric(self) -> None:
        """``alpha_k = observation_mix * decay ** k``."""
        config = ReasoningLoopConfig(
            observation_mix=0.8, observation_mix_decay=0.5
        )
        weights = [config.observation_weight(k) for k in range(4)]
        assert weights == pytest.approx([0.8, 0.4, 0.2, 0.1])

    @pytest.mark.parametrize("mix", [-0.1, 1.1])
    def test_invalid_mix_rejected(self, mix: float) -> None:
        """``observation_mix`` outside [0, 1] raises."""
        with pytest.raises(ValueError):
            ReasoningLoopConfig(observation_mix=mix)

    @pytest.mark.parametrize("decay", [-0.1, 1.1])
    def test_invalid_decay_rejected(self, decay: float) -> None:
        """``observation_mix_decay`` outside [0, 1] raises."""
        with pytest.raises(ValueError):
            ReasoningLoopConfig(observation_mix_decay=decay)

    def test_negative_iteration_rejected(self) -> None:
        """``observation_weight`` rejects a negative iteration index."""
        with pytest.raises(ValueError):
            ReasoningLoopConfig().observation_weight(-1)


class TestEndogenousOrigination:
    """Tests for the generated-input path."""

    def loop_with_generator(
        self, **config_kwargs: object
    ) -> tuple[ReasoningLoop, OriginationHead]:
        """internal: a loop wired to a real origination generator."""
        generator = OriginationHead(32)
        loop = ReasoningLoop(
            tiny_operator(),
            ReasoningLoopConfig(**config_kwargs),  # type: ignore[arg-type]
            origination=generator,
        )
        return loop, generator

    def test_default_mix_never_calls_the_generator(self) -> None:
        """With ``alpha=1`` the loop is the pre-origination loop.

        This is the strict-generalisation check: the generator is attached
        but must not be consulted, so behaviour cannot have changed.
        """
        loop, _ = self.loop_with_generator(num_iterations=4)
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        assert loop.last_observation_weights == [1.0, 1.0, 1.0]
        assert loop.last_generated_inputs == []

    def test_default_mix_matches_no_generator(self) -> None:
        """``alpha=1`` gives the same state as attaching no generator."""
        torch.manual_seed(0)
        operator = tiny_operator()
        config = ReasoningLoopConfig(num_iterations=3)
        plain = ReasoningLoop(operator, config)
        wired = ReasoningLoop(operator, config, origination=OriginationHead(32))
        observation = torch.randn(1, 4, 32)
        torch.manual_seed(1)
        plain_out = plain(tiny_pcs(), observation).get_bank("working").clone()
        torch.manual_seed(1)
        wired_out = wired(tiny_pcs(), observation).get_bank("working").clone()
        assert torch.allclose(plain_out, wired_out)

    def test_generator_called_when_mix_below_one(self) -> None:
        """A sub-1.0 ``alpha`` routes the next input through ``G``."""
        loop, _ = self.loop_with_generator(
            num_iterations=3, observation_mix=0.5
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        assert loop.last_observation_weights == [0.5, 0.5]
        assert len(loop.last_generated_inputs) == 2
        for generated in loop.last_generated_inputs:
            assert generated.shape == (1, 4, 32)

    def test_generated_input_shape_follows_the_observation(self) -> None:
        """The generated stream keeps the observation's token count."""
        loop, _ = self.loop_with_generator(
            num_iterations=2, observation_mix=0.0
        )
        loop(tiny_pcs(), torch.randn(1, 11, 32))
        assert loop.last_generated_inputs[0].shape == (1, 11, 32)

    def test_fully_endogenous_input_drops_the_observation(self) -> None:
        """``alpha=0`` feeds the generated stream verbatim."""
        loop, _ = self.loop_with_generator(
            num_iterations=2, observation_mix=0.0
        )
        loop(tiny_pcs(), torch.randn(1, 5, 32))
        weights = loop.last_observation_weights
        assert weights == [0.0]

    def test_mix_is_against_the_original_observation(self) -> None:
        """The blend uses ``O_0``, not the previous generated stream."""
        generator = OriginationHead(32)
        loop = ReasoningLoop(
            tiny_operator(),
            ReasoningLoopConfig(num_iterations=2, observation_mix=0.25),
            origination=generator,
        )
        pcs = tiny_pcs()
        observation = torch.randn(1, 4, 32)
        drifted = torch.randn(1, 4, 32)
        # ``current`` is deliberately unrelated to ``observation`` so a mix
        # against the wrong tensor would show up.
        mixed = loop.next_observation(pcs, observation, drifted, 0)
        generated = loop.last_generated_inputs[-1]
        expected = 0.25 * observation + 0.75 * generated
        assert torch.allclose(mixed, expected)
        assert not torch.allclose(mixed, 0.25 * drifted + 0.75 * generated)

    def test_generator_gradient_reaches_the_intent_bank(self) -> None:
        """A loss after the loop trains the origination generator."""
        generator = OriginationHead(32)
        operator = tiny_operator()
        loop = ReasoningLoop(
            operator,
            ReasoningLoopConfig(num_iterations=3, observation_mix=0.5),
            origination=generator,
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        working = loop.differentiable_bank("working")
        assert working is not None
        working.pow(2).mean().backward()
        assert all(p.grad is not None for p in generator.parameters())

    def test_no_generator_ignores_a_low_mix(self) -> None:
        """Without ``G`` the loop still feeds the real observation."""
        loop = ReasoningLoop(
            tiny_operator(),
            ReasoningLoopConfig(num_iterations=3, observation_mix=0.0),
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        assert loop.last_generated_inputs == []

    def test_reset_clears_origination_traces(self) -> None:
        """``reset`` drops the recorded weights and generated inputs."""
        loop, _ = self.loop_with_generator(
            num_iterations=3, observation_mix=0.5
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        loop.reset()
        assert loop.last_observation_weights == []
        assert loop.last_generated_inputs == []


class TestDifferentiableStateCarry:
    """Tests for the loop's differentiable state hand-off."""

    def test_differentiable_bank_is_none_before_forward(self) -> None:
        """No carried tensors exist before the first forward pass."""
        loop = ReasoningLoop(tiny_operator())
        assert loop.differentiable_bank("working") is None

    def test_differentiable_bank_has_autograd_history(self) -> None:
        """The carried working tensor is still attached to the graph.

        Reading the bank off the PCS instead returns a parameter that the
        ``no_grad`` write-back has detached from the transition.
        """
        loop = ReasoningLoop(
            tiny_operator(), ReasoningLoopConfig(num_iterations=2)
        )
        pcs = loop(tiny_pcs(), torch.randn(1, 4, 32))
        working = loop.differentiable_bank("working")
        assert working is not None
        assert working.grad_fn is not None
        assert working.shape == (64, 32)
        assert pcs.get_bank("working").grad_fn is None

    def test_differentiable_bank_covers_every_bank(self) -> None:
        """Every bank is carried, not just working memory."""
        loop = ReasoningLoop(
            tiny_operator(), ReasoningLoopConfig(num_iterations=1)
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        for name in (
            "working",
            "long_term",
            "goal",
            "episode",
            "task",
            "memory_index",
        ):
            bank = loop.differentiable_bank(name)
            assert bank is not None
            assert bank.grad_fn is not None

    def test_loss_on_carried_bank_reaches_operator(self) -> None:
        """A loss on the carried tensor trains the operator weights."""
        op = tiny_operator()
        loop = ReasoningLoop(op, ReasoningLoopConfig(num_iterations=3))
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        working = loop.differentiable_bank("working")
        assert working is not None
        working.pow(2).mean().backward()
        with_grad = [
            name for name, p in op.named_parameters() if p.grad is not None
        ]
        assert len(with_grad) == len(list(op.named_parameters()))

    def test_intermediates_are_differentiable(self) -> None:
        """Captured JEPA intermediates carry gradients."""
        loop = ReasoningLoop(
            tiny_operator(),
            ReasoningLoopConfig(num_iterations=3, capture_intermediates=True),
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        assert len(loop.last_intermediates) == 3
        for snapshot in loop.last_intermediates:
            assert snapshot.grad_fn is not None

    def test_reset_clears_carried_tensors(self) -> None:
        """``reset`` drops the carried tensors along with the KV cache."""
        loop = ReasoningLoop(
            tiny_operator(), ReasoningLoopConfig(num_iterations=1)
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        loop.reset()
        assert loop.last_bank_tensors is None
        assert loop.differentiable_bank("working") is None

    def test_operator_without_carry_falls_back(self) -> None:
        """Operators that expose no carried tensors still work.

        The loop must degrade to detached PCS reads rather than raise, so
        alternative operators (Mamba, RWKV) need no changes.
        """

        class BareOperator(StateTransitionOperator):
            """internal: an operator that carries no differentiable state."""

            @property
            def name(self) -> str:
                return "bare"

            def forward(
                self,
                cstate: PersistentCognitiveState,
                observation: Tensor,
            ) -> PersistentCognitiveState:
                return cstate

            def initialize(self) -> None:
                return None

            def reset(self) -> None:
                return None

        loop = ReasoningLoop(
            BareOperator(),
            ReasoningLoopConfig(num_iterations=2, capture_intermediates=True),
        )
        loop(tiny_pcs(), torch.randn(1, 4, 32))
        assert loop.differentiable_bank("working") is None
        assert len(loop.last_intermediates) == 2
        for snapshot in loop.last_intermediates:
            assert snapshot.grad_fn is None
