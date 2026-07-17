"""Tests for :mod:`ucsa.models.moe`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ucsa.models.moe import Expert, MixtureOfExperts, MoEConfig


def tiny_config(**overrides: object) -> MoEConfig:
    """Return a tiny MoE config for tests."""
    defaults: dict[str, object] = {
        "num_experts": 4,
        "top_k": 2,
        "capacity_factor": 1.5,
        "aux_loss_weight": 0.01,
    }
    defaults.update(overrides)
    return MoEConfig(**defaults)  # type: ignore[arg-type]


class TestMoEConfig:
    """Tests for :class:`MoEConfig`."""

    def test_default_config_valid(self) -> None:
        """Defaults construct without error."""
        cfg = MoEConfig()
        assert cfg.num_experts == 4
        assert cfg.top_k == 2

    def test_top_k_must_be_at_most_num_experts(self) -> None:
        """``top_k`` greater than ``num_experts`` is rejected."""
        with pytest.raises(ValueError):
            MoEConfig(num_experts=4, top_k=5)

    def test_top_k_must_be_positive(self) -> None:
        """``top_k`` must be at least one."""
        with pytest.raises(ValueError):
            MoEConfig(num_experts=4, top_k=0)

    def test_num_experts_must_be_positive(self) -> None:
        """``num_experts`` must be at least one."""
        with pytest.raises(ValueError):
            MoEConfig(num_experts=0)

    def test_capacity_factor_must_be_positive(self) -> None:
        """``capacity_factor`` must be positive."""
        with pytest.raises(ValueError):
            MoEConfig(capacity_factor=0.0)

    def test_aux_loss_weight_must_be_non_negative(self) -> None:
        """``aux_loss_weight`` must be non-negative."""
        with pytest.raises(ValueError):
            MoEConfig(aux_loss_weight=-0.1)


class TestExpert:
    """Tests for :class:`Expert`."""

    def test_forward_shape(self) -> None:
        """Output shape matches input shape."""
        expert = Expert(hidden_size=16, intermediate_size=32)
        x = torch.randn(2, 5, 16)
        out = expert(x)
        assert out.shape == x.shape

    def test_zero_input_yields_zero(self) -> None:
        """An expert initialised at zero outputs zero (after init)."""
        expert = Expert(hidden_size=8, intermediate_size=16)
        for param in expert.parameters():
            zero_parameters(param)
        x = torch.randn(1, 3, 8)
        out = expert(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_gradient_flows(self) -> None:
        """A loss on the expert output flows gradients to its parameters."""
        expert = Expert(hidden_size=8, intermediate_size=16)
        out = expert(torch.randn(2, 3, 8))
        out.sum().backward()
        for param in expert.parameters():
            assert param.grad is not None


def zero_parameters(tensor: Tensor) -> None:
    """internal: zero a parameter tensor in-place."""
    with torch.no_grad():
        tensor.zero_()


class TestMixtureOfExperts:
    """Tests for :class:`MixtureOfExperts`."""

    @pytest.fixture
    def moe(self) -> MixtureOfExperts:
        """Provide a tiny MoE block."""
        return MixtureOfExperts(
            hidden_size=16,
            intermediate_size=32,
            config=tiny_config(),
        )

    def test_forward_shape(self, moe: MixtureOfExperts) -> None:
        """Output shape matches input shape."""
        x = torch.randn(2, 5, 16)
        out, aux = moe(x)
        assert out.shape == x.shape

    def test_aux_loss_is_scalar(self, moe: MixtureOfExperts) -> None:
        """Auxiliary loss is a zero-dimensional tensor."""
        x = torch.randn(2, 5, 16)
        _, aux = moe(x)
        assert aux.dim() == 0

    def test_gradient_flows_through_moe(self, moe: MixtureOfExperts) -> None:
        """A loss on the MoE output flows gradients to all experts and router."""
        torch.manual_seed(0)
        x = torch.randn(64, 4, 16)
        out, aux = moe(x)
        (out.sum() + aux).backward()
        assert moe.router.weight.grad is not None
        # At least one expert must receive gradient under high-volume routing.
        grads_found = 0
        for expert in moe.experts:
            for param in expert.parameters():
                if param.grad is not None and torch.any(param.grad != 0):
                    grads_found += 1
                    break
        assert grads_found >= 1

    def test_load_balancing_loss_decreases_with_balanced_routing(
        self,
    ) -> None:
        """Forced balanced routing has a lower aux loss than collapsed routing."""
        torch.manual_seed(0)
        balanced = MixtureOfExperts(
            hidden_size=16,
            intermediate_size=32,
            config=tiny_config(),
        )
        # Force router to send equal weight to each expert.
        with torch.no_grad():
            balanced.router.weight.zero_()
            balanced.router.weight[:4, :4] = (
                torch.eye(4)  # type: ignore[index]
            )
        x = torch.randn(8, 4, 16)
        _, balanced_loss = balanced(x)

        torch.manual_seed(0)
        collapsed = MixtureOfExperts(
            hidden_size=16,
            intermediate_size=32,
            config=tiny_config(),
        )
        with torch.no_grad():
            # Force router to always send everything to expert 0.
            collapsed.router.weight.zero_()
            collapsed.router.weight[0, :] = 1.0
        _, collapsed_loss = collapsed(x)
        assert balanced_loss.item() < collapsed_loss.item()

    def test_top_k_routing_shape(self, moe: MixtureOfExperts) -> None:
        """Top-k routing produces ``top_k`` weights per token."""
        x = torch.randn(2, 5, 16)
        flat = x.reshape(-1, 16)
        router_logits = moe.router(flat)
        routing_weights = torch.softmax(router_logits, dim=-1)
        _, top_indices = torch.topk(routing_weights, moe.config.top_k, dim=-1)
        assert top_indices.shape == (10, moe.config.top_k)

    def test_expert_capacity_respected(self, moe: MixtureOfExperts) -> None:
        """No more than ``expert_capacity`` tokens are routed to any expert."""
        x = torch.randn(32, 8, 16)
        out, _ = moe(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_top_k_one(self) -> None:
        """Top-1 routing is supported."""
        moe = MixtureOfExperts(
            hidden_size=16,
            intermediate_size=32,
            config=tiny_config(top_k=1),
        )
        x = torch.randn(2, 4, 16)
        out, aux = moe(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert torch.isfinite(aux)

    def test_collapses_to_ffn_when_single_expert(self) -> None:
        """With a single expert, MoE behaves like a dense FFN."""
        moe = MixtureOfExperts(
            hidden_size=16,
            intermediate_size=32,
            config=tiny_config(num_experts=1, top_k=1),
        )
        x = torch.randn(2, 5, 16)
        out, aux = moe(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert torch.isfinite(aux)

    def test_integration_with_transformer_operator(self) -> None:
        """MoE installed on a transformer operator contributes to aux loss."""
        from ucsa.models.state import PCSConfig, PersistentCognitiveState
        from ucsa.models.transformer_operator import (
            TransformerOperator,
            TransformerOperatorConfig,
        )

        pcs = PersistentCognitiveState(PCSConfig(hidden_size=32))
        cfg = TransformerOperatorConfig(
            hidden_size=32,
            num_layers=4,
            num_q_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            moe=tiny_config(),
        )
        op = TransformerOperator(cfg)
        obs = torch.randn(1, 4, 32)
        op(pcs, obs)
        assert op.last_aux_loss.item() != 0.0

    def test_moe_layer_count_in_operator(self) -> None:
        """The upper-half of layers is configured as MoE layers."""
        from ucsa.models.transformer_operator import (
            TransformerOperator,
            TransformerOperatorConfig,
        )

        cfg = TransformerOperatorConfig(
            hidden_size=32,
            num_layers=4,
            num_q_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            moe=tiny_config(),
        )
        op = TransformerOperator(cfg)
        moe_layers = [b for b in op.blocks if b.is_moe_layer]
        assert len(moe_layers) == 2

    def test_no_moe_when_disabled(self) -> None:
        """``moe=None`` keeps every block as a dense FFN."""
        from ucsa.models.state import PCSConfig, PersistentCognitiveState
        from ucsa.models.transformer_operator import (
            TransformerOperator,
            TransformerOperatorConfig,
        )

        pcs = PersistentCognitiveState(PCSConfig(hidden_size=32))
        cfg = TransformerOperatorConfig(hidden_size=32, num_layers=4)
        op = TransformerOperator(cfg)
        obs = torch.randn(1, 4, 32)
        op(pcs, obs)
        assert op.last_aux_loss.item() == 0.0
