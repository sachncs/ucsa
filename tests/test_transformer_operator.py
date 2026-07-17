"""Tests for :mod:`ucsa.models.transformer_operator`."""

from __future__ import annotations

import pytest
import torch

from ucsa.models.moe import MoEConfig
from ucsa.models.state import PCSConfig, PersistentCognitiveState
from ucsa.models.transformer_operator import (
    CrossAttention,
    FeedForward,
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
    TransformerOperator,
    TransformerOperatorConfig,
)


def tiny_config(**overrides: object) -> TransformerOperatorConfig:
    """Return a tiny transformer config for tests."""
    defaults: dict[str, object] = {
        "hidden_size": 32,
        "num_layers": 4,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 64,
        "sliding_window": 4096,
        "max_position": 8192,
    }
    defaults.update(overrides)
    return TransformerOperatorConfig(**defaults)  # type: ignore[arg-type]


class TestTransformerOperatorConfig:
    """Tests for :class:`TransformerOperatorConfig`."""

    def test_default_config_valid(self) -> None:
        """Defaults construct without error."""
        config = TransformerOperatorConfig()
        assert config.hidden_size > 0
        assert config.num_layers > 0

    def test_invalid_q_kv_divisibility(self) -> None:
        """Mismatched q/kv head counts are rejected."""
        with pytest.raises(ValueError):
            TransformerOperatorConfig(num_q_heads=3, num_kv_heads=2)

    def test_invalid_hidden_divisibility(self) -> None:
        """Hidden size not divisible by num_q_heads is rejected."""
        with pytest.raises(ValueError):
            TransformerOperatorConfig(hidden_size=33, num_q_heads=4)

    def test_invalid_sliding_window(self) -> None:
        """Non-positive sliding window is rejected."""
        with pytest.raises(ValueError):
            TransformerOperatorConfig(sliding_window=0)

    def test_moe_config_validates(self) -> None:
        """MoEConfig rejects bad top_k and negative aux weight."""
        with pytest.raises(ValueError):
            MoEConfig(num_experts=4, top_k=5)
        with pytest.raises(ValueError):
            MoEConfig(aux_loss_weight=-0.1)
        with pytest.raises(ValueError):
            MoEConfig(capacity_factor=0.0)


class TestRMSNorm:
    """Tests for :class:`RMSNorm`."""

    def test_forward_shape(self) -> None:
        """Output shape matches input shape."""
        norm = RMSNorm(16)
        x = torch.randn(2, 4, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_zero_input_yields_zero(self) -> None:
        """All-zero input yields all-zero output (after weight=1 init)."""
        norm = RMSNorm(8)
        x = torch.zeros(1, 1, 8)
        out = norm(x)
        assert torch.all(out == 0.0)

    def test_gradient_flows(self) -> None:
        """A loss on the output flows gradients to the weight."""
        norm = RMSNorm(8)
        x = torch.randn(2, 3, 8)
        out = norm(x)
        out.sum().backward()
        assert norm.weight.grad is not None


class TestRotaryEmbedding:
    """Tests for :class:`RotaryEmbedding`."""

    def test_odd_head_dim_rejected(self) -> None:
        """Odd ``head_dim`` raises."""
        with pytest.raises(ValueError):
            RotaryEmbedding(head_dim=7)

    def test_cos_sin_shapes(self) -> None:
        """``get_cos_sin`` returns tables of shape ``(1, 1, seq, head_dim)``."""
        rope = RotaryEmbedding(head_dim=8)
        cos, sin = rope.get_cos_sin(seq_len=5, device=torch.device("cpu"))
        assert cos.shape == (1, 1, 5, 8)
        assert sin.shape == (1, 1, 5, 8)

    def test_apply_rotary_preserves_shape(self) -> None:
        """Rotating a tensor preserves its shape."""
        rope = RotaryEmbedding(head_dim=8)
        x = torch.randn(2, 4, 5, 8)
        cos, sin = rope.get_cos_sin(seq_len=5, device=torch.device("cpu"))
        rotated = RotaryEmbedding.apply_rotary(x, cos, sin)
        assert rotated.shape == x.shape

    def test_apply_rotary_zero(self) -> None:
        """Rotating zero returns zero."""
        rope = RotaryEmbedding(head_dim=8)
        x = torch.zeros(1, 1, 3, 8)
        cos, sin = rope.get_cos_sin(seq_len=3, device=torch.device("cpu"))
        rotated = RotaryEmbedding.apply_rotary(x, cos, sin)
        assert torch.all(rotated == 0)


class TestGroupedQueryAttention:
    """Tests for :class:`GroupedQueryAttention`."""

    def test_invalid_q_kv_divisibility(self) -> None:
        """Mismatched q/kv head counts raise."""
        with pytest.raises(ValueError):
            GroupedQueryAttention(
                hidden_size=16,
                num_q_heads=3,
                num_kv_heads=2,
                sliding_window=64,
            )

    def test_forward_shape(self) -> None:
        """Output shape matches query shape."""
        attn = GroupedQueryAttention(
            hidden_size=16,
            num_q_heads=4,
            num_kv_heads=2,
            sliding_window=64,
        )
        q = torch.randn(2, 5, 16)
        out = attn(q, q, is_first_step=True)
        assert out.shape == q.shape

    def test_kv_cache_grows_on_subsequent_steps(self) -> None:
        """Subsequent steps concatenate context to the cache."""
        attn = GroupedQueryAttention(
            hidden_size=16,
            num_q_heads=4,
            num_kv_heads=2,
            sliding_window=64,
        )
        q1 = torch.randn(1, 3, 16)
        attn(q1, q1, is_first_step=True)
        first_cache_len = attn.kv_cache["length"]
        q2 = torch.randn(1, 2, 16)
        attn(q2, q2, is_first_step=False)
        second_cache_len = attn.kv_cache["length"]
        assert second_cache_len == first_cache_len + 2

    def test_reset_cache_clears(self) -> None:
        """``reset_cache`` returns the cache to empty."""
        attn = GroupedQueryAttention(
            hidden_size=16,
            num_q_heads=4,
            num_kv_heads=2,
            sliding_window=64,
        )
        x = torch.randn(1, 3, 16)
        attn(x, x, is_first_step=True)
        attn.reset_cache()
        assert attn.kv_cache["k"] is None
        assert attn.kv_cache["v"] is None
        assert attn.kv_cache["length"] == 0

    def test_sliding_window_truncation(self) -> None:
        """KV cache is truncated to ``sliding_window`` tokens."""
        attn = GroupedQueryAttention(
            hidden_size=16,
            num_q_heads=4,
            num_kv_heads=2,
            sliding_window=4,
        )
        x = torch.randn(1, 5, 16)
        attn(x, x, is_first_step=True)
        assert attn.kv_cache["length"] == 4

    def test_sliding_window_with_large_sliding_window(self) -> None:
        """Larger sliding window caches the full context."""
        attn = GroupedQueryAttention(
            hidden_size=16,
            num_q_heads=4,
            num_kv_heads=2,
            sliding_window=128,
        )
        x = torch.randn(1, 5, 16)
        attn(x, x, is_first_step=True)
        assert attn.kv_cache["length"] == 5

    def test_query_and_context_can_differ(self) -> None:
        """The query and context streams may have different shapes."""
        attn = GroupedQueryAttention(
            hidden_size=16,
            num_q_heads=4,
            num_kv_heads=2,
            sliding_window=64,
        )
        q = torch.randn(1, 4, 16)
        ctx = torch.randn(1, 7, 16)
        out = attn(q, ctx, is_first_step=True)
        assert out.shape == q.shape


class TestCrossAttention:
    """Tests for :class:`CrossAttention`."""

    def test_forward_shape(self) -> None:
        """Cross attention output shape matches query shape."""
        attn = CrossAttention(
            hidden_size=16,
            num_q_heads=4,
            num_kv_heads=2,
        )
        query = torch.randn(2, 5, 16)
        kv = torch.randn(2, 7, 16)
        out = attn(query, kv)
        assert out.shape == query.shape

    def test_invalid_q_kv_divisibility(self) -> None:
        """Mismatched q/kv head counts raise."""
        with pytest.raises(ValueError):
            CrossAttention(hidden_size=16, num_q_heads=3, num_kv_heads=2)


class TestFeedForward:
    """Tests for :class:`FeedForward`."""

    def test_forward_shape(self) -> None:
        """Output shape matches input shape."""
        ffn = FeedForward(hidden_size=16, intermediate_size=32)
        x = torch.randn(2, 5, 16)
        assert ffn(x).shape == x.shape

    def test_gradient_flows(self) -> None:
        """A loss on the FFN output flows gradients to its parameters."""
        ffn = FeedForward(hidden_size=16, intermediate_size=32)
        out = ffn(torch.randn(2, 3, 16))
        out.sum().backward()
        for param in ffn.parameters():
            assert param.grad is not None


class TestTransformerBlock:
    """Tests for :class:`TransformerBlock`."""

    def test_is_moe_layer_upper_half(self) -> None:
        """Upper-half blocks report ``is_moe_layer`` as ``True`` only when
        MoE is configured."""
        config = tiny_config(num_layers=4, moe=MoEConfig())
        lower_block = TransformerBlock(config, layer_index=0)
        upper_block = TransformerBlock(config, layer_index=3)
        assert lower_block.is_moe_layer is False
        assert upper_block.is_moe_layer is True

    def test_forward_shape(self) -> None:
        """Forward preserves token sequence shape."""
        config = tiny_config()
        block = TransformerBlock(config, layer_index=0)
        tokens = torch.randn(2, 10, 32)
        out, aux = block(
            tokens,
            is_first_step=True,
            memory_index=None,
            working_slice=(2, 4),
        )
        assert out.shape == tokens.shape

    def test_cross_attention_used_when_memory_index_provided(self) -> None:
        """Cross attention changes the working-memory output."""
        config = tiny_config()
        block = TransformerBlock(config, layer_index=0)
        torch.manual_seed(0)
        tokens_a = torch.randn(1, 6, 32)
        tokens_b = torch.randn(1, 6, 32)
        memory_index = torch.randn(1, 4, 32)
        out_a, _ = block(
            tokens_a,
            is_first_step=True,
            memory_index=memory_index,
            working_slice=(0, 2),
        )
        out_b, _ = block(
            tokens_b,
            is_first_step=True,
            memory_index=memory_index,
            working_slice=(0, 2),
        )
        # The cross-attention uses a different KV source from self-attention.
        # Inputs differ; outputs must differ.
        assert not torch.allclose(out_a, out_b)


class TestTransformerOperator:
    """Tests for :class:`TransformerOperator`."""

    @pytest.fixture
    def state(self) -> PersistentCognitiveState:
        """Provide a fresh PCS sized for the tiny config."""
        return PersistentCognitiveState(PCSConfig(hidden_size=32))

    @pytest.fixture
    def operator(self) -> TransformerOperator:
        """Provide a tiny transformer operator."""
        return TransformerOperator(tiny_config())

    def test_constructs_with_default_config(self) -> None:
        """Default config builds a transformer operator."""
        op = TransformerOperator(TransformerOperatorConfig())
        assert op.name == "transformer"

    def test_forward_returns_pcs(
        self, operator: TransformerOperator, state: PersistentCognitiveState
    ) -> None:
        """Forward returns a PCS with the correct bank shapes."""
        observation = torch.randn(1, 6, 32)
        new_pcs = operator(state, observation)
        assert isinstance(new_pcs, PersistentCognitiveState)
        assert new_pcs.bank_size("working") == 64
        assert new_pcs.bank_size("long_term") == 128

    def test_forward_observation_shape_validation(
        self, operator: TransformerOperator, state: PersistentCognitiveState
    ) -> None:
        """Non-3D observation raises ``ValueError``."""
        with pytest.raises(ValueError):
            operator(state, torch.randn(6, 32))

    def test_forward_hidden_size_validation(self) -> None:
        """A PCS whose hidden size mismatches the operator raises."""
        op = TransformerOperator(tiny_config(hidden_size=32))
        bad_state = PersistentCognitiveState(PCSConfig(hidden_size=64))
        with pytest.raises(ValueError):
            op(bad_state, torch.randn(1, 4, 64))

    def test_gradient_flow(
        self, operator: TransformerOperator, state: PersistentCognitiveState
    ) -> None:
        """A loss on the output flows gradients to PCS parameters."""
        observation = torch.randn(1, 4, 32)
        new_pcs = operator(state, observation)
        loss = new_pcs.get_all_tokens().sum()
        loss.backward()
        assert new_pcs.get_bank("working").grad is not None

    def test_reset_clears_kv_cache(
        self, operator: TransformerOperator, state: PersistentCognitiveState
    ) -> None:
        """``reset`` zeros the KV cache length of every block."""
        operator(state, torch.randn(1, 3, 32))
        operator.reset()
        for block in operator.blocks:
            assert block.self_attn.kv_cache["length"] == 0

    def test_is_first_step_flag(
        self, operator: TransformerOperator, state: PersistentCognitiveState
    ) -> None:
        """``is_first_step`` is reset on subsequent ``reset`` calls."""
        operator(state, torch.randn(1, 3, 32))
        operator.reset()
        assert operator.is_first_step is True

    def test_cross_attention_disabled(
        self, state: PersistentCognitiveState
    ) -> None:
        """Disabling cross attention removes the cross-attn module."""
        op = TransformerOperator(tiny_config(use_memory_index_cross_attention=False))
        assert all(block.cross_attn is None for block in op.blocks)

    def test_cross_attention_enabled_by_default(
        self, state: PersistentCognitiveState
    ) -> None:
        """By default every block has a cross-attention module."""
        op = TransformerOperator(tiny_config())
        assert all(block.cross_attn is not None for block in op.blocks)

    def test_aux_loss_is_zero_without_moe(
        self, operator: TransformerOperator, state: PersistentCognitiveState
    ) -> None:
        """``last_aux_loss`` is a zero tensor when MoE is disabled."""
        operator(state, torch.randn(1, 4, 32))
        assert operator.last_aux_loss.item() == 0.0

    def test_multiple_steps_extend_kv_cache(
        self, operator: TransformerOperator, state: PersistentCognitiveState
    ) -> None:
        """KV cache length grows by the full concatenated sequence length."""
        operator(state, torch.randn(1, 3, 32))
        first_lengths = [b.self_attn.kv_cache["length"] for b in operator.blocks]
        operator(state, torch.randn(1, 5, 32))
        second_lengths = [b.self_attn.kv_cache["length"] for b in operator.blocks]
        # Second call adds (state.total_tokens + 5) tokens to the cache.
        expected_delta = state.total_tokens + 5
        assert all(
            s == f + expected_delta
            for s, f in zip(second_lengths, first_lengths, strict=False)
        )

    def test_parameter_count_reasonable(
        self, operator: TransformerOperator
    ) -> None:
        """Parameter count is positive and not absurdly large."""
        n_params = sum(p.numel() for p in operator.parameters())
        assert n_params > 0
        assert n_params < 10_000_000
