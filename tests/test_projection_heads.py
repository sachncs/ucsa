"""Tests for :mod:`ucsa.models.projection_heads`."""

from __future__ import annotations

import pytest
import torch

from ucsa.models.projection_heads import (
    HeadConfig,
    LanguageHead,
    MemoryHead,
    PlanningHead,
    ProjectionHeads,
    ToolHead,
)


def tiny_config(**overrides: object) -> HeadConfig:
    """Return a tiny head config for tests."""
    defaults: dict[str, object] = {
        "hidden_size": 32,
        "vocab_size": 100,
        "num_plan_tokens": 16,
        "num_tools": 8,
        "memory_query_dim": 24,
    }
    defaults.update(overrides)
    return HeadConfig(**defaults)  # type: ignore[arg-type]


class TestHeadConfig:
    """Tests for :class:`HeadConfig`."""

    def test_default_config_valid(self) -> None:
        """Defaults construct without error."""
        config = HeadConfig()
        assert config.hidden_size > 0

    def test_zero_hidden_size_rejected(self) -> None:
        """``hidden_size`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            HeadConfig(hidden_size=0)

    def test_zero_vocab_size_rejected(self) -> None:
        """``vocab_size`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            HeadConfig(vocab_size=0)

    def test_zero_num_plan_tokens_rejected(self) -> None:
        """``num_plan_tokens`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            HeadConfig(num_plan_tokens=0)

    def test_zero_num_tools_rejected(self) -> None:
        """``num_tools`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            HeadConfig(num_tools=0)

    def test_zero_memory_query_dim_rejected(self) -> None:
        """``memory_query_dim`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            HeadConfig(memory_query_dim=0)


class TestLanguageHead:
    """Tests for :class:`LanguageHead`."""

    def test_forward_shape(self) -> None:
        """Output shape matches ``(batch, seq, vocab)``."""
        head = LanguageHead(hidden_size=32, vocab_size=100)
        x = torch.randn(2, 5, 32)
        out = head(x)
        assert out.shape == (2, 5, 100)

    def test_gradient_flows(self) -> None:
        """Loss on output flows to the projection matrix."""
        head = LanguageHead(hidden_size=32, vocab_size=100)
        x = torch.randn(2, 5, 32)
        loss = head(x).sum()
        loss.backward()
        assert head.proj.weight.grad is not None


class TestPlanningHead:
    """Tests for :class:`PlanningHead`."""

    def test_forward_shape(self) -> None:
        """Output shape matches ``(batch, seq, num_plan_tokens)``."""
        head = PlanningHead(hidden_size=32, num_plan_tokens=16)
        x = torch.randn(1, 4, 32)
        out = head(x)
        assert out.shape == (1, 4, 16)

    def test_gradient_flows(self) -> None:
        """Loss on output flows to the projection matrix."""
        head = PlanningHead(hidden_size=32, num_plan_tokens=16)
        head(torch.randn(2, 3, 32)).sum().backward()
        assert head.proj.weight.grad is not None


class TestToolHead:
    """Tests for :class:`ToolHead`."""

    def test_forward_shape(self) -> None:
        """Output shape matches ``(batch, seq, num_tools)``."""
        head = ToolHead(hidden_size=32, num_tools=8)
        x = torch.randn(2, 3, 32)
        out = head(x)
        assert out.shape == (2, 3, 8)

    def test_gradient_flows(self) -> None:
        """Loss on output flows to the projection matrix."""
        head = ToolHead(hidden_size=32, num_tools=8)
        head(torch.randn(1, 2, 32)).sum().backward()
        assert head.proj.weight.grad is not None


class TestMemoryHead:
    """Tests for :class:`MemoryHead`."""

    def test_forward_shape(self) -> None:
        """Output shape matches ``(batch, seq, memory_query_dim)``."""
        head = MemoryHead(hidden_size=32, memory_query_dim=24)
        x = torch.randn(2, 3, 32)
        out = head(x)
        assert out.shape == (2, 3, 24)

    def test_gradient_flows(self) -> None:
        """Loss on output flows to the projection matrix."""
        head = MemoryHead(hidden_size=32, memory_query_dim=24)
        head(torch.randn(1, 2, 32)).sum().backward()
        assert head.proj.weight.grad is not None


class TestProjectionHeads:
    """Tests for :class:`ProjectionHeads`."""

    @pytest.fixture
    def heads(self) -> ProjectionHeads:
        """Provide a tiny head bundle."""
        return ProjectionHeads(tiny_config())

    def test_construction_has_all_heads(self, heads: ProjectionHeads) -> None:
        """The bundle contains every head."""
        assert isinstance(heads.language, LanguageHead)
        assert isinstance(heads.planning, PlanningHead)
        assert isinstance(heads.tool, ToolHead)
        assert isinstance(heads.memory, MemoryHead)

    def test_forward_returns_all_outputs(
        self, heads: ProjectionHeads
    ) -> None:
        """``forward`` returns language, planning, tool, and memory outputs."""
        x = torch.randn(2, 4, 32)
        out = heads(x)
        assert set(out) == {
            "language", "planning", "tool", "memory", "input_reconstruct"
        }
        assert out["language"].shape == (2, 4, 100)
        assert out["planning"].shape == (2, 4, 16)
        assert out["tool"].shape == (2, 4, 8)
        assert out["memory"].shape == (2, 4, 24)

    def test_head_outputs_alias(self, heads: ProjectionHeads) -> None:
        """``head_outputs`` is an alias for ``forward``."""
        x = torch.randn(1, 4, 32)
        assert torch.allclose(heads(x)["language"], heads.head_outputs(x)["language"])

    def test_heads_have_independent_parameters(
        self, heads: ProjectionHeads
    ) -> None:
        """Each head has its own parameters, none shared."""
        language_params = {id(p) for p in heads.language.parameters()}
        planning_params = {id(p) for p in heads.planning.parameters()}
        tool_params = {id(p) for p in heads.tool.parameters()}
        memory_params = {id(p) for p in heads.memory.parameters()}
        all_params = language_params | planning_params | tool_params | memory_params
        assert len(all_params) == sum(
            len(p) for p in (
                language_params,
                planning_params,
                tool_params,
                memory_params,
            )
        )

    def test_gradient_isolation(self, heads: ProjectionHeads) -> None:
        """Loss on one head does not affect the other heads' gradients."""
        x = torch.randn(1, 4, 32)
        heads.zero_grad(set_to_none=True)
        out = heads(x)
        out["language"].sum().backward()
        assert heads.language.proj.weight.grad is not None
        # Other heads did not receive gradient.
        assert heads.planning.proj.weight.grad is None
        assert heads.tool.proj.weight.grad is None
        assert heads.memory.proj.weight.grad is None

    def test_heads_no_state(self, heads: ProjectionHeads) -> None:
        """Heads hold no PCS-like state."""
        for name, _ in heads.named_parameters():
            assert "meta_" not in name

    def test_inputs_only_working_memory(
        self, heads: ProjectionHeads
    ) -> None:
        """Changing the input changes every head's output."""
        x_a = torch.randn(1, 4, 32)
        x_b = x_a + 0.1
        out_a = heads(x_a)
        out_b = heads(x_b)
        for key in ("language", "planning", "tool", "memory"):
            assert not torch.allclose(out_a[key], out_b[key])

    def test_parameter_count_reasonable(
        self, heads: ProjectionHeads
    ) -> None:
        """Parameter count is positive and below a sanity ceiling."""
        n = sum(p.numel() for p in heads.parameters())
        assert n > 0
        assert n < 10_000_000
