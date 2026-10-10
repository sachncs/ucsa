"""Tests for :mod:`ucsa.models.projection`."""

from __future__ import annotations

import pytest
import torch

from ucsa.models import projection


def tiny_config(**overrides: object) -> projection.Config:
    """Return a tiny head config for tests."""
    defaults: dict[str, object] = {
        "hidden_size": 32,
        "vocab_size": 100,
        "num_plan_tokens": 16,
        "num_tools": 8,
        "memory_query_dim": 24,
    }
    defaults.update(overrides)
    return projection.Config(**defaults)  # type: ignore[arg-type]


class TestHeadConfig:
    """Tests for :class:`projection.Config`."""

    def test_default_config_valid(self) -> None:
        """Defaults construct without error."""
        config = projection.Config()
        assert config.hidden_size > 0

    def test_zero_hidden_size_rejected(self) -> None:
        """``hidden_size`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            projection.Config(hidden_size=0)

    def test_zero_vocab_size_rejected(self) -> None:
        """``vocab_size`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            projection.Config(vocab_size=0)

    def test_zero_num_plan_tokens_rejected(self) -> None:
        """``num_plan_tokens`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            projection.Config(num_plan_tokens=0)

    def test_zero_num_tools_rejected(self) -> None:
        """``num_tools`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            projection.Config(num_tools=0)

    def test_zero_memory_query_dim_rejected(self) -> None:
        """``memory_query_dim`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            projection.Config(memory_query_dim=0)


class TestLanguageHead:
    """Tests for :class:`projection.Language`."""

    def test_forward_shape(self) -> None:
        """Output shape matches ``(batch, seq, vocab)``."""
        head = projection.Language(hidden_size=32, vocab_size=100)
        x = torch.randn(2, 5, 32)
        out = head(x)
        assert out.shape == (2, 5, 100)

    def test_gradient_flows(self) -> None:
        """Loss on output flows to the projection matrix."""
        head = projection.Language(hidden_size=32, vocab_size=100)
        x = torch.randn(2, 5, 32)
        loss = head(x).sum()
        loss.backward()
        assert head.proj.weight.grad is not None


class TestPlanningHead:
    """Tests for :class:`projection.Planning`."""

    def test_forward_shape(self) -> None:
        """Output shape matches ``(batch, seq, num_plan_tokens)``."""
        head = projection.Planning(hidden_size=32, num_plan_tokens=16)
        x = torch.randn(1, 4, 32)
        out = head(x)
        assert out.shape == (1, 4, 16)

    def test_gradient_flows(self) -> None:
        """Loss on output flows to the projection matrix."""
        head = projection.Planning(hidden_size=32, num_plan_tokens=16)
        head(torch.randn(2, 3, 32)).sum().backward()
        assert head.proj.weight.grad is not None


class TestToolHead:
    """Tests for :class:`projection.Tool`."""

    def test_forward_shape(self) -> None:
        """Output shape matches ``(batch, seq, num_tools)``."""
        head = projection.Tool(hidden_size=32, num_tools=8)
        x = torch.randn(2, 3, 32)
        out = head(x)
        assert out.shape == (2, 3, 8)

    def test_gradient_flows(self) -> None:
        """Loss on output flows to the projection matrix."""
        head = projection.Tool(hidden_size=32, num_tools=8)
        head(torch.randn(1, 2, 32)).sum().backward()
        assert head.proj.weight.grad is not None


class TestMemoryHead:
    """Tests for :class:`projection.Memory`."""

    def test_forward_shape(self) -> None:
        """Output shape matches ``(batch, seq, memory_query_dim)``."""
        head = projection.Memory(hidden_size=32, memory_query_dim=24)
        x = torch.randn(2, 3, 32)
        out = head(x)
        assert out.shape == (2, 3, 24)

    def test_gradient_flows(self) -> None:
        """Loss on output flows to the projection matrix."""
        head = projection.Memory(hidden_size=32, memory_query_dim=24)
        head(torch.randn(1, 2, 32)).sum().backward()
        assert head.proj.weight.grad is not None


class TestOriginationHead:
    """Tests for :class:`projection.Origination` (the generator ``G``)."""

    @pytest.fixture
    def head(self) -> projection.Origination:
        """Provide a tiny origination generator."""
        return projection.Origination(hidden_size=32)

    def test_output_matches_the_observation_shape(
        self, head: projection.Origination
    ) -> None:
        """``G`` emits one token per input token, whatever the count."""
        for tokens in (1, 4, 13):
            out = head(
                torch.randn(16, 32),
                torch.randn(64, 32),
                torch.randn(1, tokens, 32),
            )
            assert out.shape == (1, tokens, 32)

    def test_gradient_reaches_intent_and_working(
        self, head: projection.Origination
    ) -> None:
        """Both context banks receive gradient from the generated stream."""
        intent = torch.randn(16, 32, requires_grad=True)
        working = torch.randn(64, 32, requires_grad=True)
        head(intent, working, torch.randn(1, 4, 32)).sum().backward()
        assert intent.grad is not None
        assert working.grad is not None

    def test_output_depends_on_intent(
        self, head: projection.Origination
    ) -> None:
        """Changing the intent bank changes the generated stream.

        If this ever fails, ``G`` has learned to ignore its origination
        input, which is the collapse mode the Phase C diagnostic watches.
        """
        working = torch.randn(64, 32)
        observation = torch.randn(1, 4, 32)
        first = head(torch.randn(16, 32), working, observation)
        second = head(torch.randn(16, 32), working, observation)
        assert not torch.allclose(first, second)

    def test_rejects_non_3d_observation(
        self, head: projection.Origination
    ) -> None:
        """A 2D observation raises."""
        with pytest.raises(ValueError):
            head(torch.randn(16, 32), torch.randn(64, 32), torch.randn(4, 32))

    def test_rejects_batched_banks(self, head: projection.Origination) -> None:
        """Banks must be 2D ``(tokens, hidden)``."""
        with pytest.raises(ValueError):
            head(
                torch.randn(1, 16, 32),
                torch.randn(64, 32),
                torch.randn(1, 4, 32),
            )

    def test_rejects_hidden_size_mismatch(
        self, head: projection.Origination
    ) -> None:
        """A hidden-size disagreement raises rather than broadcasting."""
        with pytest.raises(ValueError):
            head(
                torch.randn(16, 8),
                torch.randn(64, 32),
                torch.randn(1, 4, 32),
            )


class TestProjectionHeads:
    """Tests for :class:`projection.Heads`."""

    @pytest.fixture
    def heads(self) -> projection.Heads:
        """Provide a tiny head bundle."""
        return projection.Heads(tiny_config())

    def test_construction_has_all_heads(self, heads: projection.Heads) -> None:
        """The bundle contains every head."""
        assert isinstance(heads.language, projection.Language)
        assert isinstance(heads.planning, projection.Planning)
        assert isinstance(heads.tool, projection.Tool)
        assert isinstance(heads.memory, projection.Memory)
        assert isinstance(heads.origination, projection.Origination)

    def test_origination_is_not_a_forward_output(
        self, heads: projection.Heads
    ) -> None:
        """``G`` stays out of the head dict; the loop calls it directly.

        It reads the intent bank and the input stream, so it cannot share
        the working-memory-only head signature.
        """
        assert "origination" not in heads(torch.randn(2, 4, 32))

    def test_forward_returns_all_outputs(self, heads: projection.Heads) -> None:
        """``forward`` returns language, planning, tool, and memory outputs."""
        x = torch.randn(2, 4, 32)
        out = heads(x)
        assert set(out) == {
            "language",
            "planning",
            "tool",
            "memory",
            "input_reconstruct",
        }
        assert out["language"].shape == (2, 4, 100)
        assert out["planning"].shape == (2, 4, 16)
        assert out["tool"].shape == (2, 4, 8)
        assert out["memory"].shape == (2, 4, 24)

    def test_head_outputs_alias(self, heads: projection.Heads) -> None:
        """``head_outputs`` is an alias for ``forward``."""
        x = torch.randn(1, 4, 32)
        assert torch.allclose(
            heads(x)["language"], heads.head_outputs(x)["language"]
        )

    def test_heads_have_independent_parameters(
        self, heads: projection.Heads
    ) -> None:
        """Each head has its own parameters, none shared."""
        language_params = {id(p) for p in heads.language.parameters()}
        planning_params = {id(p) for p in heads.planning.parameters()}
        tool_params = {id(p) for p in heads.tool.parameters()}
        memory_params = {id(p) for p in heads.memory.parameters()}
        all_params = (
            language_params | planning_params | tool_params | memory_params
        )
        assert len(all_params) == sum(
            len(p)
            for p in (
                language_params,
                planning_params,
                tool_params,
                memory_params,
            )
        )

    def test_gradient_isolation(self, heads: projection.Heads) -> None:
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

    def test_heads_no_state(self, heads: projection.Heads) -> None:
        """Heads hold no PCS-like state."""
        for name, _ in heads.named_parameters():
            assert "meta_" not in name

    def test_inputs_only_working_memory(self, heads: projection.Heads) -> None:
        """Changing the input changes every head's output."""
        x_a = torch.randn(1, 4, 32)
        x_b = x_a + 0.1
        out_a = heads(x_a)
        out_b = heads(x_b)
        for key in ("language", "planning", "tool", "memory"):
            assert not torch.allclose(out_a[key], out_b[key])

    def test_parameter_count_reasonable(self, heads: projection.Heads) -> None:
        """Parameter count is positive and below a sanity ceiling."""
        n = sum(p.numel() for p in heads.parameters())
        assert n > 0
        assert n < 10_000_000
