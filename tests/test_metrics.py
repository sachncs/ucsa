"""Tests for :mod:`ucsa.training.metrics`."""

from __future__ import annotations

import pytest
import torch

from ucsa.training.metrics import (
    DEFAULT_METRIC_NAMES,
    MetricsRegistry,
    attention_entropy,
    build_default_registry,
    expert_utilization,
    flatten_metrics,
    gpu_memory_bytes,
    intent_gate_entropy,
    intent_gate_mutual_info,
    intent_gate_usage,
    intent_read_share,
    intent_state_variance,
    memory_replacement_rate,
    memory_utilization,
    perplexity_from_loss,
    throughput,
)


class TestMetricsRegistry:
    """Tests for :class:`MetricsRegistry`."""

    def test_default_names(self) -> None:
        """The default metric set covers the spec plus intent diagnostics."""
        assert len(DEFAULT_METRIC_NAMES) == 22
        assert {
            "intent_state_variance",
            "intent_gate_entropy",
            "intent_gate_mutual_info",
            "intent_read_share",
        } <= set(DEFAULT_METRIC_NAMES)

    def test_construction(self) -> None:
        """A registry with custom names initialises correctly."""
        registry = MetricsRegistry(["a", "b"])
        assert registry.names == ("a", "b")
        assert registry.value("a") == 0.0
        assert registry.value("b") == 0.0

    def test_update_running_average(self) -> None:
        """``update`` accumulates a running average."""
        registry = MetricsRegistry(["loss"])
        registry.update("loss", 2.0)
        registry.update("loss", 1.0)
        registry.update("loss", 0.5)
        assert registry.value("loss") == pytest.approx(1.166666, abs=1e-4)

    def test_last_value(self) -> None:
        """``last`` returns the most recent value."""
        registry = MetricsRegistry(["loss"])
        registry.update("loss", 5.0)
        registry.update("loss", 2.0)
        assert registry.last("loss") == 2.0

    def test_unknown_metric_raises(self) -> None:
        """Updating an unknown metric raises ``KeyError``."""
        registry = MetricsRegistry(["a"])
        with pytest.raises(KeyError):
            registry.update("nope", 1.0)

    def test_reset(self) -> None:
        """``reset`` zeroes the running sums and history."""
        registry = MetricsRegistry(["a"])
        registry.update("a", 1.0)
        registry.update("a", 2.0)
        registry.reset()
        assert registry.value("a") == 0.0
        assert registry.last("a") == 0.0

    def test_snapshot(self) -> None:
        """``snapshot`` returns every metric's average."""
        registry = MetricsRegistry(["a", "b"])
        registry.update("a", 1.0)
        registry.update("b", 4.0)
        snapshot = registry.snapshot()
        assert snapshot == {"a": 1.0, "b": 4.0}

    def test_tensorboard_log(self) -> None:
        """``tensorboard_log`` writes a scalar for every metric."""
        registry = MetricsRegistry(["a"])
        registry.update("a", 1.0)

        class FakeWriter:
            def __init__(self) -> None:
                self.scalars: list[tuple[str, float, int]] = []

            def add_scalar(self, name: str, value: float, step: int) -> None:
                self.scalars.append((name, value, step))

        writer = FakeWriter()
        registry.tensorboard_log(writer, step=1)
        assert writer.scalars == [("a", 1.0, 1)]


class TestPerplexity:
    """Tests for :func:`perplexity_from_loss`."""

    def test_zero_loss(self) -> None:
        """Loss of zero yields perplexity of 1."""
        assert perplexity_from_loss(0.0) == 1.0

    def test_positive_loss(self) -> None:
        """Loss of 2.0 yields perplexity of approximately e^2."""
        assert perplexity_from_loss(2.0) == pytest.approx(7.389, abs=1e-3)

    def test_negative_loss_treated_as_one(self) -> None:
        """Negative loss is treated as perplexity of 1."""
        assert perplexity_from_loss(-1.0) == 1.0


class TestExpertUtilization:
    """Tests for :func:`expert_utilization`."""

    def test_shape(self) -> None:
        """Output has shape ``(num_experts,)``."""
        util = expert_utilization(torch.randn(16, 4))
        assert util.shape == (4,)

    def test_in_unit_interval(self) -> None:
        """Each utilisation is in ``[0, 1]``."""
        util = expert_utilization(torch.randn(16, 4))
        assert torch.all(util >= 0.0)
        assert torch.all(util <= 1.0)

    def test_balanced_sums_to_one(self) -> None:
        """A balanced distribution sums to 1."""
        util = expert_utilization(torch.zeros(16, 4))
        assert util.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_empty_routing_returns_zeros(self) -> None:
        """Empty routing returns zeros."""
        util = expert_utilization(torch.zeros(0, 4))
        assert torch.all(util == 0.0)


class TestAttentionEntropy:
    """Tests for :func:`attention_entropy`."""

    def test_uniform_distribution_high_entropy(self) -> None:
        """Uniform attention has high entropy."""
        weights = torch.full((1, 1, 4), 0.25)
        assert attention_entropy(weights) == pytest.approx(1.386, abs=1e-3)

    def test_peaked_distribution_low_entropy(self) -> None:
        """Peaked attention has near-zero entropy."""
        weights = torch.zeros(1, 1, 4)
        weights[0, 0, 0] = 1.0
        assert attention_entropy(weights) == pytest.approx(0.0, abs=1e-3)

    def test_empty_attention(self) -> None:
        """Empty attention has zero entropy."""
        assert attention_entropy(torch.zeros(0)) == 0.0


class TestThroughput:
    """Tests for :func:`throughput`."""

    def test_throughput_computation(self) -> None:
        """Tokens per second is computed correctly."""
        assert throughput(100, 2.0) == 50.0

    def test_zero_elapsed_returns_zero(self) -> None:
        """Zero elapsed time yields zero throughput."""
        assert throughput(100, 0.0) == 0.0

    def test_negative_elapsed_returns_zero(self) -> None:
        """Negative elapsed time yields zero throughput."""
        assert throughput(100, -1.0) == 0.0


class TestGpuMemory:
    """Tests for :func:`gpu_memory_bytes`."""

    def test_returns_zero_on_cpu(self) -> None:
        """On CPU devices, the function returns zero."""
        assert gpu_memory_bytes(torch.device("cpu")) == 0


class TestMemoryUtilization:
    """Tests for :func:`memory_utilization`."""

    def test_full(self) -> None:
        """Full memory has utilisation 1.0."""
        assert memory_utilization(64, 64) == 1.0

    def test_half(self) -> None:
        """Half-full memory has utilisation 0.5."""
        assert memory_utilization(32, 64) == 0.5

    def test_empty(self) -> None:
        """Empty memory has utilisation 0.0."""
        assert memory_utilization(0, 64) == 0.0

    def test_zero_capacity(self) -> None:
        """Zero capacity yields zero utilisation."""
        assert memory_utilization(10, 0) == 0.0


class TestMemoryReplacementRate:
    """Tests for :func:`memory_replacement_rate`."""

    def test_basic(self) -> None:
        """Replacement rate is recycled / total."""
        assert memory_replacement_rate(8, 32) == 0.25

    def test_zero_total(self) -> None:
        """Zero total yields zero rate."""
        assert memory_replacement_rate(5, 0) == 0.0


class TestDefaultRegistry:
    """Tests for :func:`build_default_registry`."""

    def test_default_registry(self) -> None:
        """``build_default_registry`` returns a registry with all metrics."""
        registry = build_default_registry()
        assert set(registry.names) == set(DEFAULT_METRIC_NAMES)


class TestFlattenMetrics:
    """Tests for :func:`flatten_metrics`."""

    def test_flattens_to_floats(self) -> None:
        """All values are plain floats."""
        out = flatten_metrics({"a": 1.0, "b": 2})
        assert all(isinstance(v, float) for v in out.values())


class TestIntentCollapseDiagnostics:
    """Tests for the intent-collapse metric functions."""

    def test_state_variance_zero_for_identical_states(self) -> None:
        """A bank that never changes reports exactly zero variance.

        This is the reading that catches a static intent parameter: one
        constant signal preceding every action.
        """
        state = torch.randn(4, 8)
        assert intent_state_variance([state, state.clone()]) == 0.0

    def test_state_variance_positive_when_states_differ(self) -> None:
        """Differing origination states report positive variance."""
        value = intent_state_variance([torch.randn(4, 8), torch.randn(4, 8)])
        assert value > 0.0

    def test_state_variance_needs_two_inputs(self) -> None:
        """Fewer than two inputs cannot express a variance."""
        assert intent_state_variance([]) == 0.0
        assert intent_state_variance([torch.randn(4, 8)]) == 0.0

    def test_gate_usage_is_a_distribution(self) -> None:
        """Usage fractions sum to one over the used slots."""
        weights = torch.tensor([[[0.5, 0.5, 0.0, 0.0]]])
        usage = intent_gate_usage(weights, 4)
        assert usage.shape == (4,)
        assert usage.sum().item() == pytest.approx(1.0)
        assert usage[2].item() == 0.0

    def test_gate_usage_empty_weights(self) -> None:
        """No gate weights means no usage."""
        assert intent_gate_usage(torch.zeros(0), 4).sum().item() == 0.0

    def test_gate_entropy_zero_for_single_slot(self) -> None:
        """One slot absorbing every query has zero entropy."""
        usage = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert intent_gate_entropy(usage) == pytest.approx(0.0)

    def test_gate_entropy_maximal_for_uniform(self) -> None:
        """A uniform gate reaches ``log(num_slots)`` and is uninformative."""
        usage = torch.full((4,), 0.25)
        assert intent_gate_entropy(usage) == pytest.approx(
            float(torch.tensor(4.0).log())
        )

    def test_gate_mutual_info_zero_when_gate_ignores_input(self) -> None:
        """Identical routing for every input carries no information.

        This is the collapse signature the diagnostic exists to catch.
        """
        usage = torch.tensor([0.5, 0.5, 0.0, 0.0])
        assert intent_gate_mutual_info([usage, usage.clone()]) == pytest.approx(
            0.0
        )

    def test_gate_mutual_info_positive_when_routing_differs(self) -> None:
        """Different inputs routing to different slots carries information."""
        first = torch.tensor([1.0, 0.0, 0.0, 0.0])
        second = torch.tensor([0.0, 1.0, 0.0, 0.0])
        assert intent_gate_mutual_info([first, second]) > 0.0

    def test_gate_mutual_info_needs_two_inputs(self) -> None:
        """Fewer than two inputs cannot express mutual information."""
        assert intent_gate_mutual_info([torch.tensor([1.0, 0.0])]) == 0.0

    def test_read_share_zero_when_intent_read_is_empty(self) -> None:
        """An all-zero intent read means the generator ignores the bank."""
        share = intent_read_share(torch.zeros(2, 4), torch.randn(2, 4))
        assert share == pytest.approx(0.0)

    def test_read_share_half_for_equal_magnitudes(self) -> None:
        """Equal read magnitudes split the share evenly."""
        vector = torch.ones(2, 4)
        assert intent_read_share(vector, vector.clone()) == pytest.approx(0.5)

    def test_read_share_zero_for_two_empty_reads(self) -> None:
        """Two empty reads report zero rather than dividing by zero."""
        zeros = torch.zeros(2, 4)
        assert intent_read_share(zeros, zeros.clone()) == 0.0
