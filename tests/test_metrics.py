"""Tests for :mod:`ucsa.training.metrics`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ucsa.training.metrics import (
    DEFAULT_METRIC_NAMES,
    MetricsRegistry,
    attention_entropy,
    build_default_registry,
    expert_utilization,
    flatten_metrics,
    gpu_memory_bytes,
    memory_replacement_rate,
    memory_utilization,
    perplexity_from_loss,
    throughput,
)


class TestMetricsRegistry:
    """Tests for :class:`MetricsRegistry`."""

    def test_default_names(self) -> None:
        """The default metric set has 12 entries."""
        assert len(DEFAULT_METRIC_NAMES) == 12

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