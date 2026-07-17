"""Training metrics.

Implements the eleven metrics called out in the project specification:

- ``training_loss``     -- running loss across the training step.
- ``perplexity``        -- exp(training_loss).
- ``jepa_loss``         -- JEPA auxiliary loss.
- ``memory_utilization``-- fraction of long-term capacity in use.
- ``memory_retention``  -- mean retention score across long-term slots.
- ``memory_replacement_rate`` -- fraction of long-term slots recycled
  per unit time.
- ``reasoning_iterations`` -- number of iterations performed per forward
  pass.
- ``expert_utilization`` -- per-expert routing probability averaged over
  the batch.
- ``attention_entropy`` -- entropy of attention weights per layer.
- ``gpu_memory``        -- peak GPU memory in bytes (optional, zero on CPU).
- ``throughput``        -- tokens processed per second.
- ``inference_latency`` -- wall-clock seconds per inference step.

Each metric supports ``update``, ``reset``, ``value``, and
``tensorboard_log``.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class MetricState:
    """internal: holds the running state of a single metric."""

    running_sum: float = 0.0
    running_count: int = 0
    last_value: float = 0.0
    history: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    started_at: float = field(default_factory=time.time)


class MetricsRegistry:
    """Registry of named metrics with running statistics."""

    def __init__(self, names: Sequence[str]) -> None:
        """Initialise the registry.

        Args:
            names: Names of the metrics to track.
        """
        self.names: tuple[str, ...] = tuple(names)
        self.states: dict[str, MetricState] = {
            name: MetricState() for name in self.names
        }

    def reset(self) -> None:
        """Reset every metric's running sum, count, and history."""
        for name in self.names:
            self.states[name] = MetricState()

    def update(self, name: str, value: float) -> None:
        """Update a metric with a new value.

        Args:
            name: Metric name. Must be in ``self.names``.
            value: New sample.
        """
        if name not in self.states:
            raise KeyError(f"Unknown metric '{name}'.")
        state = self.states[name]
        state.running_sum += float(value)
        state.running_count += 1
        state.last_value = float(value)
        state.history.append(float(value))

    def value(self, name: str) -> float:
        """Return the running average of a metric.

        Args:
            name: Metric name.

        Returns:
            Running mean, or ``0.0`` if no samples have been recorded.
        """
        state = self.states[name]
        if state.running_count == 0:
            return 0.0
        return state.running_sum / state.running_count

    def last(self, name: str) -> float:
        """Return the most recent value of a metric."""
        return self.states[name].last_value

    def snapshot(self) -> dict[str, float]:
        """Return a snapshot of every metric's running average."""
        return {name: self.value(name) for name in self.names}

    def tensorboard_log(self, writer: object, step: int) -> None:
        """Log every metric to a TensorBoard writer.

        Args:
            writer: A ``tensorboardX.SummaryWriter`` (or compatible).
            step: Global step.
        """
        for name in self.names:
            writer.add_scalar(name, self.value(name), step)


def perplexity_from_loss(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    if loss <= 0.0:
        return 1.0
    return float(torch.exp(torch.tensor(loss)).item())


def expert_utilization(router_logits: Tensor) -> Tensor:
    """Compute per-expert utilisation from router logits.

    Args:
        router_logits: Tensor of shape ``(num_tokens, num_experts)``.

    Returns:
        Tensor of shape ``(num_experts,)`` with per-expert utilisation
        fractions in ``[0, 1]``.
    """
    if router_logits.numel() == 0:
        return torch.zeros(router_logits.shape[-1])
    probabilities = torch.softmax(router_logits, dim=-1)
    return probabilities.mean(dim=0)


def attention_entropy(attention_weights: Tensor) -> float:
    """Compute entropy of attention weights.

    Args:
        attention_weights: Tensor of shape
            ``(batch, heads, query, key)`` or ``(batch, query, key)``.

    Returns:
        Mean entropy across the query positions in nats.
    """
    if attention_weights.numel() == 0:
        return 0.0
    weights = attention_weights.clamp(min=1e-9)
    entropy = -(weights * weights.log()).sum(dim=-1)
    return float(entropy.mean().item())


def throughput(
    num_tokens: int, elapsed_seconds: float
) -> float:
    """Compute tokens-per-second throughput."""
    if elapsed_seconds <= 0.0:
        return 0.0
    return float(num_tokens) / elapsed_seconds


def gpu_memory_bytes(device: torch.device | None = None) -> int:
    """Return peak GPU memory in bytes. Returns 0 on CPU."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def memory_utilization(
    used: int, capacity: int
) -> float:
    """Compute memory utilisation in ``[0, 1]``."""
    if capacity <= 0:
        return 0.0
    return float(used) / float(capacity)


def memory_replacement_rate(
    num_recycled: int, total_slots: int
) -> float:
    """Compute replacement rate as recycled / total slots."""
    if total_slots <= 0:
        return 0.0
    return float(num_recycled) / float(total_slots)


__all__ = [
    "MetricState",
    "MetricsRegistry",
    "attention_entropy",
    "expert_utilization",
    "gpu_memory_bytes",
    "memory_replacement_rate",
    "memory_utilization",
    "perplexity_from_loss",
    "throughput",
]


DEFAULT_METRIC_NAMES: tuple[str, ...] = (
    "training_loss",
    "perplexity",
    "jepa_loss",
    "jepa_prediction",
    "jepa_gaussian_reg",
    "jepa_steps",
    "reconstruction_loss",
    "memory_loss",
    "router_loss",
    "memory_utilization",
    "memory_retention",
    "memory_replacement_rate",
    "reasoning_iterations",
    "expert_utilization",
    "attention_entropy",
    "gpu_memory",
    "throughput",
    "inference_latency",
)


def build_default_registry() -> MetricsRegistry:
    """internal: build a :class:`MetricsRegistry` with the default metric set."""
    return MetricsRegistry(DEFAULT_METRIC_NAMES)


def flatten_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """internal: ensure all metric values are plain floats."""
    return {name: float(value) for name, value in metrics.items()}
