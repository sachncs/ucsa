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

Plus the intent-collapse diagnostics, which answer one question: is the
origination generator actually *using* the intent bank, or has it learned to
ignore it so the reasoning loop quietly degenerates to feeding the same
observation every iteration while every loss still looks healthy? That
failure is silent in every other metric here, so it gets its own readings:

- ``intent_state_variance`` -- how much the origination state differs
  across differing inputs. Zero means one constant signal precedes every
  action.
- ``intent_gate_entropy``   -- entropy of the gate's slot usage. Zero means
  one slot wins everything; ``log(num_slots)`` means the gate is uniform and
  carries no information.
- ``intent_gate_mutual_info`` -- mutual information between which input was
  seen and which slots were gated. Zero is the collapse signature: the gate
  is not conditioning on the input.
- ``intent_read_share``     -- fraction of the generator's output magnitude
  contributed by the intent read rather than the working read.

Each metric supports ``update``, ``reset``, ``value``, and
``tensorboard_log``.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor


class SummaryWriterLike(Protocol):
    """The one method the metrics registry needs from a TB writer."""

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """Record a scalar under ``tag`` at ``step``."""


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

    def tensorboard_log(self, writer: SummaryWriterLike, step: int) -> None:
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


def throughput(num_tokens: int, elapsed_seconds: float) -> float:
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


def memory_utilization(used: int, capacity: int) -> float:
    """Compute memory utilisation in ``[0, 1]``."""
    if capacity <= 0:
        return 0.0
    return float(used) / float(capacity)


def memory_replacement_rate(num_recycled: int, total_slots: int) -> float:
    """Compute replacement rate as recycled / total slots."""
    if total_slots <= 0:
        return 0.0
    return float(num_recycled) / float(total_slots)


def intent_state_variance(states: Sequence[Tensor]) -> float:
    """Mean per-feature variance of the origination state across inputs.

    Args:
        states: One origination-state tensor per input, each of shape
            ``(intent_tokens, hidden_size)``.

    Returns:
        Mean variance across inputs, or ``0.0`` for fewer than two inputs.
        Exactly ``0.0`` means the same signal preceded every action, which
        is what a static bank gives you.
    """
    if len(states) < 2:
        return 0.0
    stacked = torch.stack([state.detach().float() for state in states])
    return float(stacked.var(dim=0, unbiased=False).mean().item())


def intent_gate_usage(gate_weights: Tensor, num_slots: int) -> Tensor:
    """Fraction of query tokens routed to each intent slot.

    Args:
        gate_weights: Gate weights of shape ``(..., num_slots)``.
        num_slots: Number of intent slots.

    Returns:
        Tensor of shape ``(num_slots,)`` summing to ``1.0`` when any slot
        was used.
    """
    if gate_weights.numel() == 0:
        return torch.zeros(num_slots)
    used = (gate_weights.detach() > 0).float().reshape(-1, num_slots)
    counts = used.sum(dim=0)
    total = counts.sum()
    if float(total) <= 0.0:
        return torch.zeros(num_slots)
    return counts / total


def intent_gate_entropy(usage: Tensor) -> float:
    """Entropy of the gate's slot-usage distribution, in nats.

    Args:
        usage: Slot-usage distribution of shape ``(num_slots,)``.

    Returns:
        Entropy in nats. ``0.0`` means a single slot absorbed every query;
        ``log(num_slots)`` means the gate is uniform and therefore carries
        no information about the input.
    """
    if usage.numel() == 0:
        return 0.0
    probabilities = usage.detach().float()
    support = probabilities > 0.0
    if not bool(support.any()):
        return 0.0
    kept = probabilities[support]
    return float(-(kept * kept.log()).sum().item())


def intent_gate_mutual_info(usages: Sequence[Tensor]) -> float:
    """Mutual information between input identity and gated slot, in nats.

    Treats each input as equally likely and each per-input usage vector as
    ``p(slot | input)``. ``I(input; slot) = H(slot) - E_input[H(slot |
    input)]``. Zero means the gate routes the same way whatever it is
    shown, which is the collapse signature: the origination is not
    conditioning on the situation.

    Args:
        usages: One usage distribution per input, from
            :func:`intent_gate_usage`.

    Returns:
        Mutual information in nats, or ``0.0`` for fewer than two inputs.
    """
    if len(usages) < 2:
        return 0.0
    stacked = torch.stack([usage.detach().float() for usage in usages])
    if float(stacked.sum()) <= 0.0:
        return 0.0
    marginal = stacked.mean(dim=0)
    conditional = float(
        sum(intent_gate_entropy(usage) for usage in usages) / len(usages)
    )
    return max(0.0, intent_gate_entropy(marginal) - conditional)


def intent_read_share(intent_read: Tensor, working_read: Tensor) -> float:
    """Fraction of the generator's read magnitude coming from intent.

    Args:
        intent_read: The gated intent read.
        working_read: The dense working-memory read.

    Returns:
        Value in ``[0, 1]``. Near ``0.0`` means the generator is driven by
        working memory and is ignoring the origination bank.
    """
    intent_norm = float(intent_read.detach().float().norm())
    working_norm = float(working_read.detach().float().norm())
    total = intent_norm + working_norm
    if total <= 0.0:
        return 0.0
    return intent_norm / total


__all__ = [
    "MetricState",
    "MetricsRegistry",
    "attention_entropy",
    "expert_utilization",
    "gpu_memory_bytes",
    "intent_gate_entropy",
    "intent_gate_mutual_info",
    "intent_gate_usage",
    "intent_read_share",
    "intent_state_variance",
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
    "intent_state_variance",
    "intent_gate_entropy",
    "intent_gate_mutual_info",
    "intent_read_share",
)


def build_default_registry() -> MetricsRegistry:
    """internal: build a :class:`MetricsRegistry` with the default metric set."""
    return MetricsRegistry(DEFAULT_METRIC_NAMES)


def flatten_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """internal: ensure all metric values are plain floats."""
    return {name: float(value) for name, value in metrics.items()}
