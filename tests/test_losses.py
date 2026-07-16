"""Tests for :mod:`ucsa.models.losses`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ucsa.models.losses import (
    AutoregressiveLoss,
    JEPALoss,
    LossWeights,
    MemoryStabilityLoss,
    RouterLoadBalancingLoss,
    UCSACombinedLoss,
)


class TestLossWeights:
    """Tests for :class:`LossWeights`."""

    def test_default_weights_valid(self) -> None:
        """Defaults are non-negative."""
        weights = LossWeights()
        assert weights.jepa >= 0
        assert weights.memory >= 0
        assert weights.router >= 0

    def test_negative_jepa_rejected(self) -> None:
        """Negative ``jepa`` is rejected."""
        with pytest.raises(ValueError):
            LossWeights(jepa=-0.1)

    def test_negative_memory_rejected(self) -> None:
        """Negative ``memory`` is rejected."""
        with pytest.raises(ValueError):
            LossWeights(memory=-0.1)

    def test_negative_router_rejected(self) -> None:
        """Negative ``router`` is rejected."""
        with pytest.raises(ValueError):
            LossWeights(router=-0.1)


class TestAutoregressiveLoss:
    """Tests for :class:`AutoregressiveLoss`."""

    def test_loss_finite(self) -> None:
        """Loss is finite for random logits."""
        loss_fn = AutoregressiveLoss()
        logits = torch.randn(2, 5, 100)
        targets = torch.randint(0, 100, (2, 5))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_loss_zero_for_perfect_predictions(self) -> None:
        """A perfect prediction has very low loss."""
        loss_fn = AutoregressiveLoss()
        targets = torch.tensor([[0, 1, 2]])
        logits = torch.full((1, 3, 100), -10.0)
        logits[0, 0, 0] = 10.0
        logits[0, 1, 1] = 10.0
        logits[0, 2, 2] = 10.0
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_ignore_index(self) -> None:
        """``ignore_index`` excludes those targets from the loss."""
        loss_fn = AutoregressiveLoss(ignore_index=-100)
        logits = torch.randn(1, 4, 10)
        targets = torch.tensor([[0, -100, 1, -100]])
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_gradient_flows(self) -> None:
        """Loss flows gradients to the logits."""
        loss_fn = AutoregressiveLoss()
        logits = torch.randn(1, 3, 10, requires_grad=True)
        targets = torch.tensor([[0, 1, 2]])
        loss_fn(logits, targets).backward()
        assert logits.grad is not None


class TestJEPALoss:
    """Tests for :class:`JEPALoss`."""

    def test_zero_for_perfect_match(self) -> None:
        """Loss is zero when predicted equals target."""
        loss_fn = JEPALoss()
        x = torch.randn(2, 4, 32)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_increases_with_distance(self) -> None:
        """Farther predictions produce larger loss."""
        loss_fn = JEPALoss()
        target = torch.zeros(1, 1, 32)
        close = 0.01 * torch.randn(1, 1, 32)
        far = 5.0 * torch.randn(1, 1, 32)
        assert loss_fn(close, target).item() < loss_fn(far, target).item()

    def test_invalid_alpha(self) -> None:
        """``alpha`` outside ``[0, 1]`` is rejected."""
        with pytest.raises(ValueError):
            JEPALoss(alpha=-0.1)
        with pytest.raises(ValueError):
            JEPALoss(alpha=1.5)

    def test_shape_mismatch_raises(self) -> None:
        """Shape mismatch between predicted and target raises."""
        loss_fn = JEPALoss()
        with pytest.raises(ValueError):
            loss_fn(torch.randn(2, 4, 32), torch.randn(2, 5, 32))

    def test_gradient_flows(self) -> None:
        """Loss flows gradients to the predicted embeddings."""
        loss_fn = JEPALoss()
        pred = torch.randn(2, 4, 32, requires_grad=True)
        target = torch.randn(2, 4, 32)
        loss_fn(pred, target).backward()
        assert pred.grad is not None


class TestMemoryStabilityLoss:
    """Tests for :class:`MemoryStabilityLoss`."""

    def test_zero_when_no_drift(self) -> None:
        """Loss is zero when ``long_term`` equals the baseline."""
        loss_fn = MemoryStabilityLoss()
        x = torch.randn(8, 32)
        loss = loss_fn(x, baseline=x)
        assert loss.item() == 0.0

    def test_loss_with_detached_baseline(self) -> None:
        """Default baseline is a detached copy."""
        loss_fn = MemoryStabilityLoss()
        x = torch.randn(8, 32, requires_grad=True)
        loss = loss_fn(x)
        assert torch.isfinite(loss)
        loss.backward()
        assert x.grad is not None


class TestRouterLoadBalancingLoss:
    """Tests for :class:`RouterLoadBalancingLoss`."""

    def test_balanced_routing_has_loss_one(self) -> None:
        """A perfectly balanced routing yields loss = 1.0."""
        loss_fn = RouterLoadBalancingLoss()
        # Construct logits where each expert is the argmax of exactly one
        # token, giving a perfectly balanced assignment.
        router_logits = torch.full((4, 4), -10.0)
        for i in range(4):
            router_logits[i, i] = 10.0
        loss = loss_fn(router_logits)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_loss_positive_for_collapsed_routing(self) -> None:
        """Collapsed routing (all to one expert) gives a positive loss."""
        loss_fn = RouterLoadBalancingLoss()
        router_logits = torch.full((16, 4), -10.0)
        router_logits[:, 0] = 10.0
        loss = loss_fn(router_logits)
        assert loss.item() > 0.0

    def test_empty_routing_returns_zero(self) -> None:
        """Zero tokens yields zero loss."""
        loss_fn = RouterLoadBalancingLoss()
        router_logits = torch.zeros(0, 4)
        loss = loss_fn(router_logits)
        assert loss.item() == 0.0

    def test_gradient_flows(self) -> None:
        """Loss flows gradients to router logits."""
        loss_fn = RouterLoadBalancingLoss()
        router_logits = torch.randn(8, 4, requires_grad=True)
        loss_fn(router_logits).backward()
        assert router_logits.grad is not None


class TestUCSACombinedLoss:
    """Tests for :class:`UCSACombinedLoss`."""

    def test_ar_only(self) -> None:
        """Without auxiliary inputs, only AR loss contributes."""
        loss_fn = UCSACombinedLoss()
        logits = torch.randn(2, 5, 100)
        targets = torch.randint(0, 100, (2, 5))
        total, comps = loss_fn(logits, targets)
        assert "ar" in comps
        assert "jepa" not in comps
        assert "memory" not in comps
        assert "router" not in comps
        assert torch.allclose(total, torch.tensor(comps["ar"]))

    def test_with_jepa(self) -> None:
        """JEPA auxiliary is added when both tensors are provided."""
        loss_fn = UCSACombinedLoss(LossWeights(jepa=0.5))
        logits = torch.randn(2, 5, 100)
        targets = torch.randint(0, 100, (2, 5))
        jepa_pred = torch.randn(2, 5, 32)
        jepa_target = torch.randn(2, 5, 32)
        total, comps = loss_fn(
            logits, targets, jepa_predicted=jepa_pred, jepa_target=jepa_target
        )
        assert "jepa" in comps
        expected = comps["ar"] + 0.5 * comps["jepa"]
        assert torch.allclose(total, torch.tensor(expected), atol=1e-4)

    def test_with_memory(self) -> None:
        """Memory stability is added when ``long_term`` is provided."""
        loss_fn = UCSACombinedLoss(LossWeights(memory=0.1))
        logits = torch.randn(2, 5, 100)
        targets = torch.randint(0, 100, (2, 5))
        long_term = torch.randn(8, 32)
        total, comps = loss_fn(logits, targets, long_term=long_term)
        assert "memory" in comps

    def test_with_router(self) -> None:
        """Router loss is added when ``router_logits`` is provided."""
        loss_fn = UCSACombinedLoss(LossWeights(router=0.01))
        logits = torch.randn(2, 5, 100)
        targets = torch.randint(0, 100, (2, 5))
        router_logits = torch.randn(8, 4)
        total, comps = loss_fn(logits, targets, router_logits=router_logits)
        assert "router" in comps

    def test_full_combination(self) -> None:
        """All four components contribute when all inputs are provided."""
        loss_fn = UCSACombinedLoss()
        logits = torch.randn(2, 5, 100)
        targets = torch.randint(0, 100, (2, 5))
        jepa_pred = torch.randn(2, 5, 32)
        jepa_target = torch.randn(2, 5, 32)
        long_term = torch.randn(8, 32)
        router_logits = torch.randn(8, 4)
        total, comps = loss_fn(
            logits,
            targets,
            jepa_predicted=jepa_pred,
            jepa_target=jepa_target,
            long_term=long_term,
            router_logits=router_logits,
        )
        assert set(comps) == {"ar", "jepa", "memory", "router"}
        assert torch.isfinite(total)

    def test_gradient_flows(self) -> None:
        """Backward pass through the combined loss propagates to all inputs."""
        loss_fn = UCSACombinedLoss()
        logits = torch.randn(2, 5, 100, requires_grad=True)
        targets = torch.randint(0, 100, (2, 5))
        jepa_pred = torch.randn(2, 5, 32, requires_grad=True)
        jepa_target = torch.randn(2, 5, 32)
        long_term = torch.randn(8, 32, requires_grad=True)
        router_logits = torch.randn(8, 4, requires_grad=True)
        total, _ = loss_fn(
            logits,
            targets,
            jepa_predicted=jepa_pred,
            jepa_target=jepa_target,
            long_term=long_term,
            router_logits=router_logits,
        )
        total.backward()
        assert logits.grad is not None
        assert jepa_pred.grad is not None
        assert long_term.grad is not None
        assert router_logits.grad is not None