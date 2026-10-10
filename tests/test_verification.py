"""Tests for :mod:`ucsa.models.verification`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ucsa.models import cognitive, tiers, verification
from ucsa.models.verification import Verifier


def tiny_pcs() -> cognitive.State:
    """Return a fresh PCS sized for tests."""
    return cognitive.State(cognitive.Config(hidden_size=32))


def tiny_update(
    tokens: Tensor | None = None,
    importance: Tensor | None = None,
    confidence: float = 1.0,
) -> tiers.Update:
    """Return a small :class:`tiers.Update`."""
    if tokens is None:
        tokens = torch.randn(4, 32)
    if importance is None:
        importance = torch.ones(tokens.shape[0])
    return tiers.Update(
        tokens=tokens,
        importance=importance,
        confidence=confidence,
        source="working",
    )


class TestVerifierABC:
    """Tests for the :class:`Verifier` ABC contract."""

    def test_abstract_methods_must_be_implemented(self) -> None:
        """A subclass missing abstract methods cannot be instantiated."""

        class IncompleteVerifier(Verifier):
            pass

        with pytest.raises(TypeError):
            IncompleteVerifier()  # type: ignore[abstract]

    def test_subclass_is_module(self) -> None:
        """A subclass is also a :class:`torch.nn.Module`."""
        verifier = verification.Heuristic()
        assert isinstance(verifier, Verifier)
        assert isinstance(verifier, torch.nn.Module)


class TestHeuristicVerifier:
    """Tests for :class:`verification.Heuristic`."""

    @pytest.fixture
    def verifier(self) -> verification.Heuristic:
        """Provide a default heuristic verifier."""
        return verification.Heuristic()

    def test_construction_default_weights(
        self, verifier: verification.Heuristic
    ) -> None:
        """Default weights are non-negative and sum to one."""
        total = (
            verifier.confidence_weight
            + verifier.novelty_weight
            + verifier.recency_weight
            + verifier.usage_weight
        )
        assert total == pytest.approx(1.0)

    def test_zero_weights_rejected(self) -> None:
        """All-zero weights are rejected."""
        with pytest.raises(ValueError):
            verification.Heuristic(
                confidence_weight=0.0,
                novelty_weight=0.0,
                recency_weight=0.0,
                usage_weight=0.0,
            )

    def test_verify_returns_score_and_decision(
        self, verifier: verification.Heuristic
    ) -> None:
        """``verify`` returns a (score, accept) tuple."""
        update = tiny_update(confidence=0.9)
        score, accept = verifier.verify(update, tiny_pcs())
        assert 0.0 <= score <= 1.0
        assert isinstance(accept, bool)

    def test_high_confidence_increases_score(
        self, verifier: verification.Heuristic
    ) -> None:
        """Higher confidence raises the score."""
        high = tiny_update(confidence=1.0)
        low = tiny_update(confidence=0.0)
        high_score, _ = verifier.verify(high, tiny_pcs())
        low_score, _ = verifier.verify(low, tiny_pcs())
        assert high_score > low_score

    def test_acceptance_threshold(self) -> None:
        """Custom threshold changes accept/reject decisions."""
        permissive = verification.Heuristic(acceptance_threshold=0.0)
        strict = verification.Heuristic(acceptance_threshold=1.5)
        update = tiny_update()
        _, accept_permissive = permissive.verify(update, tiny_pcs())
        _, accept_strict = strict.verify(update, tiny_pcs())
        assert accept_permissive is True
        assert accept_strict is False

    def test_novelty_high_when_long_term_empty(
        self, verifier: verification.Heuristic
    ) -> None:
        """With an empty long-term bank, novelty is 1.0."""
        novelty = verification.Heuristic.novelty(
            torch.randn(4, 32),
            torch.zeros(8, 32),
            torch.zeros(8),
        )
        assert novelty == pytest.approx(1.0)

    def test_novelty_low_for_duplicate(
        self, verifier: verification.Heuristic
    ) -> None:
        """A candidate identical to long-term memory has low novelty."""
        tokens = torch.randn(4, 32)
        novelty = verification.Heuristic.novelty(
            tokens,
            tokens,
            torch.ones(4),
        )
        assert novelty < 0.1

    def test_recency_in_unit_interval(
        self, verifier: verification.Heuristic
    ) -> None:
        """Recency is in ``[0, 1]``."""
        importance = torch.tensor([0.0, 1.0, 5.0, 10.0])
        recency = verification.Heuristic.recency(importance)
        assert 0.0 <= recency <= 1.0

    def test_usage_signal_above_half(
        self, verifier: verification.Heuristic
    ) -> None:
        """Usage signal is at least 0.5."""
        usage = verification.Heuristic.usage_signal(torch.tensor([1.0, 2.0]))
        assert usage >= 0.5

    def test_update_signal_returns_zero(
        self, verifier: verification.Heuristic
    ) -> None:
        """The heuristic verifier has no learned parameters; loss is zero."""
        loss = verifier.update_signal(tiny_update(), tiny_pcs(), was_used=True)
        assert loss.item() == 0.0

    def test_score_bounded_zero_one(
        self, verifier: verification.Heuristic
    ) -> None:
        """Output score is always in ``[0, 1]``."""
        for _ in range(20):
            update = tiny_update(
                tokens=torch.randn(8, 32) * 10,
                importance=torch.randn(8) * 10,
                confidence=torch.rand(1).item(),
            )
            score, _ = verifier.verify(update, tiny_pcs())
            assert 0.0 <= score <= 1.0


class TestLearnedVerifier:
    """Tests for :class:`verification.Learned`."""

    @pytest.fixture
    def verifier(self) -> verification.Learned:
        """Provide a default learned verifier."""
        return verification.Learned(hidden_size=32, cstate_summary_size=8)

    def test_construction(self, verifier: verification.Learned) -> None:
        """Constructed MLP has the expected shape."""
        assert isinstance(verifier.mlp, torch.nn.Sequential)

    def test_verify_returns_score_and_decision(
        self, verifier: verification.Learned
    ) -> None:
        """``verify`` returns a (score, accept) tuple."""
        score, accept = verifier.verify(tiny_update(), tiny_pcs())
        assert 0.0 <= score <= 1.0
        assert isinstance(accept, bool)

    def test_pool_candidate(self, verifier: verification.Learned) -> None:
        """``pool_candidate`` returns a vector of size ``hidden_size``."""
        pooled = verifier.pool_candidate(tiny_update())
        assert pooled.shape == (32,)

    def test_summarize_cstate(self, verifier: verification.Learned) -> None:
        """``summarize_cstate`` returns a vector of size ``cstate_summary_size``."""
        summary = verifier.summarize_cstate(tiny_pcs())
        assert summary.shape == (8,)

    def test_update_signal_returns_loss(
        self, verifier: verification.Learned
    ) -> None:
        """``update_signal`` returns a scalar loss tensor."""
        loss = verifier.update_signal(tiny_update(), tiny_pcs(), was_used=True)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_update_signal_negative_when_used(self) -> None:
        """Used candidates drive the loss toward the positive target."""
        torch.manual_seed(0)
        verifier = verification.Learned(hidden_size=32, cstate_summary_size=8)
        update = tiny_update()
        # Multiple updates with the same signal should lower the loss.
        first = verifier.update_signal(update, tiny_pcs(), was_used=True).item()
        for _ in range(10):
            verifier.update_signal(update, tiny_pcs(), was_used=True)
        last = verifier.update_signal(update, tiny_pcs(), was_used=True).item()
        assert last < first

    def test_gradient_flows_through_verifier(
        self, verifier: verification.Learned
    ) -> None:
        """Loss on update_signal flows gradients to MLP and summarizer."""
        loss = verifier.update_signal(tiny_update(), tiny_pcs(), was_used=True)
        loss.backward()
        for param in verifier.mlp.parameters():
            assert param.grad is not None
        for param in verifier.summarizer.parameters():
            assert param.grad is not None

    def test_no_grad_in_verify(self, verifier: verification.Learned) -> None:
        """``verify`` does not build a computation graph."""
        update = tiny_update()
        pcs = tiny_pcs()
        score, _ = verifier.verify(update, pcs)
        assert score == score  # NaN check
        # No gradient was retained.
        assert all(p.grad is None for p in verifier.parameters())

    def test_acceptance_threshold(
        self,
    ) -> None:
        """Custom threshold affects accept/reject decisions."""
        permissive = verification.Learned(
            hidden_size=32, cstate_summary_size=8, acceptance_threshold=0.0
        )
        strict = verification.Learned(
            hidden_size=32, cstate_summary_size=8, acceptance_threshold=1.5
        )
        update = tiny_update()
        pcs = tiny_pcs()
        _, accept_permissive = permissive.verify(update, pcs)
        _, accept_strict = strict.verify(update, pcs)
        assert accept_permissive is True
        assert accept_strict is False

    def test_trainable_parameters(self, verifier: verification.Learned) -> None:
        """The learned verifier exposes trainable parameters."""
        params = list(verifier.parameters())
        assert any(p.requires_grad for p in params)
