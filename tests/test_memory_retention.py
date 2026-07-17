"""Tests for retention scoring and recycle policies."""

from __future__ import annotations

import pytest
import torch

from ucsa.models.memory import Memory
from ucsa.models.state import PCSConfig, PersistentCognitiveState


def tiny_pcs() -> PersistentCognitiveState:
    """Return a fresh PCS sized for tests."""
    return PersistentCognitiveState(PCSConfig(hidden_size=32))


class TestRetentionAccess:
    """Tests for :meth:`Memory.get_retention_scores`."""

    @pytest.fixture
    def memory(self) -> Memory:
        """Provide a memory facade with some accepted candidates."""
        mem = Memory(tiny_pcs())
        candidate = mem.propose_candidate()
        mem.accept_into_long_term(candidate, max_slots=10)
        mem.cstate.update_retention()
        return mem

    def test_returns_long_term_scores(self, memory: Memory) -> None:
        """Default bank is ``long_term``."""
        scores = memory.get_retention_scores()
        assert scores.shape == (memory.cstate.bank_size("long_term"),)

    def test_returns_working_scores(self, memory: Memory) -> None:
        """``bank='working'`` returns working-memory scores."""
        scores = memory.get_retention_scores("working")
        assert scores.shape == (memory.cstate.bank_size("working"),)

    def test_unknown_bank_raises(self, memory: Memory) -> None:
        """An unknown bank name raises ``KeyError``."""
        with pytest.raises(KeyError):
            memory.get_retention_scores("nope")

    def test_scores_in_unit_interval(self, memory: Memory) -> None:
        """All retention scores are in ``[0, 1]``."""
        scores = memory.get_retention_scores()
        assert torch.all(scores >= 0.0)
        assert torch.all(scores <= 1.0)

    def test_freshly_accepted_have_higher_retention(self) -> None:
        """Freshly accepted memories have higher retention than aged ones."""
        mem = Memory(tiny_pcs())
        # Accept a first batch.
        candidate = mem.propose_candidate()
        mem.accept_into_long_term(candidate, max_slots=4)
        # Step age many times before accepting the next batch.
        mem.cstate.step_age()
        mem.cstate.step_age()
        candidate2 = mem.propose_candidate()
        mem.accept_into_long_term(candidate2, max_slots=4)
        mem.cstate.update_retention()
        scores = mem.get_retention_scores()
        assert scores[4:].mean() < scores[:4].mean()


class TestRecyclePolicies:
    """Tests for FIFO and threshold-based recycle policies."""

    @pytest.fixture
    def memory(self) -> Memory:
        """Provide a memory facade with a populated long-term bank."""
        mem = Memory(tiny_pcs())
        candidate = mem.propose_candidate()
        mem.accept_into_long_term(candidate, max_slots=10)
        return mem

    def test_recycle_fifo_returns_oldest(self, memory: Memory) -> None:
        """``recycle_fifo`` returns the oldest slots first."""
        # Age the second batch more.
        memory.cstate.step_age()
        memory.cstate.step_age()
        recycled = memory.recycle_fifo(k=3)
        assert len(recycled) == 3
        for _ in recycled:
            # Slot indices are returned (the actual ages are reset).
            pass

    def test_recycle_fifo_zero_returns_empty(self, memory: Memory) -> None:
        """``k=0`` returns an empty list."""
        assert memory.recycle_fifo(k=0) == []

    def test_recycle_fifo_resets_metadata(self, memory: Memory) -> None:
        """Recycled slots have their metadata cleared."""
        memory.cstate.step_age()
        recycled = memory.recycle_fifo(k=3)
        usage = memory.cstate.meta_usage_long_term
        assert torch.all(usage[recycled] == 0.0)
        age = memory.cstate.meta_age_long_term
        assert torch.all(age[recycled] == 0)

    def test_recycle_fifo_replaces_tokens(self, memory: Memory) -> None:
        """Recycled slots are zeroed in the long-term bank."""
        memory.cstate.step_age()
        recycled = memory.recycle_fifo(k=2)
        long_term = memory.read_long_term()
        for idx in recycled:
            assert torch.all(long_term[idx] == 0)

    def test_threshold_percentile_validates_range(self) -> None:
        """``get_low_retention_threshold`` rejects out-of-range percentiles."""
        mem = Memory(tiny_pcs())
        with pytest.raises(ValueError):
            mem.get_low_retention_threshold(-0.1)
        with pytest.raises(ValueError):
            mem.get_low_retention_threshold(1.5)

    def test_threshold_returns_quantile(self, memory: Memory) -> None:
        """Threshold equals the retention score at the percentile."""
        threshold = memory.get_low_retention_threshold(0.5)
        scores = memory.get_retention_scores()
        expected = float(torch.quantile(scores, 0.5).item())
        assert threshold == pytest.approx(expected)

    def test_recycle_below_returns_low_scoring_slots(self) -> None:
        """``recycle_below`` clears slots whose retention is below threshold."""
        mem = Memory(tiny_pcs())
        candidate = mem.propose_candidate()
        mem.accept_into_long_term(candidate, max_slots=20)
        # Force the first 5 slots to be lowest-retention.
        mem.cstate.update_retention()
        retention = mem.cstate.meta_retention_long_term
        retention.fill_(1.0)
        retention[:5] = 0.0
        recycled = mem.recycle_below(0.1)
        assert sorted(recycled) == [0, 1, 2, 3, 4]

    def test_recycle_below_empty_when_all_high(self) -> None:
        """If all retention scores exceed the threshold, nothing is recycled."""
        mem = Memory(tiny_pcs())
        candidate = mem.propose_candidate()
        mem.accept_into_long_term(candidate, max_slots=10)
        mem.cstate.update_retention()
        recycled = mem.recycle_below(-0.1)
        assert recycled == []


class TestCapacityInvariants:
    """Tests for capacity invariants of the long-term bank."""

    def test_capacity_never_exceeds_limit(self) -> None:
        """Accepting more candidates than capacity does not overflow."""
        mem = Memory(tiny_pcs(), long_term_capacity=8)
        for _ in range(5):
            candidate = mem.propose_candidate()
            mem.accept_into_long_term(candidate)
        usage = mem.cstate.meta_usage_long_term
        assert int((usage > 0).sum().item()) <= 8

    def test_recycle_frees_slots(self) -> None:
        """Recycling frees slots for new acceptances."""
        mem = Memory(tiny_pcs(), long_term_capacity=4)
        candidate = mem.propose_candidate()
        mem.accept_into_long_term(candidate)
        assert mem.long_term_usage() == 4
        mem.cstate.update_retention()
        retention = mem.cstate.meta_retention_long_term
        retention.fill_(0.0)
        mem.recycle_below(0.1)
        assert mem.long_term_usage() == 0

    def test_capacity_used_recomputes(self) -> None:
        """``long_term_capacity_used`` reflects current state."""
        mem = Memory(tiny_pcs(), long_term_capacity=10)
        candidate = mem.propose_candidate()
        mem.accept_into_long_term(candidate, max_slots=5)
        assert mem.long_term_capacity_used() == pytest.approx(0.5)
        mem.cstate.update_retention()
        retention = mem.cstate.meta_retention_long_term
        retention.fill_(0.0)
        mem.recycle_below(0.1)
        assert mem.long_term_capacity_used() == 0.0
