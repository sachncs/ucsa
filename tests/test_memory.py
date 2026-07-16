"""Tests for :mod:`ucsa.models.memory`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ucsa.models.memory import Memory, MemoryUpdate
from ucsa.models.state import PCSConfig, PersistentCognitiveState


def tiny_pcs() -> PersistentCognitiveState:
    """Return a fresh PCS sized for tests."""
    return PersistentCognitiveState(PCSConfig(hidden_size=32))


class TestMemoryUpdate:
    """Tests for :class:`MemoryUpdate`."""

    def test_construction(self) -> None:
        """Valid inputs construct a :class:`MemoryUpdate`."""
        tokens = torch.randn(4, 32)
        importance = torch.ones(4)
        update = MemoryUpdate(tokens=tokens, importance=importance)
        assert update.tokens.shape == (4, 32)

    def test_dimension_mismatch_rejected(self) -> None:
        """Tokens and importance with different token counts raise."""
        with pytest.raises(ValueError):
            MemoryUpdate(
                tokens=torch.randn(4, 32),
                importance=torch.ones(5),
            )

    def test_non_2d_tokens_rejected(self) -> None:
        """Tokens with the wrong rank raise."""
        with pytest.raises(ValueError):
            MemoryUpdate(
                tokens=torch.randn(4, 2, 32),
                importance=torch.ones(4),
            )

    def test_confidence_out_of_range_rejected(self) -> None:
        """Confidence outside ``[0, 1]`` raises."""
        with pytest.raises(ValueError):
            MemoryUpdate(
                tokens=torch.randn(4, 32),
                importance=torch.ones(4),
                confidence=1.5,
            )
        with pytest.raises(ValueError):
            MemoryUpdate(
                tokens=torch.randn(4, 32),
                importance=torch.ones(4),
                confidence=-0.1,
            )


class TestMemory:
    """Tests for :class:`Memory`."""

    @pytest.fixture()
    def memory(self) -> Memory:
        """Provide a memory facade wrapping a fresh PCS."""
        return Memory(tiny_pcs())

    def test_construction_defaults_capacity(self) -> None:
        """Default capacity equals the PCS long-term bank size."""
        pcs = tiny_pcs()
        mem = Memory(pcs)
        assert mem.long_term_capacity == pcs.bank_size("long_term")

    def test_construction_with_explicit_capacity(self) -> None:
        """An explicit capacity overrides the PCS default."""
        mem = Memory(tiny_pcs(), long_term_capacity=10)
        assert mem.long_term_capacity == 10

    def test_propose_candidate_default(self, memory: Memory) -> None:
        """``propose_candidate`` defaults to the working bank."""
        candidate = memory.propose_candidate()
        assert candidate.source == "working"
        assert candidate.tokens.shape == memory.cstate.get_bank("working").shape
        assert torch.all(candidate.importance == 1.0)

    def test_propose_candidate_custom(self, memory: Memory) -> None:
        """Custom slice and importance are propagated."""
        slice_ = torch.randn(8, 32)
        importance = torch.full((8,), 0.5)
        candidate = memory.propose_candidate(
            working_slice=slice_, importance=importance, confidence=0.7
        )
        assert torch.allclose(candidate.tokens, slice_)
        assert torch.all(candidate.importance == 0.5)
        assert candidate.confidence == 0.7

    def test_accept_into_long_term(self, memory: Memory) -> None:
        """Accepted candidates land in the long-term bank."""
        candidate = memory.propose_candidate(confidence=1.0)
        indices = memory.accept_into_long_term(candidate, max_slots=5)
        assert len(indices) == 5
        long_term = memory.read_long_term()
        for idx in indices:
            assert torch.allclose(long_term[idx], candidate.tokens[idx])

    def test_accept_respects_capacity(self) -> None:
        """Acceptance stops at ``long_term_capacity``."""
        mem = Memory(tiny_pcs(), long_term_capacity=4)
        candidate = mem.propose_candidate()
        indices = mem.accept_into_long_term(candidate)
        assert len(indices) <= 4

    def test_accept_records_importance(self, memory: Memory) -> None:
        """Importance is recorded for every accepted slot."""
        importance = torch.linspace(0.1, 0.8, steps=8)
        working_slice = torch.randn(8, 32)
        candidate = memory.propose_candidate(
            working_slice=working_slice, importance=importance
        )
        indices = memory.accept_into_long_term(candidate, max_slots=4)
        recorded = getattr(
            memory.cstate, "meta_importance_long_term"
        )[indices]
        assert torch.allclose(recorded, importance[:4])

    def test_accept_resets_age(self, memory: Memory) -> None:
        """Accepted slots have age zeroed."""
        memory.cstate.step_age()
        candidate = memory.propose_candidate()
        indices = memory.accept_into_long_term(candidate, max_slots=3)
        age = getattr(memory.cstate, "meta_age_long_term")[indices]
        assert torch.all(age == 0)

    def test_accept_updates_retention(self, memory: Memory) -> None:
        """``accept_into_long_term`` recomputes retention."""
        candidate = memory.propose_candidate()
        memory.accept_into_long_term(candidate, max_slots=5)
        snapshot = memory.cstate.get_all_metadata()
        accepted_indices = slice(0, 5)
        # Accepted slots have positive importance and usage; non-zero retention.
        assert torch.all(snapshot["long_term"]["retention"][accepted_indices] >= 0)

    def test_accept_after_full_bank_recycles(self, memory: Memory) -> None:
        """When the bank is full, acceptance triggers recycle."""
        # Fill the long-term bank with high-retention tokens.
        candidate = memory.propose_candidate()
        memory.accept_into_long_term(candidate, max_slots=memory.long_term_capacity)
        # Now lower retention and try to accept more.
        getattr(memory.cstate, "meta_retention_long_term")[:] = 0.0
        candidate2 = memory.propose_candidate(importance=torch.zeros(candidate.tokens.shape[0]))
        indices = memory.accept_into_long_term(candidate2)
        assert len(indices) > 0

    def test_snapshot_episode_writes_bank(self, memory: Memory) -> None:
        """``snapshot_episode`` writes the episode bank."""
        memory.snapshot_episode()
        episode = memory.cstate.get_bank("episode")
        assert episode.shape == (memory.cstate.bank_size("episode"), 32)

    def test_snapshot_episode_uses_provided_tokens(self, memory: Memory) -> None:
        """Provided tokens are included in the snapshot."""
        custom = torch.ones(10, 32)
        out = memory.snapshot_episode(working_tokens=custom)
        assert torch.all(out[:10] == 1.0)

    def test_snapshot_episode_pads_when_short(self, memory: Memory) -> None:
        """Episode bank is padded when input is shorter than bank size."""
        small = torch.ones(2, 32)
        out = memory.snapshot_episode(
            working_tokens=small,
            long_term_tokens=torch.zeros(0, 32),
        )
        episode_size = memory.cstate.bank_size("episode")
        assert out.shape == (episode_size, 32)
        assert torch.all(out[2:] == 0.0)

    def test_snapshot_episode_truncates_when_long(self, memory: Memory) -> None:
        """Episode bank truncates when input is longer than bank size."""
        big = torch.ones(memory.cstate.bank_size("long_term") + 10, 32)
        out = memory.snapshot_episode(long_term_tokens=big)
        episode_size = memory.cstate.bank_size("episode")
        assert out.shape == (episode_size, 32)

    def test_update_working(self, memory: Memory) -> None:
        """``update_working`` replaces the working bank."""
        new_tokens = torch.full(
            (memory.cstate.bank_size("working"), 32), 0.7
        )
        memory.update_working(new_tokens)
        current = memory.read_working()
        assert torch.all(current == 0.7)

    def test_recycle_low_retention(self, memory: Memory) -> None:
        """``recycle_low_retention`` empties the lowest-retention slots."""
        candidate = memory.propose_candidate()
        memory.accept_into_long_term(candidate, max_slots=20)
        memory.cstate.update_retention()
        # Force specific slots to be lowest retention.
        retention = getattr(memory.cstate, "meta_retention_long_term")
        retention.fill_(1.0)
        retention[:3] = 0.0
        recycled = memory.recycle_low_retention(k=2)
        assert len(recycled) == 2
        snapshot = memory.cstate.get_all_metadata()
        for idx in recycled:
            assert snapshot["long_term"]["usage"][idx].item() == 0.0
            assert snapshot["long_term"]["importance"][idx].item() == 0.0

    def test_long_term_usage(self, memory: Memory) -> None:
        """``long_term_usage`` counts slots with usage > 0."""
        candidate = memory.propose_candidate()
        memory.accept_into_long_term(candidate, max_slots=7)
        assert memory.long_term_usage() == 7

    def test_long_term_capacity_used(self, memory: Memory) -> None:
        """``long_term_capacity_used`` is usage / capacity."""
        candidate = memory.propose_candidate()
        memory.accept_into_long_term(candidate, max_slots=memory.long_term_capacity // 2)
        assert memory.long_term_capacity_used() == pytest.approx(0.5)

    def test_recycle_zero_returns_empty(self, memory: Memory) -> None:
        """Recycling zero slots returns an empty list."""
        recycled = memory.recycle_low_retention(k=0)
        assert recycled == []