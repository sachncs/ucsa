"""Memory hierarchy.

UCSA's memory hierarchy has four levels:

================== =====================================================
Level              Description
================== =====================================================
Scratch            Local computation tokens; not persistently stored.
Working            Updated every forward pass by the reasoning loop.
Episode            Per-request context that survives the request lifetime.
Long-term          Accepted knowledge retained across many requests.
================== =====================================================

This module wraps :class:`PersistentCognitiveState` and exposes the
semantic operations of the hierarchy (propose candidate, accept into
long-term, snapshot to episode, recycle low-retention slots).

Memory updates NEVER occur inside transition blocks. They happen
explicitly via the methods here, typically driven by the background
:class:`ucsa.models.memory_service.MemoryService` worker.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ucsa.models.state import PersistentCognitiveState


@dataclass(frozen=True)
class MemoryUpdate:
    """A candidate memory proposed for the long-term bank.

    Attributes:
        tokens: Tensor of shape ``(num_tokens, hidden_size)``.
        importance: Per-token importance score.
        confidence: Optional scalar confidence in the candidate.
        source: Free-form provenance string ("working", "external", ...).
    """

    tokens: Tensor
    importance: Tensor
    confidence: float = 1.0
    source: str = "working"

    def __post_init__(self) -> None:
        if self.tokens.dim() != 2:
            raise ValueError(
                f"MemoryUpdate tokens must be 2D, got shape "
                f"{tuple(self.tokens.shape)}."
            )
        if self.tokens.shape[0] != self.importance.shape[0]:
            raise ValueError(
                f"MemoryUpdate tokens ({self.tokens.shape[0]}) and "
                f"importance ({self.importance.shape[0]}) must agree on "
                f"the token count."
            )
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}."
            )


class Memory:
    """Hierarchical memory facade over :class:`PersistentCognitiveState`.

    The Memory class exposes semantic operations (snapshot, propose, accept,
    recycle) but delegates bank storage and retention math to the PCS.
    """

    def __init__(
        self,
        cstate: PersistentCognitiveState,
        long_term_capacity: int | None = None,
    ) -> None:
        """Initialise the memory facade.

        Args:
            cstate: The PCS this memory facade wraps.
            long_term_capacity: Maximum number of long-term tokens. Defaults
                to the PCS long-term bank size.
        """
        self.cstate = cstate
        if long_term_capacity is None:
            long_term_capacity = cstate.bank_size("long_term")
        self.long_term_capacity = long_term_capacity

    def snapshot_episode(
        self,
        working_tokens: Tensor | None = None,
        long_term_tokens: Tensor | None = None,
    ) -> Tensor:
        """Copy a slice of working/long-term into the episode bank.

        Args:
            working_tokens: Optional working-memory tokens to include.
                Defaults to the current working bank.
            long_term_tokens: Optional long-term tokens to include. Pass
                an empty tensor (shape ``(0, hidden)``) to disable.

        Returns:
            Tensor stored in the episode bank.
        """
        if working_tokens is None:
            working_tokens = self.cstate.get_bank("working")
        if long_term_tokens is None:
            long_term_tokens = self.cstate.get_bank("long_term")
        if long_term_tokens.shape[0] == 0:
            combined = working_tokens
        else:
            combined = torch.cat([working_tokens, long_term_tokens], dim=0)
        episode_size = self.cstate.bank_size("episode")
        if combined.shape[0] >= episode_size:
            combined = combined[:episode_size]
        else:
            pad = torch.zeros(
                episode_size - combined.shape[0],
                combined.shape[1],
                device=combined.device,
                dtype=combined.dtype,
            )
            combined = torch.cat([combined, pad], dim=0)
        self.cstate.set_bank("episode", combined)
        return combined

    def propose_candidate(
        self,
        working_slice: Tensor | None = None,
        importance: Tensor | None = None,
        confidence: float = 1.0,
    ) -> MemoryUpdate:
        """Propose a memory update derived from working memory.

        Args:
            working_slice: Optional custom token slice. Defaults to the
                current working bank.
            importance: Optional per-token importance. Defaults to unit
                importance for every token.
            confidence: Confidence in the proposal.

        Returns:
            A :class:`MemoryUpdate` candidate.
        """
        if working_slice is None:
            working_slice = self.cstate.get_bank("working")
        if importance is None:
            importance = torch.ones(working_slice.shape[0])
        return MemoryUpdate(
            tokens=working_slice.detach().clone(),
            importance=importance.detach().clone(),
            confidence=confidence,
            source="working",
        )

    def accept_into_long_term(
        self,
        candidate: MemoryUpdate,
        max_slots: int | None = None,
    ) -> list[int]:
        """Accept a candidate into the long-term bank.

        Slots are filled in order. Importance is recorded; age is reset.
        Retention is recomputed afterwards.

        Args:
            candidate: The candidate to accept.
            max_slots: Optional cap on slots to write. Defaults to the
                candidate's token count or the bank's remaining capacity,
                whichever is smaller.

        Returns:
            List of slot indices that were written.
        """
        long_term = self.cstate.get_bank("long_term")
        capacity = self.long_term_capacity
        usage_buffer = self.cstate.metadata("long_term", "usage")
        already_used = int((usage_buffer > 0).sum().item())
        remaining = max(0, capacity - already_used)
        k = candidate.tokens.shape[0]
        if max_slots is not None:
            k = min(k, max_slots)
        k = min(k, remaining)
        if k <= 0:
            return []

        tokens_to_write = candidate.tokens[:k].to(
            device=long_term.device, dtype=long_term.dtype
        )
        importance_to_write = candidate.importance[:k].to(
            device=long_term.device,
            dtype=self.cstate.metadata("long_term", "importance").dtype,
        )
        empty = self._empty_long_term_indices(limit=k)
        empty_indices = torch.tensor(empty, dtype=torch.long)
        if len(empty) < k:
            # Recycle the lowest-retention slots to make room.
            recycled = self.cstate.recycle_bottom_k("long_term", k - len(empty))
            empty_indices = torch.cat([empty_indices, recycled])[:k]

        with torch.no_grad():
            long_term[empty_indices] = tokens_to_write
            self.cstate.metadata("long_term", "importance")[
                empty_indices
            ] = importance_to_write
            self.cstate.metadata("long_term", "age")[empty_indices] = 0
            self.cstate.metadata("long_term", "usage")[empty_indices] = 1.0
        self.cstate.update_retention()
        return [int(i) for i in empty_indices]

    def _empty_long_term_indices(self, limit: int) -> list[int]:
        """Return up to ``limit`` long-term slots with zero usage."""
        usage = self.cstate.metadata("long_term", "usage")
        empty = torch.nonzero(usage == 0, as_tuple=False).squeeze(-1)
        return [int(i) for i in empty[:limit].tolist()]

    def read_long_term(self) -> Tensor:
        """Return the long-term bank contents."""
        return self.cstate.get_bank("long_term")

    def update_working(self, new_tokens: Tensor) -> None:
        """Replace the working-memory bank with ``new_tokens``.

        Args:
            new_tokens: Tensor of shape
                ``(working_size, hidden_size)``.
        """
        self.cstate.set_bank("working", new_tokens)

    def read_working(self) -> Tensor:
        """Return the working-memory bank contents."""
        return self.cstate.get_bank("working")

    def recycle_low_retention(self, k: int) -> list[int]:
        """Recycle the ``k`` lowest-retention long-term slots.

        Args:
            k: Number of slots to recycle.

        Returns:
            Indices of recycled slots.
        """
        self.cstate.update_retention()
        recycled = self.cstate.recycle_bottom_k("long_term", k)
        return [int(i) for i in recycled.tolist()]

    def long_term_usage(self) -> int:
        """Return the number of used long-term slots."""
        usage = self.cstate.metadata("long_term", "usage")
        return int((usage > 0).sum().item())

    def long_term_capacity_used(self) -> float:
        """Return the fraction of long-term capacity in use."""
        return self.long_term_usage() / max(1, self.long_term_capacity)

    def get_retention_scores(self, bank: str = "long_term") -> Tensor:
        """Return the retention scores of every slot in ``bank``.

        Args:
            bank: Bank identifier.

        Returns:
            Tensor of shape ``(num_tokens,)`` with retention scores.
        """
        if bank not in self.cstate.bank_specs:
            raise KeyError(f"Unknown bank '{bank}'.")
        self.cstate.update_retention()
        return self.cstate.metadata(bank, "retention").clone()

    def recycle_fifo(self, k: int) -> list[int]:
        """Recycle the ``k`` oldest (highest-age) slots in long-term.

        Recycles by age, regardless of retention score. Used when a FIFO
        policy is preferred over retention-based recycling.

        Args:
            k: Number of slots to recycle.

        Returns:
            Indices of recycled slots, oldest first.
        """
        if k <= 0:
            return []
        age = self.cstate.metadata("long_term", "age")
        num_tokens = age.shape[0]
        k_eff = min(k, num_tokens)
        _, indices = torch.topk(age.float(), k_eff, largest=True)
        target = self.cstate.get_bank("long_term")
        replacement = torch.zeros(
            k_eff,
            self.cstate.config.hidden_size,
            device=target.device,
            dtype=target.dtype,
        )
        with torch.no_grad():
            target[indices] = replacement
            for field in ("importance", "usage", "age", "retention"):
                buffer = getattr(self.cstate, f"meta_{field}_long_term")
                buffer[indices] = 0.0 if field != "age" else 0
        return [int(i) for i in indices.tolist()]

    def get_low_retention_threshold(self, percentile: float) -> float:
        """Return the retention-score percentile used as recycle cutoff.

        Args:
            percentile: Percentile in ``[0, 1]``.

        Returns:
            Scalar threshold.
        """
        if not 0.0 <= percentile <= 1.0:
            raise ValueError(f"percentile must be in [0, 1], got {percentile}.")
        scores = self.get_retention_scores("long_term")
        if scores.numel() == 0:
            return 0.0
        return float(torch.quantile(scores, percentile).item())

    def recycle_below(self, threshold: float) -> list[int]:
        """Recycle every long-term slot whose retention score is below
        ``threshold``.

        Args:
            threshold: Retention score cutoff.

        Returns:
            Indices of recycled slots.
        """
        scores = self.cstate.metadata("long_term", "retention")
        mask = scores < threshold
        if not mask.any():
            return []
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        target = self.cstate.get_bank("long_term")
        replacement = torch.zeros(
            indices.shape[0],
            self.cstate.config.hidden_size,
            device=target.device,
            dtype=target.dtype,
        )
        with torch.no_grad():
            target[indices] = replacement
            for field in ("importance", "usage", "age", "retention"):
                buffer = getattr(self.cstate, f"meta_{field}_long_term")
                buffer[indices] = 0.0 if field != "age" else 0
        return [int(i) for i in indices.tolist()]


__all__ = ["Memory", "MemoryUpdate"]
