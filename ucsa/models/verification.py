"""Verification pipeline.

A :class:`Verifier` is the gate between candidate memory and the long-term
bank. It receives a :class:`ucsa.models.memory.MemoryUpdate` and the current
PCS, returns a score in ``[0, 1]`` and an accept/reject decision.

Two implementations ship:

- :class:`HeuristicVerifier` -- a score-based blend of confidence, novelty,
  recency, and usage.
- :class:`LearnedVerifier` -- a small MLP head trained on the retention
  signal (whether the candidate was re-used in subsequent requests).

Both share the :class:`Verifier` interface.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from ucsa.models.memory import MemoryUpdate
    from ucsa.models.state import PersistentCognitiveState


class Verifier(nn.Module, abc.ABC):
    """Abstract verifier base class."""

    def __init__(self) -> None:
        """Initialise the verifier base."""
        super().__init__()

    @abc.abstractmethod
    def verify(
        self,
        candidate: "MemoryUpdate",
        cstate: "PersistentCognitiveState",
    ) -> tuple[float, bool]:
        """Score and decide whether to accept ``candidate``.

        Args:
            candidate: The proposed memory update.
            cstate: The current PCS.

        Returns:
            Tuple ``(score, accepted)``. ``score`` is in ``[0, 1]``;
            ``accepted`` is the verifier's decision.
        """

    @abc.abstractmethod
    def update_signal(
        self,
        candidate: "MemoryUpdate",
        cstate: "PersistentCognitiveState",
        was_used: bool,
    ) -> Tensor:
        """Update the verifier's training signal.

        Args:
            candidate: The candidate that was evaluated.
            cstate: The current PCS.
            was_used: Whether the candidate was re-accessed in subsequent
                requests.

        Returns:
            Scalar tensor loss for backpropagation.
        """


class HeuristicVerifier(Verifier):
    """Score-based verifier blending confidence, novelty, recency, usage.

    Final score: ``0.4 * confidence + 0.3 * novelty + 0.2 * recency
    + 0.1 * usage``. ``novelty`` is ``1 - cosine_similarity`` to the
    nearest long-term token. ``recency`` decays with age. ``usage`` is
    ``sigmoid(usage_count)``.
    """

    def __init__(
        self,
        confidence_weight: float = 0.4,
        novelty_weight: float = 0.3,
        recency_weight: float = 0.2,
        usage_weight: float = 0.1,
        acceptance_threshold: float = 0.5,
    ) -> None:
        """Initialise the heuristic verifier.

        Args:
            confidence_weight: Weight for the candidate's confidence term.
            novelty_weight: Weight for the novelty term.
            recency_weight: Weight for the recency term.
            usage_weight: Weight for the usage term.
            acceptance_threshold: Minimum score required for acceptance.
        """
        super().__init__()
        total = confidence_weight + novelty_weight + recency_weight + usage_weight
        if total <= 0.0:
            raise ValueError("At least one verifier weight must be positive.")
        self.confidence_weight = confidence_weight / total
        self.novelty_weight = novelty_weight / total
        self.recency_weight = recency_weight / total
        self.usage_weight = usage_weight / total
        self.acceptance_threshold = acceptance_threshold

    def verify(
        self,
        candidate: "MemoryUpdate",
        cstate: "PersistentCognitiveState",
    ) -> tuple[float, bool]:
        """Score the candidate against the current PCS.

        Args:
            candidate: The proposed memory update.
            cstate: The current PCS.

        Returns:
            Tuple ``(score, accepted)``.
        """
        long_term = cstate.get_bank("long_term")
        usage = getattr(cstate, "meta_usage_long_term")
        novelty = HeuristicVerifier.novelty(candidate.tokens, long_term, usage)
        recency = HeuristicVerifier.recency(candidate.importance)
        usage_score = HeuristicVerifier.usage_signal(candidate.importance)
        confidence = candidate.confidence
        score = (
            self.confidence_weight * confidence
            + self.novelty_weight * novelty
            + self.recency_weight * recency
            + self.usage_weight * usage_score
        )
        score = float(max(0.0, min(1.0, score)))
        return score, score >= self.acceptance_threshold

    def update_signal(
        self,
        candidate: "MemoryUpdate",
        cstate: "PersistentCognitiveState",
        was_used: bool,
    ) -> Tensor:
        """Heuristic verifier does not learn; returns a zero tensor."""
        del candidate, cstate, was_used
        return torch.zeros(())

    @staticmethod
    def novelty(
        candidate_tokens: Tensor,
        long_term: Tensor,
        usage: Tensor,
    ) -> float:
        """Compute average novelty of ``candidate_tokens`` against ``long_term``.

        A long-term bank with no used slots returns novelty ``1.0``.
        """
        used_mask = usage > 0
        if not used_mask.any():
            return 1.0
        used_long_term = long_term[used_mask]
        candidate_norm = torch.nn.functional.normalize(
            candidate_tokens, p=2, dim=-1
        )
        long_term_norm = torch.nn.functional.normalize(
            used_long_term, p=2, dim=-1
        )
        similarity = candidate_norm @ long_term_norm.T
        max_similarity = similarity.max(dim=-1).values
        novelty = (1.0 - max_similarity).mean()
        return float(novelty.clamp(0.0, 1.0).item())

    @staticmethod
    def recency(importance: Tensor) -> float:
        """Compute a recency score from importance values.

        A linear scaling of importance, normalised to ``[0, 1]`` via a
        sigmoid of mean importance. Newer/cleaner candidates carry
        higher importance and thus higher recency.
        """
        mean_importance = float(importance.mean().item())
        return float(torch.sigmoid(torch.tensor(mean_importance)).item())

    @staticmethod
    def usage_signal(importance: Tensor) -> float:
        """Compute a usage signal from importance values.

        Returns ``0.5 + 0.5 * sigmoid(mean(importance))`` to keep the term
        in ``[0.5, 1]``.
        """
        mean_importance = float(importance.mean().item())
        sig = float(torch.sigmoid(torch.tensor(mean_importance)).item())
        return 0.5 + 0.5 * sig


class LearnedVerifier(Verifier):
    """MLP-based verifier trained on the retention signal.

    The model is a 2-layer MLP that takes the concatenation of the
    candidate's pooled embedding and a PCS summary and outputs a single
    logit. Training signal comes from :meth:`update_signal`, which returns
    a binary cross-entropy loss.
    """

    def __init__(
        self,
        hidden_size: int,
        cstate_summary_size: int,
        acceptance_threshold: float = 0.5,
    ) -> None:
        """Initialise the learned verifier.

        Args:
            hidden_size: Size of the candidate hidden dimensionality.
            cstate_summary_size: Size of the PCS summary vector fed to the
                verifier alongside the candidate.
            acceptance_threshold: Decision threshold.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.cstate_summary_size = cstate_summary_size
        self.acceptance_threshold = acceptance_threshold
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + cstate_summary_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.summarizer = nn.Linear(hidden_size, cstate_summary_size)

    def pool_candidate(self, candidate: "MemoryUpdate") -> Tensor:
        """Mean-pool the candidate's tokens into a single vector.

        Args:
            candidate: The candidate.

        Returns:
            Tensor of shape ``(hidden_size,)``.
        """
        return candidate.tokens.mean(dim=0)

    def summarize_cstate(self, cstate: "PersistentCognitiveState") -> Tensor:
        """Compute a PCS summary vector.

        Args:
            cstate: The PCS.

        Returns:
            Tensor of shape ``(cstate_summary_size,)``.
        """
        all_tokens = cstate.get_all_tokens()
        pooled = all_tokens.mean(dim=0)
        return self.summarizer(pooled)

    def verify(
        self,
        candidate: "MemoryUpdate",
        cstate: "PersistentCognitiveState",
    ) -> tuple[float, bool]:
        """Score and decide using the MLP head.

        Args:
            candidate: The candidate.
            cstate: The PCS.

        Returns:
            Tuple ``(score, accepted)``.
        """
        with torch.no_grad():
            pooled = self.pool_candidate(candidate)
            summary = self.summarize_cstate(cstate)
            logit = self.mlp(torch.cat([pooled, summary], dim=-1))
            score = float(torch.sigmoid(logit).item())
        return score, score >= self.acceptance_threshold

    def update_signal(
        self,
        candidate: "MemoryUpdate",
        cstate: "PersistentCognitiveState",
        was_used: bool,
    ) -> Tensor:
        """Compute BCE loss against the retention signal.

        Args:
            candidate: The candidate.
            cstate: The PCS at the time of evaluation.
            was_used: Whether the candidate was re-accessed.

        Returns:
            Scalar tensor loss.
        """
        pooled = self.pool_candidate(candidate)
        summary = self.summarize_cstate(cstate)
        logit = self.mlp(torch.cat([pooled, summary], dim=-1))
        target = torch.tensor(
            1.0 if was_used else 0.0, device=logit.device
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit.squeeze(-1), target
        )
        return loss


__all__ = [
    "HeuristicVerifier",
    "LearnedVerifier",
    "Verifier",
]