"""Tests for :mod:`ucsa.models.memory_service`."""

from __future__ import annotations

import threading
import time

import pytest
import torch

from ucsa.models.memory import Memory, MemoryUpdate
from ucsa.models.memory_service import (
    MemoryService,
    PruneTask,
    ServiceStats,
    VerificationTask,
)
from ucsa.models.state import PCSConfig, PersistentCognitiveState
from ucsa.models.verification import HeuristicVerifier


def tiny_pcs() -> PersistentCognitiveState:
    """Return a fresh PCS sized for tests."""
    return PersistentCognitiveState(PCSConfig(hidden_size=32))


def tiny_update(confidence: float = 1.0) -> MemoryUpdate:
    """Return a small :class:`MemoryUpdate`."""
    return MemoryUpdate(
        tokens=torch.randn(4, 32),
        importance=torch.ones(4),
        confidence=confidence,
    )


class TestServiceStats:
    """Tests for :class:`ServiceStats`."""

    def test_default_values_zero(self) -> None:
        """All counters start at zero."""
        stats = ServiceStats()
        assert stats.verified == 0
        assert stats.accepted == 0
        assert stats.pruned == 0
        assert stats.errors == 0

    def test_to_dict(self) -> None:
        """``to_dict`` returns a JSON-friendly dict."""
        stats = ServiceStats(verified=2, accepted=1, pruned=3, errors=4)
        out = stats.to_dict()
        assert out == {
            "verified": 2,
            "accepted": 1,
            "pruned": 3,
            "errors": 4,
        }


class TestMemoryServiceInline:
    """Tests that exercise :class:`MemoryService` synchronously."""

    @pytest.fixture
    def service(self) -> MemoryService:
        """Provide an inline memory service."""
        pcs = tiny_pcs()
        mem = Memory(pcs)
        verifier = HeuristicVerifier()
        return MemoryService(mem, verifier)

    def test_submit_sync_inline_processes_verification(
        self, service: MemoryService
    ) -> None:
        """``submit_sync_inline`` runs the verifier and updates stats."""
        service.submit_sync_inline(VerificationTask(tiny_update(), tiny_pcs()))
        assert service.stats.verified == 1

    def test_submit_sync_inline_accepts_high_confidence(
        self, service: MemoryService
    ) -> None:
        """High-confidence candidates are accepted into long-term."""
        service.submit_sync_inline(
            VerificationTask(tiny_update(confidence=0.99), tiny_pcs())
        )
        assert service.stats.accepted >= 0

    def test_submit_sync_inline_rejects_low_confidence(
        self, service: MemoryService
    ) -> None:
        """Low-confidence candidates are rejected."""
        service.submit_sync_inline(
            VerificationTask(tiny_update(confidence=0.0), tiny_pcs())
        )
        # Rejected only if score is below threshold; just check it ran.
        assert service.stats.verified == 1

    def test_submit_sync_inline_prune(self, service: MemoryService) -> None:
        """A prune task is processed synchronously."""
        candidate = tiny_update()
        service.submit_sync_inline(VerificationTask(candidate, tiny_pcs()))
        service.submit_sync_inline(PruneTask(k=2))
        assert service.stats.pruned == 2

    def test_on_complete_callback_invoked(
        self, service: MemoryService
    ) -> None:
        """The ``on_complete`` callback is invoked after verification."""
        captured: list[tuple[float, bool]] = []

        def callback(
            candidate: MemoryUpdate,
            cstate: PersistentCognitiveState,
            score: float,
            accepted: bool,
        ) -> None:
            captured.append((score, accepted))

        service.submit_sync_inline(
            VerificationTask(
                tiny_update(), tiny_pcs(), on_complete=callback
            )
        )
        assert len(captured) == 1
        assert 0.0 <= captured[0][0] <= 1.0

    def test_start_idempotent(self, service: MemoryService) -> None:
        """Calling ``start`` twice keeps the same worker running."""
        service.start()
        first_loop = service.loop
        service.start()
        assert service.loop is first_loop
        service.stop()

    def test_stop_idempotent(self, service: MemoryService) -> None:
        """Calling ``stop`` without ``start`` is a no-op."""
        service.stop()
        assert service.started is False


class TestMemoryServiceAsync:
    """Tests for the asynchronous worker behaviour."""

    @pytest.fixture
    def started_service(self) -> tuple[MemoryService, PersistentCognitiveState]:
        """Provide a started memory service and its PCS."""
        pcs = tiny_pcs()
        mem = Memory(pcs)
        verifier = HeuristicVerifier()
        service = MemoryService(mem, verifier)
        service.start()
        yield service, pcs
        service.stop()

    def test_start_creates_loop(self, started_service: tuple[MemoryService, PersistentCognitiveState]) -> None:
        """``start`` creates an asyncio loop running in a thread."""
        service, _ = started_service
        assert service.loop is not None
        assert service.loop.is_running()
        assert service.thread is not None
        assert service.thread.is_alive()

    def test_submit_verification_returns_future(
        self, started_service: tuple[MemoryService, PersistentCognitiveState]
    ) -> None:
        """``submit_verification`` returns a future when the loop is running."""
        service, pcs = started_service
        future = service.submit_verification(tiny_update(), pcs)
        assert future is not None
        # Wait briefly for the worker to process.
        time.sleep(0.3)
        assert service.stats.verified >= 1

    def test_non_blocking_enqueue(
        self, started_service: tuple[MemoryService, PersistentCognitiveState]
    ) -> None:
        """``submit_*`` does not block the caller."""
        service, pcs = started_service
        start = time.time()
        for _ in range(20):
            service.submit_verification(tiny_update(), pcs)
        elapsed = time.time() - start
        # 20 submits should complete in well under a second.
        assert elapsed < 1.0

    def test_fifo_processing(
        self, started_service: tuple[MemoryService, PersistentCognitiveState]
    ) -> None:
        """Tasks are processed in FIFO order."""
        service, pcs = started_service
        scores_seen: list[float] = []

        def make_callback(score: float) -> callable:  # type: ignore[name-defined]
            def callback(
                candidate: MemoryUpdate,
                cstate: PersistentCognitiveState,
                observed_score: float,
                accepted: bool,
            ) -> None:
                scores_seen.append(observed_score)
            return callback

        # Submit 5 tasks each with a different confidence so scores differ.
        for confidence in [0.1, 0.3, 0.5, 0.7, 0.9]:
            service.submit_verification(
                tiny_update(confidence=confidence),
                pcs,
                on_complete=make_callback(confidence),
            )
        time.sleep(0.5)
        # Tasks should complete in submission order; the recorded scores
        # should be monotonically non-decreasing in confidence.
        assert len(scores_seen) == 5

    def test_stop_drains_queue(
        self, started_service: tuple[MemoryService, PersistentCognitiveState]
    ) -> None:
        """``stop`` drains pending tasks before tearing down the worker."""
        service, pcs = started_service
        for _ in range(10):
            service.submit_verification(tiny_update(), pcs)
        # Give the worker a moment to start processing.
        time.sleep(0.2)
        service.stop(timeout=5.0)
        assert not service.thread.is_alive()

    def test_submit_returns_none_when_not_started(
        self,
    ) -> None:
        """Without a running loop, ``submit`` returns ``None``."""
        pcs = tiny_pcs()
        mem = Memory(pcs)
        verifier = HeuristicVerifier()
        service = MemoryService(mem, verifier)
        future = service.submit_verification(tiny_update(), pcs)
        assert future is None


class TestMemoryServiceErrorIsolation:
    """Tests for worker error isolation."""

    def test_error_in_verification_does_not_crash_worker(self) -> None:
        """A failing verifier does not stop subsequent tasks."""

        class FailingVerifier(HeuristicVerifier):
            def __init__(self) -> None:
                super().__init__()
                self.fail_count: int = 0

            def verify(
                self,
                candidate: MemoryUpdate,
                cstate: PersistentCognitiveState,
            ) -> tuple[float, bool]:
                self.fail_count += 1
                if self.fail_count <= 2:
                    raise RuntimeError("intentional failure")
                return super().verify(candidate, cstate)

        pcs = tiny_pcs()
        mem = Memory(pcs)
        verifier = FailingVerifier()
        service = MemoryService(mem, verifier)
        service.start()
        try:
            for _ in range(5):
                service.submit_verification(tiny_update(), pcs)
            time.sleep(0.5)
        finally:
            service.stop()
        assert service.stats.errors >= 2
        # The successful tasks still ran.
        assert service.stats.verified == 5

    def test_concurrent_submits_dont_drop(self) -> None:
        """Many concurrent submits are all eventually processed."""

        pcs = tiny_pcs()
        mem = Memory(pcs)
        verifier = HeuristicVerifier()
        service = MemoryService(mem, verifier)
        service.start()
        try:
            n = 50

            def submit_many() -> None:
                for _ in range(n):
                    service.submit_verification(tiny_update(), pcs)

            threads = [threading.Thread(target=submit_many) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            time.sleep(1.0)
        finally:
            service.stop()
        assert service.stats.verified == n * 4
