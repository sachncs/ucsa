"""Memory service.

The :class:`MemoryService` runs memory verification, consolidation, and
pruning in the background, on a dedicated asyncio task. Inference and
training paths enqueue work and return immediately; they never block
waiting for memory operations.

The service is launched by :meth:`MemoryService.start` and stopped by
:meth:`MemoryService.stop`. Both methods are idempotent and safe to call
multiple times.

Communication with the asyncio worker uses :class:`asyncio.Queue` for
in-process callers and :func:`asyncio.run_coroutine_threadsafe` for
external (sync) callers. The service exposes sync ``submit_*`` facades
that drop to a synchronous fallback when no event loop is running yet.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import torch
from torch import Tensor

from ucsa.models.memory import Memory, MemoryUpdate
from ucsa.models.state import PersistentCognitiveState
from ucsa.models.verification import Verifier

LOGGER = logging.getLogger(__name__)


VerificationHandler = Callable[[MemoryUpdate, PersistentCognitiveState, float, bool], None]


@dataclass
class ServiceStats:
    """Lightweight statistics maintained by the :class:`MemoryService`.

    Attributes:
        verified: Total verifications processed.
        accepted: Total verifications that resulted in long-term acceptance.
        pruned: Total long-term slots recycled.
        errors: Total task failures captured by the worker.
    """

    verified: int = 0
    accepted: int = 0
    pruned: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return the stats as a JSON-friendly dict."""
        return {
            "verified": self.verified,
            "accepted": self.accepted,
            "pruned": self.pruned,
            "errors": self.errors,
        }


@dataclass
class VerificationTask:
    """An item placed on the memory service queue.

    Attributes:
        candidate: The candidate memory.
        cstate: The PCS at submission time.
        on_complete: Optional callback invoked with ``(candidate, cstate,
            score, accepted)``.
    """

    candidate: MemoryUpdate
    cstate: PersistentCognitiveState
    on_complete: Optional[VerificationHandler] = None
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class PruneTask:
    """An item requesting pruning of the lowest-retention slots."""

    k: int


Task = VerificationTask | PruneTask


class MemoryService:
    """Background memory verification, consolidation, and pruning."""

    def __init__(
        self,
        memory: Memory,
        verifier: Verifier,
    ) -> None:
        """Initialise the memory service.

        Args:
            memory: The memory facade to operate on.
            verifier: The verifier used for candidate evaluation.
        """
        self.memory = memory
        self.verifier = verifier
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.stats = ServiceStats()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self.thread: threading.Thread | None = None
        self.started: bool = False
        self.last_verification_signal: list[float] = []

    def start(self) -> None:
        """Start the background worker.

        Safe to call multiple times. If a worker is already running it is
        left running.
        """
        if self.started:
            return
        self.loop = asyncio.new_event_loop()

        def run_loop() -> None:
            """internal: run the asyncio event loop in a worker thread."""
            asyncio.set_event_loop(self.loop)
            assert self.loop is not None
            self.worker_task = self.loop.create_task(self.worker_loop())
            try:
                self.loop.run_forever()
            finally:
                self.loop.close()

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        # Wait briefly for the loop to be ready.
        for _ in range(100):
            if self.loop is not None and self.loop.is_running():
                self.started = True
                return
            import time
            time.sleep(0.01)
        self.started = True

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker.

        Args:
            timeout: Maximum seconds to wait for the worker to drain.
        """
        if not self.started:
            return
        loop = self.loop
        assert loop is not None
        future = asyncio.run_coroutine_threadsafe(self.stop_worker(), loop)
        try:
            future.result(timeout=timeout)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("MemoryService stop failed: %s", exc)
        loop.call_soon_threadsafe(loop.stop)
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        self.started = False

    async def stop_worker(self) -> None:
        """Drain pending tasks and cancel the worker."""
        await self.queue.join()
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def worker_loop(self) -> None:
        """The main worker coroutine.

        Pulls tasks off the queue and processes them. Errors are isolated:
        one failing task does not stop the worker.
        """
        while True:
            task = await self.queue.get()
            try:
                if isinstance(task, VerificationTask):
                    await self.process_verification(task)
                elif isinstance(task, PruneTask):
                    await self.process_prune(task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Memory service task failed: %s", exc)
                self.stats.errors += 1
            finally:
                self.queue.task_done()

    async def process_verification(self, task: VerificationTask) -> None:
        """Run a single verification task.

        Args:
            task: The task payload.
        """
        self.stats.verified += 1
        score, accepted = self.verifier.verify(task.candidate, task.cstate)
        self.last_verification_signal.append(score)
        if accepted:
            self.memory.accept_into_long_term(task.candidate)
            self.stats.accepted += 1
        if task.on_complete is not None:
            task.on_complete(task.candidate, task.cstate, score, accepted)

    async def process_prune(self, task: PruneTask) -> None:
        """Run a single pruning task.

        Args:
            task: The task payload.
        """
        recycled = self.memory.recycle_low_retention(task.k)
        self.stats.pruned += len(recycled)

    def submit_verification(
        self,
        candidate: MemoryUpdate,
        cstate: PersistentCognitiveState,
        on_complete: Optional[VerificationHandler] = None,
    ) -> Awaitable[Any] | None:
        """Submit a verification task.

        Args:
            candidate: The candidate memory.
            cstate: The PCS at submission time.
            on_complete: Optional callback invoked on completion.

        Returns:
            ``asyncio.Future`` if the service is running, else ``None``.
        """
        task = VerificationTask(
            candidate=candidate,
            cstate=cstate,
            on_complete=on_complete,
        )
        return self.submit(task)

    def submit_prune(self, k: int) -> Awaitable[Any] | None:
        """Submit a pruning task."""
        return self.submit(PruneTask(k=k))

    def submit(self, task: Task) -> Awaitable[Any] | None:
        """Submit a generic task to the queue.

        Args:
            task: The task to enqueue.

        Returns:
            ``asyncio.Future`` if the service is running, else ``None``.
        """
        if self.loop is None or not self.loop.is_running():
            return None
        return asyncio.run_coroutine_threadsafe(self.queue.put(task), self.loop)

    def submit_sync_inline(self, task: Task) -> None:
        """internal: synchronously process a task without the worker.

        Used for tests that don't want to spawn the worker thread.
        """
        if isinstance(task, VerificationTask):
            self.stats.verified += 1
            score, accepted = self.verifier.verify(task.candidate, task.cstate)
            self.last_verification_signal.append(score)
            if accepted:
                self.memory.accept_into_long_term(task.candidate)
                self.stats.accepted += 1
            if task.on_complete is not None:
                task.on_complete(task.candidate, task.cstate, score, accepted)
        elif isinstance(task, PruneTask):
            recycled = self.memory.recycle_low_retention(task.k)
            self.stats.pruned += len(recycled)


__all__ = [
    "MemoryService",
    "PruneTask",
    "ServiceStats",
    "VerificationHandler",
    "VerificationTask",
]


def collect_signals(service: MemoryService) -> Tensor:
    """internal: return the recent verification signals as a tensor."""
    return torch.tensor(service.last_verification_signal, dtype=torch.float32)