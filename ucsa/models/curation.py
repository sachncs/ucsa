"""Background curation of memory: verification, consolidation, pruning.

A `Curator` takes memory work off the training and inference paths. Callers
enqueue a task and return at once; a worker thread drains the queue in order.

Design:

* One code path. `Curator.handle` is the only place a task is carried out;
  the worker thread and direct synchronous calls both use it, so the two can
  never disagree.
* Failure isolation. An exception in one task (including in its callback) is
  logged and counted in `Stats.errors`; the worker keeps serving the queue.
* Restartability. Each `start` builds a fresh event loop and a queue bound to
  it, so `stop` followed by `start` always yields a working curator.
* No lost work. `stop` waits for every queued task before shutting down.
"""

import asyncio
import contextlib
import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import torch

from ucsa.models import cognitive, tiers, verification

LOGGER = logging.getLogger(__name__)

VerificationHandler = Callable[
    [tiers.Update, cognitive.State, float, bool], None
]

# How long `start` waits for the worker loop to begin running, in seconds.
START_TIMEOUT = 1.0


@dataclasses.dataclass
class Stats:
    """Counters maintained by a `Curator`.

    Attributes:
      verified: Verifications attempted, including ones that failed.
      accepted: Verifications that wrote a candidate into long-term memory.
      pruned: Long-term slots recycled.
      errors: Tasks (or callbacks) that raised inside the worker.
    """

    verified: int = 0
    accepted: int = 0
    pruned: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, int]:
        """Returns the counters as a JSON-friendly dict."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class VerifyTask:
    """A candidate memory to verify, and maybe accept.

    Attributes:
      candidate: The candidate memory.
      cstate: The cognitive state at submission time.
      on_complete: Optional callback `(candidate, cstate, score, accepted)`.
      trace: Free-form data carried with the task.
    """

    candidate: tiers.Update
    cstate: cognitive.State
    on_complete: VerificationHandler | None = None
    trace: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PruneTask:
    """A request to recycle the `k` lowest-retention long-term slots."""

    k: int


Task = VerifyTask | PruneTask


class Curator:
    """Runs memory verification and pruning, inline or on a worker thread."""

    def __init__(
        self, memory: tiers.Memory, verifier: verification.Verifier
    ) -> None:
        """Initialises a stopped curator.

        Args:
          memory: The memory facade to operate on.
          verifier: Decides whether a candidate is accepted.
        """
        self.memory = memory
        self.verifier = verifier
        self.stats = Stats()
        self.last_verification_signal: list[float] = []
        self.queue: asyncio.Queue[Task] | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self.thread: threading.Thread | None = None
        self.started = False

    def handle(self, task: Task) -> None:
        """Carries out one task synchronously.

        This is the single implementation used by the worker thread and by
        direct callers.

        Args:
          task: The task to carry out.

        Raises:
          TypeError: If `task` is not a `VerifyTask` or `PruneTask`.
          ValueError: If a `PruneTask` has a negative `k`.
        """
        if isinstance(task, VerifyTask):
            self.stats.verified += 1
            score, accepted = self.verifier.verify(task.candidate, task.cstate)
            self.last_verification_signal.append(score)
            if accepted:
                self.memory.accept_into_long_term(task.candidate)
                self.stats.accepted += 1
            if task.on_complete is not None:
                task.on_complete(task.candidate, task.cstate, score, accepted)
        elif isinstance(task, PruneTask):
            if task.k < 0:
                raise ValueError(f"k must be non-negative, got {task.k}.")
            self.stats.pruned += len(self.memory.recycle_low_retention(task.k))
        else:
            raise TypeError(f"unsupported task type: {type(task).__name__}")

    def start(self) -> None:
        """Starts the worker thread; does nothing if already running."""
        if self.started:
            return
        self.loop = asyncio.new_event_loop()
        loop = self.loop

        def run() -> None:
            asyncio.set_event_loop(loop)
            # The queue must be created on the loop that will use it.
            self.queue = asyncio.Queue()
            self.worker_task = loop.create_task(self.work())
            try:
                loop.run_forever()
            finally:
                loop.close()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            if loop.is_running() and self.queue is not None:
                break
            time.sleep(0.005)
        self.started = True

    def stop(self, timeout: float = 5.0) -> None:
        """Waits for queued work, then stops the worker thread.

        Args:
          timeout: Maximum seconds to wait for the queue and the thread.
        """
        if not self.started:
            return
        loop = self.loop
        assert loop is not None
        future = asyncio.run_coroutine_threadsafe(self.drain(), loop)
        try:
            future.result(timeout=timeout)
        except Exception as error:  # Shutting down must always complete.
            LOGGER.warning("Curator drain failed: %s", error)
        loop.call_soon_threadsafe(loop.stop)
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        self.started = False
        self.queue = self.worker_task = None

    async def drain(self) -> None:
        """Waits until every queued task is done, then cancels the worker."""
        assert self.queue is not None
        await self.queue.join()
        if self.worker_task is not None:
            self.worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker_task

    async def work(self) -> None:
        """Serves the queue forever; a failing task never stops it."""
        assert self.queue is not None
        while True:
            task = await self.queue.get()
            try:
                self.handle(task)
            except Exception as error:
                LOGGER.exception("Curator task failed: %s", error)
                self.stats.errors += 1
            finally:
                self.queue.task_done()

    def submit(self, task: Task) -> Future[None] | None:
        """Queues a task for the worker.

        Args:
          task: The task to queue.

        Returns:
          A future resolved once the task is queued, or None if the curator is
          not running (the task is not queued; use `handle` to run it inline).
        """
        if not self.started or self.loop is None or self.queue is None:
            return None
        return asyncio.run_coroutine_threadsafe(self.queue.put(task), self.loop)

    def submit_verification(
        self,
        candidate: tiers.Update,
        cstate: cognitive.State,
        on_complete: VerificationHandler | None = None,
    ) -> Future[None] | None:
        """Queues a verification; see `submit` for the return value."""
        return self.submit(VerifyTask(candidate, cstate, on_complete))

    def submit_prune(self, k: int) -> Future[None] | None:
        """Queues a pruning task; see `submit` for the return value."""
        return self.submit(PruneTask(k))


def collect_signals(curator: Curator) -> torch.Tensor:
    """Returns the verification scores recorded so far, in order."""
    return torch.tensor(curator.last_verification_signal, dtype=torch.float32)
