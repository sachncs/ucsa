"""Guarantees of the memory curator.

Grouped by what is promised, not by method: one code path for every task,
deterministic effects, rejection of misuse, isolation of failures, recovery
across restarts, and behaviour under concurrency.
"""

import threading
import time

import pytest
import torch

from ucsa.models import cognitive, curation, tiers, verification


def make_state() -> cognitive.State:
    torch.manual_seed(0)  # the banks start random; make them reproducible
    return cognitive.State(cognitive.Config(hidden_size=32))


def make_update(confidence: float = 1.0, seed: int = 0) -> tiers.Update:
    gen = torch.Generator().manual_seed(seed)
    return tiers.Update(
        tokens=torch.randn(4, 32, generator=gen),
        importance=torch.ones(4),
        confidence=confidence,
    )


def make_curator(verifier=None) -> curation.Curator:
    return curation.Curator(
        tiers.Memory(make_state()), verifier or verification.Heuristic()
    )


class AcceptAll(verification.Verifier):
    """Accepts everything with a fixed score."""

    def verify(self, candidate, state):
        return 1.0, True

    def update_signal(self, *args, **kwargs):
        return None


class Rejecting(AcceptAll):
    def verify(self, candidate, state):
        return 0.0, False


class Exploding(AcceptAll):
    """Raises on the second call, succeeds otherwise."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def verify(self, candidate, state):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")
        return 1.0, True


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def running():
    curator = make_curator(AcceptAll())
    curator.start()
    yield curator
    curator.stop()


# ----------------------------------------------------------- deterministic flow


class TestDeterministicFlow:
    def test_stats_start_at_zero(self):
        assert make_curator().stats.to_dict() == {
            "verified": 0,
            "accepted": 0,
            "pruned": 0,
            "errors": 0,
        }

    def test_accepted_candidates_are_counted_and_written(self):
        curator = make_curator(AcceptAll())
        before = curator.memory.cstate.get_bank("long_term").clone()
        curator.handle(curation.VerifyTask(make_update(), make_state()))
        assert curator.stats.verified == 1
        assert curator.stats.accepted == 1
        assert not torch.equal(
            curator.memory.cstate.get_bank("long_term"), before
        )

    def test_rejected_candidates_leave_memory_untouched(self):
        curator = make_curator(Rejecting())
        before = curator.memory.cstate.get_bank("long_term").clone()
        curator.handle(curation.VerifyTask(make_update(), make_state()))
        assert curator.stats.verified == 1
        assert curator.stats.accepted == 0
        assert torch.equal(curator.memory.cstate.get_bank("long_term"), before)

    def test_callback_receives_candidate_score_and_decision(self):
        seen = []
        curator = make_curator(AcceptAll())
        update, state = make_update(), make_state()
        curator.handle(
            curation.VerifyTask(update, state, lambda *a: seen.append(a))
        )
        assert seen == [(update, state, 1.0, True)]

    def test_identical_task_sequences_give_identical_memory(self):
        def run():
            curator = make_curator(AcceptAll())
            for seed in range(5):
                curator.handle(
                    curation.VerifyTask(make_update(seed=seed), make_state())
                )
            return curator.memory.cstate.get_bank("long_term").clone()

        assert torch.equal(run(), run())

    def test_pruning_reports_how_many_slots_it_recycled(self):
        curator = make_curator(AcceptAll())
        curator.handle(curation.VerifyTask(make_update(), make_state()))
        curator.handle(curation.PruneTask(k=2))
        assert curator.stats.pruned == 2

    def test_signals_are_recorded_in_submission_order(self):
        curator = make_curator(AcceptAll())
        for _ in range(3):
            curator.handle(curation.VerifyTask(make_update(), make_state()))
        assert curation.collect_signals(curator).tolist() == [1.0] * 3


class TestOneCodePathForEveryTask:
    def test_worker_and_inline_processing_give_identical_results(self):
        tasks = [
            curation.VerifyTask(make_update(seed=s), make_state())
            for s in range(4)
        ]
        inline = make_curator(AcceptAll())
        for t in tasks:
            inline.handle(t)

        threaded = make_curator(AcceptAll())
        threaded.start()
        for t in tasks:
            threaded.submit(t).result(timeout=5)
        threaded.stop()

        assert threaded.stats.to_dict() == inline.stats.to_dict()
        assert torch.equal(
            threaded.memory.cstate.get_bank("long_term"),
            inline.memory.cstate.get_bank("long_term"),
        )


# ------------------------------------------------------------------- bad flows


class TestMisuse:
    def test_submitting_before_start_is_reported_not_silently_dropped(self):
        curator = make_curator()
        assert curator.submit_prune(1) is None
        assert curator.stats.verified == 0

    def test_unknown_task_types_are_rejected(self):
        with pytest.raises(TypeError, match="task"):
            make_curator().handle(object())

    def test_negative_prune_counts_are_rejected(self):
        with pytest.raises(ValueError, match="k"):
            make_curator().handle(curation.PruneTask(k=-1))

    def test_start_and_stop_are_idempotent(self):
        curator = make_curator()
        curator.stop()  # before start: harmless
        curator.start()
        thread = curator.thread
        curator.start()  # second start must not spawn another worker
        assert curator.thread is thread
        curator.stop()
        curator.stop()
        assert not curator.started


# ---------------------------------------------------- breakage and recovery


class TestFailureIsolationAndRecovery:
    def test_a_failing_task_does_not_stop_the_worker(self):
        curator = make_curator(Exploding())
        curator.start()
        for _ in range(4):
            curator.submit(
                curation.VerifyTask(make_update(), make_state())
            ).result(timeout=5)
        curator.stop()
        assert curator.stats.errors == 1
        assert curator.stats.accepted == 3  # the other three went through

    def test_a_failing_callback_is_contained_and_counted(self):
        def bad_callback(*args):
            raise ValueError("callback bug")

        curator = make_curator(AcceptAll())
        curator.start()
        curator.submit(
            curation.VerifyTask(make_update(), make_state(), bad_callback)
        ).result(timeout=5)
        curator.submit(curation.VerifyTask(make_update(), make_state())).result(
            timeout=5
        )
        curator.stop()
        assert curator.stats.errors == 1
        assert curator.stats.verified == 2

    def test_the_curator_can_be_restarted_after_stop(self):
        """Regression: the queue was bound to the first event loop, so any
        task submitted after a restart failed."""
        curator = make_curator(AcceptAll())
        curator.start()
        curator.submit(curation.VerifyTask(make_update(), make_state())).result(
            timeout=5
        )
        curator.stop()

        curator.start()
        future = curator.submit(
            curation.VerifyTask(make_update(), make_state())
        )
        assert future is not None
        future.result(timeout=5)
        curator.stop()
        assert curator.stats.verified == 2

    def test_repeated_restart_cycles_stay_healthy(self):
        curator = make_curator(AcceptAll())
        for cycle in range(5):
            curator.start()
            curator.submit(
                curation.VerifyTask(make_update(), make_state())
            ).result(timeout=5)
            curator.stop()
            assert not curator.thread.is_alive(), cycle
        assert curator.stats.verified == 5

    def test_stop_waits_for_queued_work_to_finish(self):
        done = []

        class Slow(AcceptAll):
            def verify(self, candidate, state):
                time.sleep(0.05)
                done.append(1)
                return 1.0, True

        curator = make_curator(Slow())
        curator.start()
        for _ in range(5):
            curator.submit(curation.VerifyTask(make_update(), make_state()))
        curator.stop()
        assert len(done) == 5  # nothing was discarded by shutting down

    def test_state_is_consistent_after_a_mix_of_good_and_bad_tasks(self):
        curator = make_curator(Exploding())
        for _ in range(5):
            with contextlib.suppress(RuntimeError):
                curator.handle(curation.VerifyTask(make_update(), make_state()))
        stats = curator.stats
        assert stats.accepted == 4
        assert stats.verified == 5  # attempts are counted, even failed ones


# ------------------------------------------------------------------ concurrency


class TestConcurrency:
    def test_many_producer_threads_lose_no_tasks(self, running):
        per_thread, threads = 25, 8

        def produce():
            for _ in range(per_thread):
                running.submit(curation.VerifyTask(make_update(), make_state()))

        workers = [threading.Thread(target=produce) for _ in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        assert wait_for(lambda: running.stats.verified == per_thread * threads)

    def test_submission_does_not_block_on_slow_work(self):
        class Slow(AcceptAll):
            def verify(self, candidate, state):
                time.sleep(0.2)
                return 1.0, True

        curator = make_curator(Slow())
        curator.start()
        start = time.time()
        for _ in range(5):
            curator.submit(curation.VerifyTask(make_update(), make_state()))
        assert time.time() - start < 0.5  # enqueue returns immediately
        curator.stop()

    def test_tasks_are_processed_in_fifo_order(self):
        order = []
        curator = make_curator(AcceptAll())
        curator.start()
        futures = [
            curator.submit(
                curation.VerifyTask(
                    make_update(),
                    make_state(),
                    lambda *a, i=i: order.append(i),
                )
            )
            for i in range(20)
        ]
        for f in futures:
            f.result(timeout=5)
        curator.stop()
        assert order == list(range(20))
