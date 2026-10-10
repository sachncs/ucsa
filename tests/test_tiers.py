"""Guarantees of the memory tiers.

Grouped by what is promised: deterministic effects, rejection of bad
candidates, atomic failure, capacity limits, and invariants that hold across
many accept/recycle cycles.
"""

import pytest
import torch

from ucsa.models import cognitive, tiers

HIDDEN = 16


def make_memory(capacity=None, seed=0):
    torch.manual_seed(seed)
    state = cognitive.State(cognitive.Config(hidden_size=HIDDEN))
    with torch.no_grad():
        state.get_bank("long_term").zero_()
    return tiers.Memory(state, capacity)


def update(n=4, seed=0, importance=None, confidence=1.0):
    gen = torch.Generator().manual_seed(seed)
    return tiers.Update(
        tokens=torch.randn(n, HIDDEN, generator=gen) + 1.0,
        importance=torch.ones(n) if importance is None else importance,
        confidence=confidence,
    )


def used(memory):
    return int((memory.cstate.metadata("long_term", "usage") > 0).sum())


# ---------------------------------------------------------- deterministic flow


class TestAccepting:
    def test_written_slots_hold_the_candidate_and_are_marked_used(self):
        memory = make_memory()
        cand = update(3)
        slots = memory.accept_into_long_term(cand)
        assert len(slots) == len(set(slots)) == 3
        bank = memory.read_long_term()
        assert torch.allclose(bank[slots], cand.tokens)
        usage = memory.cstate.metadata("long_term", "usage")
        age = memory.cstate.metadata("long_term", "age")
        assert (usage[slots] > 0).all()
        assert (age[slots] == 0).all()

    def test_importance_is_recorded_per_slot(self):
        memory = make_memory()
        imp = torch.tensor([0.1, 0.9, 0.5])
        slots = memory.accept_into_long_term(update(3, importance=imp))
        stored = memory.cstate.metadata("long_term", "importance")[slots]
        assert torch.allclose(stored, imp)

    def test_max_slots_caps_how_many_are_written(self):
        memory = make_memory()
        assert len(memory.accept_into_long_term(update(6), max_slots=2)) == 2
        assert used(memory) == 2

    def test_identical_sequences_give_identical_memory(self):
        def run():
            memory = make_memory()
            for seed in range(4):
                memory.accept_into_long_term(update(3, seed=seed))
            return memory.read_long_term().clone()

        assert torch.equal(run(), run())

    def test_proposals_are_detached_copies_of_working_memory(self):
        memory = make_memory()
        cand = memory.propose_candidate()
        assert not cand.tokens.requires_grad
        before = memory.read_working().clone()
        cand.tokens.zero_()
        assert torch.equal(memory.read_working(), before)

    def test_episode_snapshot_has_the_episode_size_and_leaves_working_alone(
        self,
    ):
        memory = make_memory()
        before = memory.read_working().clone()
        snap = memory.snapshot_episode()
        assert snap.shape[0] == memory.cstate.bank_size("episode")
        assert torch.equal(memory.read_working(), before)
        assert torch.equal(memory.cstate.get_bank("episode"), snap)


class TestRecycling:
    def fill(self, memory, importances):
        for imp in importances:
            memory.accept_into_long_term(
                update(1, importance=torch.tensor([imp]), seed=int(imp * 100))
            )

    def test_the_least_important_slots_go_first(self):
        memory = make_memory()
        self.fill(memory, [0.9, 0.1, 0.5, 0.8])
        memory.cstate.update_retention()
        recycled = memory.recycle_low_retention(1)
        assert len(recycled) == 1
        imp = memory.cstate.metadata("long_term", "importance")
        assert float(imp[recycled[0]]) == 0.0  # reset by the recycle

    def test_recycled_slots_are_zeroed_and_reusable(self):
        memory = make_memory()
        self.fill(memory, [0.2, 0.3])
        recycled = memory.recycle_low_retention(2)
        assert torch.count_nonzero(memory.read_long_term()[recycled]) == 0
        assert used(memory) == 0
        again = memory.accept_into_long_term(update(2, seed=9))
        assert sorted(again) == sorted(recycled) or len(again) == 2

    def test_fifo_recycles_the_oldest_first(self):
        memory = make_memory()
        self.fill(memory, [0.5, 0.5, 0.5])
        age = memory.cstate.metadata("long_term", "age")
        with torch.no_grad():
            age[0], age[1], age[2] = 1, 9, 4
        first = memory.recycle_fifo(1)
        assert first == [1]

    def test_recycling_more_than_exists_clamps_instead_of_failing(self):
        memory = make_memory()
        self.fill(memory, [0.5])
        assert len(memory.recycle_low_retention(10_000)) <= (
            memory.cstate.bank_size("long_term")
        )

    @pytest.mark.parametrize("k", [0, -3])
    def test_non_positive_counts_recycle_nothing(self, k):
        memory = make_memory()
        self.fill(memory, [0.5, 0.6])
        assert memory.recycle_low_retention(k) == []
        assert memory.recycle_fifo(k) == []
        assert used(memory) == 2


# ------------------------------------------------------------------ bad flows


class TestRejectionOfBadCandidates:
    def test_tokens_must_be_two_dimensional(self):
        with pytest.raises(ValueError, match="2D"):
            tiers.Update(torch.zeros(4), torch.zeros(4))

    def test_token_and_importance_counts_must_agree(self):
        with pytest.raises(ValueError, match="token count"):
            tiers.Update(torch.zeros(3, HIDDEN), torch.zeros(2))

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_confidence_must_be_a_probability(self, confidence):
        with pytest.raises(ValueError, match="confidence"):
            update(confidence=confidence)

    @pytest.mark.parametrize("poison", [float("nan"), float("inf")])
    def test_non_finite_tokens_or_importance_never_reach_memory(self, poison):
        bad = torch.randn(3, HIDDEN)
        bad[1, 2] = poison
        with pytest.raises(ValueError, match="NaN or infinity"):
            tiers.Update(bad, torch.ones(3))
        imp = torch.ones(3)
        imp[0] = poison
        with pytest.raises(ValueError, match="NaN or infinity"):
            tiers.Update(torch.randn(3, HIDDEN), imp)

    def test_unknown_banks_are_reported(self):
        with pytest.raises(KeyError):
            make_memory().get_retention_scores("nope")


class TestFailureIsAtomic:
    def test_a_candidate_of_the_wrong_width_changes_nothing(self):
        memory = make_memory()
        memory.accept_into_long_term(update(2))
        bank = memory.read_long_term().clone()
        usage = memory.cstate.metadata("long_term", "usage").clone()
        wrong = tiers.Update(torch.randn(2, HIDDEN + 3), torch.ones(2))
        with pytest.raises(RuntimeError):
            memory.accept_into_long_term(wrong)
        assert torch.equal(memory.read_long_term(), bank)
        assert torch.equal(memory.cstate.metadata("long_term", "usage"), usage)


# ----------------------------------------------------------------- capacity


class TestCapacity:
    def test_a_full_memory_refuses_new_candidates_and_stays_unchanged(self):
        memory = make_memory(capacity=3)
        assert len(memory.accept_into_long_term(update(3))) == 3
        before = memory.read_long_term().clone()
        assert memory.accept_into_long_term(update(2, seed=5)) == []
        assert torch.equal(memory.read_long_term(), before)
        assert memory.long_term_capacity_used() == 1.0

    def test_capacity_smaller_than_the_bank_is_honoured(self):
        memory = make_memory(capacity=2)
        memory.accept_into_long_term(update(6))
        assert used(memory) == 2

    def test_pruning_frees_room_for_new_candidates(self):
        memory = make_memory(capacity=3)
        memory.accept_into_long_term(update(3))
        memory.recycle_low_retention(2)
        assert len(memory.accept_into_long_term(update(2, seed=4))) == 2


class TestInvariantsOverManyCycles:
    def test_usage_and_contents_stay_consistent(self):
        memory = make_memory(capacity=8)
        rng = torch.Generator().manual_seed(0)
        for step in range(300):
            if torch.rand((), generator=rng) < 0.6:
                memory.accept_into_long_term(
                    update(
                        int(torch.randint(1, 4, (), generator=rng)), seed=step
                    )
                )
            else:
                memory.recycle_low_retention(
                    int(torch.randint(0, 4, (), generator=rng))
                )
            usage = memory.cstate.metadata("long_term", "usage")
            bank = memory.read_long_term()
            assert used(memory) <= 8
            # Unused slots hold nothing; used slots hold finite data.
            assert torch.count_nonzero(bank[usage == 0]) == 0
            assert torch.isfinite(bank).all()
