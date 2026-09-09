"""The headline localisation claim, as an executable test.

Everything else in the suite checks a mechanism in isolation: that the gate
routes to ``top_k`` slots, that a probe restores the PCS, that a rate lands
in ``[0, 1]``. None of them assert the claim this work exists to make, so
every part could keep passing while the claim quietly stopped being true.

The claim, from the specification:

    for a given emitted action, k intent slots carry the gradient and the
    rest do not; intervening on those slots moves behaviour in the
    direction the forward model predicted, and intervening on other slots
    does not.

That has four measurable parts, and this module trains a small model on a
learnable task and asserts each one:

1. sparsity      -- a strict minority of intent slots carry the gradient;
2. controllability -- every attributed slot moves the emitted action;
3. specificity   -- no unattributed slot moves it;
4. direction     -- attributed slots move it the way the forward model
   predicted, not merely somewhere.

Also asserted: the claim is *false* with origination disabled. A localisation
result that showed up with ``alpha=1``, where the generator is never called,
would be measuring an artefact of the probe rather than the mechanism.

Marked ``slow``: this trains for real, which the unit tests do not. It is
excluded from CI's ``-m "not integration"`` run only if that marker is added;
it is kept in the default local run on purpose, because a claim nobody
re-runs is a claim nobody notices breaking.
"""

from __future__ import annotations

import math

import pytest
import torch

from ucsa.models.losses import LossWeights, UCSACombinedLoss
from ucsa.models.origination import (
    counterfactual_controllability,
    intent_collapse_report,
)
from ucsa.models.ucsa import UCSA, UCSAConfig
from ucsa.training.curriculum import Curriculum, CurriculumSchedule
from ucsa.training.metrics import build_default_registry
from ucsa.training.trainer import Trainer, TrainerConfig

VOCAB = 32
HALF = 4
STEPS = 1200
NUM_SLOTS = 16


def copy_task_batches(
    count: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(inputs, targets)`` pairs that repeat their first half.

    A learnable structure matters: on a task the model cannot fit there is
    no settled behaviour for an intervention to move, and the claim cannot
    be tested either way.

    Args:
        count: Number of pairs.
        seed: RNG seed.

    Returns:
        List of ``(inputs, targets)`` pairs.
    """
    generator = torch.Generator().manual_seed(seed)
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(count):
        chunk = torch.randint(1, VOCAB, (1, HALF), generator=generator)
        sequence = torch.cat([chunk, chunk], dim=1)
        pairs.append((sequence[:, :-1], sequence[:, 1:]))
    return pairs


def train_model(observation_mix: float, balance: bool = False) -> UCSA:
    """Train a small UCSA on the copy task.

    Args:
        observation_mix: ``alpha_0``. ``1.0`` disables origination.
        balance: Whether to weight the gate's load-balancing loss.

    Returns:
        The trained model.
    """
    torch.manual_seed(0)
    model = UCSA(
        UCSAConfig(
            hidden_size=64,
            num_layers=4,
            num_q_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
            vocab_size=VOCAB,
            reasoning_iterations=4,
            observation_mix=observation_mix,
            intent_update_scale=0.1,
            origination_top_k=2,
        )
    )
    trainer = Trainer(
        model=model,
        loss_fn=UCSACombinedLoss(
            LossWeights(origination=0.01 if balance else 0.0)
        ),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        config=TrainerConfig(
            learning_rate=1e-3,
            max_steps=STEPS,
            warmup_steps=20,
            amp_dtype=torch.float32,
        ),
        curriculum=Curriculum(CurriculumSchedule(1, 2, 3)),
        metrics=build_default_registry(),
    )
    for inputs, targets in copy_task_batches(STEPS, seed=7):
        trainer.train_step((inputs, targets))
        assert trainer.state.last_loss == trainer.state.last_loss, "NaN loss"
    # The claim is only meaningful once the task is actually learned. This
    # guard is load-bearing: at 900 steps the same setup sits at ppl 31.85,
    # i.e. chance for a 32-token vocabulary, and there is no settled
    # behaviour for an intervention to move.
    perplexity = math.exp(min(20.0, trainer.state.last_loss))
    assert perplexity < VOCAB / 4, perplexity
    return model


@pytest.mark.slow
class TestLocalisationClaim:
    """The four parts of the headline claim, on a trained model."""

    @pytest.fixture(scope="class")
    @classmethod
    def trained(cls) -> UCSA:
        """Train once and share it across the assertions."""
        return train_model(observation_mix=0.5)

    @pytest.fixture(scope="class")
    @classmethod
    def probe_input(cls) -> torch.Tensor:
        """A held-out input, unseen during training."""
        return copy_task_batches(1, seed=123)[0][0]

    def test_diagnostic_is_not_collapsed(
        self, trained: UCSA, probe_input: torch.Tensor
    ) -> None:
        """Phase C first: no other number counts while this is red."""
        probes = [pair[0] for pair in copy_task_batches(6, seed=99)]
        report = intent_collapse_report(trained, probes)
        assert report.state_variance > 0.0
        assert report.read_share > 0.05

    def test_gradient_is_sparse_over_slots(
        self, trained: UCSA, probe_input: torch.Tensor
    ) -> None:
        """Part 1: a strict minority of slots carry the gradient."""
        report = counterfactual_controllability(trained, probe_input)
        active = report.attribution.active_slots
        assert 0 < len(active) <= NUM_SLOTS // 2, active

    def test_every_attributed_slot_moves_the_action(
        self, trained: UCSA, probe_input: torch.Tensor
    ) -> None:
        """Part 2: controllability."""
        report = counterfactual_controllability(trained, probe_input)
        assert report.controllability == pytest.approx(1.0)

    def test_no_unattributed_slot_moves_the_action(
        self, trained: UCSA, probe_input: torch.Tensor
    ) -> None:
        """Part 3: specificity."""
        report = counterfactual_controllability(trained, probe_input)
        assert report.specificity == pytest.approx(1.0)
        assert report.silent_moved == []

    def test_movement_follows_the_forward_model(
        self, trained: UCSA, probe_input: torch.Tensor
    ) -> None:
        """Part 4: direction, not merely magnitude."""
        report = counterfactual_controllability(trained, probe_input)
        assert report.directed_controllability == pytest.approx(1.0)
        assert report.mean_direction_agreement is not None
        assert report.mean_direction_agreement > 0.5

    def test_attributed_effects_dominate_unattributed_ones(
        self, trained: UCSA, probe_input: torch.Tensor
    ) -> None:
        """The separation is an order of magnitude, not a hair."""
        report = counterfactual_controllability(trained, probe_input)
        attributed = set(report.attribution.active_slots)
        effects = {
            item.slot: item.action_delta for item in report.interventions
        }
        strongest_attributed = max(
            (delta for slot, delta in effects.items() if slot in attributed),
            default=0.0,
        )
        strongest_other = max(
            (
                delta
                for slot, delta in effects.items()
                if slot not in attributed
            ),
            default=0.0,
        )
        assert strongest_attributed > 10.0 * strongest_other


@pytest.mark.slow
class TestClaimRequiresOrigination:
    """The claim must be absent when the mechanism is switched off.

    If localisation showed up at ``alpha=1``, where the generator is never
    called and no slot can reach an action, the probe would be reporting an
    artefact of itself rather than a property of the model.
    """

    def test_no_localisation_without_origination(self) -> None:
        """With ``alpha=1`` nothing is attributed and nothing is gated."""
        model = train_model(observation_mix=1.0)
        probe_input = copy_task_batches(1, seed=123)[0][0]
        report = counterfactual_controllability(model, probe_input)
        assert report.attribution.gated_slots == []
        assert report.attribution.active_slots == []
        assert report.controllability == 0.0
        assert report.directed_controllability == 0.0
