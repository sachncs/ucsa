"""Tests for :mod:`ucsa.models.intent_descent`."""

from __future__ import annotations

import pytest
import torch

from ucsa.models.intent_descent import (
    DescentReport,
    compute_matched_comparison,
    ema_outputs,
    jepa_chain_error,
    optimize_intent,
    outcome_correlation,
    realized_outcome,
)
from ucsa.models.ucsa import UCSA, UCSAConfig
from ucsa.training.ema import EMATargetEncoder


def tiny_model(**overrides: object) -> UCSA:
    """Return a tiny UCSA with the origination path active."""
    defaults: dict[str, object] = {
        "hidden_size": 32,
        "num_layers": 2,
        "vocab_size": 100,
        "reasoning_iterations": 4,
        "observation_mix": 0.5,
    }
    defaults.update(overrides)
    torch.manual_seed(0)
    return UCSA(UCSAConfig(**defaults))  # type: ignore[arg-type]


def tiny_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return an ``(inputs, targets)`` pair."""
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 100, (1, 8), generator=generator)
    return ids, torch.roll(ids, -1, dims=1)


class TestJepaChainError:
    """Tests for :func:`jepa_chain_error`."""

    def test_zero_without_a_chain(self) -> None:
        """No chain and no single pair means no objective."""
        assert float(jepa_chain_error({})) == 0.0

    def test_reads_the_multi_step_chain(self) -> None:
        """The error is the mean over the chain's pairs."""
        model = tiny_model()
        outputs = model(tiny_inputs()[0])
        value = jepa_chain_error(outputs)
        assert value.requires_grad
        assert float(value.detach()) >= 0.0

    def test_falls_back_to_the_single_pair(self) -> None:
        """A single predicted/target pair is enough."""
        predicted = torch.ones(2, 4, requires_grad=True)
        target = torch.zeros(2, 4)
        value = jepa_chain_error(
            {"jepa_predicted": predicted, "jepa_target": target}
        )
        assert float(value.detach()) > 0.0

    def test_ema_targets_replace_the_self_targets(self) -> None:
        """Supplying target outputs changes the objective.

        Without EMA targets both sides of every pair come from the same
        forward pass, so moving the intent bank moves the prediction and
        its target together.
        """
        predicted = torch.ones(2, 4, requires_grad=True)
        outputs = {"jepa_multi_step": [(predicted, torch.ones(2, 4))]}
        ema = {"jepa_multi_step": [(torch.zeros(2, 4), torch.zeros(2, 4))]}
        assert float(jepa_chain_error(outputs).detach()) == pytest.approx(0.0)
        assert float(jepa_chain_error(outputs, ema).detach()) > 0.0


class TestRealizedOutcome:
    """Tests for :func:`realized_outcome`."""

    def test_none_without_targets(self) -> None:
        """No target means no realised outcome."""
        assert (
            realized_outcome({"language": torch.randn(1, 4, 10)}, None) is None
        )

    def test_scores_against_targets(self) -> None:
        """A target produces a finite cross-entropy."""
        value = realized_outcome(
            {"language": torch.randn(1, 4, 10)}, torch.zeros(1, 4).long()
        )
        assert value is not None
        assert value > 0.0

    def test_none_without_logits(self) -> None:
        """No language head output means no realised outcome."""
        assert realized_outcome({}, torch.zeros(1, 4).long()) is None

    def test_scores_the_trailing_positions(self) -> None:
        """Alignment must match the trainer's left-padding.

        ``Trainer.compute_loss`` left-pads short targets, so the supervised
        positions are the *last* ``len(targets)`` logits. Scoring the
        leading positions reads slots the model was never trained on, which
        pinned the readout near chance regardless of how well the model had
        learned.
        """
        vocab, span = 10, 3
        logits = torch.full((1, 8, vocab), -20.0)
        targets = torch.tensor([[1, 2, 3]])
        for offset, token in enumerate([1, 2, 3]):
            logits[0, -span + offset, token] = 20.0
        value = realized_outcome({"language": logits}, targets)
        assert value is not None
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_leading_positions_are_ignored(self) -> None:
        """Getting the leading positions right must not score as correct."""
        vocab = 10
        logits = torch.full((1, 8, vocab), -20.0)
        targets = torch.tensor([[1, 2, 3]])
        for offset, token in enumerate([1, 2, 3]):
            logits[0, offset, token] = 20.0
        value = realized_outcome({"language": logits}, targets)
        assert value is not None
        assert value > 1.0


class TestMatchedCompute:
    """Tests for :func:`compute_matched_comparison`."""

    def test_arms_share_the_operator_call_budget(self) -> None:
        """Both controls must cost the same as the optimisation arm.

        A ``K=0`` arm is cheaper, not matched, so comparing against it
        would credit the optimisation for compute rather than origination.
        """
        model = tiny_model()
        inputs, targets = tiny_inputs()
        rows = compute_matched_comparison(
            model, [(inputs, targets)], intent_steps=2
        )
        assert {row.arm for row in rows} == {
            "intent-optimization",
            "more-reasoning",
            "repeat-and-average",
        }
        budgets = {row.operator_calls for row in rows}
        assert len(budgets) == 1
        base = model.reasoning_loop.config.num_iterations
        assert budgets.pop() == (2 + 2) * base

    def test_restores_the_loop_config_and_state(self) -> None:
        """The comparison is read-only."""
        model = tiny_model()
        inputs, targets = tiny_inputs()
        before_iters = model.reasoning_loop.config.num_iterations
        before = {
            name: model.pcs.get_bank(name).detach().clone()
            for name in model.pcs.bank_order
        }
        compute_matched_comparison(model, [(inputs, targets)], intent_steps=2)
        assert model.reasoning_loop.config.num_iterations == before_iters
        for name, tensor in before.items():
            assert torch.allclose(model.pcs.get_bank(name).detach(), tensor)

    def test_rejects_non_positive_steps(self) -> None:
        """There is nothing to compare at ``K=0``."""
        model = tiny_model()
        inputs, targets = tiny_inputs()
        with pytest.raises(ValueError):
            compute_matched_comparison(
                model, [(inputs, targets)], intent_steps=0
            )

    def test_rejects_empty_pairs(self) -> None:
        """No probe pairs means no comparison."""
        with pytest.raises(ValueError):
            compute_matched_comparison(tiny_model(), [], intent_steps=1)

    def test_rows_are_serialisable(self) -> None:
        """``to_dict`` returns plain values."""
        model = tiny_model()
        inputs, targets = tiny_inputs()
        rows = compute_matched_comparison(
            model, [(inputs, targets)], intent_steps=1
        )
        payload = rows[0].to_dict()
        assert set(payload) >= {"arm", "operator_calls", "realized"}


class TestEmaOutputs:
    """Tests for :func:`ema_outputs`."""

    def test_none_encoder_returns_none(self) -> None:
        """No encoder means no targets."""
        assert ema_outputs(None, tiny_inputs()[0]) is None

    def test_encoder_state_is_restored(self) -> None:
        """Consulting the encoder must not move its own banks.

        Its forward rewrites its PCS, so without restoration the target
        drifts every time it is read and the objective moves under the
        optimiser.
        """
        model = tiny_model()
        encoder = EMATargetEncoder(model, momentum=0.99)
        before = {
            name: encoder.target.pcs.get_bank(name).detach().clone()
            for name in encoder.target.pcs.bank_order
        }
        ema_outputs(encoder, tiny_inputs()[0])
        for name, tensor in before.items():
            assert torch.allclose(
                encoder.target.pcs.get_bank(name).detach(), tensor
            )


class TestOptimizeIntent:
    """Tests for :func:`optimize_intent`."""

    def test_zero_steps_is_a_no_op(self) -> None:
        """``K=0`` is the default and must not move the bank."""
        model = tiny_model()
        inputs, targets = tiny_inputs()
        before = model.pcs.get_bank("intent").detach().clone()
        report = optimize_intent(model, inputs, targets=targets)
        assert report.steps == []
        assert report.intent_shift == 0.0
        assert torch.allclose(model.pcs.get_bank("intent").detach(), before)

    def test_negative_steps_rejected(self) -> None:
        """A negative ``K`` raises."""
        with pytest.raises(ValueError):
            optimize_intent(tiny_model(), tiny_inputs()[0], num_steps=-1)

    def test_negative_learning_rate_rejected(self) -> None:
        """A negative step size raises."""
        with pytest.raises(ValueError):
            optimize_intent(tiny_model(), tiny_inputs()[0], learning_rate=-0.1)

    def test_only_the_intent_bank_moves(self) -> None:
        """Weights are frozen; the origination bank is the only variable."""
        model = tiny_model()
        inputs, targets = tiny_inputs()
        bank = model.pcs.get_bank("intent")
        others = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter is not bank
        }
        optimize_intent(
            model, inputs, num_steps=3, learning_rate=0.1, targets=targets
        )
        for name, tensor in others.items():
            current = dict(model.named_parameters())[name]
            assert torch.allclose(current.detach(), tensor), name

    def test_requires_grad_flags_are_restored(self) -> None:
        """Freezing is temporary."""
        model = tiny_model()
        optimize_intent(model, tiny_inputs()[0], num_steps=2)
        assert all(p.requires_grad for p in model.parameters())

    def test_restore_rolls_the_bank_back(self) -> None:
        """``restore=True`` leaves the model untouched."""
        model = tiny_model()
        inputs, targets = tiny_inputs()
        before = {
            name: model.pcs.get_bank(name).detach().clone()
            for name in model.pcs.bank_order
        }
        optimize_intent(
            model,
            inputs,
            num_steps=3,
            learning_rate=0.2,
            targets=targets,
            restore=True,
        )
        for name, tensor in before.items():
            assert torch.allclose(model.pcs.get_bank(name).detach(), tensor)

    def test_descent_moves_the_bank(self) -> None:
        """With ``K>0`` and a live gradient the bank actually moves."""
        model = tiny_model()
        inputs, targets = tiny_inputs()
        report = optimize_intent(
            model,
            inputs,
            num_steps=3,
            learning_rate=0.1,
            targets=targets,
            restore=True,
        )
        assert report.steps
        assert report.intent_shift > 0.0

    def test_context_banks_are_not_polluted_by_rollouts(self) -> None:
        """Speculative rollouts must not commit cognitive state.

        A forward pass rewrites the PCS, so K rollouts would otherwise each
        start somewhere new and the measured improvement would be drift.
        """
        model = tiny_model()
        inputs, targets = tiny_inputs()
        before = model.pcs.get_bank("working").detach().clone()
        optimize_intent(
            model, inputs, num_steps=4, learning_rate=0.1, targets=targets
        )
        assert torch.allclose(model.pcs.get_bank("working").detach(), before)

    def test_early_stop_on_small_gradient(self) -> None:
        """A high threshold stops the loop on the first step."""
        model = tiny_model()
        report = optimize_intent(
            model,
            tiny_inputs()[0],
            num_steps=5,
            grad_norm_threshold=1e9,
            restore=True,
        )
        assert report.stopped_early
        assert len(report.steps) == 1

    def test_relative_early_stop_is_configurable(self) -> None:
        """A relative threshold stops against the input's own gradient.

        An absolute threshold cannot adapt: intent gradient norms sit in a
        narrow band across inputs, so any value stops every input at the
        same step or none of them. Measured: absolute thresholds produced
        step counts of exactly {1} across 24 inputs, while a relative
        threshold produced {2, 3, 4, 5}.
        """
        model = tiny_model()
        inputs, targets = tiny_inputs()
        report = optimize_intent(
            model,
            inputs,
            num_steps=6,
            learning_rate=0.05,
            targets=targets,
            restore=True,
            grad_norm_relative_threshold=0.99,
        )
        assert 1 <= len(report.steps) <= 6

    def test_relative_threshold_of_zero_is_disabled(self) -> None:
        """``0.0`` must not stop the loop."""
        model = tiny_model()
        report = optimize_intent(
            model,
            tiny_inputs()[0],
            num_steps=3,
            learning_rate=0.05,
            restore=True,
            grad_norm_relative_threshold=0.0,
        )
        assert len(report.steps) == 3
        assert not report.stopped_early

    def test_relative_threshold_of_one_stops_immediately(self) -> None:
        """A threshold at the starting norm stops on the first check."""
        model = tiny_model()
        report = optimize_intent(
            model,
            tiny_inputs()[0],
            num_steps=6,
            learning_rate=0.05,
            restore=True,
            grad_norm_relative_threshold=1.0,
        )
        assert report.stopped_early
        assert len(report.steps) == 1

    def test_forward_passes_are_counted(self) -> None:
        """The cost is reported so claims can be made at matched compute."""
        model = tiny_model()
        report = optimize_intent(
            model, tiny_inputs()[0], num_steps=3, restore=True
        )
        assert report.forward_passes == 2 + len(report.steps)

    def test_report_is_serialisable(self) -> None:
        """``to_dict`` nests the per-step records."""
        model = tiny_model()
        inputs, targets = tiny_inputs()
        payload = optimize_intent(
            model, inputs, num_steps=2, targets=targets, restore=True
        ).to_dict()
        assert set(payload) >= {
            "steps",
            "initial_objective",
            "final_objective",
            "forward_passes",
            "forward_model_gamed",
        }

    def test_works_with_an_ema_target_encoder(self) -> None:
        """The EMA path runs and reports."""
        model = tiny_model()
        inputs, targets = tiny_inputs()
        encoder = EMATargetEncoder(model, momentum=0.99)
        report = optimize_intent(
            model,
            inputs,
            num_steps=2,
            targets=targets,
            restore=True,
            target_encoder=encoder,
        )
        assert report.initial_objective >= 0.0


class TestForwardModelGaming:
    """Tests for the forward-model-hacking detector."""

    def test_flags_predicted_gain_with_realised_loss(self) -> None:
        """Predicted better, realised worse, is gaming."""
        report = DescentReport(
            initial_objective=1.0,
            final_objective=0.5,
            initial_realized=2.0,
            final_realized=3.0,
        )
        assert report.objective_improved
        assert report.realized_improved is False
        assert report.forward_model_gamed

    def test_not_flagged_when_both_improve(self) -> None:
        """Both improving is the good case."""
        report = DescentReport(
            initial_objective=1.0,
            final_objective=0.5,
            initial_realized=2.0,
            final_realized=1.0,
        )
        assert not report.forward_model_gamed

    def test_not_flagged_without_a_realised_measurement(self) -> None:
        """No realised outcome means no verdict."""
        report = DescentReport(initial_objective=1.0, final_objective=0.5)
        assert report.realized_improved is None
        assert not report.forward_model_gamed


class TestOutcomeCorrelation:
    """Tests for :func:`outcome_correlation`."""

    def test_zero_for_too_few_reports(self) -> None:
        """One report cannot express a correlation."""
        assert outcome_correlation([]) == 0.0

    def test_positive_when_prediction_tracks_outcome(self) -> None:
        """Aligned deltas correlate positively."""
        reports = [
            DescentReport(
                initial_objective=1.0,
                final_objective=1.0 - delta,
                initial_realized=2.0,
                final_realized=2.0 - delta,
            )
            for delta in (0.1, 0.2, 0.3)
        ]
        assert outcome_correlation(reports) > 0.9

    def test_negative_when_prediction_opposes_outcome(self) -> None:
        """Anti-aligned deltas correlate negatively.

        A non-positive correlation means the objective and the outcome have
        come apart and the descent is exploiting the predictor.
        """
        reports = [
            DescentReport(
                initial_objective=1.0,
                final_objective=1.0 - delta,
                initial_realized=2.0,
                final_realized=2.0 + delta,
            )
            for delta in (0.1, 0.2, 0.3)
        ]
        assert outcome_correlation(reports) < -0.9

    def test_zero_for_constant_series(self) -> None:
        """A constant series has no correlation to report."""
        reports = [
            DescentReport(
                initial_objective=1.0,
                final_objective=1.0,
                initial_realized=2.0,
                final_realized=2.0,
            )
            for _ in range(3)
        ]
        assert outcome_correlation(reports) == 0.0
