"""Tests for :mod:`ucsa.models.origination`."""

from __future__ import annotations

import pytest
import torch

from ucsa.models.origination import (
    chain_latents,
    counterfactual_controllability,
    direction_agreement,
    intent_attribution,
    intent_collapse_report,
    intervene_intent,
)
from ucsa.models.ucsa import UCSA, UCSAConfig


def tiny_model(**overrides: object) -> UCSA:
    """Return a tiny UCSA with origination enabled."""
    defaults: dict[str, object] = {
        "hidden_size": 32,
        "num_layers": 2,
        "vocab_size": 100,
        "reasoning_iterations": 3,
        "observation_mix": 0.0,
        "origination_top_k": 2,
    }
    defaults.update(overrides)
    torch.manual_seed(0)
    return UCSA(UCSAConfig(**defaults))  # type: ignore[arg-type]


def tiny_inputs() -> torch.Tensor:
    """Return a fixed token batch."""
    generator = torch.Generator().manual_seed(1)
    return torch.randint(0, 100, (1, 6), generator=generator)


class TestIntentAttribution:
    """Tests for :func:`intent_attribution`."""

    def test_one_score_per_slot(self) -> None:
        """The map has an entry for every intent slot."""
        model = tiny_model()
        report = intent_attribution(model, tiny_inputs())
        assert len(report.scores) == model.pcs.bank_size("intent")
        assert len(report.gate_usage) == model.pcs.bank_size("intent")
        assert all(score >= 0.0 for score in report.scores)

    def test_top_slots_are_sorted_by_score(self) -> None:
        """``top_slots`` orders slots by descending score."""
        report = intent_attribution(tiny_model(), tiny_inputs())
        ordered = [report.scores[i] for i in report.top_slots]
        assert ordered == sorted(ordered, reverse=True)

    def test_gated_slots_respect_the_sparse_gate(self) -> None:
        """Only ``top_k`` slots per query token can be routed to.

        With ``top_k=2`` and six query tokens at most twelve of sixteen
        slots can be gated, and in practice far fewer. This is the
        bottleneck that makes the origination addressable.
        """
        model = tiny_model(origination_top_k=2)
        report = intent_attribution(model, tiny_inputs())
        assert report.gated_slots
        assert len(report.gated_slots) < model.pcs.bank_size("intent")
        assert all(report.gate_usage[i] > 0.0 for i in report.gated_slots)

    def test_dense_gate_uses_every_slot(self) -> None:
        """A ``top_k`` at the bank size removes the bottleneck."""
        model = tiny_model(origination_top_k=16)
        report = intent_attribution(model, tiny_inputs())
        assert len(report.gated_slots) == model.pcs.bank_size("intent")

    def test_gate_usage_empty_when_generator_unused(self) -> None:
        """With ``alpha=1`` the generator never runs, so no slot is gated."""
        model = tiny_model(observation_mix=1.0)
        report = intent_attribution(model, tiny_inputs())
        assert report.gated_slots == []

    def test_probe_leaves_no_gradients_behind(self) -> None:
        """Attribution clears the gradients it created."""
        model = tiny_model()
        intent_attribution(model, tiny_inputs())
        assert all(p.grad is None for p in model.parameters())

    def test_report_is_serialisable(self) -> None:
        """``to_dict`` returns plain Python values."""
        payload = intent_attribution(tiny_model(), tiny_inputs()).to_dict()
        assert set(payload) >= {"scores", "gated_slots", "top_slots"}
        assert all(isinstance(v, float) for v in payload["scores"])


class TestIntervention:
    """Tests for :func:`intervene_intent`."""

    def test_ablation_restores_the_bank(self) -> None:
        """The probe is read-only: the bank comes back unchanged."""
        model = tiny_model()
        before = model.pcs.get_bank("intent").detach().clone()
        intervene_intent(model, tiny_inputs(), 0)
        assert torch.allclose(model.pcs.get_bank("intent").detach(), before)

    def test_swap_restores_both_slots(self) -> None:
        """A swap restores both the slot and its partner."""
        model = tiny_model()
        before = model.pcs.get_bank("intent").detach().clone()
        intervene_intent(model, tiny_inputs(), 0, mode="swap", swap_with=3)
        assert torch.allclose(model.pcs.get_bank("intent").detach(), before)

    def test_reports_finite_effect_sizes(self) -> None:
        """The measured deltas are finite numbers."""
        report = intervene_intent(tiny_model(), tiny_inputs(), 1)
        assert report.action_delta >= 0.0
        assert report.action_delta == report.action_delta
        assert isinstance(report.top_token_changed, bool)

    def test_out_of_range_slot_rejected(self) -> None:
        """An out-of-range slot raises."""
        model = tiny_model()
        with pytest.raises(IndexError):
            intervene_intent(model, tiny_inputs(), 999)

    def test_out_of_range_partner_rejected(self) -> None:
        """An out-of-range swap partner raises."""
        model = tiny_model()
        with pytest.raises(IndexError):
            intervene_intent(model, tiny_inputs(), 0, mode="swap", swap_with=99)

    def test_unknown_mode_rejected(self) -> None:
        """An unrecognised mode raises."""
        model = tiny_model()
        with pytest.raises(ValueError):
            intervene_intent(model, tiny_inputs(), 0, mode="scramble")


class TestDirectionAgreement:
    """Tests for the directional half of the localisation claim."""

    def test_none_without_latents(self) -> None:
        """No chain means no directional verdict."""
        assert direction_agreement(None, None) is None
        pair = (torch.randn(2, 4), torch.randn(2, 4))
        assert direction_agreement(pair, None) is None

    def test_none_when_nothing_moved(self) -> None:
        """A zero change has no direction."""
        pair = (torch.randn(2, 4), torch.randn(2, 4))
        assert direction_agreement(pair, pair) is None

    def test_positive_when_changes_align(self) -> None:
        """Predicted and realised moving together scores near +1."""
        base = (torch.zeros(2, 4), torch.zeros(2, 4))
        step = torch.ones(2, 4)
        assert direction_agreement(base, (step, step)) == pytest.approx(1.0)

    def test_negative_when_changes_oppose(self) -> None:
        """Predicted and realised moving apart scores near -1.

        This is the case that separates "the intervention had an effect"
        from the spec's claim that it moves behaviour *in the direction the
        forward model predicted*.
        """
        base = (torch.zeros(2, 4), torch.zeros(2, 4))
        step = torch.ones(2, 4)
        agreement = direction_agreement(base, (step, -step))
        assert agreement == pytest.approx(-1.0)

    def test_chain_latents_returns_a_pair(self) -> None:
        """The probe reads the last pair of the JEPA chain."""
        model = tiny_model()
        latents = chain_latents(model, tiny_inputs())
        assert latents is not None
        predicted, realized = latents
        assert predicted.shape == realized.shape
        assert predicted.grad_fn is None

    def test_intervention_reports_direction(self) -> None:
        """Interventions on a gated slot carry a directional score.

        Uses a four-iteration loop, which is the configuration the reported
        results use. With only three iterations the chain is short enough
        that its predicted side can be a step the origination never
        reaches, leaving no direction to report even though the realised
        latents moved.

        An ungated slot has no path through ``G`` at all, so ablating it
        moves nothing and ``None`` is then the correct answer, not a number.
        """
        model = tiny_model(reasoning_iterations=4)
        inputs = tiny_inputs()
        gated = intent_attribution(model, inputs).gated_slots
        assert gated
        report = intervene_intent(model, inputs, gated[0])
        assert report.direction_agreement is not None
        assert -1.0 <= report.direction_agreement <= 1.0

    def test_ungated_slot_has_no_direction(self) -> None:
        """A slot the gate never selected moves nothing at all."""
        model = tiny_model()
        inputs = tiny_inputs()
        attribution = intent_attribution(model, inputs)
        ungated = [
            slot
            for slot in range(len(attribution.scores))
            if slot not in set(attribution.gated_slots)
        ]
        assert ungated
        report = intervene_intent(model, inputs, ungated[0])
        assert report.action_delta == pytest.approx(0.0, abs=1e-6)


class TestControllability:
    """Tests for :func:`counterfactual_controllability`."""

    def test_probes_every_slot_by_default(self) -> None:
        """Every slot gets an intervention unless a subset is given."""
        model = tiny_model()
        report = counterfactual_controllability(model, tiny_inputs())
        assert len(report.interventions) == model.pcs.bank_size("intent")

    def test_probes_only_requested_slots(self) -> None:
        """A slot subset limits the interventions."""
        report = counterfactual_controllability(
            tiny_model(), tiny_inputs(), slots=[0, 1]
        )
        assert [i.slot for i in report.interventions] == [0, 1]

    def test_rates_are_fractions(self) -> None:
        """Controllability and specificity are in [0, 1]."""
        report = counterfactual_controllability(tiny_model(), tiny_inputs())
        assert 0.0 <= report.controllability <= 1.0
        assert 0.0 <= report.specificity <= 1.0
        assert 0.0 <= report.directed_controllability <= 1.0

    def test_directed_controllability_is_not_looser(self) -> None:
        """Directed control is a subset of plain control.

        A slot only counts as directed if it moved the action *and* agreed
        with the forward model, so the directed rate can never exceed the
        undirected one.
        """
        report = counterfactual_controllability(tiny_model(), tiny_inputs())
        assert set(report.directed_moved) <= set(
            report.attribution.active_slots
        )
        assert report.directed_controllability <= report.controllability + 1e-9

    def test_reports_mean_direction_agreement(self) -> None:
        """The sweep summarises directional agreement across slots."""
        report = counterfactual_controllability(tiny_model(), tiny_inputs())
        assert report.mean_direction_agreement is not None
        assert -1.0 <= report.mean_direction_agreement <= 1.0

    def test_report_is_serialisable(self) -> None:
        """``to_dict`` nests the attribution and interventions."""
        payload = counterfactual_controllability(
            tiny_model(), tiny_inputs(), slots=[0]
        ).to_dict()
        assert "attribution" in payload
        assert len(payload["interventions"]) == 1

    def test_bank_unchanged_after_full_sweep(self) -> None:
        """A full sweep leaves the model exactly as it was."""
        model = tiny_model()
        before = model.pcs.get_bank("intent").detach().clone()
        counterfactual_controllability(model, tiny_inputs())
        assert torch.allclose(model.pcs.get_bank("intent").detach(), before)


def probe_batches(count: int = 4) -> list[torch.Tensor]:
    """Return several differing token batches."""
    generator = torch.Generator().manual_seed(3)
    return [
        torch.randint(0, 100, (1, 6), generator=generator) for _ in range(count)
    ]


class TestCollapseReport:
    """Tests for :func:`intent_collapse_report`."""

    def test_flags_a_generator_that_never_ran(self) -> None:
        """With ``alpha=1`` the origination path is inert, so it collapses."""
        report = intent_collapse_report(
            tiny_model(observation_mix=1.0), probe_batches()
        )
        assert report.collapsed
        assert any("never ran" in reason for reason in report.reasons)

    def test_flags_a_static_intent_bank(self) -> None:
        """``intent_update_scale=0`` gives one signal for every input.

        A bank that cannot vary across inputs is a constant bias, not an
        origination signal, and its variance is zero by construction.
        """
        report = intent_collapse_report(
            tiny_model(intent_update_scale=0.0), probe_batches()
        )
        assert report.state_variance == 0.0
        assert report.collapsed
        assert any("constant across inputs" in r for r in report.reasons)

    def test_active_origination_varies_across_inputs(self) -> None:
        """With the refresh enabled the state depends on the input."""
        report = intent_collapse_report(
            tiny_model(intent_update_scale=0.1), probe_batches()
        )
        assert report.state_variance > 0.0

    def test_entropy_is_bounded_by_the_uniform_ceiling(self) -> None:
        """Gate entropy cannot exceed ``log(num_slots)``."""
        report = intent_collapse_report(tiny_model(), probe_batches())
        assert 0.0 <= report.gate_entropy <= report.gate_entropy_max + 1e-6
        assert report.num_slots == 16

    def test_sparse_gate_uses_a_subset_of_slots(self) -> None:
        """A ``top_k`` gate cannot route to every slot for one input."""
        report = intent_collapse_report(
            tiny_model(origination_top_k=1), probe_batches(2)
        )
        assert 0 < report.slots_used <= report.num_slots

    def test_requires_two_inputs(self) -> None:
        """Variance and mutual information need differing inputs."""
        with pytest.raises(ValueError):
            intent_collapse_report(tiny_model(), probe_batches(1))

    def test_probe_restores_the_pcs(self) -> None:
        """The diagnostic is read-only."""
        model = tiny_model()
        before = {
            name: model.pcs.get_bank(name).detach().clone()
            for name in model.pcs.bank_order
        }
        intent_collapse_report(model, probe_batches())
        for name, tensor in before.items():
            assert torch.allclose(model.pcs.get_bank(name).detach(), tensor)

    def test_report_is_serialisable(self) -> None:
        """``to_dict`` returns plain Python values."""
        payload = intent_collapse_report(
            tiny_model(), probe_batches()
        ).to_dict()
        assert set(payload) >= {
            "state_variance",
            "gate_entropy",
            "gate_mutual_info",
            "read_share",
            "collapsed",
            "reasons",
        }
