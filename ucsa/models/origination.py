"""Origination probes: attribution and causal intervention.

The point of routing the next input through a sparse gate over the
``intent`` bank is that the origination of an action becomes *addressable*.
This module turns that into two readings:

- :func:`intent_attribution` -- a per-slot ``grad x activation`` map. For a
  given emitted action, which intent slots carried the gradient?
- :func:`intervene_intent` -- ablate or swap a slot and measure how far the
  emitted action moved. Attribution alone is a correlation; the
  intervention is the causal claim.

:func:`counterfactual_controllability` combines them into the number that
matters: of the slots attribution named, how many actually move behaviour
when you touch them, and do the slots it did *not* name stay quiet?

Every probe here is read-only with respect to the model. That takes more
than restoring one slot: the operator writes *every* bank back on *every*
iteration, so merely running a forward pass moves the PCS. A baseline and a
perturbed forward launched back to back would therefore start from
different states, and the measured difference would conflate the
perturbation with that drift. Each probe snapshots the whole PCS, runs from
it, and restores it.

Two paths, one bank
-------------------

The intent bank reaches an action two ways: through ``G``, where the sparse
gate makes the contribution attributable, and as ordinary PCS context that
the operator attends over, where it does not. ``gate_usage`` and
``gated_slots`` report the first path; ``scores`` reports the sum of both.
A slot can therefore be gradient-touched without ever having been gated,
which caps how clean the localisation claim can be while the bank stays in
``BANK_NAMES``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from ucsa.models.state import INTENT_BANK
from ucsa.training.metrics import (
    intent_gate_entropy,
    intent_gate_mutual_info,
    intent_gate_usage,
    intent_read_share,
    intent_state_variance,
)

if TYPE_CHECKING:
    from ucsa.models.ucsa import UCSA


@dataclass
class AttributionReport:
    """Per-slot attribution for one emitted action.

    Attributes:
        scores: Per-slot ``grad x activation`` magnitude, one entry per
            intent slot.
        gate_usage: Fraction of query tokens that routed to each slot
            through the sparse gate, or an empty list when the generator
            was never called.
        gated_slots: Slots the sparse gate actually routed to. These are
            the only slots that reached the action *through* ``G``. Slots
            outside this set can still show a non-zero score because the
            intent bank is also ordinary PCS context for the operator --
            see the module note on the two paths.
        readout: The scalar action readout that was differentiated.
        top_slots: Slot indices ordered by descending score.
        active_slots: Slots with a non-negligible score.
        silent_slots: Slots whose score is at or below ``tolerance``.
        tolerance: Threshold separating active from silent slots.
    """

    scores: list[float]
    gate_usage: list[float]
    gated_slots: list[int]
    readout: float
    top_slots: list[int]
    active_slots: list[int]
    silent_slots: list[int]
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "scores": self.scores,
            "gate_usage": self.gate_usage,
            "gated_slots": self.gated_slots,
            "readout": self.readout,
            "top_slots": self.top_slots,
            "active_slots": self.active_slots,
            "silent_slots": self.silent_slots,
            "tolerance": self.tolerance,
        }


@dataclass
class InterventionReport:
    """Effect of perturbing one intent slot.

    Attributes:
        slot: The perturbed slot index.
        mode: ``"ablate"`` or ``"swap"``.
        action_delta: L2 distance between the baseline and perturbed action
            logits, normalised by the baseline norm.
        top_token_changed: Whether the arg-max action token changed.
        readout_delta: Signed change in the baseline action's own logit.
        direction_agreement: Cosine similarity between the change the
            forward model *predicted* and the change that actually
            happened, in ``[-1, 1]``, or ``None`` when the JEPA chain was
            unavailable. This is what distinguishes "the intervention had
            an effect" from the spec's stronger claim, that it "moves
            behaviour in the direction the forward model predicted". A
            magnitude alone cannot say that.
    """

    slot: int
    mode: str
    action_delta: float
    top_token_changed: bool
    readout_delta: float
    direction_agreement: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "slot": self.slot,
            "mode": self.mode,
            "action_delta": self.action_delta,
            "top_token_changed": self.top_token_changed,
            "readout_delta": self.readout_delta,
            "direction_agreement": self.direction_agreement,
        }


@dataclass
class ControllabilityReport:
    """Whether attribution predicts what interventions do.

    Attributes:
        attribution: The attribution map that named the slots.
        interventions: One entry per probed slot.
        active_moved: Attributed slots whose ablation moved the action by
            more than ``effect_threshold``.
        directed_moved: Attributed slots that moved the action *and* whose
            realised change agreed in direction with the forward model's
            predicted change.
        silent_moved: Non-attributed slots that moved it anyway.
        controllability: Fraction of attributed slots that moved the
            action. ``0.0`` when attribution named no slots.
        directed_controllability: Fraction of attributed slots that moved
            it *in the predicted direction*. This is the spec's claim;
            ``controllability`` alone only says an effect existed.
        mean_direction_agreement: Mean cosine between predicted and
            realised change over the probed slots that reported one.
        specificity: Fraction of non-attributed slots that stayed quiet.
            ``1.0`` when every slot was attributed.
        effect_threshold: Normalised ``action_delta`` counted as an effect.
    """

    attribution: AttributionReport
    interventions: list[InterventionReport] = field(default_factory=list)
    active_moved: list[int] = field(default_factory=list)
    directed_moved: list[int] = field(default_factory=list)
    silent_moved: list[int] = field(default_factory=list)
    controllability: float = 0.0
    directed_controllability: float = 0.0
    mean_direction_agreement: float | None = None
    specificity: float = 1.0
    effect_threshold: float = 1e-3

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "attribution": self.attribution.to_dict(),
            "interventions": [i.to_dict() for i in self.interventions],
            "active_moved": self.active_moved,
            "directed_moved": self.directed_moved,
            "silent_moved": self.silent_moved,
            "controllability": self.controllability,
            "directed_controllability": self.directed_controllability,
            "mean_direction_agreement": self.mean_direction_agreement,
            "specificity": self.specificity,
            "effect_threshold": self.effect_threshold,
        }


def pcs_snapshot(model: UCSA) -> dict[str, Tensor]:
    """Return a copy of every PCS bank.

    Args:
        model: The model whose PCS is snapshotted.

    Returns:
        Mapping from bank name to a detached clone.
    """
    return {
        name: model.pcs.get_bank(name).detach().clone()
        for name in model.pcs.bank_order
    }


def pcs_restore(model: UCSA, snapshot: dict[str, Tensor]) -> None:
    """Write a snapshot back into the PCS.

    Args:
        model: The model whose PCS is restored.
        snapshot: A mapping produced by :func:`pcs_snapshot`.
    """
    for name, tensor in snapshot.items():
        model.pcs.set_bank(name, tensor)


def action_logits(model: UCSA, inputs: Tensor) -> Tensor:
    """Run the model and return the emitted action logits.

    Args:
        model: A :class:`~ucsa.models.ucsa.UCSA`-like model whose forward
            returns a dict with a ``language`` entry.
        inputs: Token ids of shape ``(batch, seq)``.

    Returns:
        The ``language`` logits.

    Raises:
        KeyError: If the model's output has no ``language`` entry.
    """
    device = model.pcs.get_bank(INTENT_BANK).device
    outputs = model(inputs.to(device))
    if "language" not in outputs:
        raise KeyError("model output has no 'language' entry.")
    logits: Tensor = outputs["language"]
    return logits


def action_readout(logits: Tensor) -> tuple[Tensor, int]:
    """Reduce action logits to the scalar that gets differentiated.

    The readout is the winning token's own logit at the last position,
    which is the closest scalar analogue of "the action that was emitted".

    Args:
        logits: Tensor of shape ``(batch, tokens, vocab)``.

    Returns:
        Tuple ``(scalar_readout, token_id)``.
    """
    last = logits[0, -1]
    token_id = int(last.argmax().item())
    return last[token_id], token_id


def intent_attribution(
    model: UCSA,
    inputs: Tensor,
    tolerance: float = 1e-8,
) -> AttributionReport:
    """Return the per-slot ``grad x activation`` map for one action.

    Args:
        model: The model to probe.
        inputs: Token ids of shape ``(batch, seq)``.
        tolerance: Scores at or below this count as silent.

    Returns:
        The :class:`AttributionReport`.
    """
    bank = model.pcs.get_bank(INTENT_BANK)
    snapshot = pcs_snapshot(model)
    model.zero_grad(set_to_none=True)
    logits = action_logits(model, inputs)
    readout, _ = action_readout(logits)
    readout.backward()  # type: ignore[no-untyped-call]
    grad = bank.grad
    if grad is None:
        scores_tensor = torch.zeros(bank.shape[0], device=bank.device)
    else:
        scores_tensor = (grad * snapshot[INTENT_BANK]).abs().sum(dim=-1)
    scores = [float(v) for v in scores_tensor.detach().cpu()]
    gate_weights = getattr(model.heads.origination, "last_gate_weights", None)
    if gate_weights is None:
        gate_usage: list[float] = []
    else:
        used = (gate_weights.detach() > 0).float().reshape(-1, bank.shape[0])
        gate_usage = [float(v) for v in used.mean(dim=0).cpu()]
    model.zero_grad(set_to_none=True)
    pcs_restore(model, snapshot)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return AttributionReport(
        scores=scores,
        gate_usage=gate_usage,
        gated_slots=[i for i, v in enumerate(gate_usage) if v > 0.0],
        readout=float(readout.detach()),
        top_slots=order,
        active_slots=[i for i, v in enumerate(scores) if v > tolerance],
        silent_slots=[i for i, v in enumerate(scores) if v <= tolerance],
        tolerance=tolerance,
    )


def chain_latents(model: UCSA, inputs: Tensor) -> tuple[Tensor, Tensor] | None:
    """Return the forward model's forecasts and the latents they forecast.

    Concatenates *every* pair of the multi-step JEPA chain. Taking only the
    last pair is not enough: the origination generator shapes iteration 2
    onward, so for a short loop the last pair's predicted side can be the
    one step intent never reaches, giving a delta of exactly zero and no
    directional verdict at all even though the realised latents moved.

    Args:
        model: The model to run.
        inputs: Token ids of shape ``(batch, seq)``.

    Returns:
        Tuple ``(predicted, realized)`` stacked over the chain, or ``None``
        when the chain is absent.
    """
    device = model.pcs.get_bank(INTENT_BANK).device
    with torch.no_grad():
        outputs = model(inputs.to(device))
    pairs = outputs.get("jepa_multi_step") or []
    if not pairs:
        return None
    predicted = torch.cat([p.detach().reshape(-1) for p, _ in pairs])
    realized = torch.cat([t.detach().reshape(-1) for _, t in pairs])
    return predicted.clone(), realized.clone()


def direction_agreement(
    baseline: tuple[Tensor, Tensor] | None,
    perturbed: tuple[Tensor, Tensor] | None,
) -> float | None:
    """Cosine between the predicted change and the realised change.

    Args:
        baseline: ``(predicted, realized)`` before the intervention.
        perturbed: ``(predicted, realized)`` after it.

    Returns:
        Cosine similarity in ``[-1, 1]``, or ``None`` when either latent
        pair is missing or neither quantity moved.
    """
    if baseline is None or perturbed is None:
        return None
    predicted_delta = (perturbed[0] - baseline[0]).flatten()
    realized_delta = (perturbed[1] - baseline[1]).flatten()
    if float(predicted_delta.norm()) <= 0.0:
        return None
    if float(realized_delta.norm()) <= 0.0:
        return None
    return float(
        torch.nn.functional.cosine_similarity(
            predicted_delta, realized_delta, dim=0
        ).item()
    )


def intervene_intent(
    model: UCSA,
    inputs: Tensor,
    slot: int,
    mode: str = "ablate",
    swap_with: int | None = None,
) -> InterventionReport:
    """Perturb one intent slot and measure the change in the action.

    The bank is restored before returning, so the model is unchanged.

    Args:
        model: The model to probe.
        inputs: Token ids of shape ``(batch, seq)``.
        slot: Index of the slot to perturb.
        mode: ``"ablate"`` zeros the slot; ``"swap"`` exchanges it with
            ``swap_with``.
        swap_with: Partner slot for ``"swap"``. Defaults to the next slot,
            wrapping around.

    Returns:
        The :class:`InterventionReport`.

    Raises:
        IndexError: If ``slot`` is out of range.
        ValueError: If ``mode`` is not recognised.
    """
    bank = model.pcs.get_bank(INTENT_BANK)
    num_slots = bank.shape[0]
    if not 0 <= slot < num_slots:
        raise IndexError(f"slot {slot} out of range for {num_slots} slots.")
    if mode not in ("ablate", "swap"):
        raise ValueError(f"mode must be 'ablate' or 'swap', got {mode!r}.")
    partner = (slot + 1) % num_slots if swap_with is None else swap_with
    if not 0 <= partner < num_slots:
        raise IndexError(
            f"swap_with {partner} out of range for {num_slots} slots."
        )

    snapshot = pcs_snapshot(model)
    original = snapshot[INTENT_BANK]
    try:
        with torch.no_grad():
            baseline = action_logits(model, inputs).detach().clone()
        pcs_restore(model, snapshot)
        baseline_latents = chain_latents(model, inputs)
        pcs_restore(model, snapshot)
        with torch.no_grad():
            if mode == "ablate":
                bank[slot] = torch.zeros_like(bank[slot])
            else:
                bank[slot] = original[partner]
                bank[partner] = original[slot]
            perturbed = action_logits(model, inputs).detach().clone()
        perturbed_intent = bank.detach().clone()
        pcs_restore(model, snapshot)
        with torch.no_grad():
            bank.copy_(perturbed_intent)
        perturbed_latents = chain_latents(model, inputs)
    finally:
        pcs_restore(model, snapshot)
    baseline_last = baseline[0, -1]
    baseline_token = int(baseline_last.argmax().item())

    perturbed_last = perturbed[0, -1]
    denominator = float(baseline.norm())
    delta = float((perturbed - baseline).norm())
    action_delta = delta / denominator if denominator > 0.0 else delta
    return InterventionReport(
        slot=slot,
        mode=mode,
        action_delta=action_delta,
        top_token_changed=int(perturbed_last.argmax().item()) != baseline_token,
        readout_delta=float(
            perturbed_last[baseline_token] - baseline_last[baseline_token]
        ),
        direction_agreement=direction_agreement(
            baseline_latents, perturbed_latents
        ),
    )


def counterfactual_controllability(
    model: UCSA,
    inputs: Tensor,
    slots: Sequence[int] | None = None,
    mode: str = "ablate",
    effect_threshold: float = 1e-3,
    tolerance: float = 1e-8,
) -> ControllabilityReport:
    """Check that attribution predicts what interventions actually do.

    This is the localisation claim in one call: the slots attribution names
    should move the action when perturbed, and the slots it does not name
    should leave it alone.

    Args:
        model: The model to probe.
        inputs: Token ids of shape ``(batch, seq)``.
        slots: Slots to intervene on. Defaults to every slot.
        mode: Passed to :func:`intervene_intent`.
        effect_threshold: Normalised ``action_delta`` counted as an effect.
        tolerance: Passed to :func:`intent_attribution`.

    Returns:
        The :class:`ControllabilityReport`.
    """
    attribution = intent_attribution(model, inputs, tolerance=tolerance)
    num_slots = len(attribution.scores)
    probe = list(range(num_slots)) if slots is None else list(slots)
    interventions = [
        intervene_intent(model, inputs, slot, mode=mode) for slot in probe
    ]
    active = set(attribution.active_slots)
    moved = {
        report.slot
        for report in interventions
        if report.action_delta > effect_threshold
    }
    probed_active = [s for s in probe if s in active]
    probed_silent = [s for s in probe if s not in active]
    directed = {
        report.slot
        for report in interventions
        if report.action_delta > effect_threshold
        and report.direction_agreement is not None
        and report.direction_agreement > 0.0
    }
    active_moved = [s for s in probed_active if s in moved]
    directed_moved = [s for s in probed_active if s in directed]
    silent_moved = [s for s in probed_silent if s in moved]
    controllability = (
        len(active_moved) / len(probed_active) if probed_active else 0.0
    )
    directed_controllability = (
        len(directed_moved) / len(probed_active) if probed_active else 0.0
    )
    agreements = [
        report.direction_agreement
        for report in interventions
        if report.direction_agreement is not None
    ]
    specificity = (
        1.0 - len(silent_moved) / len(probed_silent) if probed_silent else 1.0
    )
    return ControllabilityReport(
        attribution=attribution,
        interventions=interventions,
        active_moved=active_moved,
        directed_moved=directed_moved,
        silent_moved=silent_moved,
        controllability=controllability,
        directed_controllability=directed_controllability,
        mean_direction_agreement=(
            sum(agreements) / len(agreements) if agreements else None
        ),
        specificity=specificity,
        effect_threshold=effect_threshold,
    )


@dataclass
class CollapseReport:
    """Whether the generator is actually using the origination bank.

    Attributes:
        state_variance: Mean variance of the origination state across the
            probed inputs. Exactly ``0.0`` means one constant signal
            preceded every action.
        gate_entropy: Mean entropy of the gate's slot usage, in nats.
        gate_entropy_max: ``log(num_slots)``, the uniform-gate ceiling.
        gate_mutual_info: Mutual information between input identity and
            gated slot, in nats. ``0.0`` is the collapse signature.
        read_share: Mean fraction of the generator's read magnitude coming
            from intent rather than working memory.
        slots_used: Number of distinct slots gated across all inputs.
        num_slots: Size of the intent bank.
        collapsed: ``True`` when the diagnostic says the bank is being
            ignored.
        reasons: Human-readable reasons behind ``collapsed``.
    """

    state_variance: float
    gate_entropy: float
    gate_entropy_max: float
    gate_mutual_info: float
    read_share: float
    slots_used: int
    num_slots: int
    collapsed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "state_variance": self.state_variance,
            "gate_entropy": self.gate_entropy,
            "gate_entropy_max": self.gate_entropy_max,
            "gate_mutual_info": self.gate_mutual_info,
            "read_share": self.read_share,
            "slots_used": self.slots_used,
            "num_slots": self.num_slots,
            "collapsed": self.collapsed,
            "reasons": self.reasons,
        }


def intent_collapse_report(
    model: UCSA,
    inputs: Sequence[Tensor],
    variance_floor: float = 1e-8,
    mutual_info_floor: float = 1e-4,
    read_share_floor: float = 0.05,
) -> CollapseReport:
    """Measure whether ``G`` is ignoring the intent bank.

    This is the diagnostic that has to be read *before* trusting any other
    origination number. Collapse is silent everywhere else: the generator
    learns to ignore the bank, the loop degenerates to feeding the same
    observation every iteration, and every loss still looks healthy.

    Args:
        model: The model to probe.
        inputs: Two or more differing token batches. Variance and mutual
            information are measured across them.
        variance_floor: Origination-state variance at or below this counts
            as a constant signal.
        mutual_info_floor: Gate mutual information at or below this counts
            as a gate that ignores its input.
        read_share_floor: Intent read share at or below this counts as a
            generator driven by working memory alone.

    Returns:
        The :class:`CollapseReport`.

    Raises:
        ValueError: If fewer than two inputs are given.
    """
    if len(inputs) < 2:
        raise ValueError(
            f"need at least two differing inputs, got {len(inputs)}."
        )
    num_slots = model.pcs.bank_size(INTENT_BANK)
    generator = model.heads.origination
    states: list[Tensor] = []
    usages: list[Tensor] = []
    shares: list[float] = []
    used_slots: set[int] = set()
    snapshot = pcs_snapshot(model)
    try:
        for batch in inputs:
            with torch.no_grad():
                action_logits(model, batch)
            pcs_restore(model, snapshot)
            loop_states = model.reasoning_loop.last_intent_states
            if loop_states:
                states.append(loop_states[-1].detach())
            gate_weights = generator.last_gate_weights
            if gate_weights is not None:
                usage = intent_gate_usage(gate_weights, num_slots)
                usages.append(usage)
                used_slots.update(
                    int(i) for i in (usage > 0.0).nonzero().flatten()
                )
            intent_read = generator.last_intent_read
            working_read = generator.last_working_read
            if intent_read is not None and working_read is not None:
                shares.append(intent_read_share(intent_read, working_read))
    finally:
        pcs_restore(model, snapshot)

    variance = intent_state_variance(states)
    entropy = (
        float(sum(intent_gate_entropy(u) for u in usages) / len(usages))
        if usages
        else 0.0
    )
    mutual_info = intent_gate_mutual_info(usages)
    share = float(sum(shares) / len(shares)) if shares else 0.0
    ceiling = float(torch.tensor(float(num_slots)).log()) if num_slots else 0.0

    reasons: list[str] = []
    if not states:
        reasons.append("generator never ran: no origination state observed")
    if variance <= variance_floor:
        reasons.append(
            f"origination state is constant across inputs "
            f"(variance {variance:.3e} <= {variance_floor:.3e})"
        )
    if mutual_info <= mutual_info_floor:
        reasons.append(
            f"gate does not condition on the input "
            f"(mutual info {mutual_info:.3e} <= {mutual_info_floor:.3e})"
        )
    if share <= read_share_floor:
        reasons.append(
            f"generator is driven by working memory, not intent "
            f"(read share {share:.3f} <= {read_share_floor:.3f})"
        )
    return CollapseReport(
        state_variance=variance,
        gate_entropy=entropy,
        gate_entropy_max=ceiling,
        gate_mutual_info=mutual_info,
        read_share=share,
        slots_used=len(used_slots),
        num_slots=num_slots,
        collapsed=bool(reasons),
        reasons=reasons,
    )


__all__ = [
    "INTENT_BANK",
    "AttributionReport",
    "CollapseReport",
    "ControllabilityReport",
    "InterventionReport",
    "action_logits",
    "action_readout",
    "chain_latents",
    "direction_agreement",
    "counterfactual_controllability",
    "intent_collapse_report",
    "intent_attribution",
    "intervene_intent",
]
