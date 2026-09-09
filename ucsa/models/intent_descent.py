"""Inference-time optimisation of the origination state.

Phase D of the endogenous-origination work: having made the signal that
precedes an action explicit (the ``intent`` bank) and localisable (the
sparse gate plus the probes in :mod:`ucsa.models.origination`), this module
*updates* it. Weights stay frozen; only the intent bank moves.

The objective is the model's own forward model. The multi-step JEPA chain
already predicts each next latent from the previous one, so the chain's
prediction error is a score the model can compute for a candidate
origination *before* committing to the action it would cause. Descending
that error with respect to the intent bank alone is latent-space model
predictive control, or active inference, over the origination signal.

Every candidate origination is evaluated from the *same* cognitive state.
A forward pass rewrites the PCS, so K rollouts launched back to back would
each start somewhere new and the "improvement" would be state drift rather
than a better origination -- measured, not assumed: the first version of
this module reported the objective falling 0.0069 to 0.0028 purely from
drift, with the intent bank barely moving. The banks are therefore restored
before every rollout, and the entry state is restored afterwards, so
speculative rollouts never commit. Only the intent bank carries over, which
is the point.

Two guards, because both failure modes are real:

- ``K`` defaults to ``0``. Nothing here runs unless it is asked for.
- :func:`outcome_correlation` measures agreement between the predicted
  outcome and the realised one *after* optimising. Descent that improves
  the predicted score while the realised score gets worse is the forward
  model being gamed, not the origination being improved, and the caller can
  see that in the report rather than having to trust the objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from ucsa.models.origination import pcs_restore, pcs_snapshot
from ucsa.models.state import INTENT_BANK
from ucsa.models.verification import LearnedVerifier

if TYPE_CHECKING:
    from ucsa.models.ucsa import UCSA


@dataclass
class DescentStep:
    """One inner-loop step.

    Attributes:
        step: Zero-based step index.
        objective: The predicted-outcome error being minimised.
        grad_norm: L2 norm of the gradient with respect to the intent bank.
        critic_score: Optional outcome-critic score for this step.
    """

    step: int
    objective: float
    grad_norm: float
    critic_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "step": self.step,
            "objective": self.objective,
            "grad_norm": self.grad_norm,
            "critic_score": self.critic_score,
        }


@dataclass
class DescentReport:
    """Result of optimising the origination state for one input.

    Attributes:
        steps: One entry per inner step actually taken.
        initial_objective: Predicted-outcome error before descending.
        final_objective: Predicted-outcome error after descending.
        initial_realized: Realised outcome (autoregressive loss proxy)
            before descending, when a target was supplied.
        final_realized: Realised outcome after descending.
        stopped_early: Whether the gradient-norm threshold ended the loop.
        intent_shift: Relative L2 movement of the intent bank.
        forward_passes: Number of extra forward passes spent, for
            matched-compute accounting.
    """

    steps: list[DescentStep] = field(default_factory=list)
    initial_objective: float = 0.0
    final_objective: float = 0.0
    initial_realized: float | None = None
    final_realized: float | None = None
    stopped_early: bool = False
    intent_shift: float = 0.0
    forward_passes: int = 0

    @property
    def objective_improved(self) -> bool:
        """Whether the predicted-outcome error went down."""
        return self.final_objective < self.initial_objective

    @property
    def realized_improved(self) -> bool | None:
        """Whether the realised outcome went down, if it was measured."""
        if self.initial_realized is None or self.final_realized is None:
            return None
        return self.final_realized < self.initial_realized

    @property
    def forward_model_gamed(self) -> bool:
        """Whether the predicted score improved while the real one did not.

        This is the second failure mode: descent finds an origination the
        JEPA predictor likes and the decoder does not.
        """
        realized = self.realized_improved
        if realized is None:
            return False
        return self.objective_improved and not realized

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "initial_realized": self.initial_realized,
            "final_realized": self.final_realized,
            "stopped_early": self.stopped_early,
            "intent_shift": self.intent_shift,
            "forward_passes": self.forward_passes,
            "objective_improved": self.objective_improved,
            "realized_improved": self.realized_improved,
            "forward_model_gamed": self.forward_model_gamed,
        }


def jepa_chain_error(
    outputs: dict[str, Any],
    target_outputs: dict[str, Any] | None = None,
) -> Tensor:
    """Return the multi-step JEPA prediction error from a forward pass.

    This is the model's own forward model scoring itself: each step of the
    chain predicts the next working-memory latent.

    Pass ``target_outputs`` from an EMA target encoder whenever the error is
    being minimised *with respect to the intent bank*. Without it both sides
    of every pair come from the same forward pass, so moving the origination
    moves the prediction and its target together: the error is then a
    self-consistency measure whose gradient with respect to intent is
    roughly noise (measured at ~3e-7, with the objective refusing to fall
    even when the bank moved by 2.2x its own norm). The EMA copy holds its
    own frozen intent bank, so its latents do not follow the optimisation
    and the comparison becomes a real prediction error.

    Args:
        outputs: A :meth:`ucsa.models.ucsa.UCSA.forward` result.
        target_outputs: Optional forward result from an EMA target encoder,
            supplying the targets.

    Returns:
        Scalar tensor. Zero when the chain is absent.
    """
    pairs = outputs.get("jepa_multi_step") or []
    if not pairs:
        predicted = outputs.get("jepa_predicted")
        target = outputs.get("jepa_target")
        if predicted is None or target is None:
            return torch.zeros(())
        pairs = [(predicted, target)]
    targets: list[Tensor] = [target for _, target in pairs]
    if target_outputs is not None:
        ema_pairs = target_outputs.get("jepa_multi_step") or []
        if len(ema_pairs) >= len(pairs):
            targets = [ema_pairs[k][0] for k in range(len(pairs))]
        else:
            ema_predicted = target_outputs.get("jepa_predicted")
            if ema_predicted is not None:
                targets = [ema_predicted for _ in pairs]
    errors = [
        torch.nn.functional.smooth_l1_loss(predicted, target.detach())
        for (predicted, _), target in zip(pairs, targets, strict=True)
    ]
    return torch.stack(errors).mean()


def ema_outputs(
    target_encoder: torch.nn.Module | None, inputs: Tensor
) -> dict[str, Any] | None:
    """Run the EMA target encoder once, under ``no_grad``.

    The encoder keeps its own frozen copy of every bank, so its latents are
    unaffected by the intent bank being optimised. That independence is the
    whole reason to use it as the target.

    Args:
        target_encoder: The encoder, or ``None``.
        inputs: Token ids of shape ``(batch, seq)``.

    Returns:
        The encoder's forward result, or ``None``.
    """
    if target_encoder is None:
        return None
    inner: UCSA | None = getattr(target_encoder, "target", None)
    has_pcs = inner is not None and hasattr(inner, "pcs")
    snapshot = pcs_snapshot(inner) if has_pcs and inner is not None else None
    try:
        with torch.no_grad():
            result = target_encoder(inputs)
    finally:
        # The encoder's own forward rewrites its banks, so without this the
        # target moves every time it is consulted and the objective drifts
        # under the optimiser's feet.
        if snapshot is not None and inner is not None:
            pcs_restore(inner, snapshot)
    return dict(result) if isinstance(result, dict) else None


def jepa_step_errors(
    outputs: dict[str, Any],
    target_outputs: dict[str, Any] | None = None,
) -> list[float]:
    """Return the JEPA prediction error at each step of the chain.

    The origination generator only shapes iteration 2 onward, so if intent
    is really steering the loop the *late* steps of the chain should improve
    more than the early ones. A uniform profile means the origination is not
    doing the work.

    Args:
        outputs: A :meth:`ucsa.models.ucsa.UCSA.forward` result.
        target_outputs: Optional EMA target encoder result.

    Returns:
        One error per chain step, earliest first.
    """
    pairs = outputs.get("jepa_multi_step") or []
    if not pairs:
        return []
    targets: list[Tensor] = [target for _, target in pairs]
    if target_outputs is not None:
        ema_pairs = target_outputs.get("jepa_multi_step") or []
        if len(ema_pairs) >= len(pairs):
            targets = [ema_pairs[k][0] for k in range(len(pairs))]
    return [
        float(
            torch.nn.functional.smooth_l1_loss(
                predicted, target.detach()
            ).item()
        )
        for (predicted, _), target in zip(pairs, targets, strict=True)
    ]


def realized_outcome(
    outputs: dict[str, Any], targets: Tensor | None
) -> float | None:
    """Score the action the model actually emitted.

    Aligned the way the trainer aligns them. ``Trainer.compute_loss``
    left-pads short targets, so the supervised positions are the *last*
    ``len(targets)`` of the logits, not the first. Scoring the leading
    positions instead reads slots the model was never trained on, which
    pins the result near chance however well the model has learned: on a
    model at training loss 0.13 the leading-position readout reported 3.64
    against a chance level of 3.43.

    Args:
        outputs: A forward-pass result.
        targets: Target token ids of shape ``(batch, seq)``, or ``None``.

    Returns:
        Cross-entropy against ``targets``, or ``None`` when no target was
        supplied.
    """
    if targets is None:
        return None
    logits = outputs.get("language")
    if logits is None:
        return None
    span = min(logits.shape[1], targets.shape[1])
    if span == 0:
        return None
    aligned_logits = logits[:, -span:, :]
    aligned_targets = targets[:, -span:].to(logits.device)
    return float(
        torch.nn.functional.cross_entropy(
            aligned_logits.reshape(-1, aligned_logits.shape[-1]),
            aligned_targets.reshape(-1),
        ).item()
    )


def critic_score(model: UCSA, outputs: dict[str, Any]) -> float | None:
    """Score the current working memory with the learned verifier.

    The verifier is UCSA's existing acceptance critic. It is optional here:
    only a :class:`~ucsa.models.verification.LearnedVerifier` produces a
    differentiable-quality signal worth reading during descent.

    Args:
        model: The model whose verifier is consulted.
        outputs: A forward-pass result.

    Returns:
        The critic's scalar score, or ``None`` when unavailable.
    """
    verifier = getattr(model, "verifier", None)
    if not isinstance(verifier, LearnedVerifier):
        return None
    working = outputs.get("language")
    if working is None:
        return None
    with torch.no_grad():
        candidate = model.pcs.get_bank("working").detach()
        pooled = candidate.mean(dim=0)
        summary = verifier.summarize_cstate(model.pcs)
        full = torch.cat([pooled, summary], dim=-1)
        logit = verifier.mlp(full).reshape(-1)[0]
    return float(logit.item())


def critic_objective(
    model: UCSA,
    outputs: dict[str, Any],
) -> Tensor | None:
    """Verifier logit as a *differentiable* objective.

    The spec lists the LearnedVerifier as the outcome critic. The
    ``critic_score`` helper above is a logged metric, taken under
    ``no_grad``; this one is the *objective* the descent can descend on.
    It exists as a separate path so the logged score does not need to
    carry gradient just to be reported.

    The score is the verifier's confidence in the current working memory.
    The working bank is the model's *current* state, not the pre-rewrite
    ``no_grad`` snapshot the log uses, so descending it actually moves the
    bank's influence on the verifier's next assessment.

    Args:
        model: The model whose verifier is consulted.
        outputs: A forward-pass result. Currently unused -- the objective
            is computed from the working bank's current contents rather
            than the forward output, so the bank is the bridge for the
            gradient. Kept on the signature so callers do not have to
            special-case which side of the call surface they pass.

    Returns:
        Verifier logit as a tensor with grad, or ``None`` when the model
        does not carry a :class:`LearnedVerifier`.
    """
    verifier = getattr(model, "verifier", None)
    if not isinstance(verifier, LearnedVerifier):
        return None
    candidate = model.pcs.get_bank("working")
    pooled = candidate.mean(dim=0)
    summary = verifier.summarize_cstate(model.pcs)
    full = torch.cat([pooled, summary], dim=-1)
    logit: Tensor = verifier.mlp(full).reshape(-1)[0]
    return logit


def _resolved_objective(model: UCSA, requested: str) -> str:
    """Pick the actual objective to descend on.

    Args:
        model: The model whose verifier availability matters.
        requested: One of ``"auto"``, ``"jepa"``, ``"critic"``.

    Returns:
        ``"critic"`` when the request is satisfied by a
        :class:`LearnedVerifier` on the model, ``"jepa"`` otherwise, and the
        literal request when it is not ``"auto"``.
    """
    if requested != "auto":
        return requested
    verifier = getattr(model, "verifier", None)
    if isinstance(verifier, LearnedVerifier):
        return "critic"
    return "jepa"


def optimize_intent(
    model: UCSA,
    inputs: Tensor,
    num_steps: int = 0,
    learning_rate: float = 0.05,
    grad_norm_threshold: float = 0.0,
    grad_norm_relative_threshold: float = 0.0,
    targets: Tensor | None = None,
    restore: bool = False,
    normalize_gradient: bool = True,
    target_encoder: torch.nn.Module | None = None,
    objective: str = "auto",
) -> DescentReport:
    """Descend on the ``intent`` bank only, with the weights frozen.

    Args:
        model: The model to optimise. Left in eval mode; its parameters are
            never updated.
        inputs: Token ids of shape ``(batch, seq)``.
        num_steps: ``K``, the number of inner steps. ``0`` (the default)
            does nothing beyond one scoring pass.
        learning_rate: Step size on the intent bank.
        grad_norm_threshold: Stop once the intent gradient norm falls to or
            below this absolute value. ``0.0`` disables it. An absolute
            threshold does not adapt: intent gradient norms sit in a narrow
            band across inputs, so any given value stops every input at the
            same step or none of them.
        grad_norm_relative_threshold: Stop once the gradient norm falls to
            or below this fraction of its value at the first step. ``0.0``
            disables it. This is the criterion that actually varies per
            input, because it is measured against that input's own starting
            gradient.
        targets: Optional target token ids. When given, the realised
            outcome is measured before and after so the caller can see
            whether the forward model was gamed.
        restore: When ``True`` the optimised bank is rolled back before
            returning, leaving the model untouched. Useful for measurement.
        normalize_gradient: Step along the unit-normalised gradient, so
            ``learning_rate`` is a step size in bank-norm units rather than
            being at the mercy of the gradient's scale. The raw intent
            gradient here is around ``1e-7``, so no fixed raw step size
            would move the bank at all on one model and not diverge on
            another. Set ``False`` for plain gradient descent.
        target_encoder: Optional EMA target encoder supplying the JEPA
            targets. Strongly recommended: without it the objective's
            targets move with the origination being optimised and the
            gradient is close to noise.

    Returns:
        The :class:`DescentReport`.

    Raises:
        ValueError: If ``num_steps`` or ``learning_rate`` is negative.
    """
    if num_steps < 0:
        raise ValueError(f"num_steps must be >= 0, got {num_steps}.")
    if learning_rate < 0.0:
        raise ValueError(
            f"learning_rate must be non-negative, got {learning_rate}."
        )
    if objective not in ("auto", "jepa", "critic"):
        raise ValueError(
            f"objective must be 'auto', 'jepa' or 'critic', got {objective!r}."
        )
    objective_mode = _resolved_objective(model, objective)
    bank = model.pcs.get_bank(INTENT_BANK)
    original = bank.detach().clone()
    frozen = [
        parameter
        for parameter in model.parameters()
        if parameter is not bank and parameter.requires_grad
    ]
    previous_requires_grad = [parameter.requires_grad for parameter in frozen]
    for parameter in frozen:
        parameter.requires_grad_(False)
    was_training = model.training
    model.eval()
    report = DescentReport()
    initial_grad_norm = 0.0
    snapshot = pcs_snapshot(model)
    context = {
        name: tensor for name, tensor in snapshot.items() if name != INTENT_BANK
    }
    try:
        pcs_restore(model, context)
        target_outputs = ema_outputs(target_encoder, inputs)
        outputs = model(inputs)
        report.initial_objective = float(
            jepa_chain_error(outputs, target_outputs).item()
        )
        report.initial_realized = realized_outcome(outputs, targets)
        report.final_objective = report.initial_objective
        report.final_realized = report.initial_realized
        report.forward_passes = 1
        for step in range(num_steps):
            model.zero_grad(set_to_none=True)
            pcs_restore(model, context)
            outputs = model(inputs)
            report.forward_passes += 1
            objective_tensor: Tensor | None
            if objective_mode == "jepa":
                objective_tensor = jepa_chain_error(outputs, target_outputs)
            else:  # "critic"
                objective_tensor = critic_objective(model, outputs)
            if objective_tensor is None or not objective_tensor.requires_grad:
                break
            objective_tensor.backward()  # type: ignore[no-untyped-call]
            gradient = bank.grad
            grad_norm = (
                0.0 if gradient is None else float(gradient.norm().item())
            )
            objective_value = float(objective_tensor.item())
            report.steps.append(
                DescentStep(
                    step=step,
                    objective=objective_value,
                    grad_norm=grad_norm,
                    critic_score=critic_score(model, outputs),
                )
            )
            if step == 0:
                initial_grad_norm = grad_norm
            if grad_norm <= grad_norm_threshold:
                report.stopped_early = True
                break
            if (
                grad_norm_relative_threshold > 0.0
                and initial_grad_norm > 0.0
                and grad_norm
                <= grad_norm_relative_threshold * initial_grad_norm
            ):
                report.stopped_early = True
                break
            if gradient is not None:
                direction = gradient
                if normalize_gradient and grad_norm > 0.0:
                    direction = gradient / grad_norm
                with torch.no_grad():
                    bank.sub_(learning_rate * direction)
        model.zero_grad(set_to_none=True)
        pcs_restore(model, context)
        with torch.no_grad():
            final_outputs = model(inputs)
        report.forward_passes += 1
        report.final_objective = float(
            jepa_chain_error(final_outputs, target_outputs).item()
        )
        report.final_realized = realized_outcome(final_outputs, targets)
        denominator = float(original.norm())
        shift = float((bank.detach() - original).norm())
        report.intent_shift = (
            shift / denominator if denominator > 0.0 else shift
        )
    finally:
        optimized = bank.detach().clone()
        pcs_restore(model, snapshot)
        if not restore:
            with torch.no_grad():
                bank.copy_(optimized)
        for parameter, flag in zip(
            frozen, previous_requires_grad, strict=False
        ):
            parameter.requires_grad_(flag)
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
    return report


@dataclass
class MatchedComputeRow:
    """One arm of a compute-matched comparison.

    Attributes:
        arm: ``"intent-optimization"`` or ``"more-reasoning"``.
        operator_calls: Transition-operator invocations per input, the
            common currency the two arms are matched on.
        forward_passes: Model forward passes per input.
        reasoning_iterations: Loop iterations per forward pass.
        intent_steps: Inner descent steps per input.
        realized: Mean realised outcome (lower is better).
    """

    arm: str
    operator_calls: int
    forward_passes: int
    reasoning_iterations: int
    intent_steps: int
    realized: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "arm": self.arm,
            "operator_calls": self.operator_calls,
            "forward_passes": self.forward_passes,
            "reasoning_iterations": self.reasoning_iterations,
            "intent_steps": self.intent_steps,
            "realized": self.realized,
        }


def compute_matched_comparison(
    model: UCSA,
    pairs: Sequence[tuple[Tensor, Tensor]],
    intent_steps: int,
    learning_rate: float = 0.05,
    target_encoder: torch.nn.Module | None = None,
) -> list[MatchedComputeRow]:
    """Compare intent optimisation against spending the same compute on
    more reasoning.

    A ``K=0`` arm is *cheaper*, not matched, so comparing against it would
    credit intent optimisation for compute rather than for origination. The
    honest control spends the same budget the other way the architecture
    allows: more passes through the transition operator.

    Both arms are costed in operator calls per input. The optimisation arm
    makes ``(2 + K)`` forward passes at the model's configured
    ``reasoning_iterations``. Two controls spend the same budget:

    - ``more-reasoning`` raises the loop's iteration count. Note this runs
      the model out of distribution -- it was trained at a fixed iteration
      count -- so it is a weak control and its score degrades as the budget
      grows. Reported for completeness, not relied on.
    - ``repeat-and-average`` makes the same number of forward passes and
      averages their logits. Every pass is in distribution, which makes
      this the control the comparison should be judged against.

    Args:
        model: The model to measure. Restored before returning.
        pairs: ``(inputs, targets)`` probe pairs.
        intent_steps: ``K`` for the optimisation arm.
        learning_rate: Step size on the intent bank.
        target_encoder: Optional EMA target encoder for the objective.

    Returns:
        One :class:`MatchedComputeRow` per arm.

    Raises:
        ValueError: If ``intent_steps`` is not positive or ``pairs`` is
            empty.
    """
    if intent_steps <= 0:
        raise ValueError(
            f"intent_steps must be positive to compare, got {intent_steps}."
        )
    if not pairs:
        raise ValueError("need at least one probe pair.")
    base_iterations = model.reasoning_loop.config.num_iterations
    forward_passes = 2 + intent_steps
    budget = forward_passes * base_iterations

    optimised: list[float] = []
    for inputs, targets in pairs:
        report = optimize_intent(
            model,
            inputs,
            num_steps=intent_steps,
            learning_rate=learning_rate,
            targets=targets,
            restore=True,
            target_encoder=target_encoder,
        )
        if report.final_realized is not None:
            optimised.append(report.final_realized)

    control: list[float] = []
    original_config = model.reasoning_loop.config
    snapshot = pcs_snapshot(model)
    try:
        model.reasoning_loop.config = replace(
            original_config, num_iterations=budget
        )
        for inputs, targets in pairs:
            pcs_restore(model, snapshot)
            with torch.no_grad():
                outputs = model(inputs)
            value = realized_outcome(outputs, targets)
            if value is not None:
                control.append(value)
    finally:
        model.reasoning_loop.config = original_config
        pcs_restore(model, snapshot)

    ensemble: list[float] = []
    try:
        for inputs, targets in pairs:
            pcs_restore(model, snapshot)
            accumulated: Tensor | None = None
            with torch.no_grad():
                for _ in range(forward_passes):
                    logits = model(inputs)["language"]
                    accumulated = (
                        logits.clone()
                        if accumulated is None
                        else accumulated + logits
                    )
            if accumulated is None:
                continue
            averaged = accumulated / float(forward_passes)
            value = realized_outcome({"language": averaged}, targets)
            if value is not None:
                ensemble.append(value)
    finally:
        pcs_restore(model, snapshot)

    return [
        MatchedComputeRow(
            arm="intent-optimization",
            operator_calls=budget,
            forward_passes=forward_passes,
            reasoning_iterations=base_iterations,
            intent_steps=intent_steps,
            realized=sum(optimised) / len(optimised) if optimised else 0.0,
        ),
        MatchedComputeRow(
            arm="more-reasoning",
            operator_calls=budget,
            forward_passes=1,
            reasoning_iterations=budget,
            intent_steps=0,
            realized=sum(control) / len(control) if control else 0.0,
        ),
        MatchedComputeRow(
            arm="repeat-and-average",
            operator_calls=budget,
            forward_passes=forward_passes,
            reasoning_iterations=base_iterations,
            intent_steps=0,
            realized=sum(ensemble) / len(ensemble) if ensemble else 0.0,
        ),
    ]


def outcome_correlation(reports: Sequence[DescentReport]) -> float:
    """Correlation between predicted-outcome and realised-outcome change.

    The forward-model-hacking check. Descent minimises the *predicted*
    error, so a positive correlation across inputs means improving the
    prediction really does improve the emitted action. A correlation at or
    below zero means the objective and the outcome have come apart and the
    descent is exploiting the predictor.

    Args:
        reports: Reports from :func:`optimize_intent`, each with a realised
            outcome measured.

    Returns:
        Pearson correlation of the two deltas, or ``0.0`` when fewer than
        two reports carry a realised outcome or either series is constant.
    """
    predicted: list[float] = []
    realized: list[float] = []
    for report in reports:
        if report.initial_realized is None or report.final_realized is None:
            continue
        predicted.append(report.final_objective - report.initial_objective)
        realized.append(report.final_realized - report.initial_realized)
    if len(predicted) < 2:
        return 0.0
    first = torch.tensor(predicted)
    second = torch.tensor(realized)
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(first.norm() * second.norm())
    if denominator <= 0.0:
        return 0.0
    return float((first * second).sum().item() / denominator)


__all__ = [
    "DescentReport",
    "DescentStep",
    "MatchedComputeRow",
    "compute_matched_comparison",
    "critic_objective",
    "critic_score",
    "jepa_chain_error",
    "jepa_step_errors",
    "optimize_intent",
    "outcome_correlation",
    "realized_outcome",
]
