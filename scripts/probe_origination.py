"""Origination probe: is the intent bank used, localised, and useful?

Produces the three readings the endogenous-origination work is judged on,
as one JSON file:

1. **Collapse** -- is ``G`` actually using the intent bank, or has it
   learned to ignore it so the reasoning loop quietly degenerates to
   feeding the same observation every iteration? Read this first; nothing
   else means anything while it is red.
2. **Localisation** -- for a given emitted action, do ``k`` intent slots
   carry the gradient and the rest not? Does ablating an attributed slot
   move the action while ablating an unattributed one does not?
3. **Descent** -- does optimising the origination at inference improve the
   outcome, and at what compute? Reported with the forward-pass count and
   with the predicted-versus-realised correlation, because descent that
   improves the forward model's score while the real outcome worsens is
   the model gaming its own predictor.

Run without ``--ckpt`` to probe a freshly initialised model, which is only
useful as a smoke test: an untrained origination has nothing to say.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import yaml

from ucsa.models.intent_descent import (
    compute_matched_comparison,
    jepa_step_errors,
    optimize_intent,
    outcome_correlation,
)
from ucsa.models.origination import (
    counterfactual_controllability,
    intent_collapse_report,
)
from ucsa.train import build_model
from ucsa.training.ema import EMATargetEncoder
from ucsa.utils.checkpoint import load_state_dict_compat
from ucsa.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="UCSA origination probe")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--ucsa-config", default="ucsa/configs/default.yaml")
    parser.add_argument("--out-json", default="runs/origination-probe.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inputs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument(
        "--intent-steps",
        nargs="*",
        type=int,
        default=[0, 1, 3, 5],
        help="K values to sweep. Each K costs 2 + K forward passes.",
    )
    parser.add_argument("--intent-learning-rate", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1e-3)
    parser.add_argument("--observation-mix", type=float, default=0.5)
    return parser.parse_args()


def build_probe_inputs(
    vocab_size: int, count: int, seq_len: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(inputs, targets)`` pairs with a learnable structure.

    The sequences repeat their first half, so there is something for the
    model to get right and therefore something for an origination to
    improve.

    Args:
        vocab_size: Tokeniser vocabulary size.
        count: Number of pairs.
        seq_len: Total sequence length before the shift.
        seed: RNG seed.

    Returns:
        List of ``(inputs, targets)`` pairs.
    """
    generator = torch.Generator().manual_seed(seed)
    half = max(1, seq_len // 2)
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(count):
        chunk = torch.randint(
            0, max(2, vocab_size // 2), (1, half), generator=generator
        )
        sequence = torch.cat([chunk, chunk], dim=1)
        pairs.append((sequence[:, :-1], sequence[:, 1:]))
    return pairs


def descent_sweep(
    model: torch.nn.Module,
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    steps: list[int],
    learning_rate: float,
) -> list[dict[str, object]]:
    """Sweep ``K`` and report cost alongside quality.

    Args:
        model: The model to optimise.
        pairs: Probe ``(inputs, targets)`` pairs.
        steps: ``K`` values to try.
        learning_rate: Step size on the intent bank.

    Returns:
        One row per ``K``.
    """
    encoder = EMATargetEncoder(model, momentum=0.99)
    rows: list[dict[str, object]] = []
    for k in steps:
        reports = [
            optimize_intent(
                model,
                inputs,
                num_steps=k,
                learning_rate=learning_rate,
                targets=targets,
                restore=True,
                target_encoder=encoder,
            )
            for inputs, targets in pairs
        ]
        total = len(reports) or 1
        rows.append(
            {
                "intent_steps": k,
                "forward_passes_per_input": reports[0].forward_passes,
                "predicted_improved": sum(
                    r.objective_improved for r in reports
                ),
                "realized_improved": sum(
                    bool(r.realized_improved) for r in reports
                ),
                "forward_model_gamed": sum(
                    r.forward_model_gamed for r in reports
                ),
                "num_inputs": len(reports),
                "mean_intent_shift": sum(r.intent_shift for r in reports)
                / total,
                "outcome_correlation": outcome_correlation(reports),
            }
        )
    return rows


def print_human(report: dict[str, object]) -> None:
    """Print a compact summary."""
    collapse = report["collapse"]
    assert isinstance(collapse, dict)
    print("\nOrigination probe", flush=True)
    verdict = "COLLAPSED" if collapse["collapsed"] else "healthy"
    print(f"  collapse: {verdict}", flush=True)
    print(
        f"    state variance {collapse['state_variance']:.3e}  "
        f"gate MI {collapse['gate_mutual_info']:.4f}  "
        f"gate H {collapse['gate_entropy']:.3f}/"
        f"{collapse['gate_entropy_max']:.3f}  "
        f"intent read share {collapse['read_share']:.3f}  "
        f"slots {collapse['slots_used']}/{collapse['num_slots']}",
        flush=True,
    )
    for reason in collapse["reasons"]:
        print(f"    - {reason}", flush=True)
    localisation = report["localisation"]
    assert isinstance(localisation, dict)
    attribution = localisation["attribution"]
    assert isinstance(attribution, dict)
    print(
        f"  localisation: {len(attribution['active_slots'])}"
        f"/{len(attribution['scores'])} slots carry gradient, "
        f"gated {attribution['gated_slots']}",
        flush=True,
    )
    print(
        f"    controllability {localisation['controllability']:.3f}  "
        f"specificity {localisation['specificity']:.3f}",
        flush=True,
    )
    chain = report["jepa_step_errors"]
    assert isinstance(chain, list)
    if chain:
        rendered = "  ".join(f"k{k}={v:.5f}" for k, v in enumerate(chain))
        print(f"  jepa chain error by step: {rendered}", flush=True)
    print("  descent (matched compute is the pass count):", flush=True)
    rows = report["descent"]
    assert isinstance(rows, list)
    for row in rows:
        print(
            f"    K={row['intent_steps']:<2} "
            f"passes/input={row['forward_passes_per_input']:<2} "
            f"predicted+ {row['predicted_improved']}/{row['num_inputs']}  "
            f"realised+ {row['realized_improved']}/{row['num_inputs']}  "
            f"gamed {row['forward_model_gamed']}/{row['num_inputs']}  "
            f"corr {row['outcome_correlation']:+.3f}",
            flush=True,
        )
    matched_rows = report["matched_compute"]
    assert isinstance(matched_rows, list)
    if matched_rows:
        print("  matched compute (equal operator calls per input):", flush=True)
        for row in matched_rows:
            print(
                f"    K={row['intent_steps']:<2} {row['arm']:20s} "
                f"calls={row['operator_calls']:<3} "
                f"realised={row['realized']:.5f}",
                flush=True,
            )


def main() -> None:
    """Entry point."""
    args = parse_args()
    set_seed(args.seed)

    with open(args.ucsa_config) as handle:
        cfg = yaml.safe_load(handle)
    cfg["model"]["max_seq_len"] = cfg["model"].get("max_seq_len", 1024)
    cfg["model"]["observation_mix"] = args.observation_mix
    model = build_model(cfg)
    notes: list[str] = []
    if args.ckpt:
        from safetensors.torch import load_file

        state = load_file(args.ckpt)
        renamed = {
            name.removeprefix("model."): tensor
            for name, tensor in state.items()
        }
        notes = load_state_dict_compat(model, renamed, strict=False)
        for note in notes:
            print(f"ckpt compat: {note}", flush=True)
    model.eval()

    vocab_size = int(cfg["model"]["vocab_size"])
    pairs = build_probe_inputs(
        vocab_size, args.num_inputs, args.seq_len, args.seed
    )
    inputs_only = [inputs for inputs, _ in pairs]

    with torch.no_grad():
        chain_outputs = model(inputs_only[0])
    chain = jepa_step_errors(chain_outputs)

    collapse = intent_collapse_report(model, inputs_only)
    localisation = counterfactual_controllability(
        model, inputs_only[0], effect_threshold=args.effect_threshold
    )
    descent = descent_sweep(
        model, pairs, list(args.intent_steps), args.intent_learning_rate
    )

    matched: list[dict[str, object]] = []
    for k in sorted({s for s in args.intent_steps if s > 0}):
        matched.extend(
            row.to_dict()
            for row in compute_matched_comparison(
                model,
                pairs,
                intent_steps=k,
                learning_rate=args.intent_learning_rate,
                target_encoder=EMATargetEncoder(model, momentum=0.99),
            )
        )

    report: dict[str, object] = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "observation_mix": args.observation_mix,
        "ckpt_compat_notes": notes,
        "jepa_step_errors": chain,
        "collapse": collapse.to_dict(),
        "localisation": localisation.to_dict(),
        "descent": descent,
        "matched_compute": matched,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {args.out_json}", flush=True)
    print_human(report)


if __name__ == "__main__":
    main()
