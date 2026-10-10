"""Evaluates a trained checkpoint against published reference results.

Reports zero-shot accuracy on HellaSwag, ARC-Easy, ARC-Challenge, PIQA and
WinoGrande over the full evaluation splits, held-out perplexity on the
last 64 positions of fixed windows, and a comparison with the numbers in
`paper/reference_results.json`.

    python scripts/eval.py --recurrent-ckpt ckpts/r-small/final.pt \
        --out-json runs/eval-r-small.json
"""

import argparse
import json
import os

import small_config
import torch
import yaml
from safetensors import torch as safetensors_torch

from ucsa import train as ucsa_train
from ucsa.models import perception
from ucsa.training import engine, eval_harness, prefix, reference, shards
from ucsa.utils import checkpoint

PPL_SETS = {"fineweb-edu": "val.bin", "wikitext-103": "wikitext_test.bin"}


@torch.no_grad()
def heldout_ppl(
    model: torch.nn.Module,
    kind: str,
    shard: shards.Shard,
    device: torch.device,
    batches: int,
    seq_len: int = 1024,
) -> float:
    """Measures perplexity of the last 64 positions of fixed windows.

    Args:
      model: A `recurrent.Model` (`kind="recurrent"`) or the original slot
        `UCSA` (`kind="ucsa"`, which sees only the prefix).
      kind: Model family.
      shard: Held-out tokens.
      device: Device holding the model.
      batches: Maximum number of windows to score.
      seq_len: Window length.

    Returns:
      Perplexity over the scored positions.
    """
    k = prefix.DEFAULT_NUM_TARGETS
    model.eval()
    nll, count = 0.0, 0
    stream = shard.batches(1, seq_len, seed=None, loop=False)
    for n, (x, y) in enumerate(stream):
        if n >= batches:
            break
        x, y = x.to(device), y.to(device)
        if kind == "ucsa":
            head, target = prefix.split_prefix_targets(x, y, k)
            logits = model(head)["language"][:, :k, :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                reduction="sum",
            )
            nll, count = nll + loss.item(), count + target.numel()
        else:
            total, tokens = prefix.tail_nll(model(x)["logits"], y, k)
            nll, count = nll + total, count + tokens
    return prefix.perplexity(nll, count)


def run_model(
    name: str,
    model: torch.nn.Module,
    kind: str,
    args: argparse.Namespace,
    device: torch.device,
    report: dict,
) -> None:
    """Runs the benchmarks and perplexities for one model into `report`.

    Args:
      name: Report key prefix.
      model: Model to evaluate.
      kind: `recurrent` or `ucsa`.
      args: Parsed command-line arguments.
      device: Device holding the model.
      report: Dict updated in place.
    """
    tokenizer = perception.Tokenizer(
        tokenizer_name="gpt2", max_seq_len=args.max_seq_len
    )
    results = eval_harness.evaluate_all(args.tasks, model, tokenizer, device)
    report[f"{name}_tasks"] = {r.name: r.to_dict() for r in results}
    report[f"{name}_avg_acc"] = sum(r.accuracy for r in results) / max(
        1, len(results)
    )
    report[f"{name}_ppl"] = {}
    for label, filename in PPL_SETS.items():
        path = os.path.join(args.data, filename)
        if not os.path.exists(path):
            continue
        ppl = heldout_ppl(
            model,
            kind,
            shards.Shard(path),
            device,
            args.ppl_batches,
            args.max_seq_len,
        )
        report[f"{name}_ppl"][label] = ppl
        print(f"{name} {label} ppl_last64: {ppl:.1f}", flush=True)
    report[f"{name}_params"] = sum(p.numel() for p in model.parameters())


def main() -> None:
    """Parses arguments, evaluates, prints the comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recurrent-ckpt", default=None, help="UCSA-R .pt")
    parser.add_argument("--ucsa-ckpt", default=None, help="original UCSA")
    parser.add_argument("--ucsa-config", default="ucsa/config.yaml")
    parser.add_argument(
        "--tasks", nargs="*", default=list(eval_harness.TASK_REGISTRY)
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Cap per task; 0 (default) runs the full split.",
    )
    parser.add_argument("--data", default="data")
    parser.add_argument("--ppl-batches", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="runs/eval.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    for spec in eval_harness.TASK_REGISTRY.values():
        spec.max_examples = args.max_examples or None

    device = engine.pick_device()
    print(f"Device: {device}", flush=True)
    report: dict = {"device": str(device)}

    if args.recurrent_ckpt:
        model = engine.load_model(args.recurrent_ckpt, device)
        run_model("recurrent", model, "recurrent", args, device, report)
    if args.ucsa_ckpt:
        with open(args.ucsa_config) as f:
            config = yaml.safe_load(f)
        config["training"]["batch_size"] = 1
        small_config.apply_small_overrides(config)
        config["model"]["max_seq_len"] = args.max_seq_len
        model = ucsa_train.build_model(config)
        state = safetensors_torch.load_file(args.ucsa_ckpt)
        renamed = {k.removeprefix("model."): v for k, v in state.items()}
        checkpoint.load_state_dict_compat(model, renamed, strict=True)
        run_model("ucsa", model.to(device), "ucsa", args, device, report)

    primary = report.get("recurrent_tasks") or report.get("ucsa_tasks")
    if primary:
        ref = reference.load_reference()
        rows = reference.compare(primary, ref)
        report["reference_comparison"] = rows
        print(reference.format_table(rows, ref), flush=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
