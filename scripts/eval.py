"""Evaluates a trained checkpoint against published reference results.

Reports zero-shot accuracy on HellaSwag, ARC-Easy, ARC-Challenge, PIQA and
WinoGrande over the full evaluation splits, held-out perplexity on the
last 64 positions of fixed windows, and a comparison with the numbers in
`paper/reference_results.json`.

    python scripts/eval.py --ckpt ckpts/r-small/final.pt \
        --out-json runs/eval-r-small.json
"""

import argparse
import json
import os

import torch
import transformers

from ucsa.models import recurrent
from ucsa.training import engine, eval_harness, reference, scoring, shards

PPL_SETS = {"fineweb-edu": "val.bin", "wikitext-103": "wikitext_test.bin"}


@torch.no_grad()
def heldout_ppl(
    model: recurrent.Model,
    shard: shards.Shard,
    device: torch.device,
    batches: int,
    seq_len: int = 1024,
) -> float:
    """Measures perplexity of the last 64 positions of fixed windows.

    Args:
      model: Model to score.
      shard: Held-out tokens.
      device: Device holding the model.
      batches: Maximum number of windows to score.
      seq_len: Window length.

    Returns:
      Perplexity over the scored positions.
    """
    k = scoring.DEFAULT_NUM_TARGETS
    model.eval()
    nll, count = 0.0, 0
    stream = shard.batches(1, seq_len, seed=None, loop=False)
    for n, (x, y) in enumerate(stream):
        if n >= batches:
            break
        x, y = x.to(device), y.to(device)
        total, tokens = scoring.tail_nll(model(x)["logits"], y, k)
        nll, count = nll + total, count + tokens
    return scoring.perplexity(nll, count)


def main() -> None:
    """Parses arguments, evaluates, prints the comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="UCSA-R checkpoint")
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
    device = engine.pick_device()
    print(f"Device: {device}", flush=True)
    model = engine.load_model(args.ckpt, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    results = eval_harness.evaluate_all(
        args.tasks,
        model,
        tokenizer,
        device,
        max_examples=args.max_examples or None,
    )
    report: dict = {
        "device": str(device),
        "tasks": {r.name: r.to_dict() for r in results},
        "avg_acc": sum(r.accuracy for r in results) / max(1, len(results)),
        "ppl": {},
        "params": sum(p.numel() for p in model.parameters()),
    }
    for label, filename in PPL_SETS.items():
        path = os.path.join(args.data, filename)
        if not os.path.exists(path):
            continue
        ppl = heldout_ppl(
            model, shards.Shard(path), device, args.ppl_batches, args.max_seq_len
        )
        report["ppl"][label] = ppl
        print(f"{label} ppl_last64: {ppl:.1f}", flush=True)

    ref = reference.load_reference()
    rows = reference.compare(report["tasks"], ref)
    report["reference_comparison"] = rows
    print(reference.format_table(rows, ref), flush=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
