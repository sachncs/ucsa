"""Standard-LM eval harness runner.

Loads a trained UCSA checkpoint (or runs a baseline zero-shot) and
reports accuracy on HellaSwag, ARC-e, ARC-c, PIQA, and WinoGrande.
Writes a JSON report next to ``--out-json`` for downstream
paper-writing tools.

Usage:
    .venv/bin/python scripts/eval.py \
        --ucsa-ckpt ckpts/ucsa-final.safetensors \
        --baseline-results runs/baseline.json \
        --out-json runs/eval-ucsa-small.json \
        [--tasks hellaswag arc_easy piqa]
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import yaml

from ucsa.models.perception import TokenizerWrapper
from ucsa.train import build_model
from ucsa.training.eval_harness import (
    TASK_REGISTRY,
    evaluate_all,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ucsa-ckpt", default=None,
        help="Path to a safetensors UCSA checkpoint. None = skip UCSA.",
    )
    p.add_argument(
        "--ucsa-config", default="ucsa/configs/default.yaml",
        help="Config used to build the UCSA model.",
    )
    p.add_argument(
        "--baseline-results", default=None,
        help="Path to a JSON file with a vanilla baseline's val_ppl.",
    )
    p.add_argument(
        "--tasks", nargs="*", default=list(TASK_REGISTRY.keys()),
        help="Subset of tasks to run.",
    )
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json", default="runs/eval.json",
        help="Output JSON file.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    with open(args.ucsa_config) as f:
        cfg = yaml.safe_load(f)

    device = (
        torch.device("mps", 0) if torch.backends.mps.is_available()
        else torch.device("cuda", 0) if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}", flush=True)

    report: dict = {"device": str(device), "tasks": {}, "ucsa_params": None}

    if args.ucsa_ckpt:
        print(f"Loading UCSA from {args.ucsa_ckpt} ...", flush=True)
        # Match the config to the checkpoint's architectural size.
        cfg["training"]["batch_size"] = 1
        cfg["model"]["hidden_size"] = cfg["model"].get("hidden_size", 384)
        cfg["model"]["max_seq_len"] = args.max_seq_len
        model = build_model(cfg)
        from safetensors.torch import load_file

        from ucsa.utils.checkpoint import load_state_dict_compat
        sd = load_file(args.ucsa_ckpt)
        renamed = {
            name.removeprefix("model."): t for name, t in sd.items()
        }
        for note in load_state_dict_compat(model, renamed, strict=False):
            print(f"  ckpt compat: {note}", flush=True)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report["ucsa_params"] = n_params
        model = model.to(device).eval()

        tokenizer = TokenizerWrapper(
            tokenizer_name=cfg["tokenizer"]["name"],
            max_seq_len=args.max_seq_len,
        )
        results = evaluate_all(args.tasks, model, tokenizer, device)
        for r in results:
            report["tasks"][r.name] = r.to_dict()
        avg = sum(r.accuracy for r in results) / max(1, len(results))
        report["ucsa_avg_acc"] = avg
        print(f"UCSA-small avg acc: {avg:.4f}", flush=True)
    else:
        print("(no UCSA ckpt provided; skipping UCSA eval)", flush=True)

    if args.baseline_results and os.path.exists(args.baseline_results):
        with open(args.baseline_results) as f:
            bl = json.load(f)
        report["baseline"] = bl
        print(
            f"Baseline val_ppl: {bl.get('final_val_ppl', 'n/a')}",
            flush=True,
        )

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
