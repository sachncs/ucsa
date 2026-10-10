"""Runs a short-budget ablation matrix and compares arms with paired CIs.

    python scripts/ablate.py --steps 600 --out runs/ablate
    python scripts/ablate.py --summary --out runs/ablate

Every arm trains from the same seed on the same data order and is scored on
the same held-out windows, so the per-window loss gap to the base arm is a
paired comparison. An arm is only "better" if the interval of that gap
excludes zero. Finished arms are skipped, so an interrupted run resumes.
"""

import argparse
import gc
import json
import os
import time

import torch
import train_r
import transformers

from ucsa.models import recurrent
from ucsa.training import compression, diagnostics, engine

# Arm name -> `section.key=value` overrides on top of the base preset.
ARMS: dict[str, list[str]] = {
    "base": [],
    "base-seed43": [],  # noise floor: only the seed differs (see --seed)
    "base-seed44": [],
    "base-seed45": [],
    "no-state": ["model.use_state=false"],
    "no-window": ["model.window=0"],
    "no-state-no-window": ["model.use_state=false", "model.window=0"],
    "no-jepa": ["model.jepa_weight=0"],
    "surprise": ["model.surprise_gate=true"],
    "read-gate": ["model.read_gate=true"],
    "read-gate-surprise": ["model.read_gate=true", "model.surprise_gate=true"],
    "read-every-2": ["model.read_every=2"],
    "chunk-64": ["model.chunk_size=64"],
    "slot-dropout": ["model.slot_dropout=0.5"],
    "weight-ema": ["train.weight_ema=0.99"],
    "zfilter": ["train.train_shard=train-zfilter.bin"],
    "lr-3e-4": ["train.lr=3e-4"],
    "lr-1.2e-3": ["train.lr=1.2e-3"],
    "lr-2.4e-3": ["train.lr=2.4e-3"],
}
SEED_OVERRIDES = {"base-seed43": 43, "base-seed44": 44, "base-seed45": 45}


def run_arm(
    name: str, args: argparse.Namespace, byte_lengths: torch.Tensor
) -> dict:
    """Trains one arm and returns its record, including per-window losses."""
    ns = argparse.Namespace(
        preset=args.preset,
        params=0,
        seed=SEED_OVERRIDES.get(name, args.seed),
        out_dir=os.path.join(args.out, "ckpt", name),
        set=[
            *ARMS[name],
            f"train.steps={args.steps}",
            f"train.warmup_steps={max(10, args.steps // 10)}",
            "train.eval_every=0",
            "train.ckpt_every=0",
            f"train.log_every={max(1, args.steps // 6)}",
            f"train.eval_batches={args.eval_batches}",
            f"train.batch_size={args.batch_size}",
            *args.extra,
        ],
    )
    model_config, train_config = train_r.build_configs(ns)
    model = recurrent.Model(model_config)
    train, val = train_r.batch_factories(train_config, args.data)
    started = time.time()
    record = engine.fit(
        model,
        train_config,
        train,
        val,
        byte_lengths=byte_lengths,
        log=lambda line: print(f"[{name}] {line}", flush=True),
    )
    model.eval()
    record["name"] = name
    record["overrides"] = ARMS[name]
    record["window_nll"] = diagnostics.window_nll(
        model, val(0), args.eval_batches
    ).tolist()
    record["train_seconds"] = time.time() - started
    record["tokens_per_second"] = (
        train_config.steps
        * train_config.batch_size
        * train_config.seq_len
        / record["train_seconds"]
    )
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return record


def summarise(out: str) -> str:
    """Builds the Markdown comparison of every finished arm against base."""
    records = {}
    for fname in sorted(os.listdir(out)):
        if fname.endswith(".json") and fname != "summary.json":
            with open(os.path.join(out, fname)) as f:
                rec = json.load(f)
            records[rec["name"]] = rec
    import numpy as np

    base = np.asarray(records["base"]["window_nll"])
    lines = [
        "| arm | params | tok/s | ppl | bits/byte | gap vs base (bits/token) "
        "[95% CI] | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rec in records.items():
        arm = np.asarray(rec["window_nll"])
        fin = rec["final"]
        if name == "base":
            gap, verdict = "-", "reference"
        else:
            r = diagnostics.paired_bootstrap(arm, base)
            scale = 1.0 / np.log(2.0)
            gap = (
                f"{r.mean_gap * scale:+.4f} "
                f"[{r.low * scale:+.4f}, {r.high * scale:+.4f}]"
            )
            verdict = (
                "better"
                if r.significant and r.mean_gap < 0
                else "worse" if r.significant else "no significant difference"
            )
        lines.append(
            f"| {name} | {rec['params'] / 1e6:.1f}M | "
            f"{rec.get('tokens_per_second', 0):.0f} | "
            f"{fin['ppl_all']:.1f} | {fin.get('bpb_all', float('nan')):.4f} | "
            f"{gap} | {verdict} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Parses arguments, runs missing arms, prints the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="small")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="runs/ablate")
    parser.add_argument("--only", nargs="*", help="run just these arms")
    parser.add_argument("--extra", nargs="*", default=[], help="k=v for all")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not args.summary:
        byte_lengths = compression.token_byte_lengths(
            transformers.AutoTokenizer.from_pretrained("gpt2")
        )
        for name in args.only or list(ARMS):
            path = os.path.join(args.out, f"{name}.json")
            if os.path.exists(path):
                print(f"skip {name} (done)", flush=True)
                continue
            record = run_arm(name, args, byte_lengths)
            with open(path, "w") as f:
                json.dump(record, f)
    table = summarise(args.out)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
