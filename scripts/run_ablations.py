"""Ablation matrix runner.

Calls :mod:`scripts.train` repeatedly with different ablation flags
and writes each run's structured JSON to ``runs/``. This is the
entrypoint for the paper's ablations table.

Usage:
    .venv/bin/python scripts/run_ablations.py [--max-steps N] \
        [--seeds 42 43 44]

Each ablation toggles a single feature off its default-on state:

- ``full`` — every feature on (the baseline run).
- ``no-jepa`` — IJEPALoss, no multi-step chain.
- ``no-ema`` — EMA target encoder disabled (jepa_target uses the
  previous-iteration working memory instead of EMA-tracked latent).
- ``no-recon`` — input-reconstruction head disabled (weight 0).
- ``no-tc-jepa`` — TC-JEPA sparse cross-attention conditioner disabled.
- ``no-curriculum`` — all losses on from step 1 (no 4-stage gating).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ABLATIONS = [
    {"tag": "full", "flags": []},
    {"tag": "no-jepa", "flags": ["--no-lewm"]},
    {"tag": "no-ema", "flags": ["--no-ema"]},
    {"tag": "no-recon", "flags": ["--no-recon"]},
    {"tag": "no-tc-jepa", "flags": ["--no-tc-jepa"]},
    {"tag": "no-curriculum", "flags": ["--no-curriculum"]},
    # Endogenous origination. "origination" is the treatment; the rest
    # isolate one mechanism each, all at the same step count and the same
    # forward-pass budget, so the comparison is matched-compute.
    {"tag": "origination", "flags": ["--observation-mix", "0.5"]},
    {
        "tag": "origination-no-balance",
        "flags": ["--observation-mix", "0.5", "--no-origination-balance"],
    },
    {
        "tag": "origination-static-bank",
        "flags": [
            "--observation-mix", "0.5", "--intent-update-scale", "0.0",
        ],
    },
    {
        "tag": "origination-streamed-intent",
        "flags": ["--observation-mix", "0.5", "--stream-intent-bank"],
    },
    {
        "tag": "origination-dense-gate",
        "flags": ["--observation-mix", "0.5", "--origination-top-k", "16"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--seeds", nargs="*", type=int, default=[42])
    p.add_argument(
        "--out-dir", default="runs/ablations",
        help="Per-ablation JSON files land here.",
    )
    p.add_argument("--skip-baselines", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cmd_base = [
        sys.executable, "scripts/train.py",
        "--max-steps", str(args.max_steps),
        "--ablation", "PLACEHOLDER",
        "--ckpt-every", "0",  # one final ckpt only
        "--eval-every", "200",
        "--eval-batches", "10",
        "--stage-1-end", "400",
        "--stage-2-end", "1200",
        "--stage-3-end", "2400",
    ]
    if args.skip_baselines:
        cmd_base.append("--skip-baselines")

    rows: list[dict] = []
    for seed in args.seeds:
        for abl in ABLATIONS:
            print(
                f"\n=== ablation={abl['tag']} seed={seed} ===",
                flush=True,
            )
            out_path = os.path.join(
                args.out_dir,
                f"ucsa-{abl['tag']}-seed{seed}.json",
            )
            cmd = list(cmd_base) + list(abl["flags"]) + [
                "--seed", str(seed),
                "--out-json", out_path,
            ]
            subprocess.run(cmd, check=False)
            if os.path.exists(out_path):
                with open(out_path) as f:
                    data = json.load(f)
                rows.append({
                    "ablation": abl["tag"],
                    "seed": seed,
                    "best_val_ppl": data.get("best_val_ppl"),
                    "final_val_ppl": data.get("final_val_ppl"),
                    "n_ucsa_params": data.get("n_ucsa_params"),
                })

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nWrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
