"""Memory-bank probing utility.

For each of the six PCS banks, project the bank tokens back into
vocabulary space via the tied LM head and report the top-k tokens
the bank "remembers". Plus:

- Per-bank L2 norm snapshot.
- Per-bank retention-score distribution statistics.
- A simple per-bank centroid cosine-similarity matrix — banks with
  differentiated roles will have well-separated centroids; banks
  that collapsed into one another will look identical.

The output is a single JSON file that the paper's analysis plots
consume. We use the language head's tied projection (``token_emb
weight.T``) as the readout.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from ucsa.train import build_model
from ucsa.utils.seed import set_seed

BANK_NAMES = (
    "working",
    "long_term",
    "goal",
    "episode",
    "task",
    "memory_index",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to a UCSA safetensors checkpoint.")
    p.add_argument("--ucsa-config", default="ucsa/configs/default.yaml")
    p.add_argument("--out-json", default="runs/bank-probe.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def probe_banks(model, top_k: int = 20) -> dict:
    """Compute the bank-probe report for an in-memory UCSA model."""
    token_emb = model.perception.token_embedding
    vocab_proj = token_emb.weight  # (V, H)
    hidden = token_emb.weight.shape[1]

    report: dict = {"hidden_size": hidden}
    bank_summaries: list[dict] = []
    bank_centroids: dict[str, torch.Tensor] = {}

    for name in BANK_NAMES:
        bank = model.pcs.get_bank(name)
        if bank.numel() == 0:
            bank_summaries.append({"name": name, "empty": True})
            continue
        flat = bank.detach().cpu()
        norms = flat.norm(dim=-1)
        topk_logits = flat @ vocab_proj.cpu().T
        topk = torch.topk(topk_logits, k=top_k, dim=-1)
        token_ids = topk.indices.tolist()
        tok = model.perception.tokenizer.tokenizer
        token_strs = [
            [tok.decode([t]) for t in row]
            for row in token_ids
        ]
        top1_str = [r[0] for r in token_strs]
        retention = model.memory.get_retention_scores(name)
        ret = retention.detach().cpu().numpy().tolist()
        centroid = flat.mean(dim=0)
        bank_centroids[name] = centroid
        bank_summaries.append({
            "name": name,
            "n_tokens": int(flat.shape[0]),
            "norm_mean": float(norms.mean().item()),
            "norm_std": float(norms.std().item()),
            "norm_min": float(norms.min().item()),
            "norm_max": float(norms.max().item()),
            "retention_mean": sum(ret) / max(1, len(ret)),
            "retention_min": min(ret) if ret else 0.0,
            "retention_max": max(ret) if ret else 0.0,
            "top_tokens": top1_str[:10],
            "all_top_tokens": token_strs[:5],
        })
    report["banks"] = bank_summaries

    names = [s["name"] for s in bank_summaries if not s.get("empty")]
    matrix = []
    for a in names:
        row = []
        for b in names:
            ca = bank_centroids[a]
            cb = bank_centroids[b]
            cos = F.cosine_similarity(ca, cb, dim=0).item()
            row.append(cos)
        matrix.append(row)
    report["centroid_cosine_sim"] = {"names": names, "matrix": matrix}
    return report


def _print_human(report: dict) -> None:
    bank_summaries = report["banks"]
    hidden = report["hidden_size"]
    print("\nBank probe summary:", flush=True)
    print(f"  {len(bank_summaries)} banks @ hidden={hidden}", flush=True)
    print(
        f"  {'bank':14s} {'n':>4s} {'|b|':>8s} | {'top3 tokens':<40s}",
        flush=True,
    )
    for s in bank_summaries:
        if s.get("empty"):
            print(f"  {s['name']:14s} (empty)", flush=True)
            continue
        top3 = ", ".join(s["top_tokens"][:3])
        print(
            f"  {s['name']:14s} {s['n_tokens']:>4d} "
            f"{s['norm_mean']:>8.4f} | {top3:<40s}",
            flush=True,
        )
    names = report["centroid_cosine_sim"]["names"]
    matrix = report["centroid_cosine_sim"]["matrix"]
    print("\nCentroid cosine similarity:", flush=True)
    print("        " + "  ".join(f"{n[:8]:>8s}" for n in names), flush=True)
    for i, n in enumerate(names):
        row = "  ".join(f"{matrix[i][j]:>8.3f}" for j in range(len(names)))
        print(f"  {n[:8]:>8s} {row}", flush=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with open(args.ucsa_config) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["max_seq_len"] = cfg["model"].get("max_seq_len", 1024)
    model = build_model(cfg)
    sd = load_file(args.ckpt)
    renamed = {name.removeprefix("model."): t for name, t in sd.items()}
    model.load_state_dict(renamed, strict=False)

    report = probe_banks(model, top_k=args.top_k)
    report["ckpt"] = args.ckpt
    report["seed"] = args.seed
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out_json}", flush=True)
    _print_human(report)


if __name__ == "__main__":
    main()
