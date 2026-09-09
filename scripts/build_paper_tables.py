"""Build the paper's results tables from ``runs/*.json``.

Reads:
  - ``runs/ucsa-<ablation>-seed<n>.json`` (training history / final PPL).
  - ``runs/eval-ucsa-small.json`` (downstream benchmark accuracy).
  - ``runs/baseline.json`` (matched-compute vanilla Transformer val_ppl).
  - ``runs/bank-probe.json`` (PCS bank probe).

Writes:
  - ``paper/TABLES.md`` — markdown tables ready to paste into
    ``paper/PAPER.md``.

This is the missing bridge between "I trained models" and "I wrote a
paper" — the script deliberately consumes the same JSONs the
training / eval / probe scripts emit, so a real experiment run
populates paper tables with one extra command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from collections import defaultdict


def _load_jsons(root: str, pattern: str) -> list[dict]:
    found: list[dict] = []
    rx = re.compile(pattern)
    if not os.path.isdir(root):
        return found
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json"):
            continue
        if not rx.search(name):
            continue
        path = os.path.join(root, name)
        try:
            with open(path) as f:
                found.append({"name": name, "path": path, "data": json.load(f)})
        except Exception:
            pass
    return found


def _agg(rows: list[dict], key: str) -> tuple[float, float, int]:
    vals = [
        r["data"][key]
        for r in rows
        if isinstance(r["data"].get(key), (int, float))
    ]
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return float(vals[0]), 0.0, 1
    return (
        float(statistics.mean(vals)),
        float(statistics.stdev(vals)),
        len(vals),
    )


def build_main_table(ucsa_paths: list[dict], baseline_paths: list[dict]) -> str:
    """Table 1: matched-compute comparison."""
    ucsa_loss, ucsa_sd, n_u = _agg(ucsa_paths, "best_val_ppl")
    base_ppl, base_sd, n_b = _agg(baseline_paths, "final_val_ppl")
    base_best, base_best_sd, _ = _agg(baseline_paths, "best_val_ppl")
    if not math.isnan(ucsa_loss) and not math.isnan(base_ppl):
        delta_pct = (ucsa_loss - base_ppl) / base_ppl * 100.0
    else:
        delta_pct = float("nan")
    s = "## Table 1 — Matched-compute comparison (fineweb-edu, "
    s += f"{n_u} UCSA seeds, {n_b} baseline seeds)\n\n"
    s += "| Model                      | Val PPL (mean ± std)  | Best Val PPL |\n"
    s += (
        "| -------------------------- | -------------------- | ------------ |\n"
    )
    if not math.isnan(base_ppl):
        s += (
            f"| Vanilla-Transformer (no PCS) "
            f"| {base_ppl:.2f} ± {base_sd:.2f}          "
            f"| {base_best:.2f}      |\n"
        )
    if not math.isnan(ucsa_loss):
        s += (
            f"| UCSA-small (full stack)  "
            f"| {ucsa_loss:.2f} ± {ucsa_sd:.2f}          "
            f"| {ucsa_loss:.2f}      |\n"
        )
    if not math.isnan(delta_pct):
        s += (
            f"\nUCSA vs baseline val-PPL delta: "
            f"{delta_pct:+.2f}% "
            f"(negative = UCSA better)\n"
        )
    s += "\n"
    return s


def build_ablation_table(ucsa_paths: list[dict]) -> str:
    """Table 2: ablation matrix."""
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in ucsa_paths:
        # Filename like "ucsa-full-seed42.json" or "ucsa-no-jepa-seed42.json"
        m = re.match(r"ucsa-(?P<tag>.+)-seed\d+\.json", r["name"])
        if m:
            by_tag[m.group("tag")].append(r)
    if not by_tag:
        return ""
    s = "## Table 2 — Ablation matrix (UCSA-small, fineweb-edu, "
    s += f"{sum(len(v) for v in by_tag.values())} runs, multi-seed)\n\n"
    s += "| Ablation          | Runs | Mean Val PPL | Best Val PPL |\n"
    s += "| ----------------- | ---- | ------------ | ------------ |\n"
    for tag in sorted(by_tag.keys()):
        rows = by_tag[tag]
        mean_ppl, sd_ppl, _ = _agg(rows, "best_val_ppl")
        # We don't stdev best across seeds because we average.
        if math.isnan(mean_ppl):
            continue
        s += (
            f"| {tag:17s} | {len(rows):>4d} "
            f"| {mean_ppl:.2f}{' ± '+f'{sd_ppl:.2f}' if sd_ppl > 0 else '':<7s} "
            f"| {mean_ppl:.2f}      |\n"
        )
    s += "\n"
    return s


def build_eval_table(eval_paths: list[dict]) -> str:
    """Table 3: downstream benchmark accuracy."""
    if not eval_paths:
        return ""
    s = "## Table 3 — Downstream standard-LM benchmarks\n\n"
    s += "| Task          | UCSA acc |\n"
    s += "| ------------- | -------- |\n"
    # Use the first eval JSON
    rep = eval_paths[0]["data"]
    tasks = rep.get("tasks", {})
    for name in (
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
        "winogrande",
    ):
        if name in tasks:
            acc = tasks[name].get("accuracy")
            if isinstance(acc, (int, float)):
                s += f"| {name:13s} | {acc:.4f}   |\n"
    avg = rep.get("ucsa_avg_acc")
    if isinstance(avg, (int, float)):
        s += f"| **avg**       | **{avg:.4f}** |\n"
    s += "\n"
    return s


def build_bank_probe_table(probe_paths: list[dict]) -> str:
    """Table 4: bank probe."""
    if not probe_paths:
        return ""
    s = "## Table 4 — PCS bank probe (top 3 tokens per bank)\n\n"
    s += "| Bank            | Norm ‖·‖ mean | Retention mean | Top 3 tokens |\n"
    s += "| --------------- | ------------- | -------------- | ------------ |\n"
    rep = probe_paths[0]["data"]
    for bank in rep.get("banks", []):
        if bank.get("empty"):
            continue
        top3 = ", ".join(bank.get("top_tokens", [])[:3])
        s += (
            f"| {bank['name']:15s} | "
            f"{bank['norm_mean']:>13.4f} | "
            f"{bank['retention_mean']:>14.4f} | "
            f"{top3:<15s} |\n"
        )
    sim = rep.get("centroid_cosine_sim", {})
    if sim and sim.get("names"):
        s += "\n### Bank-centroid cosine similarity\n\n"
        s += "| | " + " | ".join(sim["names"]) + " |\n"
        s += "| " + " | ".join(["---"] * len(sim["names"])) + " |\n"
        for i, name in enumerate(sim["names"]):
            row = " | ".join(
                f"{sim['matrix'][i][j]:.3f}" for j in range(len(sim["names"]))
            )
            s += f"| **{name}** | {row} |\n"
    s += "\n"
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-md", default="paper/TABLES.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs = args.runs_dir

    ucsa_files = _load_jsons(runs, r"^ucsa-(?!baseline).*-seed\d+\.json$")
    ucsa_files = [
        f for f in ucsa_files if not f["name"].startswith("ucsa-baseline-")
    ]
    baseline_files = _load_jsons(
        runs, r"^baseline.*\.json$|^ucsa-baseline-.*\.json$"
    )
    eval_files = _load_jsons(runs, r"^eval-.*\.json$")
    probe_files = _load_jsons(runs, r"^bank-probe.*\.json$")

    print(
        f"Found {len(ucsa_files)} UCSA training runs, "
        f"{len(baseline_files)} baseline runs, "
        f"{len(eval_files)} eval reports, "
        f"{len(probe_files)} bank probes",
        flush=True,
    )

    tables: list[str] = []
    tables.append(build_main_table(ucsa_files, baseline_files))
    tables.append(build_ablation_table(ucsa_files))
    tables.append(build_eval_table(eval_files))
    tables.append(build_bank_probe_table(probe_files))
    tables = [t for t in tables if t]

    md = "# Paper Tables\n\n"
    md += "_Auto-generated by `scripts/build_paper_tables.py`. "
    md += "Paste these into `paper/PAPER.md`._\n\n"
    md += "\n".join(tables)

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md)
    print(f"Wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
