"""Builds `paper/RESULTS.md` from run artifacts. No number is typed by hand.

    python scripts/report.py --runs runs --out paper/RESULTS.md

Every section reads a JSON file written by one of the experiment scripts. A
section whose artifact does not exist says so instead of showing a
placeholder, so the document can never claim a result that was not produced.
"""

import argparse
import json
import math
import os
from typing import Any


def load(path: str) -> dict[str, Any] | None:
    """Reads a JSON artifact, or returns None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def missing(name: str, how: str) -> str:
    """Renders the line shown for an artifact that was never produced."""
    return f"_Not run: `{name}` does not exist. Produce it with `{how}`._"


def section_training(record: dict[str, Any] | None) -> str:
    """Renders the model and training setup from a `record.json`."""
    if record is None:
        return missing("record.json", "scripts/train_r.py")
    mc, tc, fin = (
        record["model_config"],
        record["train_config"],
        record["final"],
    )
    tokens = tc["steps"] * tc["batch_size"] * tc["seq_len"]
    lines = [
        f"- parameters: {record['params']:,}",
        f"- width {mc['hidden']}, layers {mc['layers']}, heads {mc['heads']}, "
        f"chunk {mc['chunk_size']}, window "
        f"{mc['window'] if mc['window'] is not None else mc['chunk_size']}, "
        f"state slots {sum(n for _, n in mc['banks'])}",
        f"- steps {tc['steps']}, batch {tc['batch_size']} x {tc['seq_len']} "
        f"tokens, {tokens / 1e6:.0f}M tokens seen, seed {tc['seed']}",
        f"- training time {record['elapsed_seconds'] / 3600:.2f} h",
    ]
    if fin:
        lines.append(
            f"- held-out perplexity (last 64 positions) "
            f"{fin['ppl_last64']:.1f}; bits per byte "
            f"{fin.get('bpb_last64', math.nan):.3f}"
        )
    return "\n".join(lines)


def section_compression(report: dict[str, Any] | None) -> str:
    """Renders real compressed sizes against general-purpose compressors."""
    if report is None:
        return missing("compress.json", "scripts/compress.py")
    anchors = report["anchors_bits_per_byte"]
    return "\n".join(
        [
            "| compressor | bits per byte |",
            "|---|---|",
            f"| UCSA-R + arithmetic coding "
            f"({'verified lossless' if report['lossless_round_trip'] else 'FAILED'})"
            f" | {report['ucsa_bits_per_byte']:.3f} |",
            f"| zlib (level 9) | {anchors['zlib']:.3f} |",
            f"| LZMA (preset 9) | {anchors['lzma']:.3f} |",
            "",
            f"{report['tokens']} tokens, {report['raw_bytes']} raw bytes, "
            f"{report['compressed_bytes']} bytes coded.",
        ]
    )


def section_benchmarks(report: dict[str, Any] | None) -> str:
    """Renders benchmark accuracy beside published models and chance."""
    if report is None or "reference_comparison" not in report:
        return missing("eval.json", "scripts/eval.py --recurrent-ckpt ...")
    rows = report["reference_comparison"]
    names = [
        k
        for k in rows[0]
        if k not in {"task", "metric", "ours", "stderr", "n", "chance"}
    ]
    head = ["task", "metric", "n", "ours", "chance", *names]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [
            r["task"],
            r["metric"],
            str(r["n"]),
            f"{r['ours']:.1f} ± {r['stderr']:.1f}",
            f"{r['chance']:.1f}",
            *[f"{r[n]:.1f}" for n in names],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    ppl = report.get("recurrent_ppl", {})
    if ppl:
        lines += [
            "",
            "Held-out perplexity (last 64 positions): "
            + ", ".join(f"{k} {v:.1f}" for k, v in ppl.items()),
        ]
    return "\n".join(lines)


def section_file(path: str, how: str) -> str:
    """Includes a Markdown table written by an experiment script."""
    if not os.path.exists(path):
        return missing(os.path.basename(path), how)
    with open(path) as f:
        return f.read().strip()


def section_probe(report: dict[str, Any] | None) -> str:
    """Renders the state probe: chunk profiles and rate-distortion."""
    if report is None:
        return missing("probe-state.json", "scripts/probe_state.py")
    lines = []
    for length, entry in report["profiles"].items():
        lines.append(f"Window of {length} tokens, bits per token by chunk:")
        for key, values in entry.items():
            lines.append(f"- {key}: " + " ".join(f"{v:.2f}" for v in values))
        lines.append("")
    lines += [
        "| state slots read | state bits | bits per byte |",
        "|---|---|---|",
    ]
    for row in report["rate_distortion"]:
        lines.append(
            f"| {int(row['slots'])} | {row['state_bits']:.0f} | {row['bpb']:.4f} |"
        )
    return "\n".join(lines)


def build(runs: str, ladder: str, ablate: str) -> str:
    """Assembles the whole document from the artifacts under `runs`."""
    parts = [
        (
            "Setup",
            section_training(load(os.path.join(runs, "final-record.json"))),
        ),
        (
            "Lossless compression",
            section_compression(load(os.path.join(runs, "compress.json"))),
        ),
        (
            "Zero-shot benchmarks against published models",
            section_benchmarks(load(os.path.join(runs, "eval.json"))),
        ),
        (
            "Design choices, short runs",
            section_file(
                os.path.join(ablate, "summary.md"), "scripts/ablate.py"
            ),
        ),
        (
            "Design choices, forecast to the full run",
            section_file(
                os.path.join(ladder, "forecast.md"),
                "scripts/ladder.py --analyze",
            ),
        ),
        (
            "What the state contributes",
            section_probe(load(os.path.join(runs, "probe-state.json"))),
        ),
    ]
    body = "\n\n".join(f"## {title}\n\n{text}" for title, text in parts)
    return (
        "# Results\n\nGenerated by `scripts/report.py` from run artifacts; "
        "edit the experiments, not this file.\n\n" + body + "\n"
    )


def main() -> None:
    """Parses arguments and writes the results document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--ladder", default="runs/ladder")
    parser.add_argument("--ablate", default="runs/ablate-window")
    parser.add_argument("--out", default="paper/RESULTS.md")
    args = parser.parse_args()
    text = build(args.runs, args.ladder, args.ablate)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
