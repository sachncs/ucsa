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
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
import ladder  # noqa: E402

from ucsa.training import scaling  # noqa: E402


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
    verdict = "verified lossless" if report["lossless_round_trip"] else "FAILED"
    return "\n".join(
        [
            "| compressor | bits per byte |",
            "|---|---|",
            f"| UCSA-R + arithmetic coding "
            f"({verdict})"
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


def section_curve(record: dict[str, Any] | None) -> str:
    """Renders held-out perplexity and bits per byte through training."""
    if record is None:
        return missing("record.json", "scripts/train_r.py")
    rows = [h for h in record["history"] if "ppl_all" in h]
    lines = ["| step | perplexity | bits per byte |", "|---|---|---|"]
    for h in rows:
        lines.append(
            f"| {h['step']} | {h['ppl_all']:.1f} | "
            f"{h.get('bpb_all', math.nan):.3f} |"
        )
    return "\n".join(lines)


def section_forecast_check(
    ladder_dirs: list[str],
    arm: str,
    record: dict[str, Any] | None,
    noise_bits: float,
) -> str:
    """Compares a forecast made from short runs with the full-run outcome.

    The forecast uses only the ladder rungs, which exist before the full run;
    the outcome is the full run's final held-out loss.

    Args:
      ladder_dirs: Folders holding the ladder rungs of `arm`.
      arm: Ladder arm that was later trained in full.
      record: The full run's `record.json`.
      noise_bits: Run-to-run noise, in bits per token.

    Returns:
      A Markdown paragraph and table, or a "not run" line.
    """
    steps, losses = ladder.load_points(ladder_dirs, arm)
    if record is None or steps.size < 3:
        return missing("ladder rungs / record.json", "scripts/ladder.py")
    target = float(record["train_config"]["steps"])
    noise = noise_bits * math.log(2.0)
    fc = scaling.forecast(steps, losses, target, noise)
    final = record["final"]
    actual = math.log(final["ppl_all"])
    inside = fc.low <= actual <= fc.high
    return "\n".join(
        [
            f"Forecast for `{arm}` made from rungs "
            f"{', '.join(str(int(x)) for x in steps)} (reach "
            f"{fc.reach:.0f}x), before the {int(target)}-step run:",
            "",
            "| | loss (nats) | perplexity |",
            "|---|---|---|",
            f"| forecast | {fc.mean:.3f} [{fc.low:.3f}, {fc.high:.3f}] | "
            f"{math.exp(fc.mean):.1f} [{math.exp(fc.low):.1f}, "
            f"{math.exp(fc.high):.1f}] |",
            f"| outcome | {actual:.3f} | {final['ppl_all']:.1f} |",
            "",
            "The outcome lies "
            + ("inside" if inside else "OUTSIDE")
            + " the forecast interval"
            + (
                ""
                if inside
                else f"; the forecast was too "
                f"{'optimistic' if actual > fc.high else 'pessimistic'} by "
                f"{abs(actual - fc.mean) / math.log(2.0):.2f} bits/token"
            )
            + ".",
        ]
    )


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
        slots, bits = int(row["slots"]), row["state_bits"]
        lines.append(f"| {slots} | {bits:.0f} | {row['bpb']:.4f} |")
    return "\n".join(lines)


def build(runs: str, ladder_dir: str, ablate: str) -> str:
    """Assembles the whole document from the artifacts under `runs`."""
    record = load(os.path.join(runs, "final-record.json"))
    parts = [
        ("Setup", section_training(record)),
        ("Held-out loss through training", section_curve(record)),
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
            "Forecast against outcome",
            section_forecast_check(
                [ladder_dir, ablate], "lr-2.4e-3", record, 0.0145
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
    parser.add_argument("--ladder", default="runs/ladder-gated")
    parser.add_argument("--ablate", default="runs/ablate-gated")
    parser.add_argument("--out", default="paper/RESULTS.md")
    args = parser.parse_args()
    text = build(args.runs, args.ladder, args.ablate)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
