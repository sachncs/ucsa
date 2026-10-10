"""Published reference results and comparison against them.

UCSA-R is compared with numbers reported in the literature, not with a
baseline trained here. The reference file records its source and protocol;
`compare` lines our per-task accuracy up against each reference model and
against chance so a result is never read without its floor.
"""

import json
import os
from typing import Any

DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "paper", "reference_results.json"
)


def load_reference(path: str = DEFAULT_PATH) -> dict[str, Any]:
    """Loads the published reference results.

    Args:
      path: JSON file with `source`, `protocol`, `metrics`, `chance` and
        `models`.

    Returns:
      The parsed reference.
    """
    with open(path) as f:
        return json.load(f)


def compare(
    ours: dict[str, dict[str, Any]], reference: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Lines our per-task accuracy up against the reference models.

    Args:
      ours: Maps task name to an `EvalResult.to_dict()`. Tasks absent from
        the reference are skipped.
      reference: Reference results; loaded from `DEFAULT_PATH` when None.

    Returns:
      One row per task with `ours`, `stderr`, `chance` (all in percent), `n`,
      the metric name and one column per reference model.
    """
    ref = reference or load_reference()
    rows = []
    for task, metric in ref["metrics"].items():
        if task not in ours:
            continue
        result = ours[task]
        row: dict[str, Any] = {
            "task": task,
            "metric": metric,
            "ours": 100.0 * result["accuracy"],
            "stderr": 100.0 * result.get("stderr", 0.0),
            "n": result["n"],
            "chance": ref["chance"][task],
        }
        for name, scores in ref["models"].items():
            row[name] = scores[task]
        rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]], reference: dict[str, Any]) -> str:
    """Renders `compare` rows as a Markdown table.

    Args:
      rows: Output of `compare`.
      reference: The reference the rows were built from.

    Returns:
      Markdown text.
    """
    names = list(reference["models"])
    head = ["task", "metric", "n", "ours", "chance", *names]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        cells = [
            row["task"],
            row["metric"],
            str(row["n"]),
            f"{row['ours']:.1f} ± {row['stderr']:.1f}",
            f"{row['chance']:.1f}",
            *[f"{row[name]:.1f}" for name in names],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
