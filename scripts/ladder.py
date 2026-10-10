"""Judges design choices at the full-run length from short ladder runs.

    python scripts/ladder.py --arms base no-state surprise --out runs/ladder
    python scripts/ladder.py --analyze --target 12000 --out runs/ladder

Each arm is trained for a geometric ladder of step budgets (default 150, 300,
600, 1200), each with its own annealed schedule. The 600-step point is read
from the ablation matrix when it exists. `--analyze` fits a learning curve to
every arm, forecasts the loss at `--target` steps with an interval driven by
the measured seed-to-seed noise, flags curves that cross before the target,
and compares each arm with the base arm at the target length.
"""

import argparse
import dataclasses
import json
import os

import ablate
import numpy as np
import transformers

from ucsa.training import compression, scaling

TOKENS_PER_STEP = 8 * 1024
DEFAULT_NOISE = 0.04  # nats; replaced by the measured value when available


@dataclasses.dataclass
class Verdict:
    """How an arm compares with the base arm at the target length.

    Attributes:
      arm: Arm name.
      forecast: Predicted loss at the target length.
      gap: Predicted gap to base (negative is better).
      low: Lower end of the gap interval.
      high: Upper end of the gap interval.
      crossing: Budget where the arm and base swap order, if before target.
      reach: Extrapolation distance in multiples of the longest run.
      decision: One of `better`, `worse`, `inconclusive`.
    """

    arm: str
    forecast: float
    gap: float
    low: float
    high: float
    crossing: float | None
    reach: float
    decision: str


def load_points(dirs: list[str], arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Collects `(steps, mean window loss)` for an arm from result folders.

    Args:
      dirs: Folders holding `<arm>.json` (600 steps) and `<arm>@<n>.json`.
      arm: Arm name.

    Returns:
      Budgets and losses sorted by budget (possibly empty).
    """
    points = {}
    for folder in dirs:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            stem = name.removesuffix(".json")
            if not name.endswith(".json") or stem.split("@")[0] != arm:
                continue
            with open(os.path.join(folder, name)) as f:
                rec = json.load(f)
            steps = rec["train_config"]["steps"]
            points[steps] = float(np.mean(rec["window_nll"]))
    order = sorted(points)
    return np.array(order, dtype=float), np.array([points[s] for s in order])


def seed_noise(dirs: list[str]) -> float:
    """Estimates the loss noise from arms that differ only in the seed.

    Args:
      dirs: Result folders to search for `base`, `base-seed43`, ...

    Returns:
      The standard deviation of a single run's mean loss, or `DEFAULT_NOISE`
      if fewer than two seed replicates exist.
    """
    runs = []
    for arm in ["base", "base-seed43", "base-seed44", "base-seed45"]:
        steps, losses = load_points(dirs, arm)
        if 600 in steps:
            runs.append(losses[list(steps).index(600)])
    if len(runs) < 2:
        return DEFAULT_NOISE
    return float(np.std(runs, ddof=1))


def analyze(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    target: float,
    noise: float,
    base: str = "base",
) -> list[Verdict]:
    """Compares every arm with `base` at the target training length.

    An arm is `better` (or `worse`) only if the interval of its forecast gap
    to base excludes zero; otherwise the result is `inconclusive`, and no
    claim should be made.

    Args:
      data: Maps arm name to `(budgets, losses)`.
      target: Training length to compare at.
      noise: Run-to-run standard deviation of one measurement.
      base: Name of the reference arm.

    Returns:
      One verdict per non-base arm with at least three points.
    """
    out = []
    base_curve = scaling.fit(*data[base])
    for arm, (steps, losses) in sorted(data.items()):
        if arm == base or len(steps) < 3:
            continue
        curve = scaling.fit(steps, losses)
        gap = scaling.paired_forecast_gap(
            (steps, losses), data[base], target, noise
        )
        decision = (
            "better"
            if gap.high < 0
            else "worse" if gap.low > 0 else "inconclusive"
        )
        out.append(
            Verdict(
                arm=arm,
                forecast=curve.predict(target),
                gap=gap.mean,
                low=gap.low,
                high=gap.high,
                crossing=scaling.crossing(curve, base_curve, target),
                reach=gap.reach,
                decision=decision,
            )
        )
    return out


def render(verdicts: list[Verdict], noise: float, target: float) -> str:
    """Formats verdicts as a Markdown table (losses in bits/token)."""
    k = 1.0 / np.log(2.0)
    lines = [
        f"Forecast at {target:.0f} steps; seed noise {noise * k:.3f} "
        "bits/token per run.",
        "",
        "| arm | forecast | gap vs base [95% CI] | crosses base at | reach "
        "| decision |",
        "|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        cross = f"{v.crossing:.0f}" if v.crossing else "-"
        lines.append(
            f"| {v.arm} | {v.forecast * k:.3f} | {v.gap * k:+.3f} "
            f"[{v.low * k:+.3f}, {v.high * k:+.3f}] | {cross} | "
            f"{v.reach:.1f}x | {v.decision} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Runs missing ladder rungs, or analyses finished ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=["base"])
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[150, 300, 600, 1200]
    )
    parser.add_argument("--target", type=float, default=12000)
    parser.add_argument("--out", default="runs/ladder")
    parser.add_argument("--ablate-dir", default="runs/ablate")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--preset", default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--data", default="data")
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    dirs = [args.out, args.ablate_dir]
    os.makedirs(args.out, exist_ok=True)
    if not args.analyze:
        byte_lengths = compression.token_byte_lengths(
            transformers.AutoTokenizer.from_pretrained("gpt2")
        )
        for arm in args.arms:
            for steps in args.budgets:
                have, _ = load_points(dirs, arm)
                if steps in have:
                    print(f"skip {arm}@{steps} (done)", flush=True)
                    continue
                run_args = argparse.Namespace(
                    **{**vars(args), "steps": steps, "out": args.out}
                )
                record = ablate.run_arm(arm, run_args, byte_lengths)
                path = os.path.join(args.out, f"{arm}@{steps}.json")
                with open(path, "w") as f:
                    json.dump(record, f)
    data = {a: load_points(dirs, a) for a in ablate.ARMS}
    data = {a: p for a, p in data.items() if len(p[0]) >= 3}
    if "base" not in data:
        raise SystemExit("need at least three base points; run --arms base")
    noise = seed_noise(dirs)
    table = render(analyze(data, args.target, noise), noise, args.target)
    with open(os.path.join(args.out, "forecast.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
