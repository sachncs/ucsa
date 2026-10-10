"""Tunes UCSA-R with successive halving.

    python scripts/tune.py --preset small --n 9 --budgets 150 450 1350

Each trial trains a fresh model for the rung's step budget on the training
shard and is scored by held-out `ppl_last64`. Results are cached next to
`--out`, so an interrupted search resumes where it stopped.
"""

import argparse
import dataclasses
import gc
import json
import os
import tempfile

import torch

from ucsa.models import recurrent
from ucsa.training import engine, presets, tuning


def main() -> None:
    """Parses arguments, runs the search and writes the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="small")
    parser.add_argument("--n", type=int, default=9, help="rung-0 candidates")
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[150, 450, 1350]
    )
    parser.add_argument("--keep", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--space", default=None, help="JSON {key: [values]}")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="runs/tune.json")
    args = parser.parse_args()

    preset = presets.load(args.preset)
    space = tuning.DEFAULT_SPACE
    if args.space:
        with open(args.space) as f:
            space = json.load(f)

    def score(candidate: dict, steps: int) -> float:
        raw = {"model": dict(preset["model"]), "train": dict(preset["train"])}
        overrides = [f"{k}={json.dumps(v)}" for k, v in candidate.items()]
        presets.apply_overrides(raw, overrides)
        raw["train"].update(
            steps=steps,
            seed=args.seed,
            warmup_steps=min(raw["train"].get("warmup_steps", 400), steps // 5),
            eval_every=0,
            ckpt_every=0,
            log_every=max(1, steps),
            out_dir=tempfile.mkdtemp(prefix="ucsa-tune-"),
        )
        model_config = recurrent.Config.from_dict(raw["model"])
        train_config = engine.Config.from_dict(raw["train"])
        train, val = presets.batch_factories(train_config, args.data)
        record = engine.fit(
            recurrent.Model(model_config),
            train_config,
            train,
            val,
            log=lambda line: None,
        )
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return record["final"]["ppl_last64"]

    candidates = tuning.sample_candidates(space, args.n, args.seed)
    result = tuning.successive_halving(
        candidates,
        score,
        args.budgets,
        args.keep,
        cache_path=args.out + ".cache",
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trials = [dataclasses.asdict(t) for t in result.trials]
    with open(args.out, "w") as f:
        json.dump(
            {
                "best": result.best,
                "best_ppl_last64": result.best_score,
                "trials": trials,
            },
            f,
            indent=2,
        )
    print(f"best: {result.best}  ppl_last64={result.best_score:.1f}")


if __name__ == "__main__":
    main()
