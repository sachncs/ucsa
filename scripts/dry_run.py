"""Pre-flight check of a training configuration. Run before `train_r.py`.

    python scripts/dry_run.py --preset small
    python scripts/dry_run.py --preset small --set model.chunk_size=64

Builds the real model and batch shape, audits that every tensor is in the
library dtype, and measures memory and throughput. The auditing hook is slow
and allocates memory, which is why it lives here and not in training: the
training process carries no instrumentation. Exits non-zero if anything is
off, so it can gate a long run.
"""

import argparse
import dataclasses
import json
import os
import sys

import train_r

from ucsa import dryrun
from ucsa.models import recurrent


def main() -> None:
    """Parses arguments, dry-runs, prints and writes the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="small")
    parser.add_argument("--set", nargs="*", help="section.key=value")
    parser.add_argument("--params", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="runs/dry-run.json")
    parser.add_argument(
        "--ignore-scalars",
        action="store_true",
        help="skip 0-dim tensors (AdamW's float32 step counter in fp16)",
    )
    args = parser.parse_args()
    args.out_dir = "ckpts/dry-run"

    model_config, train_config = train_r.build_configs(args)
    report = dryrun.run(
        recurrent.Model(model_config),
        train_config,
        steps=args.steps,
        ignore_scalars=args.ignore_scalars,
    )
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(dataclasses.asdict(report), f, indent=2)
    print(json.dumps(dataclasses.asdict(report), indent=2))
    if not report.clean:
        sys.exit("dry run FAILED: see offenders / off_dtype_tensors above")
    print("dry run clean")


if __name__ == "__main__":
    main()
