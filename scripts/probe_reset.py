"""Tests whether a trained model uses what the state remembers.

    python scripts/probe_reset.py --ckpt ckpts/r-small/final.pt

Evaluates the same checkpoint on the same windows twice: as trained, and with
the state reset to its initial value at every chunk (`use_state=False` at
inference). If the loss does not move, the slots act as fixed learned context
tokens and carry nothing from earlier chunks.
"""

import argparse
import dataclasses
import json
import math
import os

import numpy as np

from ucsa.models import recurrent
from ucsa.training import diagnostics, engine, shards


def main() -> None:
    """Parses arguments, evaluates both ways and prints the paired gap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--shard", default="val.bin")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--out-json", default="runs/probe-reset.json")
    args = parser.parse_args()

    device = engine.pick_device()
    model = engine.load_model(args.ckpt, device)
    shard = shards.Shard(os.path.join(args.data, args.shard))
    reset = recurrent.Model(dataclasses.replace(model.config, use_state=False))
    reset.load_state_dict(model.state_dict())
    reset.to(device).eval()

    def losses(m: recurrent.Model) -> np.ndarray:
        batches = shard.batches(1, args.length, skip=0, seed=None, loop=False)
        return diagnostics.window_nll(m, batches, args.windows)

    kept, wiped = losses(model), losses(reset)
    gap = diagnostics.paired_bootstrap(wiped, kept)
    scale = math.log(2.0)
    report = {
        "windows": int(kept.size),
        "kept_state_bits": float(kept.mean()) / scale,
        "reset_state_bits": float(wiped.mean()) / scale,
        "reset_minus_kept_bits": diagnostics.bits_gap(gap),
        "low_bits": gap.low / scale,
        "high_bits": gap.high / scale,
    }
    print(json.dumps(report, indent=2, default=float))
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, default=float)


if __name__ == "__main__":
    main()
