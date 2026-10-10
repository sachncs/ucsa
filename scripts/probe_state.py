"""Measures what the persistent state contributes, in compression terms.

    python scripts/probe_state.py --ckpt ckpts/r-small/final.pt \
        --control ckpts/r-nostate/final.pt

Reports, in bits per token:

* the loss of each chunk position of a window. A state that compresses the
  past makes later chunks cheaper; the no-state control stays flat;
* the same profile on windows longer than any seen in training, where only a
  constant-size state can keep carrying information;
* a memory rate-distortion curve: bits per byte against the number of state
  slots the decoder may read.
"""

import argparse
import json
import math
import os

import numpy as np
import transformers

from ucsa.training import compression, diagnostics, engine, shards


def bits(nats: np.ndarray) -> list[float]:
    """Converts nats per token to bits per token."""
    return [float(v) / math.log(2.0) for v in nats]


def main() -> None:
    """Parses arguments, probes, prints and writes the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--control", default=None, help="no-state checkpoint")
    parser.add_argument("--data", default="data")
    parser.add_argument("--shard", default="val.bin")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--slots", type=int, nargs="+", default=None)
    parser.add_argument("--out-json", default="runs/probe-state.json")
    args = parser.parse_args()

    device = engine.pick_device()
    model = engine.load_model(args.ckpt, device)
    control = engine.load_model(args.control, device) if args.control else None
    shard = shards.Shard(os.path.join(args.data, args.shard))
    report: dict = {"profiles": {}, "rate_distortion": []}

    for length in args.lengths:

        def make(skip: int, length: int = length) -> engine.BatchIterator:
            return shard.batches(1, length, skip=skip, seed=None, loop=False)

        entry = {
            "state": bits(
                diagnostics.chunk_profile(model, make(0), args.windows)
            )
        }
        if control is not None:
            entry["control"] = bits(
                diagnostics.chunk_profile(control, make(0), args.windows)
            )
            entry["state_minus_control"] = [
                s - c
                for s, c in zip(entry["state"], entry["control"], strict=True)
            ]
        report["profiles"][str(length)] = entry
        print(f"\nwindow {length} tokens (bits/token per chunk):")
        for key, values in entry.items():
            print(f"  {key:20s}", " ".join(f"{v:5.2f}" for v in values))

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    lengths = compression.token_byte_lengths(tokenizer)
    total = sum(n for _, n in model.config.banks)
    slots = args.slots or sorted({1, total // 4, total // 2, total})
    base_length = args.lengths[0]
    report["rate_distortion"] = diagnostics.state_rate_distortion(
        model,
        lambda skip: shard.batches(
            1, base_length, skip=skip, seed=None, loop=False
        ),
        args.windows,
        [s for s in slots if s >= 1],
        lengths,
    )
    print("\nmemory rate-distortion (state read -> bits/byte):")
    for row in report["rate_distortion"]:
        print(
            f"  {int(row['slots']):3d} slots "
            f"({row['state_bits']:9.0f} bits) -> {row['bpb']:.4f}"
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
