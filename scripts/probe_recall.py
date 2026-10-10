"""Tests whether the state carries information past the attention window.

    python scripts/probe_recall.py --ckpt ckpts/r-small/final.pt \
        --control ckpts/r-nostate/final.pt

Each probe window is: a random token span, `gap` tokens of real text, then the
same span again. The decoder window reaches back one chunk, so with a gap
larger than that only the state can know the span. Reports the loss on the
repeated span against the loss on a fresh random span of the same length
(the floor with no memory). A model that remembers pays less on the repeat.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from torch.nn import functional

from ucsa.training import engine, shards


@torch.no_grad()
def repeat_loss(
    model, spans: torch.Tensor, filler: torch.Tensor, span_len: int
) -> tuple[float, float]:
    """Returns bits per token on a repeated span and on a fresh span.

    Args:
      model: Model to score.
      spans: Random spans, shape (n, span_len).
      filler: Real text, shape (n, gap).
      span_len: Length of each span.

    Returns:
      (loss on the repeat, loss on a fresh span), both in bits per token.
    """
    device = next(model.parameters()).device
    fresh = torch.randint_like(spans, 0, int(spans.max()) + 1)
    out = []
    for second in (spans, fresh):
        ids = torch.cat([spans, filler, second], dim=1).to(device)
        logits = model(ids[:, :-1])["logits"][:, -span_len:]
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ids[:, -span_len:].reshape(-1),
        )
        out.append(float(loss) / math.log(2.0))
    return out[0], out[1]


def main() -> None:
    """Parses arguments, probes each model and prints the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--control", default=None)
    parser.add_argument("--data", default="data")
    parser.add_argument("--span", type=int, default=32)
    parser.add_argument("--gaps", type=int, nargs="+", default=[0, 64, 512])
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--out-json", default="runs/probe-recall.json")
    args = parser.parse_args()

    device = engine.pick_device()
    shard = shards.Shard(os.path.join(args.data, "val.bin"))
    models = {"state": engine.load_model(args.ckpt, device)}
    if args.control:
        models["control"] = engine.load_model(args.control, device)
    rng = np.random.default_rng(0)
    vocab = 50257
    report: dict = {}
    for gap in args.gaps:
        spans = torch.from_numpy(
            rng.integers(0, vocab, (args.windows, args.span))
        )
        text = [
            x
            for x, _ in shard.batches(
                1, max(gap, 1), skip=0, seed=None, loop=False
            )
        ]
        filler = torch.cat(text[: args.windows], dim=0)[:, :gap]
        for name, model in models.items():
            repeat, fresh = repeat_loss(model.eval(), spans, filler, args.span)
            report[f"{name}-gap{gap}"] = {"repeat": repeat, "fresh": fresh}
            print(
                f"{name:8s} gap {gap:4d}: repeat {repeat:6.2f}  fresh {fresh:6.2f}"
            )
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
