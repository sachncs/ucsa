"""Measures UCSA-R training throughput and accelerator memory.

    python scripts/profile_r.py --batch 1 4 8 --set model.hidden=256

Reports tokens/s, step time and peak driver memory per batch size so an
optimisation is judged by measurement.
"""

import argparse
import gc
import os
import time

import torch
import yaml

from ucsa.models import recurrent
from ucsa.training import engine, shards


def synchronize(device: torch.device) -> None:
    """Blocks until queued device work finishes."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def memory_mb(device: torch.device) -> float:
    """Returns accelerator memory in use, in MiB (0 on CPU)."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    return 0.0


def measure(
    config: recurrent.Config,
    batch: int,
    seq_len: int,
    steps: int,
    shard: shards.Shard,
) -> tuple[int, float, float, float]:
    """Times full optimiser steps on real training batches.

    Args:
      config: Model configuration.
      batch: Sequences per step.
      seq_len: Tokens per sequence.
      steps: Timed steps (two warm-up steps are added).
      shard: Training shard the batch is read from.

    Returns:
      `(params, tokens_per_second, seconds_per_step, memory_mib)` using the
      median step time.
    """
    device = engine.pick_device()
    torch.manual_seed(0)
    model = recurrent.Model(config).to(device).train()
    train_config = engine.Config(seq_len=seq_len, batch_size=batch)
    optimizer = engine.build_optimizer(model, train_config)
    params = [p for p in model.parameters() if p.requires_grad]
    x, y = (t.to(device) for t in next(shard.batches(batch, seq_len, seed=0)))
    times = []
    for i in range(steps + 2):
        synchronize(device)
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(x, y)
        loss.backward()
        engine.clip_grad_norm(params, 1.0)
        optimizer.step()
        model.update_ema()
        synchronize(device)
        if i >= 2:
            times.append(time.time() - start)
    median = sorted(times)[len(times) // 2]
    result = (
        recurrent.count_parameters(model),
        batch * seq_len / median,
        median,
        memory_mb(device),
    )
    del model, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def main() -> None:
    """Parses arguments and prints one line per batch size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", nargs="*", default=[], help="model.key=value")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--data", default="data")
    args = parser.parse_args()

    fields = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        fields[key.removeprefix("model.")] = yaml.safe_load(raw)
    config = recurrent.Config.from_dict(fields)
    shard = shards.Shard(os.path.join(args.data, "train.bin"))
    for batch in args.batch:
        try:
            count, tps, seconds, mem = measure(
                config, batch, args.seq, args.steps, shard
            )
        except RuntimeError as error:  # e.g. out of memory
            print(f"batch={batch}: FAILED {str(error)[:80]}")
            continue
        print(
            f"batch={batch:2d} params={count / 1e6:.1f}M {tps:8.0f} tok/s "
            f"{seconds * 1000:6.0f} ms/step mem={mem:7.0f} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
