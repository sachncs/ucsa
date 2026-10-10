"""Training engine for UCSA-R: resumable, configurable, failure-tolerant.

Design goals, in order: a long run must survive being killed, a bad batch
must not poison the weights, and every knob must be a config field so a
tuner can drive it. Nothing here is specific to one dataset.
"""

import dataclasses
import json
import math
import os
import queue
import random
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from ucsa.models import recurrent
from ucsa.training import compression

BatchIterator = Iterator[tuple[torch.Tensor, torch.Tensor]]
Batches = Callable[[int], BatchIterator]

# Positions scored for the headline perplexity: the last 64 of each window.
SCORED_TAIL = 64

# Supported compute precisions and their autocast dtypes (None = full).
PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Optimisation and bookkeeping knobs.

    Attributes:
      steps: Optimiser steps to run.
      batch_size: Sequences per micro-batch.
      grad_accum: Micro-batches per optimiser step.
      lr: Peak learning rate.
      min_lr_ratio: Final learning rate as a fraction of `lr`.
      warmup_steps: Linear warm-up length.
      weight_decay: Decay on weight matrices (not norms, biases, embeddings).
      beta1: Adam first-moment coefficient.
      beta2: Adam second-moment coefficient.
      grad_clip: Global gradient-norm clip.
      seq_len: Tokens per sequence.
      eval_every: Steps between validations; 0 disables.
      eval_batches: Validation batches per evaluation.
      ckpt_every: Steps between checkpoints; 0 disables.
      keep_ckpts: Numbered checkpoints to keep.
      log_every: Steps between log lines.
      seed: Seed for Python and torch RNGs and data order.
      precision: Compute precision, `fp32`, `fp16` or `bf16`. 16-bit modes use
        autocast with fp32 master weights; fp16 adds a dynamic loss scaler.
      weight_ema: Decay of an exponential moving average of the weights that
        is used for evaluation and the final model; 0 disables it.
      weight_ema_every: Steps between EMA updates (the decay is raised to
        this power, so the averaging horizon is unchanged but the per-step
        cost drops).
      prefetch: Batches the data thread keeps ready.
      max_bad_steps: Consecutive non-finite steps tolerated before aborting.
      out_dir: Directory for checkpoints and the run record.
    """

    steps: int = 12000
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 400
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seq_len: int = 1024
    eval_every: int = 1000
    eval_batches: int = 20
    ckpt_every: int = 1000
    keep_ckpts: int = 2
    log_every: int = 100
    seed: int = 42
    precision: str = "fp32"
    weight_ema: float = 0.0
    weight_ema_every: int = 4
    prefetch: int = 4
    max_bad_steps: int = 20
    out_dir: str = "ckpts/r"

    def __post_init__(self) -> None:
        """Validates the fields.

        Raises:
          ValueError: If a count is not positive or a ratio is out of range.
        """
        if min(self.steps, self.batch_size, self.grad_accum) <= 0:
            raise ValueError("steps, batch_size, grad_accum must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.prefetch < 1:
            raise ValueError("prefetch must be >= 1")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
        if not 0.0 <= self.weight_ema < 1.0 or self.weight_ema_every < 1:
            raise ValueError("weight_ema must be in [0, 1), every >= 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        """Builds a config from a dict, rejecting unknown keys.

        Args:
          data: Field values.

        Returns:
          The validated config.

        Raises:
          ValueError: If `data` has an unknown key.
        """
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown train keys: {sorted(unknown)}")
        return cls(**data)


def pick_device() -> torch.device:
    """Returns the best available accelerator, else the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_at(step: int, config: TrainConfig) -> float:
    """Returns the learning rate: linear warm-up, then cosine decay.

    Args:
      step: Zero-based optimiser step.
      config: Training configuration.

    Returns:
      The learning rate for `step`, decaying to `lr * min_lr_ratio`.
    """
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    span = max(1, config.steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / span)
    floor = config.lr * config.min_lr_ratio
    return floor + 0.5 * (config.lr - floor) * (
        1.0 + math.cos(math.pi * progress)
    )


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """Builds AdamW with weight decay on matrices only.

    Args:
      model: Model whose trainable parameters are optimised.
      config: Training configuration.

    Returns:
      The optimiser; norms, biases, embeddings and the initial state are
      excluded from weight decay.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        skip = param.ndim < 2 or "embed" in name or "state0" in name
        (no_decay if skip else decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )


def clip_grad_norm(params: list[nn.Parameter], max_norm: float) -> torch.Tensor:
    """Clips the global gradient norm with a single reduction.

    `torch.nn.utils.clip_grad_norm_` launches one reduction per tensor on
    backends without foreach kernels (MPS), which cost 60 ms per step on a
    37M-parameter model. One concatenated norm is a single kernel.

    Args:
      params: Parameters whose `.grad` is clipped in place.
      max_norm: Maximum global norm.

    Returns:
      The pre-clip global norm as a device tensor (no host sync).
    """
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads]))
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(scale)
    return norm


class Prefetcher:
    """Produces batches on a background thread so data never stalls the GPU.

    A stalled or failing source surfaces as an exception in the consumer
    instead of a silent hang.
    """

    def __init__(
        self, source: BatchIterator, depth: int, timeout: float = 300.0
    ) -> None:
        """Starts the producer thread.

        Args:
          source: Iterator of CPU batches.
          depth: Maximum batches buffered.
          timeout: Seconds to wait for a batch before raising.
        """
        self.timeout = timeout
        self.items: queue.Queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self.produce, args=(source,), daemon=True
        )
        self.thread.start()

    def produce(self, source: BatchIterator) -> None:
        """Thread body: pushes batches, then an end marker or the error."""
        try:
            for item in source:
                while not self.stop.is_set():
                    try:
                        self.items.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if self.stop.is_set():
                    return
            self.items.put(StopIteration())
        except Exception as error:  # Forwarded to the consumer.
            self.items.put(error)

    def __iter__(self) -> Prefetcher:
        """Returns self."""
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next batch.

        Raises:
          StopIteration: When the source is exhausted.
          TimeoutError: If no batch arrives within `timeout` seconds.
        """
        try:
            item = self.items.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("data source stalled") from None
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        """Stops the producer thread."""
        self.stop.set()


class WeightEma:
    """Exponential moving average of a model's trainable weights.

    Averaged weights generalise better than the last iterate, which matters
    most for small models trained on little data. The shadow copy is kept in
    fp32 and updated every few steps to keep the per-step cost low.
    """

    def __init__(self, model: nn.Module, decay: float, every: int) -> None:
        """Copies the current weights.

        Args:
          model: Model to track.
          decay: Per-step decay in `[0, 1)`.
          every: Steps between updates.
        """
        self.decay = decay**every
        self.every = every
        self.shadow = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Folds the live weights in, on every `every`-th step."""
        if step % self.every:
            return
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach().float(), alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def swap(self, model: nn.Module) -> None:
        """Exchanges the live weights with the averaged ones.

        Calling it twice restores the original state.

        Args:
          model: Model whose weights are exchanged.
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                live = param.detach().clone()
                param.copy_(self.shadow[name].to(param.dtype))
                self.shadow[name] = live.float()


@torch.no_grad()
def evaluate(
    model: recurrent.RecurrentUCSA,
    batches: BatchIterator,
    count: int,
    byte_lengths: torch.Tensor | None = None,
) -> dict[str, float]:
    """Measures perplexity over all positions and over the last 64.

    The tail number is the headline metric: it is the position range used by
    every comparison in the paper, so context length is the same everywhere.

    Args:
      model: Model to evaluate.
      batches: Validation batches.
      count: Maximum number of batches.
      byte_lengths: Raw bytes per token id (`compression.token_byte_lengths`).
        When given, bits per byte is reported as well.

    Returns:
      `ppl_all` and `ppl_last64`, plus `bpb_all` and `bpb_last64` when
      `byte_lengths` is given; infinite if no batch was available.
    """
    model.eval()
    device = next(model.parameters()).device
    nll_all = torch.zeros((), device=device)
    nll_tail = torch.zeros((), device=device)
    n_all = n_tail = 0
    bytes_all = bytes_tail = 0
    lengths = None if byte_lengths is None else byte_lengths.to(device)
    for n, (x, y) in enumerate(batches):
        if n >= count:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)["logits"].float()
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        ).view(y.shape)
        tail = loss[:, -SCORED_TAIL:]
        nll_all = nll_all + loss.sum()
        nll_tail = nll_tail + tail.sum()
        n_all += loss.numel()
        n_tail += tail.numel()
        if lengths is not None:
            per_token = lengths[y]
            bytes_all = bytes_all + per_token.sum()
            bytes_tail = bytes_tail + per_token[:, -SCORED_TAIL:].sum()
    model.train()
    if n_all == 0:
        return {"ppl_all": math.inf, "ppl_last64": math.inf}
    metrics = {
        "ppl_all": math.exp(float(nll_all) / n_all),
        "ppl_last64": math.exp(float(nll_tail) / n_tail),
    }
    if lengths is not None:
        metrics["bpb_all"] = compression.bits_per_byte(
            float(nll_all), int(bytes_all)
        )
        metrics["bpb_last64"] = compression.bits_per_byte(
            float(nll_tail), int(bytes_tail)
        )
    return metrics


def save_checkpoint(path: str, payload: dict[str, Any]) -> None:
    """Writes a checkpoint atomically.

    A crash mid-save leaves the previous file intact.

    Args:
      path: Destination file.
      payload: Object passed to `torch.save`.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def prune_checkpoints(out_dir: str, keep: int) -> None:
    """Deletes all but the newest `keep` numbered checkpoints.

    Args:
      out_dir: Checkpoint directory.
      keep: Number to keep; 0 or less keeps everything.
    """
    if keep <= 0:
        return
    names = sorted(
        n
        for n in os.listdir(out_dir)
        if n.startswith("step") and n.endswith(".pt")
    )
    for name in names[:-keep]:
        os.remove(os.path.join(out_dir, name))


def load_model(path: str, device: torch.device) -> recurrent.RecurrentUCSA:
    """Loads a model saved by `fit`.

    Args:
      path: Checkpoint file.
      device: Device to place the model on.

    Returns:
      The model in eval mode.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = recurrent.RecurrentConfig.from_dict(blob["model_config"])
    model = recurrent.RecurrentUCSA(config)
    model.load_state_dict(blob["model"], strict=True)
    return model.to(device).eval()


def fit(
    model: recurrent.RecurrentUCSA,
    config: TrainConfig,
    train_batches: Batches,
    val_batches: Batches | None = None,
    resume: bool = False,
    log: Callable[[str], None] = print,
    byte_lengths: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Trains `model` and returns the run record.

    Args:
      model: Model to train.
      config: Training configuration.
      train_batches: Maps batches already consumed to an iterator, so a
        resumed run continues the stream.
      val_batches: Same, over held-out data.
      resume: Continue from `out_dir/latest.pt` when it exists.
      log: Sink for progress lines.
      byte_lengths: Raw bytes per token id; enables bits-per-byte reporting.

    Returns:
      A dict with the configs, parameter count, final metrics and history.

    Raises:
      RuntimeError: After `max_bad_steps` consecutive non-finite steps.
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = pick_device()
    model.to(device).train()
    optimizer = build_optimizer(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    os.makedirs(config.out_dir, exist_ok=True)
    dtype = PRECISIONS[config.precision]
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    averaged = (
        WeightEma(model, config.weight_ema, config.weight_ema_every)
        if config.weight_ema
        else None
    )

    step, best, history, bad = 0, math.inf, [], 0
    latest = os.path.join(config.out_dir, "latest.pt")
    if resume and os.path.exists(latest):
        blob = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=True)
        optimizer.load_state_dict(blob["optimizer"])
        if averaged is not None and "ema" in blob:
            averaged.shadow = {k: v.to(device) for k, v in blob["ema"].items()}
        step, best, history = blob["step"], blob["best"], blob["history"]
        log(f"resumed from step {step}")
    stream = Prefetcher(
        train_batches(step * config.grad_accum), config.prefetch
    )

    def save(name: str) -> None:
        save_checkpoint(
            os.path.join(config.out_dir, name),
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": averaged.shadow if averaged is not None else None,
                "step": step,
                "best": best,
                "history": history,
                "model_config": model.config.to_dict(),
                "train_config": dataclasses.asdict(config),
            },
        )

    started = time.time()
    tokens = 0
    window = torch.zeros((), device=device)
    try:
        while step < config.steps:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, config)
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            for _ in range(config.grad_accum):
                x, y = next(stream)
                x, y = x.to(device), y.to(device)
                with torch.autocast(
                    device.type, dtype=dtype, enabled=dtype is not None
                ):
                    loss, _ = model.compute_loss(x, y)
                scaler.scale(loss / config.grad_accum).backward()
                step_loss = step_loss + loss.detach() / config.grad_accum
                tokens += x.numel()
            scaler.unscale_(optimizer)
            norm = clip_grad_norm(params, config.grad_clip)
            if not bool(torch.isfinite(norm)):  # The only host sync per step.
                scaler.step(optimizer)  # Skips; lets the scaler back off.
                scaler.update()
                bad += 1
                log(f"  step={step} non-finite gradient: skipped ({bad})")
                if bad >= config.max_bad_steps:
                    raise RuntimeError(f"{bad} consecutive non-finite steps")
                continue
            bad = 0
            scaler.step(optimizer)
            scaler.update()
            model.update_ema()
            step += 1
            if averaged is not None:
                averaged.update(model, step)
            window = window + step_loss

            if step % config.log_every == 0:
                elapsed = time.time() - started
                avg = float(window) / config.log_every
                window = torch.zeros((), device=device)
                log(
                    f"  step={step:6d} loss={avg:.4f} "
                    f"lr={lr_at(step, config):.2e} "
                    f"tok/s={tokens / elapsed:.0f} elapsed={elapsed:.0f}s"
                )
                history.append({"step": step, "loss": avg, "elapsed": elapsed})
            if (
                val_batches
                and config.eval_every
                and (step % config.eval_every == 0)
            ):
                if averaged is not None:
                    averaged.swap(model)
                metrics = evaluate(
                    model, val_batches(0), config.eval_batches, byte_lengths
                )
                if averaged is not None:
                    averaged.swap(model)
                best = min(best, metrics["ppl_last64"])
                history.append({"step": step, **metrics})
                log(
                    f"  eval@{step}: ppl_last64={metrics['ppl_last64']:.1f} "
                    f"ppl_all={metrics['ppl_all']:.1f} "
                    f"bpb={metrics.get('bpb_last64', float('nan')):.3f} "
                    f"(best={best:.1f})"
                )
            if config.ckpt_every and step % config.ckpt_every == 0:
                save(f"step{step:07d}.pt")
                save("latest.pt")
                prune_checkpoints(config.out_dir, config.keep_ckpts)
    finally:
        stream.close()

    if averaged is not None:
        averaged.swap(model)  # The averaged weights are the final model.
    final = {}
    if val_batches:
        final = evaluate(
            model, val_batches(0), config.eval_batches, byte_lengths
        )
    save("final.pt")
    record = {
        "model_config": model.config.to_dict(),
        "train_config": dataclasses.asdict(config),
        "params": sum(p.numel() for p in params),
        "final": final,
        "best_ppl_last64": best,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    with open(os.path.join(config.out_dir, "record.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record
