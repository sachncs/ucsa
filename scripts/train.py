"""End-to-end training on fineweb-edu.

Profile (matches the default UCSA paper config, scaled for MPS):
- hidden_size=384, num_layers=6, num_concepts=16, reasoning_iterations=4
- batch_size=1 (UCSA PCS is per-call today), max_seq_len=1024
- 8000 steps with a 4-stage curriculum so every component warms up

Usage:
    .venv/bin/python scripts/train.py [--max-steps N] [--ckpt-dir DIR]
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from ucsa.train import build_model, build_trainer
from ucsa.training.curriculum import Curriculum, CurriculumSchedule
from ucsa.training.dataset import DatasetConfig, TextDataset
from ucsa.models.perception import TokenizerWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--ckpt-dir", default="ckpts")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--stage-1-end", type=int, default=2000)
    p.add_argument("--stage-2-end", type=int, default=4500)
    p.add_argument("--stage-3-end", type=int, default=6500)
    return p.parse_args()


class _IterableOver(torch.utils.data.IterableDataset):
    """Wrap a TextDataset in the IterableDataset protocol for DataLoader."""

    def __init__(self, ds: TextDataset) -> None:
        self.ds = ds

    def __iter__(self):
        return iter(self.ds)


def _infinite(loader: DataLoader):
    """Yield from loader forever; the outer loop controls when to stop."""
    while True:
        for batch in loader:
            yield batch


def _current_lr(trainer) -> float:
    sched = trainer.scheduler
    if hasattr(sched, "get_last_lr"):
        return sched.get_last_lr()[0]
    if hasattr(sched, "last_lr"):
        return sched.last_lr[0]
    return trainer.optimizer.param_groups[0]["lr"]


@torch.no_grad()
def _eval(trainer, loader: DataLoader, max_batches: int) -> dict[str, float]:
    """Run a quick held-out eval using trainer.compute_loss."""
    trainer.model.eval()
    total, count = 0.0, 0
    it = iter(loader)
    for i in range(max_batches):
        try:
            inputs, targets = next(it)
        except StopIteration:
            break
        inputs = inputs.to(trainer.device)
        targets = targets.to(trainer.device)
        loss, _ = trainer.compute_loss(inputs, targets)
        total += float(loss.item())
        count += 1
    trainer.model.train()
    if count == 0:
        return {"loss": 0.0, "perplexity": 0.0}
    avg = total / count
    return {"loss": avg, "perplexity": math.exp(avg)}


def main() -> None:
    args = parse_args()
    with open("ucsa/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["reasoning_iterations"] = 4
    cfg["model"]["hidden_size"] = 384
    cfg["model"]["num_layers"] = 6
    cfg["model"]["num_q_heads"] = 8
    cfg["model"]["num_kv_heads"] = 4
    cfg["model"]["intermediate_size"] = 1024
    cfg["model"]["vocab_size"] = 50257
    cfg["model"]["max_seq_len"] = 1024
    cfg["model"]["num_concepts"] = 16

    cfg["training"]["max_steps"] = args.max_steps
    cfg["training"]["warmup_steps"] = 200
    cfg["training"]["batch_size"] = 1
    cfg["training"]["log_every_n_steps"] = 100
    cfg["training"]["learning_rate"] = 6e-4
    cfg["training"]["checkpoint_every_n_steps"] = args.ckpt_every
    cfg["dataset"]["sequence_length"] = 1024

    print("Building tokenizer + dataset...", flush=True)
    tokenizer = TokenizerWrapper(
        tokenizer_name=cfg["tokenizer"]["name"],
        max_seq_len=cfg["dataset"]["sequence_length"],
    )
    ds_cfg = DatasetConfig(
        sequence_length=cfg["dataset"]["sequence_length"],
        primary_dataset=cfg["dataset"]["primary_dataset"],
        primary_split=cfg["dataset"]["primary_split"],
        streaming=True,
        pack_sequences=True,
    )
    train_ds = TextDataset(tokenizer, ds_cfg)
    val_ds = TextDataset(tokenizer, ds_cfg)  # ponytail: same stream, different cursor; real eval wants a real validation split

    print("Building model...", flush=True)
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}", flush=True)

    curriculum = Curriculum(CurriculumSchedule(
        stage_1_end=args.stage_1_end,
        stage_2_end=args.stage_2_end,
        stage_3_end=args.stage_3_end,
    ))
    trainer = build_trainer(model, cfg, curriculum=curriculum)
    print(f"Device: {trainer.device}", flush=True)
    print(
        f"Curriculum: stage1→{args.stage_1_end} stage2→{args.stage_2_end} "
        f"stage3→{args.stage_3_end} stage4→end",
        flush=True,
    )

    train_loader = DataLoader(_IterableOver(train_ds), batch_size=None)
    val_loader = DataLoader(_IterableOver(val_ds), batch_size=None)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Starting: max_steps={args.max_steps}", flush=True)

    start = time.time()
    losses: list[float] = []
    for step, batch in enumerate(_infinite(train_loader)):
        if step >= args.max_steps:
            break
        snap = trainer.train_step(batch)
        loss = snap.get("training_loss") or 0.0
        losses.append(loss)

        if step % cfg["training"]["log_every_n_steps"] == 0:
            el = time.time() - start
            window = min(cfg["training"]["log_every_n_steps"], len(losses))
            avg = sum(losses[-window:]) / window
            lr = _current_lr(trainer)
            stage = trainer.curriculum.state.current_stage.display_name
            print(
                f"  step={step:5d} loss={loss:.4f} avg={avg:.4f} "
                f"lr={lr:.2e} stage={stage} elapsed={el:.0f}s",
                flush=True,
            )

        if step > 0 and step % args.eval_every == 0:
            vm = _eval(trainer, val_loader, args.eval_batches)
            print(
                f"  eval@{step}: val_loss={vm['loss']:.4f} "
                f"val_ppl={vm['perplexity']:.1f}",
                flush=True,
            )

        if step > 0 and step % args.ckpt_every == 0:
            path = os.path.join(args.ckpt_dir, f"ucsa-step{step}.safetensors")
            trainer.save_checkpoint(path)
            print(f"  saved {path}", flush=True)

    trainer.save_checkpoint(os.path.join(args.ckpt_dir, "ucsa-final.safetensors"))
    elapsed = time.time() - start
    print(
        f"\nDone: {args.max_steps} steps in {elapsed:.0f}s "
        f"= {args.max_steps/elapsed:.2f} steps/s",
        flush=True,
    )
    print(f"first-10 losses: {[round(x, 3) for x in losses[:10]]}", flush=True)
    print(f"last-10 losses:  {[round(x, 3) for x in losses[-10:]]}", flush=True)
    print(
        f"loss delta (first5 mean - last5 mean): "
        f"{sum(losses[:5])/5 - sum(losses[-5:])/5:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
