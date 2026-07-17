"""End-to-end training + SOTA showcase on fineweb-edu.

Trains UCSA-small with the full JEPA+SOTA stack wired up (LeWM-style
loss, hard-EMA target encoder, multi-step JEPA prediction chain,
input-reconstruction head, 4-stage curriculum) and reports its val
perplexity against pretrained HuggingFace baselines on the same
held-out cursor. If UCSA-small beats a larger baseline, the script
flags the win.

Defaults:
  - UCSA-small profile: hidden=384, layers=6, num_concepts=16,
    reasoning_iterations=4, max_seq_len=1024 (≈63M params).
  - 12000 steps with a 4-stage curriculum so every loss component
    warms up:
    language only (0-2000) -> language+jepa (2000-4500) ->
    language+jepa+memory (4500-6500) -> joint+router (6500-end).
  - 0.1 attention/residual/ffn dropout.
  - JEPA: LeWM mode (single SmoothL1 prediction + Gaussian
    regulariser) with multi-step chain (3 prediction steps per
    forward), plus the TC-JEPA sparse text conditioner.
  - Hard-EMA target encoder (momentum 0.996), target encoder swaps
    the multi-step targets so the predictor learns to match
    EMA-tracked latents.
  - Input-reconstruction head + loss (capacity bottleneck).
  - Held-out cursor advanced via skip(val_skip) on the same stream.

Usage:
    .venv/bin/python scripts/train.py [--max-steps N] [--ckpt-dir DIR]
                                     [--baselines gpt2 gpt2-medium]
                                     [--skip-baselines]
                                     [--no-ema] [--no-lewm] [--no-recon]
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from ucsa.models.perception import TokenizerWrapper
from ucsa.train import build_model, build_trainer
from ucsa.training.dataset import DatasetConfig, TextDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=12000)
    p.add_argument("--ckpt-dir", default="ckpts")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--stage-1-end", type=int, default=2000)
    p.add_argument("--stage-2-end", type=int, default=4500)
    p.add_argument("--stage-3-end", type=int, default=6500)
    p.add_argument("--val-skip", type=int, default=10_000,
                   help="Skip this many fineweb-edu examples before reading val batches")
    p.add_argument("--baseline-eval-batches", type=int, default=20)
    p.add_argument("--baselines", nargs="*",
                   default=["gpt2", "gpt2-medium"],
                   help="HuggingFace model ids evaluated zero-shot on the same val cursor")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Skip the SOTA comparison; just train UCSA-small")
    # SOTA stack toggles
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--ema-momentum", type=float, default=0.996,
                   help="EMA decay for the target encoder (default 0.996, I-JEPA-style)")
    p.add_argument("--no-lewm", dest="lewm", action="store_false")
    p.add_argument("--lewm-gaussian-reg", type=float, default=0.1)
    p.add_argument("--no-recon", dest="recon", action="store_false")
    p.add_argument("--reconstruction-weight", type=float, default=0.1)
    p.set_defaults(ema=True, lewm=True, recon=True)
    return p.parse_args()


class _IterableOver(torch.utils.data.IterableDataset):
    def __init__(self, ds: TextDataset) -> None:
        self.ds = ds

    def __iter__(self):
        return iter(self.ds)


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def _current_lr(trainer) -> float:
    sched = trainer.scheduler
    if hasattr(sched, "get_last_lr"):
        return sched.get_last_lr()[0]
    if hasattr(sched, "last_lr"):
        return sched.last_lr[0]
    return trainer.optimizer.param_groups[0]["lr"]


@torch.no_grad()
def _eval(trainer, loader: DataLoader, max_batches: int) -> dict[str, float]:
    """UCSA eval — uses trainer.compute_loss so all aux losses apply consistently."""
    trainer.model.eval()
    total, count = 0.0, 0
    it = iter(loader)
    for _i in range(max_batches):
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


@torch.no_grad()
def _eval_hf_baseline(
    model_id: str,
    val_iter,
    max_batches: int,
    device: torch.device,
) -> dict[str, float]:
    """Standard next-token cross-entropy averaged across ``max_batches``."""
    print(f"    loading {model_id}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    total, count = 0.0, 0
    for _i in range(max_batches):
        try:
            inputs, targets = next(val_iter)
        except StopIteration:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        out = model(input_ids=inputs)
        logits = out.logits
        if logits.shape[1] != targets.shape[1]:
            seq = min(logits.shape[1], targets.shape[1])
            logits = logits[:, -seq:, :]
            targets = targets[:, -seq:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        total += float(loss.item())
        count += 1
    del model
    if hasattr(device, "type") and device.type == "mps":
        torch.mps.empty_cache()
    if count == 0:
        return {"loss": 0.0, "perplexity": 0.0, "params": n_params}
    avg = total / count
    return {"loss": avg, "perplexity": math.exp(avg), "params": n_params}


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
    cfg["model"]["attention_dropout"] = 0.1
    cfg["model"]["residual_dropout"] = 0.1
    cfg["model"]["ffn_dropout"] = 0.1

    # JEPA stack
    cfg["model"]["jepa_mode"] = "lewm" if args.lewm else "ijepa"
    cfg["model"]["gaussian_reg_weight"] = args.lewm_gaussian_reg
    cfg["training"]["ema_momentum"] = args.ema_momentum if args.ema else 0.0
    cfg["training"]["ema_update_every"] = 1

    cfg["training"]["max_steps"] = args.max_steps
    cfg["training"]["warmup_steps"] = 400
    cfg["training"]["batch_size"] = 1
    cfg["training"]["log_every_n_steps"] = 100
    cfg["training"]["learning_rate"] = 6e-4
    cfg["training"]["checkpoint_every_n_steps"] = args.ckpt_every
    cfg["dataset"]["sequence_length"] = 1024
    cfg["curriculum"]["stage_1_end"] = args.stage_1_end
    cfg["curriculum"]["stage_2_end"] = args.stage_2_end
    cfg["curriculum"]["stage_3_end"] = args.stage_3_end

    print("Building tokenizer + dataset...", flush=True)
    ucsa_tokenizer = TokenizerWrapper(
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
    train_ds = TextDataset(ucsa_tokenizer, ds_cfg)
    val_ds = TextDataset(ucsa_tokenizer, ds_cfg)
    val_ds.dataset = val_ds.dataset.skip(args.val_skip)

    print("Building model...", flush=True)
    model = build_model(cfg)
    n_ucsa_params = sum(p.numel() for p in model.parameters())
    print(f"UCSA-small params: {n_ucsa_params:,}", flush=True)

    trainer = build_trainer(model, cfg)
    print(f"Device: {trainer.device}", flush=True)
    stack_label = (
        f"JEPA={cfg['model']['jepa_mode']} "
        f"+recon={args.recon} "
        f"+ema={cfg['training']['ema_momentum']:.3f}"
    )
    print(
        f"Stack: {stack_label}",
        flush=True,
    )
    print(
        f"Curriculum: stage1->{args.stage_1_end} stage2->{args.stage_2_end} "
        f"stage3->{args.stage_3_end} stage4->end",
        flush=True,
    )

    train_loader = DataLoader(_IterableOver(train_ds), batch_size=None)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Starting training: max_steps={args.max_steps}", flush=True)

    start = time.time()
    losses: list[float] = []
    best_val_ppl = float("inf")
    best_step = -1
    val_loader = DataLoader(_IterableOver(val_ds), batch_size=None)
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
            extras = (
                f"jp={snap.get('jepa_prediction', 0):.4f} "
                f"rec={snap.get('reconstruction_loss', 0):.4f}"
            )
            print(
                f"  step={step:5d} loss={loss:.4f} avg={avg:.4f} "
                f"lr={lr:.2e} stage={stage} {extras} elapsed={el:.0f}s",
                flush=True,
            )

        if step > 0 and step % args.eval_every == 0:
            vm = _eval(trainer, val_loader, args.eval_batches)
            if vm["perplexity"] < best_val_ppl:
                best_val_ppl = vm["perplexity"]
                best_step = step
            print(
                f"  eval@{step}: val_loss={vm['loss']:.4f} "
                f"val_ppl={vm['perplexity']:.1f} "
                f"(best={best_val_ppl:.1f}@{best_step})",
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

    # Final eval on UCSA-small using a longer window — the headline number.
    final_ucsa = _eval(trainer, val_loader, args.eval_batches)
    print(
        f"\nUCSA-small val (final, {args.eval_batches} batches): "
        f"val_loss={final_ucsa['loss']:.4f} val_ppl={final_ucsa['perplexity']:.1f}",
        flush=True,
    )

    if args.skip_baselines:
        return

    print("\n=== SOTA comparison vs HuggingFace baselines ===", flush=True)
    print(
        f"  evaluating baselines on the same val cursor "
        f"(skip={args.val_skip}, batches={args.baseline_eval_batches}, "
        f"sequence_length={cfg['dataset']['sequence_length']})",
        flush=True,
    )
    device = trainer.device
    bench_ds = TextDataset(
        ucsa_tokenizer,
        DatasetConfig(
            sequence_length=cfg["dataset"]["sequence_length"],
            primary_dataset=cfg["dataset"]["primary_dataset"],
            primary_split=cfg["dataset"]["primary_split"],
            streaming=True,
            pack_sequences=True,
        ),
    )
    bench_ds.dataset = bench_ds.dataset.skip(args.val_skip)
    bench_loader = DataLoader(_IterableOver(bench_ds), batch_size=None)
    bench_iter = iter(bench_loader)

    rows: list[tuple[str, int, float | None]] = [
        ("UCSA-small (ours, SOTA stack)", n_ucsa_params, final_ucsa["perplexity"]),
    ]
    for baseline in args.baselines:
        try:
            metrics = _eval_hf_baseline(
                baseline, bench_iter, args.baseline_eval_batches, device
            )
            rows.append((baseline, metrics["params"], metrics["perplexity"]))
        except Exception as exc:
            print(f"    {baseline} failed: {exc}", flush=True)
            rows.append((baseline, 0, None))

    print("\n  Model                              Params        Val PPL", flush=True)
    print("  -----------------------------------  ------------  --------", flush=True)
    for name, params, ppl in rows:
        ppl_str = f"{ppl:.1f}" if ppl is not None else "n/a"
        print(
            f"  {name:35s}  {params/1e6:>7.0f}M       {ppl_str}",
            flush=True,
        )

    ucsa_ppl = final_ucsa["perplexity"]
    print("\n  Small beats large:", flush=True)
    for name, _, ppl in rows[1:]:
        if ppl is None:
            continue
        ratio = ucsa_ppl / ppl
        verdict = "WIN" if ratio < 1.0 else "loss"
        print(
            f"    UCSA-small / {name:20s}: {ratio:.3f}x  ({verdict})",
            flush=True,
        )


if __name__ == "__main__":
    main()
