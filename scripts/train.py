"""UCSA-small training + SOTA stack + ablation harness.

This script is the central entrypoint for the paper's experiments.
It supports:

- A full-stack UCSA-small training run (default), with the
  multi-step JEPA chain, hard-EMA target encoder, LeWM mode,
  input-reconstruction head, and TC-JEPA text conditioner all
  enabled.
- A per-feature ablation mode. ``--no-ema``, ``--no-lewm``,
  ``--no-recon``, ``--no-tc-jepa``, ``--no-curriculum``, and so on
  isolate the contribution of each piece.
- Structured JSON output for downstream paper-writing tools.
- Deterministic seeding (``--seed``).

Usage:
    .venv/bin/python scripts/train.py [--seed N] [--max-steps N]
        [--ablation NAME] [--out-json PATH]
        [--no-ema|--no-lewm|--no-recon|--no-tc-jepa|--no-curriculum]
        [--baselines gpt2 gpt2-medium] [--skip-baselines]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from ucsa.models.perception import TokenizerWrapper
from ucsa.train import build_model, build_trainer
from ucsa.training.dataset import DatasetConfig, TextDataset
from ucsa.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=12000)
    p.add_argument("--ckpt-dir", default="ckpts")
    p.add_argument("--ckpt-every", type=int, default=1000,
                   help="0 disables periodic checkpoints.")
    p.add_argument("--eval-every", type=int, default=500,
                   help="0 disables periodic evaluation.")
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--stage-1-end", type=int, default=2000)
    p.add_argument("--stage-2-end", type=int, default=4500)
    p.add_argument("--stage-3-end", type=int, default=6500)
    p.add_argument("--val-skip", type=int, default=10_000)
    p.add_argument("--baseline-eval-batches", type=int, default=20)
    p.add_argument("--baselines", nargs="*",
                   default=["gpt2", "gpt2-medium"],
                   help="HuggingFace model ids evaluated zero-shot on the same val cursor")
    p.add_argument("--skip-baselines", action="store_true")
    # SOTA stack toggles — ablations
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--ema-momentum", type=float, default=0.996)
    p.add_argument("--no-lewm", dest="lewm", action="store_false")
    p.add_argument("--lewm-gaussian-reg", type=float, default=0.1)
    p.add_argument("--no-recon", dest="recon", action="store_false")
    p.add_argument("--reconstruction-weight", type=float, default=0.1)
    p.add_argument("--no-tc-jepa", dest="tc_jepa", action="store_false")
    p.add_argument("--text-conditioner-scale", type=float, default=0.1)
    # Endogenous origination (intent bank) toggles
    p.add_argument("--observation-mix", type=float, default=1.0,
                   help="alpha_0. 1.0 keeps the exogenous observation and "
                        "never calls the origination generator.")
    p.add_argument("--observation-mix-decay", type=float, default=1.0)
    p.add_argument("--origination-top-k", type=int, default=2)
    p.add_argument("--no-origination-balance", dest="origination_balance",
                   action="store_false",
                   help="Drop the gate's load-balancing loss. Tightens "
                        "localisation but collapses the gate's mutual "
                        "information with the input to zero.")
    p.add_argument("--origination-weight", type=float, default=0.01)
    p.add_argument("--intent-update-scale", type=float, default=0.1,
                   help="0.0 freezes the intent bank, making the "
                        "origination signal identical for every input.")
    p.add_argument("--stream-intent-bank", action="store_true",
                   help="Also let the operator attend over the intent "
                        "bank. Destroys per-slot attribution; kept as the "
                        "ablation that shows why the exclusion is needed.")
    p.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                   help="Disable the 4-stage curriculum (all losses always on).")
    p.add_argument("--ablation", default=None,
                   help="A short tag appended to --out-json (e.g., 'no-ema').")
    p.add_argument("--out-json", default=None)
    p.set_defaults(ema=True, lewm=True, recon=True,
                   tc_jepa=True, curriculum=True, origination_balance=True)
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
def _eval_hf_baseline(model_id, val_iter, max_batches, device):
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
    set_seed(args.seed, deterministic=True)

    # Auto-name output if not specified, so ablation sweeps land
    # in separate JSON files without a shell wrapper.
    if args.out_json is None:
        ab = args.ablation or "full"
        os.makedirs("runs", exist_ok=True)
        args.out_json = f"runs/ucsa-{ab}-seed{args.seed}.json"

    print(f"Seed: {args.seed}", flush=True)
    print(f"Output: {args.out_json}", flush=True)

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

    # Ablation toggles that all map into the config
    cfg["model"]["jepa_mode"] = "lewm" if args.lewm else "ijepa"
    cfg["model"]["gaussian_reg_weight"] = args.lewm_gaussian_reg
    cfg["training"]["ema_momentum"] = (
        args.ema_momentum if args.ema else 0.0
    )
    cfg["training"]["ema_update_every"] = 1
    cfg["model"]["text_conditioner_scale"] = (
        args.text_conditioner_scale if args.tc_jepa else 0.0
    )
    cfg["model"]["observation_mix"] = args.observation_mix
    cfg["model"]["observation_mix_decay"] = args.observation_mix_decay
    cfg["model"]["origination_top_k"] = args.origination_top_k
    cfg["model"]["intent_update_scale"] = args.intent_update_scale
    cfg["model"]["stream_intent_bank"] = args.stream_intent_bank

    cfg["training"]["max_steps"] = args.max_steps
    cfg["training"]["warmup_steps"] = 400
    cfg["training"]["batch_size"] = 1
    cfg["training"]["log_every_n_steps"] = 100
    cfg["training"]["learning_rate"] = 6e-4
    cfg["training"]["checkpoint_every_n_steps"] = args.ckpt_every
    cfg["dataset"]["sequence_length"] = 1024
    if args.curriculum:
        cfg["curriculum"]["stage_1_end"] = args.stage_1_end
        cfg["curriculum"]["stage_2_end"] = args.stage_2_end
        cfg["curriculum"]["stage_3_end"] = args.stage_3_end
    else:
        # No curriculum: stage_1_end = 1 means every loss is on from
        # step 1 onward. The trainer's compute_loss gates by stage
        # name so a stage-N setup with N-1 = "any step > 0" puts
        # everything in JOINT for the whole run.
        cfg["curriculum"]["stage_1_end"] = 1
        cfg["curriculum"]["stage_2_end"] = 2
        cfg["curriculum"]["stage_3_end"] = 3

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
    train_loader = DataLoader(_IterableOver(train_ds), batch_size=None)
    val_loader = DataLoader(_IterableOver(val_ds), batch_size=None)

    model = build_model(cfg)
    n_ucsa_params = sum(p.numel() for p in model.parameters())
    print(f"UCSA-small params: {n_ucsa_params:,}", flush=True)

    trainer = build_trainer(model, cfg)
    print(f"Device: {trainer.device}", flush=True)
    stack = []
    if args.lewm:
        stack.append("JEPA=lewm(multi-step)")
    if args.ema:
        stack.append(f"EMA=0.{int(args.ema_momentum*1000):03d}")
    if args.recon:
        stack.append(f"recon(w={args.reconstruction_weight})")
    if args.tc_jepa:
        stack.append(f"tc-jepa(s={args.text_conditioner_scale})")
    if args.observation_mix < 1.0:
        stack.append(
            f"origination(alpha0={args.observation_mix},"
            f"decay={args.observation_mix_decay},"
            f"k={args.origination_top_k},"
            f"update={args.intent_update_scale},"
            f"balance={'on' if args.origination_balance else 'off'},"
            f"stream={'on' if args.stream_intent_bank else 'off'})"
        )
    else:
        stack.append("origination=off(alpha=1)")
    if args.curriculum:
        stack.append("curriculum=4stage")
    else:
        stack.append("curriculum=off(all-on)")
    print(f"Stack: {' + '.join(stack)}", flush=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Starting training: max_steps={args.max_steps}", flush=True)

    # Ablation toggles flow into the combined loss via a zero-weight
    # substitution. The forward path stays unchanged.
    if not args.recon or not args.origination_balance:
        from ucsa.models.losses import LossWeights
        trainer.loss_fn.weights = LossWeights(
            jepa=trainer.loss_fn.weights.jepa,
            memory=trainer.loss_fn.weights.memory,
            router=trainer.loss_fn.weights.router,
            reconstruction=(
                trainer.loss_fn.weights.reconstruction
                if args.recon
                else 0.0
            ),
            origination=(
                args.origination_weight
                if args.origination_balance
                else 0.0
            ),
        )
    if not args.tc_jepa:
        # Zero the field so the conditioner contributes 0.
        trainer.model.text_conditioner_scale = 0.0

    history: list[dict[str, float]] = []
    losses: list[float] = []
    best_val_ppl = float("inf")
    best_step = -1
    start = time.time()

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
                f"rec={snap.get('reconstruction_loss', 0):.4f} "
                f"steps={int(snap.get('jepa_steps', 0))}"
            )
            print(
                f"  step={step:5d} loss={loss:.4f} avg={avg:.4f} "
                f"lr={lr:.2e} stage={stage} {extras} elapsed={el:.0f}s",
                flush=True,
            )
            history.append({
                "step": step,
                "loss": loss,
                "avg": avg,
                "lr": lr,
                "stage": stage,
                "jepa_prediction": snap.get("jepa_prediction", 0),
                "reconstruction_loss": snap.get("reconstruction_loss", 0),
                "jepa_steps": int(snap.get("jepa_steps", 0)),
            })

        # ``0`` means never, matching the documented meaning of
        # ``--ckpt-every 0`` ("one final ckpt only") that
        # ``scripts/run_ablations.py`` already passes. Taking a modulo by it
        # raised ZeroDivisionError on the first step instead.
        if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
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

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            path = os.path.join(args.ckpt_dir, f"ucsa-step{step}.safetensors")
            trainer.save_checkpoint(path)
            print(f"  saved {path}", flush=True)

    trainer.save_checkpoint(os.path.join(args.ckpt_dir, "ucsa-final.safetensors"))
    elapsed = time.time() - start
    final_ucsa = _eval(trainer, val_loader, args.eval_batches)
    rows: list[dict] = []
    if not args.skip_baselines:
        device = trainer.device
        rows.append(
            {"name": "UCSA-small (this run)", "params": n_ucsa_params,
             "val_ppl": final_ucsa["perplexity"]}
        )
        bench_ds = TextDataset(
            ucsa_tokenizer,
            DatasetConfig(
                sequence_length=cfg["dataset"]["sequence_length"],
                primary_dataset=cfg["dataset"]["primary_dataset"],
                primary_split=cfg["dataset"]["primary_split"],
                streaming=True, pack_sequences=True,
            ),
        )
        bench_ds.dataset = bench_ds.dataset.skip(args.val_skip)
        bench_loader = DataLoader(_IterableOver(bench_ds), batch_size=None)
        bench_iter = iter(bench_loader)
        for baseline in args.baselines:
            try:
                metrics = _eval_hf_baseline(
                    baseline, bench_iter,
                    args.baseline_eval_batches, device,
                )
                rows.append({
                    "name": baseline, "params": metrics["params"],
                    "val_ppl": metrics["perplexity"],
                })
            except Exception as exc:
                print(f"    {baseline} failed: {exc}", flush=True)
                rows.append({"name": baseline, "params": 0, "val_ppl": None})

    report = {
        "model": "ucsa-small",
        "seed": args.seed,
        "max_steps": args.max_steps,
        "stack": stack,
        "history": history,
        "final_val_loss": final_ucsa["loss"],
        "final_val_ppl": final_ucsa["perplexity"],
        "best_val_ppl": best_val_ppl,
        "best_val_ppl_step": best_step,
        "n_ucsa_params": n_ucsa_params,
        "elapsed_seconds": elapsed,
        "baselines": rows,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out_json}", flush=True)
    print(
        f"Final: loss={final_ucsa['loss']:.4f} "
        f"val_ppl={final_ucsa['perplexity']:.1f} "
        f"(best={best_val_ppl:.1f}@{best_step})",
        flush=True,
    )


if __name__ == "__main__":
    main()
    # Exit explicitly. `main` runs to completion -- the checkpoint is
    # saved and the JSON is written -- but the interpreter then deadlocks
    # in finalisation on this torch/MPS build: no non-daemon threads and no
    # child processes remain, yet the process never exits. That makes the
    # script unusable for automation, because `scripts/run_ablations.py`
    # drives it with `subprocess.run` and blocks forever after the first
    # arm. Flush and hard-exit past finalisation.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
