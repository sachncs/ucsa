"""Matched-compute baseline trainer.

Trains a vanilla Transformer LM (no PCS, no MoE, no JEPA) on the
same fineweb-edu stream for the same number of steps / tokens as
UCSA-small. Goal: a *fair* baseline. We don't claim our UCSA beats
a state-of-the-art pretrained LM — we claim it beats a matched
baseline on the same compute.

Why a from-scratch vanilla Transformer instead of GPT-2?
- HuggingFace GPT-2 has been pretrained on a different and much larger
  corpus. Zero-shot transfer to fineweb-edu confounds pretraining
  corpus with adapter effects. A from-scratch vanilla trained on the
  exact same dataset/step budget isolates the architectural
  contribution of UCSA.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ucsa.models.perception import TokenizerWrapper
from ucsa.training.dataset import DatasetConfig, TextDataset
from ucsa.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Vanilla modern Transformer LM (no PCS, no MoE, no JEPA)
# ---------------------------------------------------------------------------


@dataclass
class VanillaConfig:
    vocab_size: int
    hidden: int = 384
    num_layers: int = 6
    num_q_heads: int = 8
    num_kv_heads: int = 4
    ffn_dim: int = 1024
    max_seq: int = 1024


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).to(x.dtype) * self.weight


def _rope_freqs(dim: int, max_seq: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(max_seq).float()
    return torch.polar(torch.ones_like(inv_freq), torch.outer(pos, inv_freq))


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    f = freqs[: x.shape[-2]].to(x.device)
    pair = x.float().reshape(*x.shape[:-1], -1, 2)
    complex_ = torch.view_as_complex(pair)
    rotated = torch.view_as_real(complex_ * f).flatten(-2)
    return rotated.to(x.dtype)


class VanillaBlock(nn.Module):
    def __init__(self, cfg: VanillaConfig):
        super().__init__()
        hd = cfg.hidden // cfg.num_q_heads
        self.norm1 = RMSNorm(cfg.hidden)
        self.norm2 = RMSNorm(cfg.hidden)
        self.q = nn.Linear(cfg.hidden, cfg.num_q_heads * hd, bias=False)
        self.k = nn.Linear(cfg.hidden, cfg.num_kv_heads * hd, bias=False)
        self.v = nn.Linear(cfg.hidden, cfg.num_kv_heads * hd, bias=False)
        self.o = nn.Linear(cfg.num_q_heads * hd, cfg.hidden, bias=False)
        self.repeat = cfg.num_q_heads // cfg.num_kv_heads
        self.gate = nn.Linear(cfg.hidden, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.hidden, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.hidden, bias=False)
        self.register_buffer(
            "freqs", _rope_freqs(hd, cfg.max_seq), persistent=False
        )
        self.hd = hd
        self.num_q = cfg.num_q_heads
        self.num_kv = cfg.num_kv_heads

    def forward(self, x):
        b, s, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).view(b, s, self.num_q, self.hd)
        k = self.k(h).view(b, s, self.num_kv, self.hd)
        v = self.v(h).view(b, s, self.num_kv, self.hd)
        q = _apply_rope(q, self.freqs)
        k = _apply_rope(k, self.freqs)
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=2)
            v = v.repeat_interleave(self.repeat, dim=2)
        out = (
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True,
            )
            .transpose(1, 2)
            .contiguous()
            .view(b, s, -1)
        )
        x = x + self.o(out)
        n = self.norm2(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


class VanillaTransformerLM(nn.Module):
    """vanilla GQA + RoPE + SwiGLU decoder for fair-compute baseline."""

    def __init__(self, cfg: VanillaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList(
            VanillaBlock(cfg) for _ in range(cfg.num_layers)
        )
        self.final = RMSNorm(cfg.hidden)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        x = self.final(x)
        # Tied output projection (standard small-LM trick).
        return x @ self.embed.weight.T


def _build_vanilla(target_params: int = 63_000_000) -> VanillaConfig:
    """Match UCSA-small's parameter count using a small grid search."""
    best = (256, 4, 4, 512)
    best_diff = 10**18
    for hidden in (256, 320, 384, 512, 640):
        for nl in (4, 6, 8):
            for ffm in (2, 3, 4):
                heads = max(4, hidden // 64)
                kv = max(2, heads // 2)
                hd = hidden // heads
                emb = 50257 * hidden
                per_block = (
                    2 * hidden * heads * hd
                    + hidden * kv * hd * 2
                    + heads * hd * hidden
                    + 3 * hidden * hidden * ffm
                )
                total = emb + nl * per_block
                if abs(total - target_params) < best_diff:
                    best_diff = abs(total - target_params)
                    best = (hidden, nl, heads, hidden * ffm)
    hidden, nl, qheads, ffn = best
    return VanillaConfig(
        vocab_size=50257,
        hidden=hidden,
        num_layers=nl,
        num_q_heads=qheads,
        num_kv_heads=max(2, qheads // 2),
        ffn_dim=ffn,
    )


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


class _IterableOver(torch.utils.data.IterableDataset):
    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        return iter(self.ds)


def _infinite(loader):
    while True:
        yield from loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=12000)
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--val-skip", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--target-params", type=int, default=63_000_000)
    p.add_argument(
        "--out-json", default=None, help="Write eval history to this JSON file"
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    with open("ucsa/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    tokenizer = TokenizerWrapper(
        tokenizer_name=cfg["tokenizer"]["name"],
        max_seq_len=args.max_seq_len,
    )
    ds_cfg = DatasetConfig(
        sequence_length=args.max_seq_len,
        primary_dataset=cfg["dataset"]["primary_dataset"],
        primary_split=cfg["dataset"]["primary_split"],
        streaming=True,
        pack_sequences=True,
    )
    train_ds = TextDataset(tokenizer, ds_cfg)
    val_ds = TextDataset(tokenizer, ds_cfg)
    val_ds.dataset = val_ds.dataset.skip(args.val_skip)
    train_loader = DataLoader(_IterableOver(train_ds), batch_size=None)

    vanilla_cfg = _build_vanilla(args.target_params)
    print(
        f"Vanilla config: hidden={vanilla_cfg.hidden} nl={vanilla_cfg.num_layers} "
        f"q_heads={vanilla_cfg.num_q_heads} ffn={vanilla_cfg.ffn_dim} "
        f"max_seq={vanilla_cfg.max_seq}",
        flush=True,
    )
    model = VanillaTransformerLM(vanilla_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}  (target {args.target_params:,})", flush=True)

    device = (
        torch.device("mps", 0)
        if torch.backends.mps.is_available()
        else (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    )
    model = model.to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min(
            (step + 1) / max(1, args.warmup_steps),
            (
                0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * (step - args.warmup_steps)
                        / max(1, args.max_steps - args.warmup_steps)
                    )
                )
                if step >= args.warmup_steps
                else (step + 1) / max(1, args.warmup_steps)
            ),
        ),
    )
    model.train()

    print(f"Device: {device}", flush=True)
    print(f"Max steps: {args.max_steps}", flush=True)

    losses: list[float] = []
    best_val = float("inf")
    history = []

    start = time.time()
    it = _infinite(train_loader)
    for step in range(args.max_steps):
        inputs, targets = next(it)
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        seq = min(logits.shape[1], targets.shape[1])
        logits = logits[:, :seq, :]
        targets = targets[:, :seq]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        losses.append(float(loss.item()))

        if step % args.log_every == 0:
            el = time.time() - start
            window = min(args.log_every, len(losses))
            avg = sum(losses[-window:]) / window
            print(
                f"  step={step:5d} loss={float(loss.item()):.4f} "
                f"avg={avg:.4f} elapsed={el:.0f}s",
                flush=True,
            )

        if step > 0 and step % args.eval_every == 0:
            vppl = _eval_ppl(
                model, val_ds, args.eval_batches, args.max_seq_len, device
            )
            if vppl < best_val:
                best_val = vppl
            history.append({"step": step, "val_ppl": vppl})
            print(
                f"  eval@{step}: val_ppl={vppl:.1f} " f"(best={best_val:.1f})",
                flush=True,
            )

    final_ppl = _eval_ppl(
        model, val_ds, args.eval_batches, args.max_seq_len, device
    )
    elapsed = time.time() - start
    print(f"\nDone: {args.max_steps} steps in {elapsed:.0f}s", flush=True)
    print(f"Final val_ppl (vanilla baseline) = {final_ppl:.1f}", flush=True)
    history.append({"step": args.max_steps, "val_ppl": final_ppl})

    out = {
        "model": "vanilla-transformer-lm",
        "params": n_params,
        "config": vars(vanilla_cfg),
        "seed": args.seed,
        "history": history,
        "final_val_ppl": final_ppl,
        "best_val_ppl": best_val,
        "elapsed_seconds": elapsed,
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out_json}", flush=True)
    else:
        print(json.dumps(out, indent=2))


@torch.no_grad()
def _eval_ppl(model, val_ds, n_batches, seq_len, device):
    model.eval()
    loader = DataLoader(_IterableOver(val_ds), batch_size=None)
    it = iter(loader)
    losses = []
    counts = []
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        seq = min(logits.shape[1], y.shape[1])
        losses.append(
            F.cross_entropy(
                logits[:, :seq, :].reshape(-1, logits.shape[-1]),
                y[:, :seq].reshape(-1),
                reduction="sum",
            ).item()
        )
        counts.append(seq)
    model.train()
    if not losses:
        return float("inf")
    return math.exp(sum(losses) / sum(counts))


if __name__ == "__main__":
    main()
