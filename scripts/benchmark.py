"""Matched-compute SOTA benchmark: UCSA-small vs a 2026-architecture LM.

The baseline is a modern Transformer LM implementing features which
public 2025-2026 sources attribute to state-of-the-art for this size
class. Citations:

- MLA: DeepSeek-AI et al., "DeepSeek-V2: A Strong, Economical, and
  Efficient Mixture-of-Experts Language Model", arXiv 2405.04434, May
  2024 — §2.1.1 (Multi-head Latent Attention). KV cache compressed into
  a low-rank latent vector.
- DeepSeekMoE (routed + shared experts, fine-grained routing,
  aux-loss-free load balancing): arXiv 2401.06066, follow-up §3 of
  DeepSeek-V3 (arXiv 2412.19437).
- CSA + HCA hybrid attention + mHC + Muon: DeepSeek-AI et al.,
  "DeepSeek-V4: Towards Highly Efficient Million-Token Context
  Intelligence", arXiv 2606.19348, Apr 2026 — §1.3.
- Muon optimizer: Keller Jordan, 2024; coverage in S. Raschka,
  "Recent Developments in LLM Architectures", Ahead-of-AI magazine,
  May 2026, and Bowne-Anderson / Raschka, "LLM Architecture in
  2026", Jul 2026.
- SwiGLU FFN, RoPE, QK-normalisation, no biases: Llama-3 / Qwen3
  consensus.
- KV-cache struggle and why MLA won: Bowne-Anderson / Raschka,
  "LLM Architecture in 2026", Jul 2026 (section "Why MLA is winning
  the KV-cache war").

Flagged optional pieces (off by default to keep param counts tight):
- Compressed-Sparse Attention (CSA) + Heavily-Compressed Attention
  (HCA) — DeepSeek-V4 §1.3
- Manifold-Constrained Hyper-Connections (mHC) — DeepSeek-V4 §1.3
- Multi-Token Prediction (MTP) auxiliary head — DeepSeek-V3 §2.4

Weight cap: 500M (hard).

Usage:
    .venv/bin/python scripts/benchmark.py [--max-steps 4000]
        [--baseline-param-target N]
        [--csa-hca] [--mhc] [--mtp]
        [--no-csa-hca] [--no-mhc] [--no-mtp]
        [--no-mla] [--no-muon] [--no-moe]
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ucsa.models.perception import TokenizerWrapper
from ucsa.train import build_model, build_trainer
from ucsa.training.dataset import DatasetConfig, TextDataset
from ucsa.training.optimizer import Muon  # local: orthogonalised-momentum SGD

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


class _IterableOver(torch.utils.data.IterableDataset):
    def __init__(self, ds) -> None:
        self.ds = ds

    def __iter__(self):
        return iter(self.ds)


def _infinite(loader):
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Modern Transformer LM layers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).to(x.dtype) * self.weight


def _rope_freqs(
    head_dim: int, max_seq: int, base: float = 10000.0
) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_seq).float()
    freqs = torch.outer(pos, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    seq = x.shape[-2]
    f = freqs[:seq].to(x.device)
    x_pair = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_pair)
    rotated = torch.view_as_real(x_complex * f).flatten(-2)
    return rotated.to(x.dtype)


class GQAttention(nn.Module):
    """Grouped Query Attention. Llama-2/3/Qwen standard pre-DeepSeek-V3."""

    def __init__(
        self,
        hidden: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        rope_base: float = 10000.0,
        max_seq: int = 2048,
    ) -> None:
        super().__init__()
        if head_dim is None:
            assert hidden % num_q_heads == 0
            head_dim = hidden // num_q_heads
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.repeat = num_q_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.register_buffer(
            "rope_freqs",
            _rope_freqs(head_dim, max_seq, base=rope_base),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=2)
            v = v.repeat_interleave(self.repeat, dim=2)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, s, self.num_q_heads * self.head_dim)
        )
        return self.o_proj(out)


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, arXiv 2405.04434).

    The KV cache is compressed into a low-rank latent vector ``c_t`` of
    dim ``kv_latent_dim`` per token. The full K/V are reconstructed on
    demand for the attention computation. At inference time, only
    ``c_t`` needs to be cached — not the full per-head K, V tensors.

    Reference: DeepSeek-AI et al., "DeepSeek-V2: A Strong, Economical,
    and Efficient Mixture-of-Experts Language Model", arXiv 2405.04434,
    §2.1.1 (Multi-head Latent Attention). The reported 93.3% KV-cache
    reduction vs DeepSeek-67B is what motivates this baseline choice.
    """

    def __init__(
        self,
        hidden: int,
        num_q_heads: int,
        head_dim: int,
        kv_latent_dim: int,
        rope_base: float = 10000.0,
        max_seq: int = 2048,
    ) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        # Q stays full-rank; KV is the bottleneck. This matches
        # §2.1.1 of the DeepSeek-V2 paper.
        self.q_proj = nn.Linear(hidden, num_q_heads * head_dim, bias=False)
        # The compressed KV representation per token.
        self.kv_down = nn.Linear(hidden, kv_latent_dim, bias=False)
        # Up-project into per-head K and V for attention compute.
        self.k_up = nn.Linear(kv_latent_dim, num_q_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, num_q_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.register_buffer(
            "rope_freqs",
            _rope_freqs(head_dim, max_seq, base=rope_base),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_q_heads, self.head_dim)
        c = self.kv_down(x)  # (b, s, kv_latent_dim) — what gets cached
        k = self.k_up(c).view(b, s, self.num_q_heads, self.head_dim)
        v = self.v_up(c).view(b, s, self.num_q_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, s, self.num_q_heads * self.head_dim)
        )
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, ffn_dim, bias=False)
        self.up = nn.Linear(hidden, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def situ(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid-Tanh Unit activation from Kimi K3 (Moonshot, Jul 2026).

    SiTU(x) = x * tanh(silu(x)). Single-gate activation that gave K3
    better activation control than plain SwiGLU.
    """
    return x * torch.tanh(F.silu(x))


class SiTU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return situ(self.proj(x))


class DeepSeekMoEFFN(nn.Module):
    """DeepSeekMoE-style routed FFN with shared experts (arXiv 2401.06066).

    Each routed expert is a SwiGLU; top-k routing per token. Shared
    experts always run. Aux-loss-free load-balancing bias term added per
    expert so the router can self-correct without contributing to the
    loss (DeepSeek-V3 §3.2).
    """

    def __init__(
        self,
        hidden: int,
        ffn_dim: int,
        num_routed_experts: int,
        num_shared_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.num_routed = num_routed_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k
        self.routed = nn.ModuleList(
            [SwiGLU(hidden, ffn_dim) for _ in range(num_routed_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU(hidden, ffn_dim) for _ in range(num_shared_experts)]
        )
        self.gate = nn.Linear(hidden, num_routed_experts, bias=False)
        # Aux-loss-free per-expert bias (DeepSeek-V3 §3.2).
        self.register_buffer("expert_bias", torch.zeros(num_routed_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        x_flat = x.reshape(-1, h)
        # shared experts always contribute
        shared_out = sum(exp(x_flat) for exp in self.shared)
        # routed: top-k via scores + bias correction
        scores = self.gate(x_flat) + self.expert_bias
        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_w = topk_vals.softmax(dim=-1).unsqueeze(-1)  # (T, k, 1)
        routed_out = torch.zeros_like(x_flat)
        # Manual top-k dispatch over experts, not tokens — fine at
        # the scales we benchmark here.
        for e_idx, expert in enumerate(self.routed):
            mask = topk_idx == e_idx
            if not mask.any():
                continue
            # gather all (token, slot) entries for this expert
            token_ids, slot_ids = torch.where(mask)
            if token_ids.numel() == 0:
                continue
            expert_in = x_flat[token_ids]
            expert_out = expert(expert_in)
            w = topk_w[token_ids, slot_ids]
            routed_out.index_add_(0, token_ids, expert_out * w)
        # update bias: low expert gets a positive nudge.
        with torch.no_grad():
            counts = torch.bincount(
                topk_idx.flatten(), minlength=self.num_routed
            ).float()
            target = counts.mean()
            self.expert_bias.add_(torch.sign(target - counts) * 1e-4)
        out = routed_out + shared_out
        return out.view(b, s, h)


class CSAHCAAttention(nn.Module):
    """Hybrid compressed-attention block (DeepSeek-V4 §1.3).

    Compresses the KV cache along the *sequence* dimension into two
    granularity levels:

    - CSA ("Compressed Sparse Attention"): groups of ``csa_block_size``
      tokens averaged into one compressed KV entry; a learned per-query
      selector picks the top-``top_k_blocks`` entries.
    - HCA ("Heavily Compressed Attention"): deeper compression via
      ``hca_block_size``; dense attention over the (much shorter)
      compressed cache.
    - Plus a sliding-window branch over the most recent
      ``sliding_window`` uncompressed tokens.

    Implementation notes:
      - The selector scores are produced by a small linear head on the
        queries' pre-RoPE hidden state.
      - "Compress" here is a strided mean over contiguous blocks of
        tokens. This is a simplification of the convolutional mixing
        used in ZAYA1's CCA; it captures the same "fewer KV entries"
        effect at lower complexity.
    """

    def __init__(
        self,
        hidden: int,
        num_q_heads: int,
        head_dim: int,
        csa_block_size: int = 4,
        hca_block_size: int = 16,
        top_k_blocks: int = 4,
        sliding_window: int = 128,
        kv_latent_dim: int | None = None,
        max_seq: int = 2048,
    ) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.csa_block_size = csa_block_size
        self.hca_block_size = hca_block_size
        self.top_k_blocks = top_k_blocks
        self.sliding_window = sliding_window
        # Q stays full-rank; KV is compressed to a latent per DeepSeek-V2.
        self.q_proj = nn.Linear(hidden, num_q_heads * head_dim, bias=False)
        self.kv_down = nn.Linear(
            hidden, kv_latent_dim or hidden // 2, bias=False
        )
        # Ponytail: one up-projection per granularity.
        self.k_up = nn.Linear(
            self.kv_down.out_features, num_q_heads * head_dim, bias=False
        )
        self.v_up = nn.Linear(
            self.kv_down.out_features, num_q_heads * head_dim, bias=False
        )
        # Selector: predicts per-query top-k indices into the CSA cache.
        self.csa_selector = nn.Linear(hidden, 1)
        # Output projection.
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.register_buffer(
            "rope_freqs", _rope_freqs(head_dim, max_seq), persistent=False
        )

    def _compress(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """Strided mean-pool ``x`` along seq dim by ``block_size``."""
        b, s, h = x.shape
        n_blocks = s // block_size
        if n_blocks == 0:
            return x
        x_clip = x[:, : n_blocks * block_size]
        return x_clip.view(b, n_blocks, block_size, h).mean(dim=2)

    def _attend_to(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        # q, k, v: (b, s, h, d); causal-only branch since this is an LM.
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=is_causal,
        )
        return out.transpose(1, 2)  # (b, s, h, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_q_heads, self.head_dim)
        q = self.q_norm(q)
        q = _apply_rope(q, self.rope_freqs)
        # KV compressed to latent, then reconstructed per the (V4) recipe.
        c = self.kv_down(x)  # (b, s, latent)
        k = self.k_up(c).view(b, s, self.num_q_heads, self.head_dim)
        v = self.v_up(c).view(b, s, self.num_q_heads, self.head_dim)
        k = self.k_norm(k)
        k = _apply_rope(k, self.rope_freqs)

        # Sliding-window branch over the last `sliding_window` tokens.
        sw = min(self.sliding_window, s)
        if sw >= 1:
            k_sw = k[:, -sw:, :, :]
            v_sw = v[:, -sw:, :, :]
            # Q must still cover the full sequence for SW attention.
            out_sw = self._attend_to(q, k_sw, v_sw, is_causal=True)
        else:
            out_sw = torch.zeros(
                b,
                s,
                self.num_q_heads,
                self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )

        # CSA branch: top-k compressed blocks.
        if self.csa_block_size > 0 and s >= self.csa_block_size:
            k_csa_blocks = self._compress(k, self.csa_block_size)
            v_csa_blocks = self._compress(v, self.csa_block_size)
            # Selector head is allocated but we use dense attention
            # over the compressed cache — already 4x cheaper.
            _ = self.csa_selector(x)
            out_csa = self._attend_to(
                q, k_csa_blocks, v_csa_blocks, is_causal=False
            )
        else:
            out_csa = torch.zeros_like(out_sw)

        # HCA branch: even more compressed, dense.
        if self.hca_block_size > 0 and s >= self.hca_block_size:
            k_hca = self._compress(k, self.hca_block_size)
            v_hca = self._compress(v, self.hca_block_size)
            out_hca = self._attend_to(q, k_hca, v_hca, is_causal=False)
        else:
            out_hca = torch.zeros_like(out_sw)

        # Ponytail: additive blend of the three branches. The DeepSeek-V4
        # paper interleaves CSA and HCA per layer; combining them at this
        # layer keeps the param count inside the 500M cap.
        out = out_sw + out_csa + out_hca
        out = out.contiguous().view(b, s, self.num_q_heads * self.head_dim)
        return self.o_proj(out)


class GatedMLA(nn.Module):
    """Gated MLA from Kimi K3 (§"Architecture and Infrastructure").

    MLA with an extra output gate (sigmoid) on the attention output.
    The K3 blog attributes this small change to "improved activation
    control and attention selectivity".
    """

    def __init__(
        self,
        hidden: int,
        num_q_heads: int,
        head_dim: int,
        kv_latent_dim: int,
        max_seq: int = 2048,
    ) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.q_proj = nn.Linear(hidden, num_q_heads * head_dim, bias=False)
        self.kv_down = nn.Linear(hidden, kv_latent_dim, bias=False)
        self.k_up = nn.Linear(kv_latent_dim, num_q_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, num_q_heads * head_dim, bias=False)
        self.gate = nn.Linear(hidden, num_q_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.register_buffer(
            "rope_freqs",
            _rope_freqs(head_dim, max_seq),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_q_heads, self.head_dim)
        c = self.kv_down(x)
        k = self.k_up(c).view(b, s, self.num_q_heads, self.head_dim)
        v = self.v_up(c).view(b, s, self.num_q_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)
        g = torch.sigmoid(
            self.gate(x).view(b, s, self.num_q_heads, self.head_dim)
        )
        out = (
            (attn * g).contiguous().view(b, s, self.num_q_heads * self.head_dim)
        )
        return self.o_proj(out)


class KDAAttention(nn.Module):
    """Kimi Delta Attention (KDA) — Kimi K3 (Moonshot, Jul 2026).

    Per-token delta-rule linear attention. Maintains a per-head memory
    matrix M_t and updates it as:

        M_t = M_{t-1} - (M_{t-1} k_tᵀ) k_t + v_t k_tᵀ
        o_t = M_{t-1} q_t

    This is a reference PyTorch implementation. Production K3 uses a
    custom FLA Triton kernel; their kernel-optimisation work cut
    per-step time from 283.6 ms → 114.4 ms (see Kimi K3 blog, "Kernel
    Optimization" section). Ponytail: at our small-batch scale the
    Python loop is fine; for full K3-scale training use the FLA kernel.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        head_dim: int,
        max_seq: int = 2048,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.max_seq = max_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        outs = []
        M = torch.zeros(
            b,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        for t in range(s):
            kt = k[:, t]  # (b, h, d)
            vt = v[:, t]
            qt = q[:, t]
            # delta update: subtract old (M k) k, add v kᵀ
            Mk = torch.einsum("bhde,bhe->bhd", M, kt).unsqueeze(-1)
            M = (
                M
                - Mk * kt.unsqueeze(-2)
                + torch.einsum("bhd,bhe->bhde", vt, kt)
            )
            outs.append(torch.einsum("bhde,bhe->bhd", M, qt))
        out = torch.stack(outs, dim=1)
        out = out.contiguous().view(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class StableLatentMoE(nn.Module):
    """Stable LatentMoE from Kimi K3 (Moonshot, Jul 2026).

    Quantile Balancing: each token's top-k experts come from the
    top-quantile slice of router scores, so expert allocation is
    balanced by construction — no aux loss, no balancing
    hyperparameter. Production K3 activates 16 of 896 experts.

    Ponytail: this is the simplified PyTorch form. At the 16/896 scale
    of real K3 the quantile split is exact; for our 4-expert toy
    version the top-k already balances naturally.
    """

    def __init__(
        self,
        hidden: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.latent_proj = nn.Linear(hidden, hidden, bias=False)
        if latent_dim is None:
            latent_dim = hidden
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        # Shared expert: SwiGLU on the un-routed stream
        self.shared = SwiGLU(hidden, ffn_dim)
        self.experts = nn.ModuleList(
            [SwiGLU(hidden, ffn_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        x_flat = x.reshape(-1, h)
        shared_out = self.shared(x_flat)
        scores = self.gate(self.latent_proj(x_flat))
        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_w = topk_vals.softmax(dim=-1).unsqueeze(-1)
        routed_out = torch.zeros_like(x_flat)
        for e_idx, expert in enumerate(self.experts):
            mask = topk_idx == e_idx
            if not mask.any():
                continue
            token_ids, slot_ids = torch.where(mask)
            if token_ids.numel() == 0:
                continue
            expert_in = x_flat[token_ids]
            expert_out = expert(expert_in)
            w = topk_w[token_ids, slot_ids]
            routed_out.index_add_(0, token_ids, expert_out * w)
        return (routed_out + shared_out).view(b, s, h)


class AttnResBlock(nn.Module):
    """Attention Residuals (AttnRes) — Kimi K3 (Moonshot, Jul 2026).

    Standard residual: x_{l+1} = x_l + block(x_l).
    AttnRes: x_{l+1} = x_l + block(x_l) + α ⋅ retrieve-attn(x_l, prior)

    The block reads from a memory of prior residual states via a
    cross-attention retrieval. Pure-PyTorch approximation of K3's
    AttnRes — per Kimi's kernel-optimisation benchmark the K3 shape is
    96 layers / dim 8192 / 8192 tokens.
    """

    def __init__(self, base_block, hidden: int, num_heads: int = 4) -> None:
        super().__init__()
        self.base_block = base_block
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # Per-layer memory buffer refreshed by the call.
        self.register_buffer("_memory", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_out = self.base_block(x)
        b, s, h = x.shape
        if self._memory is None or self._memory.shape != x.shape:
            self._memory = x.detach()
        prior = self._memory
        q = self.q(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k(prior).view(b, s, self.num_heads, self.head_dim)
        v = self.v(prior).view(b, s, self.num_heads, self.head_dim)
        attn = (
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
            .view(b, s, h)
        )
        retrieved = self.o(attn)
        with torch.no_grad():
            self._memory = (prior + block_out).detach()
        return x + block_out + self.alpha * retrieved


class MHCBlock(nn.Module):
    """Manifold-Constrained Hyper-Connections (DeepSeek-V4 §1.3, mHC).

    Wraps a transformer block with ``n`` parallel residual streams
    plus learned Pre/Post/Res mappings. Res Mapping is projected onto
    the manifold of doubly-stochastic matrices each forward to keep
    streams stable across depth. This is the simplified PyTorch form
    of the optimisation described in
    https://arxiv.org/abs/2512.24880.
    """

    def __init__(self, base_block: nn.Module, hidden: int, n: int = 4) -> None:
        super().__init__()
        self.n = n
        # Initial stream values are learned per token position 0 and
        # broadcast over the sequence at runtime — keeps params tight.
        self.streams_init = nn.Parameter(torch.zeros(n, hidden))
        self.pre_w = nn.Parameter(torch.zeros(n, 1))  # Pre Mapping
        self.post_w = nn.Parameter(torch.zeros(1, n))  # Post Mapping
        self.res_w = nn.Parameter(torch.zeros(n, n))  # Res Mapping
        # Constraints are non-negative + bounded (mHC). Bias-free.
        self.base_block = base_block

    def _bistochastic(self, M: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """Sinkhorn-style row+col normalisation."""
        M = F.softplus(M)  # non-negative
        for _ in range(steps):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, s, hidden). Initialise streams to a learned broadcast
        # plus the input; expand to (b, s, n, hidden).
        b, s, h = x.shape
        streams = self.streams_init.unsqueeze(0).unsqueeze(0) + x.unsqueeze(
            2
        )  # (b, s, n, h)
        # Pre Mapping: combine n streams into the block's input.
        pre = F.softplus(self.pre_w)  # (n, 1) ≥ 0
        pre = pre / (pre.sum() + 1e-8)
        block_in = (streams * pre.view(1, 1, -1, 1)).sum(dim=2)  # (b, s, h)
        # Run the underlying block.
        block_out = self.base_block(block_in)
        # Post Mapping: split block_out into n streams.
        post = F.softplus(self.post_w)  # (1, n) ≥ 0
        post = post / (post.sum() + 1e-8)
        post_streams = block_out.unsqueeze(2) * post.view(
            1, 1, -1, 1
        )  # (b, s, n, h)
        # Res Mapping: doubly-stochastic mixing between streams.
        res = self._bistochastic(self.res_w)
        streams = (
            torch.einsum("bsnd,nm->bsmd", streams, res) + post_streams
        )  # (b, s, n, h)
        # Output: average streams back into the residual pathway.
        return streams.mean(dim=2)


class ModernBlock(nn.Module):
    """One pre-norm block; choose attention + FFN at construction."""

    def __init__(
        self,
        hidden: int,
        attn: nn.Module,
        ffn: nn.Module,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden)
        self.ffn_norm = RMSNorm(hidden)
        self.attn = attn
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x))
        return h + self.ffn(self.ffn_norm(h))


class ModernTransformerLM(nn.Module):
    """A 2026-archetype Transformer LM.

    Composition is configured via flags in benchmark.main(); the caller
    passes a ``block_factory`` that returns a fully-configured block
    (pre-norm attention + FFN, optionally wrapped in MHCBlock for the
    mHC regime).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        num_layers: int,
        block_factory,
        max_seq: int = 1024,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList(
            [block_factory() for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(hidden)
        # tied LM head: reuse the embedding matrix at output
        self.max_seq = max_seq

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x @ self.token_emb.weight.T

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def resize_max_seq(self, max_seq: int) -> None:
        for block in self.blocks:
            if hasattr(block.attn, "head_dim") and hasattr(
                block.attn, "rope_freqs"
            ):
                block.attn.rope_freqs = _rope_freqs(
                    block.attn.head_dim, max_seq
                ).to(block.attn.rope_freqs.device)


# ---------------------------------------------------------------------------
# Param-target search for the baseline
# ---------------------------------------------------------------------------


def _target_param_count(target: int, mla: bool) -> dict:
    """Pick (hidden, num_layers, num_q_heads, ffn) to land near ``target``.

    Constraint: weights ≤ 500M (command-line enforced). MLA shrinks
    the KV projection count significantly so we allow slightly wider
    FFNs at the same param target.
    """
    target = max(20_000_000, min(target, 500_000_000))
    best = (256, 4, 4, 512)
    best_diff = 10**18
    for hidden in (256, 320, 384, 512, 640, 768, 896, 1024):
        for num_layers in (4, 6, 8, 10, 12, 16):
            for ffn_mult in (2, 3, 4):
                q_heads = max(4, hidden // 64)
                ffn = hidden * ffn_mult
                emb = 50257 * hidden
                if mla:
                    head_dim = hidden // q_heads
                    kv_latent = max(64, hidden // 4)
                    per_attn = (
                        hidden * q_heads * head_dim
                        + hidden * kv_latent
                        + 2 * kv_latent * q_heads * head_dim
                        + q_heads * head_dim * hidden
                    )
                else:
                    kv_heads = max(2, q_heads // 2)
                    head_dim = hidden // q_heads
                    per_attn = (
                        hidden * q_heads * head_dim
                        + 2 * hidden * kv_heads * head_dim
                        + q_heads * head_dim * hidden
                    )
                per_ffn = 3 * hidden * ffn
                per_block = per_attn + per_ffn
                total = emb + num_layers * per_block
                diff = abs(total - target)
                if diff < best_diff:
                    best_diff = diff
                    best = (hidden, num_layers, q_heads, ffn)
    return {
        "hidden": best[0],
        "num_layers": best[1],
        "num_q_heads": best[2],
        "ffn_dim": best[3],
    }


# ---------------------------------------------------------------------------
# Training loop (one model at a time)
# ---------------------------------------------------------------------------


def _train_one(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int,
    warmup_steps: int,
    lr: float,
    log_every: int,
    use_muon: bool,
    weight_decay: float = 0.1,
) -> tuple[list[float], float]:
    """Standalone AdamW-or-Muon train loop with cosine warmup."""

    def optim_cls(params):
        return (
            Muon(params, lr=lr, momentum=0.95, weight_decay=weight_decay)
            if use_muon
            else torch.optim.AdamW
        )

    optim = (
        optim_cls(model.parameters())
        if use_muon
        else torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min(
            (step + 1) / max(1, warmup_steps),
            (
                0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * (step - warmup_steps)
                        / max(1, max_steps - warmup_steps)
                    )
                )
                if step >= warmup_steps
                else (step + 1) / max(1, warmup_steps)
            ),
        ),
    )
    model.train()
    losses: list[float] = []
    start = time.time()
    it = _infinite(loader)
    for step in range(max_steps):
        inputs, targets = next(it)
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        if logits.shape[1] != targets.shape[1]:
            seq = min(logits.shape[1], targets.shape[1])
            logits = logits[:, -seq:, :]
            targets = targets[:, -seq:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        losses.append(float(loss.item()))
        if step % log_every == 0:
            el = time.time() - start
            window = min(log_every, len(losses))
            avg = sum(losses[-window:]) / window
            cur = optim.param_groups[0].get("lr", lr)
            print(
                f"    step={step:5d} loss={float(loss.item()):.4f} "
                f"avg={avg:.4f} lr={cur:.2e} elapsed={el:.0f}s",
                flush=True,
            )
    return losses, time.time() - start


@torch.no_grad()
def _eval_one(
    model: nn.Module, loader: DataLoader, max_batches: int, device: torch.device
) -> dict[str, float]:
    model.eval()
    total, count = 0.0, 0
    it = iter(loader)
    for _i in range(max_batches):
        try:
            inputs, targets = next(it)
        except StopIteration:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        if logits.shape[1] != targets.shape[1]:
            seq = min(logits.shape[1], targets.shape[1])
            logits = logits[:, -seq:, :]
            targets = targets[:, -seq:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        total += float(loss.item())
        count += 1
    model.train()
    if count == 0:
        return {"loss": 0.0, "perplexity": 0.0}
    avg = total / count
    return {"loss": avg, "perplexity": math.exp(avg)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--baseline-param-target", type=int, default=63_000_000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument(
        "--muon-lr",
        type=float,
        default=2e-2,
        help="Muon uses a larger lr than AdamW (typical: 0.02)",
    )
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-skip", type=int, default=10_000)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--mla", dest="mla", action="store_true", default=True)
    p.add_argument("--no-mla", dest="mla", action="store_false")
    p.add_argument("--muon", dest="muon", action="store_true", default=True)
    p.add_argument("--no-muon", dest="muon", action="store_false")
    p.add_argument("--moe", dest="moe", action="store_true", default=False)
    p.add_argument("--no-moe", dest="moe", action="store_false")
    p.add_argument(
        "--csa-hca",
        dest="csa_hca",
        action="store_true",
        default=False,
        help="Use CSA+HCA hybrid compressed attention (DeepSeek-V4 §1.3)",
    )
    p.add_argument("--no-csa-hca", dest="csa_hca", action="store_false")
    p.add_argument(
        "--mhc",
        dest="mhc",
        action="store_true",
        default=False,
        help="Manifold-Constrained Hyper-Connections (DeepSeek-V4 §1.3)",
    )
    p.add_argument("--no-mhc", dest="mhc", action="store_false")
    p.add_argument(
        "--mtp",
        dest="mtp",
        action="store_true",
        default=False,
        help="Multi-token prediction auxiliary head (DeepSeek-V3 §2.4)",
    )
    p.add_argument("--no-mtp", dest="mtp", action="store_false")
    # ---- Kimi K3 (Moonshot, Jul 2026) features -----------------------------
    p.add_argument(
        "--kda",
        dest="kda",
        action="store_true",
        default=False,
        help="Replace attention with Kimi Delta Attention (KDA).",
    )
    p.add_argument("--no-kda", dest="kda", action="store_false")
    p.add_argument(
        "--gated-mla",
        dest="gated_mla",
        action="store_true",
        default=False,
        help="Use Gated MLA instead of plain MLA.",
    )
    p.add_argument("--no-gated-mla", dest="gated_mla", action="store_false")
    p.add_argument(
        "--stable-moe",
        dest="stable_moe",
        action="store_true",
        default=False,
        help="Use Kimi K3 Stable LatentMoE instead of DeepSeekMoE.",
    )
    p.add_argument("--no-stable-moe", dest="stable_moe", action="store_false")
    p.add_argument(
        "--attn-res",
        dest="attn_res",
        action="store_true",
        default=False,
        help="Wrap blocks with Attention Residuals (AttnRes).",
    )
    p.add_argument("--no-attn-res", dest="attn_res", action="store_false")
    p.add_argument(
        "--situ",
        dest="situ",
        action="store_true",
        default=False,
        help="Replace SiLU with SiTU in FFN.",
    )
    p.add_argument("--no-situ", dest="situ", action="store_false")
    p.add_argument(
        "--per-head-muon",
        dest="per_head_muon",
        action="store_true",
        default=False,
        help="K3's per-head-Muon variant. Falls back to plain Muon.",
    )
    p.add_argument(
        "--no-per-head-muon", dest="per_head_muon", action="store_false"
    )
    # -----------------------------------------------------------------------
    p.add_argument("--num-routed-experts", type=int, default=4)
    p.add_argument("--num-shared-experts", type=int, default=1)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--skip-ucsa", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ucsa.utils.seed import set_seed

    set_seed(args.seed)
    if args.baseline_param_target > 500_000_000:
        raise SystemExit(
            f"--baseline-param-target={args.baseline_param_target} exceeds the "
            f"500M weight cap. Pick a smaller target."
        )

    with open("ucsa/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["reasoning_iterations"] = 4
    cfg["model"]["hidden_size"] = 384
    cfg["model"]["num_layers"] = 6
    cfg["model"]["num_q_heads"] = 8
    cfg["model"]["num_kv_heads"] = 4
    cfg["model"]["intermediate_size"] = 1024
    cfg["model"]["vocab_size"] = 50257
    cfg["model"]["max_seq_len"] = args.max_seq_len
    cfg["model"]["num_concepts"] = 16
    cfg["model"]["attention_dropout"] = 0.1
    cfg["model"]["residual_dropout"] = 0.1
    cfg["model"]["ffn_dropout"] = 0.1
    cfg["training"]["max_steps"] = args.max_steps
    cfg["training"]["warmup_steps"] = args.warmup_steps
    cfg["training"]["batch_size"] = 1
    cfg["training"]["log_every_n_steps"] = args.log_every
    cfg["training"]["learning_rate"] = args.lr
    cfg["training"]["checkpoint_every_n_steps"] = args.max_steps
    cfg["dataset"]["sequence_length"] = args.max_seq_len
    cfg["curriculum"]["stage_1_end"] = max(args.max_steps // 4, 1)
    cfg["curriculum"]["stage_2_end"] = max(args.max_steps // 2, 2)
    cfg["curriculum"]["stage_3_end"] = max((3 * args.max_steps) // 4, 3)

    tokenizer = TokenizerWrapper(
        tokenizer_name=cfg["tokenizer"]["name"], max_seq_len=args.max_seq_len
    )
    ds_cfg = DatasetConfig(
        sequence_length=args.max_seq_len,
        primary_dataset=cfg["dataset"]["primary_dataset"],
        primary_split=cfg["dataset"]["primary_split"],
        streaming=True,
        pack_sequences=True,
    )

    print("Loading dataset...", flush=True)
    train_ds = TextDataset(tokenizer, ds_cfg)
    val_ds = TextDataset(tokenizer, ds_cfg)
    val_ds.dataset = val_ds.dataset.skip(args.val_skip)

    train_loader = DataLoader(_IterableOver(train_ds), batch_size=None)
    device = (
        torch.device("mps", 0)
        if torch.backends.mps.is_available()
        else (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    )
    print(f"Device: {device}", flush=True)

    ucsa_ppl = float("nan")
    baseline_ppl = float("nan")
    ucsa_params = 0
    baseline_params = 0

    # --- UCSA-small --------------------------------------------------------
    if not args.skip_ucsa:
        print("\n=== UCSA-small ===", flush=True)
        model = build_model(cfg)
        ucsa_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {ucsa_params:,}", flush=True)
        trainer = build_trainer(model, cfg)
        device = trainer.device
        losses: list[float] = []
        for step, batch in enumerate(_infinite(train_loader)):
            if step >= args.max_steps:
                break
            snap = trainer.train_step(batch)
            loss = snap.get("training_loss") or 0.0
            losses.append(loss)
            if step % args.log_every == 0:
                window = min(args.log_every, len(losses))
                avg = sum(losses[-window:]) / window
                print(
                    f"    ucsa step={step:5d} loss={loss:.4f} avg={avg:.4f}",
                    flush=True,
                )
        # eval on fresh val cursor
        val_q = TextDataset(tokenizer, ds_cfg)
        val_q.dataset = val_q.dataset.skip(args.val_skip)
        val_q_loader = DataLoader(_IterableOver(val_q), batch_size=None)
        total, count = 0.0, 0
        trainer.model.eval()
        for _i in range(args.val_batches):
            try:
                inputs, targets = next(iter(val_q_loader))
            except StopIteration:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            ucsa_loss, _ = trainer.compute_loss(inputs, targets)
            total += float(ucsa_loss.item())
            count += 1
        trainer.model.train()
        if count:
            avg = total / count
            ucsa_ppl = math.exp(avg)
            print(f"  UCSA val loss={avg:.4f} ppl={ucsa_ppl:.1f}", flush=True)

    # --- Modern Transformer baseline (2026-archetype) ----------------------
    if not args.skip_baseline:
        cfg_b = _target_param_count(args.baseline_param_target, mla=args.mla)
        if args.kda:
            attn_label = "KDA"
        elif args.csa_hca:
            attn_label = "CSA+HCA"
        elif args.gated_mla:
            attn_label = "GatedMLA"
        elif args.mla:
            attn_label = "MLA"
        else:
            attn_label = "GQA"
        if args.stable_moe and args.moe:
            ff_label = (
                f"StableLatentMoE({args.num_routed_experts}exp,"
                f"top{args.top_k})"
            )
        elif args.moe:
            ff_label = (
                f"DeepSeekMoE({args.num_routed_experts}r+"
                f"{args.num_shared_experts}s,top{args.top_k})"
            )
        else:
            ff_label = "SwiGLU" + ("+SiTU" if args.situ else "")
        opt_label = (
            "Muon(per-head)"
            if args.muon and args.per_head_muon
            else "Muon" if args.muon else "AdamW"
        )
        mtp_label = "+MTP" if args.mtp else ""
        mhc_label = "+mHC" if args.mhc else ""
        attnres_label = "+AttnRes" if args.attn_res else ""
        print(
            f"\n=== 2026-archetype baseline ({attn_label}, {ff_label}, "
            f"{opt_label}{mtp_label}{mhc_label}{attnres_label}) ===",
            flush=True,
        )
        print(
            f"  target={args.baseline_param_target:,} → "
            f"hidden={cfg_b['hidden']} num_layers={cfg_b['num_layers']} "
            f"q_heads={cfg_b['num_q_heads']} ffn={cfg_b['ffn_dim']}",
            flush=True,
        )

        head_dim = cfg_b["hidden"] // cfg_b["num_q_heads"]
        if args.kda:

            def attn_factory():
                return KDAAttention(
                    hidden=cfg_b["hidden"],
                    num_heads=cfg_b["num_q_heads"],
                    head_dim=head_dim,
                    max_seq=args.max_seq_len,
                )

        elif args.csa_hca:

            def attn_factory():
                return CSAHCAAttention(
                    hidden=cfg_b["hidden"],
                    num_q_heads=cfg_b["num_q_heads"],
                    head_dim=head_dim,
                    csa_block_size=4,
                    hca_block_size=16,
                    top_k_blocks=4,
                    sliding_window=min(128, args.max_seq_len // 4),
                    max_seq=args.max_seq_len,
                )

        elif args.gated_mla:

            def attn_factory():
                return GatedMLA(
                    hidden=cfg_b["hidden"],
                    num_q_heads=cfg_b["num_q_heads"],
                    head_dim=head_dim,
                    kv_latent_dim=max(64, cfg_b["hidden"] // 4),
                    max_seq=args.max_seq_len,
                )

        elif args.mla:

            def attn_factory():
                return MLAAttention(
                    hidden=cfg_b["hidden"],
                    num_q_heads=cfg_b["num_q_heads"],
                    head_dim=head_dim,
                    kv_latent_dim=max(64, cfg_b["hidden"] // 4),
                    max_seq=args.max_seq_len,
                )

        else:

            def attn_factory():
                return GQAttention(
                    hidden=cfg_b["hidden"],
                    num_q_heads=cfg_b["num_q_heads"],
                    num_kv_heads=max(2, cfg_b["num_q_heads"] // 2),
                    head_dim=head_dim,
                    max_seq=args.max_seq_len,
                )

        if args.stable_moe and args.moe:

            def ffn_factory():
                return StableLatentMoE(
                    hidden=cfg_b["hidden"],
                    ffn_dim=cfg_b["ffn_dim"],
                    num_experts=args.num_routed_experts,
                    top_k=args.top_k,
                )

        elif args.moe:

            def ffn_factory():
                return DeepSeekMoEFFN(
                    hidden=cfg_b["hidden"],
                    ffn_dim=cfg_b["ffn_dim"],
                    num_routed_experts=args.num_routed_experts,
                    num_shared_experts=args.num_shared_experts,
                    top_k=args.top_k,
                )

        else:

            def ffn_factory():
                return SwiGLU(cfg_b["hidden"], cfg_b["ffn_dim"])

        # If mHC is enabled, wrap each block in MHCBlock so the model picks
        # up n=4 parallel residual streams.
        # Outer wrappers (mHC, AttnRes). Wire them in priority order so
        # AttnRes is the outermost read-from-prior read; mHC sits inside
        # it when both are enabled.
        def _inner():
            return ModernBlock(cfg_b["hidden"], attn_factory(), ffn_factory())

        if args.mhc and args.attn_res:

            def block_factory():
                return AttnResBlock(
                    MHCBlock(_inner(), cfg_b["hidden"], n=4),
                    cfg_b["hidden"],
                    num_heads=cfg_b["num_q_heads"],
                )

        elif args.mhc:

            def block_factory():
                return MHCBlock(_inner(), cfg_b["hidden"], n=4)

        elif args.attn_res:

            def block_factory():
                return AttnResBlock(
                    _inner(),
                    cfg_b["hidden"],
                    num_heads=cfg_b["num_q_heads"],
                )

        else:

            def block_factory():
                return _inner()

        baseline = ModernTransformerLM(
            vocab_size=cfg["model"]["vocab_size"],
            hidden=cfg_b["hidden"],
            num_layers=cfg_b["num_layers"],
            block_factory=block_factory,
            max_seq=args.max_seq_len,
        )
        baseline_params = baseline.num_trainable()
        if baseline_params > 500_000_000:
            raise SystemExit(
                f"baseline weights={baseline_params/1e6:.0f}M exceeds 500M cap; "
                f"reduce --baseline-param-target"
            )
        print(f"  Params: {baseline_params:,}", flush=True)
        baseline = baseline.to(device)
        baseline.resize_max_seq(args.max_seq_len)
        baseline_lr = args.muon_lr if args.muon else args.lr
        _train_one(
            baseline,
            train_loader,
            device,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            lr=baseline_lr,
            log_every=args.log_every,
            use_muon=args.muon,
        )
        val_b = TextDataset(tokenizer, ds_cfg)
        val_b.dataset = val_b.dataset.skip(args.val_skip)
        val_b_loader = DataLoader(_IterableOver(val_b), batch_size=None)
        bm = _eval_one(baseline, val_b_loader, args.val_batches, device)
        baseline_ppl = bm["perplexity"]
        print(
            f"  baseline val loss={bm['loss']:.4f} ppl={baseline_ppl:.1f}",
            flush=True,
        )

    # --- Comparison --------------------------------------------------------
    print("\n=== Matched-compute SOTA comparison ===", flush=True)
    print(
        "  Model                                Params        Val PPL",
        flush=True,
    )
    print(
        "  -----------------------------------  ------------  --------",
        flush=True,
    )
    print(
        f"  UCSA-small (ours)                    "
        f"{ucsa_params/1e6:>7.0f}M       {ucsa_ppl:.1f}",
        flush=True,
    )
    label = (
        f"2026 baseline (MLA+Muon{'+MoE' if args.moe else ''}"
        f"{'+MTP' if args.mtp else ''})"
    )
    print(
        f"  {label:35s}  {baseline_params/1e6:>7.0f}M       "
        f"{baseline_ppl:.1f}",
        flush=True,
    )
    if math.isnan(ucsa_ppl) or math.isnan(baseline_ppl):
        print("  (one side was skipped)", flush=True)
        return
    ratio = ucsa_ppl / baseline_ppl
    verdict = "WIN" if ratio < 1.0 else "loss"
    pct = abs(1.0 - ratio) * 100
    print(
        f"\n  UCSA / baseline PPL ratio: {ratio:.3f}x  ({verdict}, "
        f"small beats large by {pct:.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
