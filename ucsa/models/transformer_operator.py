"""Transformer state transition operator.

The :class:`TransformerOperator` is the reference implementation of
:class:`ucsa.models.transition_operator.StateTransitionOperator`. It is a
pre-norm transformer with grouped-query attention, sliding-window KV cache,
optional cross-attention to the ``memory_index`` bank, and either dense FFN
or Mixture-of-Experts FFN on the upper half of layers.

Architecture per block (when MoE is disabled)::

    x = x + SelfAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

Architecture per block (when MoE is enabled and this block is in the upper
half)::

    x = x + SelfAttention(LayerNorm(x))
    x = x + MoE(LayerNorm(x))

The KV cache holds only the current reasoning iteration's combined PCS and
observation keys/values. It is reset between requests by :meth:`reset`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from ucsa.models.moe import MixtureOfExperts, MoEConfig
from ucsa.models.state import BANK_NAMES, PersistentCognitiveState
from ucsa.models.transition_operator import StateTransitionOperator


@dataclass(frozen=True)
class TransformerOperatorConfig:
    """Configuration for :class:`TransformerOperator`.

    Attributes:
        hidden_size: Hidden dimensionality of every token.
        num_layers: Number of transformer blocks.
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads. Must divide
            ``num_q_heads`` for grouped query attention.
        intermediate_size: Hidden size of the dense FFN.
        sliding_window: Maximum number of tokens kept in the KV cache. The
            cache is a sliding window over the most recent tokens.
        attention_dropout: Dropout probability inside the attention module.
        residual_dropout: Dropout probability on residual branches.
        ffn_dropout: Dropout probability inside the FFN.
        vocab_size: Vocabulary size used by the language projection. The
            operator itself does not depend on this; it is included so the
            model can size its language head consistently.
        norm_eps: Epsilon for RMSNorm.
        rope_base: Base for rotary positional embeddings.
        use_memory_index_cross_attention: Whether each block performs cross
            attention to the ``memory_index`` bank. Defaults to ``True``.
        moe: Optional Mixture of Experts configuration. When ``None``, every
            block uses a dense FFN.
        differentiable_state_carry: When ``True`` (default) consecutive
            operator calls consume the previous call's *differentiable*
            bank tensors instead of re-reading the PCS parameters. The PCS
            write-back copies under ``torch.no_grad``, so without this the
            autograd graph is severed at every transition and no loss can
            reach the operator. Set to ``False`` to reproduce the older,
            severed behaviour for ablations.
    """

    hidden_size: int = 128
    num_layers: int = 4
    num_q_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 256
    sliding_window: int = 512
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    ffn_dropout: float = 0.0
    vocab_size: int = 50257
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    use_memory_index_cross_attention: bool = True
    max_position: int = 4096
    moe: MoEConfig | None = None
    differentiable_state_carry: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
        if self.num_q_heads <= 0:
            raise ValueError(
                f"num_q_heads must be positive, got {self.num_q_heads}."
            )
        if self.num_kv_heads <= 0:
            raise ValueError(
                f"num_kv_heads must be positive, got {self.num_kv_heads}."
            )
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})."
            )
        if self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {self.intermediate_size}."
            )
        if self.sliding_window <= 0:
            raise ValueError(
                f"sliding_window must be positive, got {self.sliding_window}."
            )
        if self.hidden_size % self.num_q_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_q_heads ({self.num_q_heads})."
            )


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    Computes ``x * rrms(x) * weight`` where ``rrms`` is the reciprocal of the
    root-mean-square of ``x`` along the feature dimension.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialise RMSNorm.

        Args:
            hidden_size: Size of the feature dimension.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm.

        Args:
            x: Tensor with final dim ``hidden_size``.

        Returns:
            Normalised tensor of the same shape and dtype as ``x``.
        """
        norm = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (norm.type_as(x)) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE).

    Applies rotary embeddings to ``q`` and ``k`` tensors. The implementation
    supports arbitrary sequence lengths and caches ``cos``/``sin`` tables
    keyed by sequence length to avoid recomputation.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        """Initialise rotary embeddings.

        Args:
            head_dim: Dimension per attention head.
            base: Base for the geometric frequency schedule.
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}.")
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cos_cache: dict[int, Tensor] = {}
        self.sin_cache: dict[int, Tensor] = {}

    def get_cos_sin(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` tables of length ``seq_len``.

        Args:
            seq_len: Required sequence length.
            device: Target device.

        Returns:
            Tuple of ``cos`` and ``sin`` tensors of shape
            ``(1, 1, seq_len, head_dim)`` ready to broadcast against
            ``(batch, heads, seq, head_dim)``.
        """
        cached = self.cos_cache.get(seq_len)
        if cached is not None and cached.device == device:
            return cached, self.sin_cache[seq_len]
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        cos = freqs.cos()
        sin = freqs.sin()
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(1)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(1)
        self.cos_cache[seq_len] = cos
        self.sin_cache[seq_len] = sin
        return cos, sin

    @staticmethod
    def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary embedding to ``x``.

        Args:
            x: Tensor of shape ``(batch, heads, seq, head_dim)``.
            cos: Cosine table of shape ``(1, seq, 1, head_dim)``.
            sin: Sine table of shape ``(1, seq, 1, head_dim)``.

        Returns:
            Rotated tensor of the same shape as ``x``.
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_e = cos[..., 0::2]
        sin_e = sin[..., 0::2]
        rotated_even = x_even * cos_e - x_odd * sin_e
        rotated_odd = x_even * sin_e + x_odd * cos_e
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        return rotated.reshape(*x.shape)


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention with optional KV cache.

    Implements ``num_q_heads`` query heads sharing ``num_kv_heads`` key/value
    heads. Supports Flash Attention via
    :func:`torch.nn.functional.scaled_dot_product_attention` and a sliding-
    window KV cache sized to ``sliding_window`` tokens.

    The KV cache stores keys/values for the **context** stream only. Queries
    come from a (possibly different) query stream. This matches the
    ``reasoning-loop`` access pattern: the observation context is constant
    across iterations, so caching it avoids recomputation while the PCS
    queries are always fresh.
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        sliding_window: int,
        attention_dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        """Initialise grouped-query attention.

        Args:
            hidden_size: Hidden dimensionality.
            num_q_heads: Number of query heads.
            num_kv_heads: Number of key/value heads.
            sliding_window: KV cache size.
            attention_dropout: Attention dropout.
            rope_base: Base for rotary embeddings.
        """
        super().__init__()
        if hidden_size % num_q_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_q_heads ({num_q_heads})."
            )
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})."
            )
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=False)
        self.attention_dropout = attention_dropout
        self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)
        self.kv_cache: dict[str, Tensor | None] = {
            "k": None,
            "v": None,
            "length": 0,
        }

    def reset_cache(self) -> None:
        """Clear the KV cache."""
        self.kv_cache = {"k": None, "v": None, "length": 0}

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        is_first_step: bool,
    ) -> Tensor:
        """Apply grouped-query attention with KV cache.

        Args:
            query: Tensor of shape ``(batch, q_seq, hidden_size)``.
            context: Tensor of shape ``(batch, c_seq, hidden_size)``.
            is_first_step: If ``True`` the KV cache is initialised with the
                context keys and values of this step.

        Returns:
            Tensor of shape ``(batch, q_seq, hidden_size)``.
        """
        batch, q_seq, _ = query.shape
        c_seq = context.shape[1]
        q = self.q_proj(query).view(
            batch, q_seq, self.num_q_heads, self.head_dim
        ).transpose(1, 2)
        k_new = self.k_proj(context).view(
            batch, c_seq, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v_new = self.v_proj(context).view(
            batch, c_seq, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos_q, sin_q = self.rotary.get_cos_sin(q_seq, query.device)
        cos_c, sin_c = self.rotary.get_cos_sin(c_seq, context.device)
        q = RotaryEmbedding.apply_rotary(q, cos_q, sin_q)
        k_new = RotaryEmbedding.apply_rotary(k_new, cos_c, sin_c)

        if is_first_step or self.kv_cache["k"] is None:
            k_full = k_new
            v_full = v_new
        else:
            k_full = torch.cat([self.kv_cache["k"], k_new], dim=2)
            v_full = torch.cat([self.kv_cache["v"], v_new], dim=2)

        if k_full.shape[2] > self.sliding_window:
            k_full = k_full[:, :, -self.sliding_window:]
            v_full = v_full[:, :, -self.sliding_window:]

        self.kv_cache["k"] = k_full
        self.kv_cache["v"] = v_full
        self.kv_cache["length"] = int(k_full.shape[2])

        k_expanded = self._expand_kv(k_full)
        v_expanded = self._expand_kv(v_full)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, q_seq, -1)
        return self.out_proj(attn_output)

    def _expand_kv(self, kv: Tensor) -> Tensor:
        """Expand KV heads to match the number of query heads.

        Args:
            kv: Tensor of shape ``(batch, num_kv_heads, seq, head_dim)``.

        Returns:
            Tensor of shape ``(batch, num_q_heads, seq, head_dim)``.
        """
        repeat = self.num_q_heads // self.num_kv_heads
        if repeat == 1:
            return kv
        batch, kv_heads, seq, head_dim = kv.shape
        return (
            kv.unsqueeze(2)
            .expand(batch, kv_heads, repeat, seq, head_dim)
            .reshape(batch, kv_heads * repeat, seq, head_dim)
        )


class CrossAttention(nn.Module):
    """Cross-attention module used for ``memory_index`` retrieval.

    Queries come from one stream; keys and values come from the
    ``memory_index`` bank. No KV cache; the memory bank is small and
    re-projected each step.
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        attention_dropout: float = 0.0,
    ) -> None:
        """Initialise cross attention.

        Args:
            hidden_size: Hidden dimensionality.
            num_q_heads: Number of query heads.
            num_kv_heads: Number of key/value heads.
            attention_dropout: Attention dropout.
        """
        super().__init__()
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})."
            )
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=False)
        self.attention_dropout = attention_dropout

    def forward(self, query: Tensor, kv_source: Tensor) -> Tensor:
        """Apply cross attention.

        Args:
            query: Tensor of shape ``(batch, q_seq, hidden_size)``.
            kv_source: Tensor of shape ``(batch, kv_seq, hidden_size)``.

        Returns:
            Tensor of shape ``(batch, q_seq, hidden_size)``.
        """
        batch, q_seq, _ = query.shape
        kv_seq = kv_source.shape[1]
        q = self.q_proj(query).view(batch, q_seq, self.num_q_heads, self.head_dim)
        k = self.k_proj(kv_source).view(
            batch, kv_seq, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(kv_source).view(
            batch, kv_seq, self.num_kv_heads, self.head_dim
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        repeat = self.num_q_heads // self.num_kv_heads
        if repeat > 1:
            k = k.unsqueeze(2).expand(
                batch, self.num_kv_heads, repeat, kv_seq, self.head_dim
            ).reshape(batch, self.num_q_heads, kv_seq, self.head_dim)
            v = v.unsqueeze(2).expand(
                batch, self.num_kv_heads, repeat, kv_seq, self.head_dim
            ).reshape(batch, self.num_q_heads, kv_seq, self.head_dim)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, q_seq, -1)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Standard transformer feed-forward block.

    Uses a gated linear unit variant: ``down(silu(gate(x)) * up(x))``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the FFN.

        Args:
            hidden_size: Hidden dimensionality.
            intermediate_size: Intermediate dimensionality.
            dropout: Dropout probability.
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the FFN.

        Args:
            x: Tensor of shape ``(batch, seq, hidden_size)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


@dataclass
class _BlockAux:
    """Container for auxiliary outputs collected per block.

    Attributes:
        moe_load: Per-block MoE load-balancing loss.
        moe_router_logits: Router logits from MoE blocks (for diagnostics).
    """

    moe_load: Tensor = field(default_factory=lambda: torch.tensor(0.0))
    moe_router_logits: Tensor | None = None


class TransformerBlock(nn.Module):
    """A single transformer block used by :class:`TransformerOperator`.

    The block performs self-attention over the combined PCS + observation
    stream, an optional cross-attention read from the ``memory_index`` bank,
    and a dense FFN or (on upper-half layers) MoE FFN.
    """

    def __init__(
        self,
        config: TransformerOperatorConfig,
        layer_index: int,
    ) -> None:
        """Initialise the block.

        Args:
            config: Operator configuration.
            layer_index: Zero-based index of this block in the stack.
        """
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.norm_self_attn = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_q_heads=config.num_q_heads,
            num_kv_heads=config.num_kv_heads,
            sliding_window=config.sliding_window,
            attention_dropout=config.attention_dropout,
            rope_base=config.rope_base,
        )
        self.norm_cross_attn: RMSNorm | None = None
        self.cross_attn: CrossAttention | None = None
        if config.use_memory_index_cross_attention:
            self.norm_cross_attn = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.cross_attn = CrossAttention(
                hidden_size=config.hidden_size,
                num_q_heads=config.num_q_heads,
                num_kv_heads=config.num_kv_heads,
                attention_dropout=config.attention_dropout,
            )
        self.norm_ffn = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn: nn.Module = FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.ffn_dropout,
        )
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    @property
    def is_moe_layer(self) -> bool:
        """Return whether this block uses MoE (upper half of layers)."""
        if self.config.moe is None:
            return False
        return self.layer_index >= self.config.num_layers // 2

    def install_moe(self, moe_module: nn.Module) -> None:
        """Swap the dense FFN for a Mixture of Experts module.

        Args:
            moe_module: The MoE module to install.
        """
        self.ffn = moe_module

    def forward(
        self,
        all_tokens: Tensor,
        is_first_step: bool,
        memory_index: Tensor | None,
        working_slice: tuple[int, int],
    ) -> tuple[Tensor, _BlockAux]:
        """Run one block.

        Args:
            all_tokens: Tensor of shape ``(batch, seq, hidden_size)``
                containing PCS tokens concatenated with observation tokens.
            is_first_step: Whether this is the first step of the iteration.
            memory_index: Optional memory index bank of shape
                ``(batch, mem_seq, hidden_size)``.
            working_slice: ``(start, end)`` offsets of the working memory
                tokens within ``all_tokens``.

        Returns:
            Tuple ``(updated_all_tokens, aux_outputs)``.
        """
        aux = _BlockAux()
        normed = self.norm_self_attn(all_tokens)
        attn_out = self.self_attn(normed, normed, is_first_step)
        all_tokens = all_tokens + self.residual_dropout(attn_out)

        if (
            self.cross_attn is not None
            and self.norm_cross_attn is not None
            and memory_index is not None
        ):
            start, end = working_slice
            working = all_tokens[:, start:end, :]
            cross_out = self.cross_attn(
                self.norm_cross_attn(working), memory_index
            )
            all_tokens = torch.cat(
                [
                    all_tokens[:, :start, :],
                    working + self.residual_dropout(cross_out),
                    all_tokens[:, end:, :],
                ],
                dim=1,
            )

        ffn_in = self.norm_ffn(all_tokens)
        ffn_out = self.ffn(ffn_in)
        if isinstance(ffn_out, tuple):
            ffn_value, ffn_aux = ffn_out
            aux.moe_load = ffn_aux
            all_tokens = all_tokens + self.residual_dropout(ffn_value)
        else:
            all_tokens = all_tokens + self.residual_dropout(ffn_out)

        return all_tokens, aux


class TransformerOperator(StateTransitionOperator):
    """Reference :class:`StateTransitionOperator` implementation.

    The operator reads the PCS, concatenates its banks and the new
    observation into a single token sequence, runs ``num_layers`` blocks of
    grouped-query self-attention with optional ``memory_index`` cross
    attention, then writes the updated banks back into the PCS.
    """

    def __init__(self, config: TransformerOperatorConfig | None = None) -> None:
        """Initialise the transformer operator.

        Args:
            config: Optional operator configuration. Defaults to
                :class:`TransformerOperatorConfig` defaults.
        """
        super().__init__()
        if config is None:
            config = TransformerOperatorConfig()
        self.config = config
        self.bank_id_embedding = nn.Embedding(
            len(BANK_NAMES) + 1, config.hidden_size
        )
        self.position_embedding = nn.Embedding(
            config.max_position, config.hidden_size
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_index=i)
            for i in range(config.num_layers)
        )
        if config.moe is not None:
            for block in self.blocks:
                if block.is_moe_layer:
                    moe_module = MixtureOfExperts(
                        hidden_size=config.hidden_size,
                        intermediate_size=config.intermediate_size,
                        config=config.moe,
                    )
                    block.install_moe(moe_module)
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.bank_offsets: dict[str, tuple[int, int]] = {}
        self.cumulative_offsets: tuple[int, ...] = ()
        self.is_first_step: bool = True
        self.last_aux_loss: Tensor = torch.zeros(())
        self.last_router_logits: Tensor | None = None  # ponytail: aggregated MoE router logits from blocks; None if no MoE
        # ponytail: the differentiable post-transition bank tensors, stashed
        # before the ``no_grad`` PCS write-back. Same lifecycle as the KV
        # cache: written every forward, cleared by ``reset``.
        self.last_bank_tensors: dict[str, Tensor] | None = None
        self.initialize()

    @property
    def name(self) -> str:
        """Return the operator's registration name."""
        return "transformer"

    def initialize(self) -> None:
        """Compute and cache bank offsets inside the PCS token stream."""
        offsets: list[tuple[str, tuple[int, int]]] = []
        cursor = 0
        for bank_name in BANK_NAMES:
            # We don't know sizes here without PCS; offsets are bound at first
            # forward. We use placeholder offsets that get re-bound dynamically.
            offsets.append((bank_name, (cursor, cursor)))
            cursor += 0
        self.bank_offsets = dict(offsets)
        self.cumulative_offsets = ()

    def reset(self) -> None:
        """Clear the KV cache, step counter, and carried state per block."""
        self.is_first_step = True
        self.last_bank_tensors = None
        for block in self.blocks:
            block.self_attn.reset_cache()

    def carried_bank_tensors(self) -> dict[str, Tensor] | None:
        """Return the differentiable banks carried from the previous call.

        Returns:
            The stashed post-transition bank tensors, or ``None`` when the
            carry is disabled or this is the first call since
            :meth:`reset`.
        """
        return self.last_bank_tensors

    def _read_bank(
        self, cstate: PersistentCognitiveState, name: str
    ) -> Tensor:
        """Read one bank, preferring the carried differentiable tensor.

        Args:
            cstate: The PCS to fall back to.
            name: Bank identifier.

        Returns:
            Tensor of shape ``(num_tokens, hidden_size)``.
        """
        carried = self.carried_bank_tensors()
        if carried is not None and name in carried:
            return carried[name]
        return cstate.get_bank(name)

    def _read_pcs_tokens(self, cstate: PersistentCognitiveState) -> Tensor:
        """Read every bank as one tensor, preferring the carried tensors.

        Args:
            cstate: The PCS to fall back to.

        Returns:
            Tensor of shape ``(total_tokens, hidden_size)``.
        """
        carried = self.carried_bank_tensors()
        if carried is None:
            return cstate.get_all_tokens()
        return torch.cat([carried[name] for name in BANK_NAMES], dim=0)

    def _bind_offsets(self, cstate: PersistentCognitiveState) -> dict[str, tuple[int, int]]:
        """Bind and return bank offsets for the given PCS.

        Args:
            cstate: The PCS whose bank sizes determine offsets.

        Returns:
            Mapping from bank name to ``(start, end)`` token offsets.
        """
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for bank_name in BANK_NAMES:
            size = cstate.bank_size(bank_name)
            offsets[bank_name] = (cursor, cursor + size)
            cursor += size
        return offsets

    def _add_position_signal(
        self,
        tokens: Tensor,
        offsets: Mapping[str, tuple[int, int]],
        bank_id_base: int,
    ) -> Tensor:
        """Add bank-id and positional embeddings to ``tokens``.

        Args:
            tokens: Tensor of shape ``(batch, seq, hidden_size)``.
            offsets: Mapping from bank name to ``(start, end)``.
            bank_id_base: Constant offset to add to the bank-id indices so
                that observation tokens get a unique id.

        Returns:
            Tokens with bank-id and position embeddings added.
        """
        batch, seq, hidden = tokens.shape
        positions = torch.arange(seq, device=tokens.device).unsqueeze(0).expand(
            batch, seq
        )
        pos_emb = self.position_embedding(positions)

        bank_ids = torch.full(
            (batch, seq), bank_id_base, device=tokens.device, dtype=torch.long
        )
        for index, bank_name in enumerate(BANK_NAMES):
            start, end = offsets[bank_name]
            if end <= seq:
                bank_ids[:, start:end] = index
        bank_emb = self.bank_id_embedding(bank_ids)
        return tokens + pos_emb + bank_emb

    def forward(
        self,
        cstate: PersistentCognitiveState,
        observation: Tensor,
    ) -> PersistentCognitiveState:
        """Run the transformer operator.

        Args:
            cstate: The current PCS.
            observation: Tensor of shape
                ``(batch, observation_tokens, hidden_size)``.

        Returns:
            The updated PCS.
        """
        if observation.dim() != 3:
            raise ValueError(
                f"observation must be 3D (batch, seq, hidden), got "
                f"{tuple(observation.shape)}."
            )
        batch = observation.shape[0]
        pcs_tokens = self._read_pcs_tokens(cstate)
        if pcs_tokens.shape[-1] != observation.shape[-1]:
            raise ValueError(
                f"PCS hidden size ({pcs_tokens.shape[-1]}) does not match "
                f"observation hidden size ({observation.shape[-1]})."
            )
        if pcs_tokens.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"Operator hidden_size ({self.config.hidden_size}) does not "
                f"match PCS hidden size ({pcs_tokens.shape[-1]})."
            )

        offsets = self._bind_offsets(cstate)
        pcs_tokens_b = pcs_tokens.unsqueeze(0).expand(batch, -1, -1).clone()
        all_tokens = torch.cat([pcs_tokens_b, observation], dim=1)
        working_start, working_end = offsets["working"]

        bank_id_base = len(BANK_NAMES)
        all_tokens = self._add_position_signal(all_tokens, offsets, bank_id_base)

        # The cross-attention keys/values are saved by autograd for the K/V
        # weight gradients. ``set_bank`` below writes the banks back in
        # place, which would bump the version counter of an expanded *view*
        # of the ``memory_index`` parameter and make backward fail with
        # "one of the variables needed for gradient computation has been
        # modified by an inplace operation". Cloning detaches the storage
        # (not the graph), exactly as ``pcs_tokens_b`` already does above.
        memory_index_tokens = (
            self._read_bank(cstate, "memory_index")
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .clone()
        )

        total_aux = torch.zeros(())
        router_pieces: list[Tensor] = []
        for block in self.blocks:
            all_tokens, aux = block(
                all_tokens,
                is_first_step=self.is_first_step,
                memory_index=memory_index_tokens if block.cross_attn is not None else None,
                working_slice=(working_start, working_end),
            )
            total_aux = total_aux + aux.moe_load
            if block.is_moe_layer and hasattr(block.ffn, "last_router_logits"):
                rl = block.ffn.last_router_logits
                if rl is not None:
                    router_pieces.append(rl)
        all_tokens = self.final_norm(all_tokens)
        self.is_first_step = False
        self.last_aux_loss = total_aux
        self.last_router_logits = torch.cat(router_pieces, dim=0) if router_pieces else None

        new_pcs = cstate
        bank_tensors: dict[str, Tensor] = {}
        for bank_name in BANK_NAMES:
            start, end = offsets[bank_name]
            bank_tensors[bank_name] = all_tokens[0, start:end, :]
        # Stash the differentiable tensors *before* the write-back, which
        # copies under ``no_grad`` and would otherwise be the end of the
        # autograd graph for this transition. Leaving the stash empty
        # reproduces the older severed behaviour end to end: the loop and
        # the heads both fall back to detached PCS reads.
        self.last_bank_tensors = (
            bank_tensors if self.config.differentiable_state_carry else None
        )
        for bank_name in BANK_NAMES:
            new_pcs.set_bank(bank_name, bank_tensors[bank_name])
        return new_pcs


__all__ = [
    "CrossAttention",
    "FeedForward",
    "GroupedQueryAttention",
    "MoEConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "TransformerBlock",
    "TransformerOperator",
    "TransformerOperatorConfig",
]
