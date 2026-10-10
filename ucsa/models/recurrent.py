"""UCSA-R: causal chunked state recurrence over a persistent cognitive state.

The original UCSA decodes its language logits from working-bank slots that
have read the whole input, so a slot can copy the token it is scored on (see
`ucsa.training.prefix`). UCSA-R keeps the central idea, one persistent
multi-bank state that everything reads and writes, and makes the information
flow causal by construction.

Data flow for a sequence cut into chunks of `chunk_size` tokens:

    tokens --embed--> x_t --chunk encoder (bidirectional, per chunk)--> e_t
    S_t = update(S_{t-1}, e_t)            # tiny sequential scan over chunks
    logits_t = decode(x_t | S_{t-1})      # causal blocks, all chunks parallel

Chunk `t` is decoded from `S_{t-1}` only, which was written from chunks
`< t`, so every logit is an exact next-token prediction. The encoder may look
both ways inside a chunk because its output is read only by later chunks.
Because the state is built from shallow chunk summaries, the only sequential
work is the small state scan; the expensive decoder runs over all chunks as
one batch, which is what makes it fast on a single accelerator.

Cost is linear in sequence length and the state has constant size, so the
same weights run over arbitrarily long streams. `use_state=False` gives the
matched chunk-local control that isolates what the state contributes.
"""

import copy
import dataclasses
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from torch.utils import checkpoint as torch_checkpoint


@dataclasses.dataclass(frozen=True)
class Config:
    """Architecture knobs of UCSA-R, validated on construction.

    Attributes:
      vocab_size: Tokenizer vocabulary size. The output head is tied to the
        embedding.
      hidden: Model width.
      layers: Number of decoder blocks.
      heads: Number of attention heads; divides `hidden`.
      ffn_dim: SwiGLU hidden width.
      chunk_size: Tokens per chunk, which is also the attention span inside a
        chunk.
      encoder_layers: Layers in the chunk encoder that feeds the state.
      banks: `(name, n_slots)` pairs making up the state.
      bank_write_bias: Initial write-gate bias per bank. More negative means
        the bank is written less readily, so it retains longer.
      read_every: The decoder reads the state in every `read_every`-th block
        (1 reads in every block); larger values save compute.
      write_top_k: Only the `k` slots with the highest write gate are updated
        per chunk. 0 writes densely.
      bptt_chunks: Detach the state every this many chunks (truncated
        backpropagation through time). 0 backpropagates through the whole
        sequence.
      use_state: False resets the state every chunk (the chunk-local control).
      slot_dropout: Probability, per training step, of reading only a random
      prefix of the state slots (elastic state). Slots are then ordered by
      importance, so memory can be shrunk at inference with graceful
      degradation. 0 disables it.
    min_slots: Smallest prefix sampled by `slot_dropout`.
    surprise_gate: Write more when the state failed to predict the chunk.
      The JEPA predictor's error for a chunk (its surprise) shifts the write
      gate through a learned per-slot gain that starts at zero, so the model
      stores the residual that prediction left unexplained. Needs
      `jepa_weight > 0`.
    jepa_weight: Weight of the causal JEPA loss. 0 disables it.
      ema_momentum: Momentum of the JEPA target encoder.
      loss_chunk: Tokens per slice of the checkpointed LM-head loss, so the
        full `(tokens, vocab)` logits are never held in memory. 0 computes the
        loss in one piece.
      dropout: Residual dropout probability.
      init_std: Standard deviation of weight initialisation.
    """

    vocab_size: int = 50257
    hidden: int = 256
    layers: int = 6
    heads: int = 4
    ffn_dim: int = 704
    chunk_size: int = 128
    encoder_layers: int = 1
    banks: tuple[tuple[str, int], ...] = (
        ("working", 16),
        ("long_term", 8),
        ("intent", 8),
    )
    bank_write_bias: tuple[tuple[str, float], ...] = (
        ("working", 0.0),
        ("long_term", -2.0),
        ("intent", -1.0),
    )
    read_every: int = 1
    write_top_k: int = 0
    bptt_chunks: int = 0
    use_state: bool = True
    slot_dropout: float = 0.0
    min_slots: int = 4
    surprise_gate: bool = False
    jepa_weight: float = 0.1
    ema_momentum: float = 0.996
    loss_chunk: int = 1024
    dropout: float = 0.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        """Validates every field.

        Raises:
          ValueError: If any field is out of range or inconsistent.
        """
        checks = (
            (self.vocab_size > 0, "vocab_size must be positive"),
            (self.hidden > 0 and self.layers > 0, "hidden/layers must be > 0"),
            (
                self.heads > 0 and self.hidden % self.heads == 0,
                "hidden must be divisible by heads",
            ),
            (
                self.hidden % self.heads == 0
                and (self.hidden // self.heads) % 2 == 0,
                "head_dim must be even",
            ),
            (self.chunk_size >= 2, "chunk_size must be >= 2"),
            (self.encoder_layers >= 1, "encoder_layers must be >= 1"),
            (
                len(self.banks) > 0 and all(n > 0 for _, n in self.banks),
                "banks must be non-empty with positive sizes",
            ),
            (
                len({name for name, _ in self.banks}) == len(self.banks),
                "bank names must be unique",
            ),
            (self.read_every >= 1, "read_every must be >= 1"),
            (
                0 <= self.write_top_k <= self.num_slots,
                f"write_top_k must be in [0, {self.num_slots}]",
            ),
            (self.bptt_chunks >= 0, "bptt_chunks must be >= 0"),
            (0.0 <= self.slot_dropout <= 1.0, "slot_dropout must be in [0, 1]"),
            (
                1 <= self.min_slots <= self.num_slots,
                f"min_slots must be in [1, {self.num_slots}]",
            ),
            (self.jepa_weight >= 0.0, "jepa_weight must be >= 0"),
            (
                not self.surprise_gate or self.jepa_weight > 0,
                "surprise_gate needs jepa_weight > 0",
            ),
            (0.0 < self.ema_momentum < 1.0, "ema_momentum must be in (0, 1)"),
            (self.loss_chunk >= 0, "loss_chunk must be >= 0"),
            (0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"),
        )
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
        unknown = {n for n, _ in self.bank_write_bias} - {
            n for n, _ in self.banks
        }
        if unknown:
            raise ValueError(f"bank_write_bias names unknown banks: {unknown}")

    @property
    def num_slots(self) -> int:
        """Total number of state slots across all banks."""
        return sum(n for _, n in self.banks)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Builds a config from a plain dict, rejecting unknown keys.

        Args:
          data: Field values. `banks` and `bank_write_bias` may be given as a
            mapping or as a sequence of pairs.

        Returns:
          The validated config.

        Raises:
          ValueError: If `data` has an unknown key or an invalid value.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        data = dict(data)
        for key, cast in (("banks", int), ("bank_write_bias", float)):
            if key in data:
                items = data[key]
                if isinstance(items, dict):
                    items = items.items()
                data[key] = tuple((str(a), cast(b)) for a, b in items)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dict that `from_dict` round-trips."""
        out = dataclasses.asdict(self)
        out["banks"] = [list(b) for b in self.banks]
        out["bank_write_bias"] = [list(b) for b in self.bank_write_bias]
        return out


def rope_tables(
    head_dim: int, length: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds rotary-embedding cosine and sine tables.

    Args:
      head_dim: Per-head width; must be even.
      length: Number of positions.
      base: Rotary frequency base.

    Returns:
      `(cos, sin)`, each of shape `(length, head_dim // 2)`.
    """
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    freqs = torch.outer(torch.arange(length).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Applies half-split rotary embeddings.

    Args:
      x: Queries or keys of shape `(batch, heads, seq, head_dim)`.
      cos: Cosine table from `rope_tables`.
      sin: Sine table from `rope_tables`.

    Returns:
      Tensor of the same shape as `x`.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[: x.shape[2]], sin[: x.shape[2]]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


class SelfAttention(nn.Module):
    """Fused-QKV multi-head self-attention with RoPE."""

    def __init__(self, dim: int, heads: int, causal: bool) -> None:
        """Initialises the layer.

        Args:
          dim: Model width.
          heads: Number of heads.
          causal: Whether positions may only attend backwards.
        """
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Attends within `x`.

        Args:
          x: Input of shape `(batch, seq, dim)`.
          cos: RoPE cosine table.
          sin: RoPE sine table.

        Returns:
          Output of shape `(batch, seq, dim)`.
        """
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal
        )
        return self.out(out.transpose(1, 2).reshape(b, t, d))


class CrossAttention(nn.Module):
    """Multi-head attention with queries from `x` and keys from a context."""

    def __init__(self, dim: int, heads: int) -> None:
        """Initialises the layer.

        Args:
          dim: Model width.
          heads: Number of heads.
        """
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key_value = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Reads `context` from the positions of `x`.

        Args:
          x: Queries of shape `(batch, seq, dim)`.
          context: Keys and values of shape `(batch, n, dim)`.

        Returns:
          Output of shape `(batch, seq, dim)`.
        """
        b, t, d = x.shape
        head_dim = d // self.heads
        q = self.query(x).view(b, t, self.heads, head_dim).transpose(1, 2)
        kv = self.key_value(context).view(
            b, context.shape[1], 2, self.heads, head_dim
        )
        k, v = kv.permute(2, 0, 3, 1, 4)
        out = functional.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    """Gated feed-forward layer."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Initialises the layer.

        Args:
          dim: Model width.
          hidden: Inner width.
        """
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the layer to `x` of shape `(..., dim)`."""
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(functional.silu(gate) * up)


class Block(nn.Module):
    """Self-attention, an optional state read, and a feed-forward layer."""

    def __init__(self, config: Config, causal: bool, read: bool) -> None:
        """Initialises the block.

        Args:
          config: Model configuration.
          causal: Whether self-attention is causal.
          read: Whether the block cross-attends to the state.
        """
        super().__init__()
        dim = config.hidden
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = SelfAttention(dim, config.heads, causal)
        self.read_norm = nn.RMSNorm(dim) if read else None
        self.read = CrossAttention(dim, config.heads) if read else None
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, config.ffn_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Runs the block.

        Args:
          x: Tokens of shape `(rows, seq, dim)`.
          state: State to read of shape `(rows, slots, dim)`, or None.
          cos: RoPE cosine table.
          sin: RoPE sine table.

        Returns:
          Output of shape `(rows, seq, dim)`.
        """
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin))
        if self.read is not None and state is not None:
            x = x + self.drop(self.read(self.read_norm(x), state))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class StateUpdater(nn.Module):
    """Gated, optionally sparse, per-bank write of a chunk into the state.

    Slots query the chunk summary. A per-slot, per-channel gate decides how much
    of the read replaces the old content, and learned per-slot biases give each
    bank its own retention.
    """

    def __init__(self, config: Config) -> None:
        """Initialises the updater.

        Args:
          config: Model configuration.
        """
        super().__init__()
        dim = config.hidden
        self.top_k = config.write_top_k
        self.read_norm = nn.RMSNorm(dim)
        self.read = CrossAttention(dim, config.heads)
        self.out_norm = nn.RMSNorm(dim)
        self.gate = nn.Linear(2 * dim, dim)
        bias = dict(config.bank_write_bias)
        slot_bias = torch.cat(
            [torch.full((n,), bias.get(name, 0.0)) for name, n in config.banks]
        )
        self.slot_bias = nn.Parameter(slot_bias.unsqueeze(-1))
        # Per-slot response to surprise; zero keeps the baseline behaviour.
        self.surprise_gain = (
            nn.Parameter(torch.zeros(config.num_slots, 1))
            if config.surprise_gate
            else None
        )

    def forward(
        self,
        state: torch.Tensor,
        chunk: torch.Tensor,
        surprise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes one chunk summary into the state.

        Args:
          state: Current state of shape `(batch, slots, dim)`.
          chunk: Chunk summary of shape `(batch, tokens, dim)`.
          surprise: Per-example surprise of shape `(batch,)`, used when the
            updater has a surprise gain.

        Returns:
          The new state and the mean write gate per slot, `(batch, slots)`.
        """
        read = self.read(self.read_norm(state), chunk)
        candidate = self.out_norm(state + read)
        logits = self.gate(torch.cat([state, read], -1)) + self.slot_bias
        if self.surprise_gain is not None and surprise is not None:
            logits = logits + self.surprise_gain * surprise.view(-1, 1, 1)
        gate = torch.sigmoid(logits)
        if self.top_k:
            top = gate.mean(-1).topk(self.top_k, dim=-1).indices
            keep = torch.zeros_like(gate[..., :1]).scatter_(
                1, top.unsqueeze(-1), 1.0
            )
            gate = gate * keep
        return state + gate * (candidate - state), gate.mean(-1)


class Model(nn.Module):
    """Causal language model whose context is a persistent multi-bank state."""

    def __init__(self, config: Config) -> None:
        """Initialises the model.

        Args:
          config: Model configuration.
        """
        super().__init__()
        self.config = config
        dim = config.hidden
        self.embed = nn.Embedding(config.vocab_size, dim)
        self.encoder = nn.ModuleList(
            Block(config, causal=False, read=False)
            for _ in range(config.encoder_layers)
        )
        self.blocks = nn.ModuleList(
            Block(config, causal=True, read=(i % config.read_every == 0))
            for i in range(config.layers)
        )
        self.final_norm = nn.RMSNorm(dim)
        self.state0 = nn.Parameter(
            torch.randn(config.num_slots, dim) * config.init_std
        )
        self.updater = StateUpdater(config)
        # Causal JEPA: predict the latent of chunk t from S_{t-1}.
        self.predictor = nn.Sequential(
            nn.RMSNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.summary = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.target_summary = copy.deepcopy(self.summary)
        for param in self.target_summary.parameters():
            param.requires_grad_(False)
        cos, sin = rope_tables(dim // config.heads, config.chunk_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self.__init_weights)
        self.target_summary.load_state_dict(self.summary.state_dict())

    def __init_weights(self, module: nn.Module) -> None:
        """Initialises linear and embedding weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, self.config.init_std)

    def initial_state(self, batch: int) -> torch.Tensor:
        """Returns the learned initial state expanded to `batch` rows."""
        return self.state0.unsqueeze(0).expand(batch, -1, -1)

    def __encode(self, x: torch.Tensor) -> torch.Tensor:
        """Summarises chunks for the state; `x` is `(rows, chunk, dim)`."""
        for block in self.encoder:
            x = block(x, None, self.cos, self.sin)
        return x

    def __decode(
        self, x: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        """Runs the causal decoder over `(rows, chunk, dim)` tokens."""
        for block in self.blocks:
            x = block(x, state, self.cos, self.sin)
        return self.final_norm(x)

    def __surprise(
        self, state: torch.Tensor, pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts a chunk's latent from the state and scores the miss.

        Args:
          state: State before the chunk, `(batch, slots, dim)`.
          pooled: Mean token embedding of the chunk, `(batch, dim)`.

        Returns:
          The prediction `(batch, dim)` and the surprise `(batch,)`, which is
          `1 - cosine(prediction, target)` and carries no gradient.
        """
        pred = self.predictor(state.mean(1))
        with torch.no_grad():
            target = self.target_summary(pooled)
            surprise = 1.0 - functional.cosine_similarity(
                pred.detach().float(), target.float(), dim=-1
            )
        return pred, surprise

    def __scan(
        self,
        state: torch.Tensor,
        summaries: torch.Tensor,
        pooled: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Writes chunk summaries into the state one chunk at a time.

        Args:
          state: Initial state of shape `(batch, slots, dim)`.
          summaries: Chunk summaries of shape `(batch, chunks, tokens, dim)`.
          pooled: Mean token embedding per chunk, `(batch, chunks, dim)`;
            when given, the JEPA prediction (and the surprise that gates the
            write) is computed for every chunk.

        Returns:
          `(reads, final, writes, preds)` where `reads[:, t]` is `S_{t-1}`,
          the state chunk `t` reads, `final` is the last state, `writes` holds
          the mean write gate per chunk and slot, and `preds` is the JEPA
          prediction per chunk (None without `pooled`).
        """
        reads, writes, preds = [], [], []
        for t in range(summaries.shape[1]):
            reads.append(state)
            surprise = None
            if pooled is not None:
                pred, surprise = self.__surprise(state, pooled[:, t])
                preds.append(pred)
            state, write = self.updater(state, summaries[:, t], surprise)
            writes.append(write)
            if (
                self.config.bptt_chunks
                and (t + 1) % self.config.bptt_chunks == 0
            ):
                state = state.detach()
        return (
            torch.stack(reads, 1),
            state,
            torch.stack(writes, 1),
            torch.stack(preds, 1) if preds else None,
        )

    def hidden_states(
        self,
        ids: torch.Tensor,
        state: torch.Tensor | None = None,
        active_slots: int | None = None,
    ) -> dict[str, Any]:
        """Computes final hidden states for every position.

        Args:
          ids: Token ids of shape `(batch, seq)`.
          state: Optional starting state of shape `(batch, slots, dim)`.
          active_slots: Read only the first `active_slots` state slots. None
            reads all of them (or a random prefix while training with
            `slot_dropout`).

        Returns:
          A dict with `hidden` `(batch, seq, dim)`, the final `state`, the JEPA
          `(predictions, targets)` pair (None when disabled) and per-chunk
          `write` gates (None without a state).
        """
        config = self.config
        batch, seq = ids.shape
        size = config.chunk_size
        chunks = -(-seq // size)
        if (
            chunks * size != seq
        ):  # Pad the tail; padding only follows real tokens.
            ids = functional.pad(ids, (0, chunks * size - seq))
        x = self.embed(ids).view(batch * chunks, size, -1)
        start = self.initial_state(batch) if state is None else state

        preds = targets = writes = None
        final = start
        if config.use_state:
            summaries = self.__encode(x).view(batch, chunks, size, -1)
            predictive = config.surprise_gate or (
                config.jepa_weight > 0 and chunks > 1
            )
            pooled = (
                x.view(batch, chunks, size, -1).mean(2) if predictive else None
            )
            reads, final, writes, preds = self.__scan(start, summaries, pooled)
            read = reads.reshape(batch * chunks, *reads.shape[2:])
            read = read[:, : self.__slots_to_read(active_slots)]
            if config.jepa_weight > 0 and chunks > 1:
                with torch.no_grad():
                    targets = self.target_summary(pooled)
            else:
                preds = None
        else:
            read = (
                start.unsqueeze(1)
                .expand(batch, chunks, -1, -1)
                .reshape(batch * chunks, -1, x.shape[-1])
            )
        hidden = self.__decode(x, read).view(batch, chunks * size, -1)[:, :seq]
        return {
            "hidden": hidden,
            "state": final,
            "jepa": (preds, targets),
            "write": writes,
        }

    def __slots_to_read(self, active_slots: int | None) -> int:
        """Returns how many leading state slots the decoder reads."""
        total = self.config.num_slots
        if active_slots is not None:
            return max(1, min(active_slots, total))
        config = self.config
        if self.training and torch.rand(()).item() < config.slot_dropout:
            return int(torch.randint(config.min_slots, total + 1, ()).item())
        return total

    def forward(
        self,
        ids: torch.Tensor,
        state: torch.Tensor | None = None,
        active_slots: int | None = None,
    ) -> dict[str, Any]:
        """Computes logits for every position.

        Args:
          ids: Token ids of shape `(batch, seq)`.
          state: Optional starting state.
          active_slots: Read only this many leading state slots.

        Returns:
          The `hidden_states` dict plus `logits` of shape `(batch, seq, vocab)`.
        """
        out = self.hidden_states(ids, state, active_slots)
        out["logits"] = out["hidden"] @ self.embed.weight.T
        return out

    def __ce_sum(
        self, hidden: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Summed cross-entropy of a slice of hidden states against targets."""
        logits = (hidden @ self.embed.weight.T).float()
        return functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="sum",
        )

    def compute_loss(
        self, ids: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes cross-entropy plus the causal JEPA loss.

        Metrics are returned as on-device tensors so logging never forces a host
        synchronisation.

        Args:
          ids: Input token ids of shape `(batch, seq)`.
          targets: Next-token ids of shape `(batch, seq)`.

        Returns:
          The scalar loss and a dict of detached metric tensors.
        """
        config = self.config
        out = self.hidden_states(ids)
        hidden = out["hidden"]
        step = config.loss_chunk or hidden.shape[1]
        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, hidden.shape[1], step):
            args = (hidden[:, i : i + step], targets[:, i : i + step])
            if self.training and config.loss_chunk:
                total = total + torch_checkpoint.checkpoint(
                    self.__ce_sum, *args, use_reentrant=False
                )
            else:
                total = total + self.__ce_sum(*args)
        ce = total / targets.numel()
        loss, metrics = ce, {"ce": ce.detach()}
        preds, tgts = out["jepa"]
        if preds is not None:
            p = functional.normalize(preds.float(), dim=-1)
            z = functional.normalize(tgts.float(), dim=-1)
            align = (1.0 - (p * z).sum(-1)).mean()
            # Keep the target latent from collapsing to a constant.
            spread = functional.relu(
                0.5 - tgts.float().flatten(0, 1).std(0)
            ).mean()
            jepa = align + spread
            loss = loss + config.jepa_weight * jepa
            metrics["jepa"] = jepa.detach()
        if out["write"] is not None:
            metrics["write_gate"] = out["write"].mean().detach()
        return loss, metrics

    @torch.no_grad()
    def update_ema(self) -> None:
        """Moves the JEPA target encoder toward the online encoder."""
        momentum = self.config.ema_momentum
        for target, online in zip(
            self.target_summary.parameters(),
            self.summary.parameters(),
            strict=True,
        ):
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)

    @torch.no_grad()
    def advance(self, state: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        """Writes one full chunk into the state (the streaming update).

        Args:
          state: State before the chunk, `(batch, slots, dim)`.
          chunk: Token ids of one full chunk, `(batch, chunk_size)`.

        Returns:
          The state after the chunk. Without `use_state` it is unchanged.
        """
        if not self.config.use_state:
            return state
        embedded = self.embed(chunk)
        surprise = None
        if self.config.surprise_gate:
            _, surprise = self.__surprise(state, embedded.mean(1))
        new_state, _ = self.updater(state, self.__encode(embedded), surprise)
        return new_state

    @torch.no_grad()
    def next_logits(
        self, state: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        """Returns the next-token logits after a partial chunk.

        This is the streaming counterpart of `forward`: with the state that
        precedes the current chunk and the chunk's tokens so far, it gives the
        logits `forward` would give at the last of those positions.

        Args:
          state: State before the current chunk, `(batch, slots, dim)`.
          current: Tokens of the current chunk so far, `(batch, n)` with
            `1 <= n <= chunk_size`.

        Returns:
          Logits of shape `(batch, vocab)`.
        """
        read = (
            state
            if self.config.use_state
            else self.initial_state(current.shape[0])
        )
        hidden = self.__decode(self.embed(current), read)
        return hidden[:, -1] @ self.embed.weight.T

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Streams tokens, carrying the state across chunks.

        Memory is constant in the generated length: only the state and the
        current partial chunk are kept.

        Args:
          prompt: Token ids of shape `(batch, prompt_len)`.
          max_new_tokens: Number of tokens to generate.
          temperature: Sampling temperature; 0 or less decodes greedily.
          top_k: If positive, sample only from the `top_k` most likely tokens.

        Returns:
          The prompt followed by the generated tokens.
        """
        self.eval()
        size = self.config.chunk_size
        state = self.initial_state(prompt.shape[0])
        current = out = prompt
        for _ in range(max_new_tokens):
            while current.shape[1] > size:
                head, current = current[:, :size], current[:, size:]
                state = self.advance(state, head)
            logits = self.next_logits(state, current)
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = logits.topk(top_k, -1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, -math.inf)
                nxt = torch.multinomial(logits.softmax(-1), 1)
            current, out = torch.cat([current, nxt], 1), torch.cat(
                [out, nxt], 1
            )
        return out


def count_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters of `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def config_for_params(target: int, **overrides: Any) -> Config:
    """Picks width and depth so the model has about `target` parameters.

    Searches a small grid with the shape rules fixed (head_dim 64, SwiGLU of
    about 8/3 width) so scaling is one number rather than several.

    Args:
      target: Desired parameter count.
      **overrides: Extra `Config` fields held fixed during the search.

    Returns:
      The config whose parameter count is closest to `target`.
    """
    best: tuple[int, Config] | None = None
    for hidden in (128, 192, 256, 320, 384, 512, 640, 768, 1024):
        for layers in (2, 3, 4, 6, 8, 10, 12, 16):
            config = Config(
                hidden=hidden,
                layers=layers,
                heads=max(2, hidden // 64),
                ffn_dim=int(round(hidden * 8 / 3 / 64)) * 64 or 64,
                **overrides,
            )
            with torch.device("meta"):  # Count without allocating.
                size = count_parameters(Model(config))
            if best is None or abs(size - target) < best[0]:
                best = (abs(size - target), config)
    return best[1]
