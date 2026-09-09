"""Projection heads.

Four independent projection heads read from the working-memory bank of the
PCS and produce:

- **LanguageHead** -- vocabulary logits for autoregressive text generation.
- **PlanningHead** -- logits over a discrete set of plan tokens.
- **ToolHead** -- logits over a discrete set of tool tokens.
- **MemoryHead** -- dense memory-query embeddings used by the retrieval
  pipeline.

Heads never share parameters, never communicate, and never store state
beyond their own projection matrices.

Two further modules live here but sit outside that contract because they
read more than working memory: **InputReconstructionHead** (the LeWM
capacity bottleneck) and **OriginationHead** (``G``, which generates the
next iteration's input from the ``intent`` bank).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ucsa.models.moe import load_balancing_loss, top_k_mask


@dataclass(frozen=True)
class HeadConfig:
    """Common configuration values for projection heads.

    Attributes:
        hidden_size: Hidden dimensionality of the working-memory stream.
        vocab_size: Vocabulary size used by the language head.
        num_plan_tokens: Discrete plan vocabulary size.
        num_tools: Discrete tool vocabulary size.
        memory_query_dim: Dimensionality of memory-query embeddings.
        reconstruction_dim: Dim of the input-reconstruction projection.
            Defaults to ``hidden_size`` so it can be compared against
            ``perception.embed_tokens`` element-wise.
        origination_top_k: Number of intent slots each query token may read
            through the origination gate. The sparsity is what makes the
            origination attributable.
        origination_aux_loss_weight: Weight on the origination gate's
            load-balancing loss, exposed as
            ``OriginationHead.last_aux_loss``.
        intent_update_scale: Weight on the residual refresh of the
            origination state from working memory. ``0.0`` freezes the
            intent bank at its parameter value, which makes it identical
            for every input.
    """

    hidden_size: int = 128
    vocab_size: int = 50257
    num_plan_tokens: int = 64
    num_tools: int = 32
    memory_query_dim: int = 64
    reconstruction_dim: int = 0  # 0 = inherit hidden_size
    origination_top_k: int = 2
    origination_aux_loss_weight: float = 0.01
    intent_update_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {self.hidden_size}."
            )
        if self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {self.vocab_size}."
            )
        if self.num_plan_tokens <= 0:
            raise ValueError(
                f"num_plan_tokens must be positive, got {self.num_plan_tokens}."
            )
        if self.num_tools <= 0:
            raise ValueError(
                f"num_tools must be positive, got {self.num_tools}."
            )
        if self.memory_query_dim <= 0:
            raise ValueError(
                f"memory_query_dim must be positive, "
                f"got {self.memory_query_dim}."
            )
        if self.origination_top_k <= 0:
            raise ValueError(
                f"origination_top_k must be positive, "
                f"got {self.origination_top_k}."
            )
        if self.origination_aux_loss_weight < 0.0:
            raise ValueError(
                f"origination_aux_loss_weight must be non-negative, "
                f"got {self.origination_aux_loss_weight}."
            )
        if self.intent_update_scale < 0.0:
            raise ValueError(
                f"intent_update_scale must be non-negative, "
                f"got {self.intent_update_scale}."
            )
        if self.reconstruction_dim <= 0:
            # Default the reconstruction dim to ``hidden_size`` so the
            # reconstructed token embeddings line up with
            # ``perception.embed_tokens`` for an element-wise loss.
            object.__setattr__(
                self, "reconstruction_dim", self.hidden_size
            )


class LanguageHead(nn.Module):
    """Project working memory to vocabulary logits."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        """Initialise the language head.

        Args:
            hidden_size: Hidden dimensionality.
            vocab_size: Vocabulary size.
        """
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, working_memory: Tensor) -> Tensor:
        """Project to vocabulary logits.

        Args:
            working_memory: Tensor of shape
                ``(batch, seq, hidden_size)``.

        Returns:
            Logits of shape ``(batch, seq, vocab_size)``.
        """
        projected: Tensor = self.proj(working_memory)
        return projected


class PlanningHead(nn.Module):
    """Project working memory to planning logits."""

    def __init__(self, hidden_size: int, num_plan_tokens: int) -> None:
        """Initialise the planning head.

        Args:
            hidden_size: Hidden dimensionality.
            num_plan_tokens: Plan vocabulary size.
        """
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_plan_tokens, bias=False)

    def forward(self, working_memory: Tensor) -> Tensor:
        """Project to planning logits.

        Args:
            working_memory: Tensor of shape
                ``(batch, seq, hidden_size)``.

        Returns:
            Logits of shape ``(batch, seq, num_plan_tokens)``.
        """
        projected: Tensor = self.proj(working_memory)
        return projected


class ToolHead(nn.Module):
    """Project working memory to tool logits."""

    def __init__(self, hidden_size: int, num_tools: int) -> None:
        """Initialise the tool head.

        Args:
            hidden_size: Hidden dimensionality.
            num_tools: Tool vocabulary size.
        """
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_tools, bias=False)

    def forward(self, working_memory: Tensor) -> Tensor:
        """Project to tool logits.

        Args:
            working_memory: Tensor of shape
                ``(batch, seq, hidden_size)``.

        Returns:
            Logits of shape ``(batch, seq, num_tools)``.
        """
        projected: Tensor = self.proj(working_memory)
        return projected


class MemoryHead(nn.Module):
    """Project working memory to memory-query embeddings."""

    def __init__(self, hidden_size: int, memory_query_dim: int) -> None:
        """Initialise the memory head.

        Args:
            hidden_size: Hidden dimensionality.
            memory_query_dim: Dimensionality of memory queries.
        """
        super().__init__()
        self.proj = nn.Linear(hidden_size, memory_query_dim, bias=False)
        self.memory_query_dim = memory_query_dim

    def forward(self, working_memory: Tensor) -> Tensor:
        """Project to memory-query embeddings.

        Args:
            working_memory: Tensor of shape
                ``(batch, seq, hidden_size)``.

        Returns:
            Queries of shape ``(batch, seq, memory_query_dim)``.
        """
        projected: Tensor = self.proj(working_memory)
        return projected


class InputReconstructionHead(nn.Module):
    """Predict input-token embeddings from working memory.

    Used for the LeWM-style "capacity bottleneck" loss: the JEPA latent
    must retain enough information to reconstruct input-token
    embeddings. Without this, the latent can collapse to a constant
    and still satisfy the JEPA prediction loss (especially with the
    LeWM Gaussian regulariser alone).
    """

    def __init__(self, hidden_size: int, reconstruction_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, reconstruction_dim, bias=False)
        self.reconstruction_dim = reconstruction_dim

    def forward(self, working_memory: Tensor) -> Tensor:
        projected: Tensor = self.proj(working_memory)
        return projected


class IntentUpdate(nn.Module):
    """Refresh the origination state from working memory.

    Without this the ``intent`` bank is a plain parameter: identical for
    every input, so it can only encode an input-independent bias and its
    variance across inputs is zero by construction. A signal that is the
    same before every reach is not an origination signal, and it also makes
    the collapse diagnostic unreadable.

    The update is a residual attention read with the intent slots as
    queries and working memory as keys and values:

    ``intent_{k+1} = intent_k + scale * Attn(q=intent_k, kv=working_k)``

    Residual on purpose. The learned (or inference-time optimised) bank
    value has to survive, or descending on it would be pointless -- an
    absolute rewrite would discard exactly what Phase D optimises.
    """

    def __init__(self, hidden_size: int, scale: float = 0.1) -> None:
        """Initialise the intent updater.

        Args:
            hidden_size: Hidden dimensionality.
            scale: Weight on the residual update. ``0.0`` freezes the
                origination state at its parameter value.

        Raises:
            ValueError: If ``scale`` is negative.
        """
        super().__init__()
        if scale < 0.0:
            raise ValueError(f"scale must be non-negative, got {scale}.")
        self.hidden_size = hidden_size
        self.scale = scale
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, intent: Tensor, working: Tensor) -> Tensor:
        """Return the refreshed origination state.

        Args:
            intent: Current origination state of shape ``(intent_tokens,
                hidden_size)``.
            working: Working bank of shape ``(working_tokens,
                hidden_size)``.

        Returns:
            Tensor shaped like ``intent``.

        Raises:
            ValueError: If either tensor is not 2D.
        """
        if intent.dim() != 2 or working.dim() != 2:
            raise ValueError(
                f"intent and working must be 2D (tokens, hidden), got "
                f"{tuple(intent.shape)} and {tuple(working.shape)}."
            )
        if self.scale == 0.0:
            return intent
        scores = torch.matmul(
            self.q_proj(intent), self.k_proj(working).transpose(-1, -2)
        ) / (self.hidden_size**0.5)
        read = torch.matmul(torch.softmax(scores, dim=-1), self.v_proj(working))
        return intent + self.scale * read


class OriginationHead(nn.Module):
    """Generate the next iteration's input from the origination state.

    This is ``G`` in ``obs_{k+1} = (1 - alpha_k) * G + alpha_k * obs``. It
    is the one place where the signal that *causes* the next state is
    computed, which is what makes that signal addressable: an intent slot
    either contributes here or it does not affect the next input at all.
    The operator does not attend over the intent bank
    (``stream_intent_bank=False``), so ``G`` really is the only route from
    the bank to an action.

    The generated stream is a cross-attention read over the intent bank and
    over working memory -- the spec's ``G(intent, working)`` -- with
    queries from the current input stream. The query side exists because
    the generated tensor has to line up with a variable-length
    observation: 16 intent slots cannot by themselves say how many input
    tokens to emit, or in what order. No observation content reaches the
    output except through the attention weights, so the mix in
    :class:`~ucsa.models.reasoning_loop.ReasoningLoop` remains the only
    path by which the real observation survives.

    The intent read is routed through a **top-k sparse gate**, reusing the
    routing machinery in :mod:`ucsa.models.moe` with intent slots in place
    of experts. The sparsity is the point: with a dense read every slot
    contributes a little to every action and the origination cannot be
    attributed to anything. The working read stays dense -- it is context,
    not origination -- and is computed under its *own* softmax so that a
    handful of gated intent slots are not made to compete for attention
    mass against every working slot.
    """

    def __init__(
        self,
        hidden_size: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
    ) -> None:
        """Initialise the origination generator.

        Args:
            hidden_size: Hidden dimensionality.
            top_k: Number of intent slots each query token may read. Values
                at or above the bank size make the gate dense.
            aux_loss_weight: Weight on the gate's load-balancing loss,
                which stops one slot from winning every query.

        Raises:
            ValueError: If ``top_k`` is not positive or ``aux_loss_weight``
                is negative.
        """
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        if aux_loss_weight < 0.0:
            raise ValueError(
                f"aux_loss_weight must be non-negative, got {aux_loss_weight}."
            )
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Gate diagnostics, overwritten every forward. The attribution
        # and collapse probes read these instead of re-deriving the
        # routing.
        self.last_gate_logits: Tensor | None = None
        self.last_gate_weights: Tensor | None = None
        self.last_gate_mask: Tensor | None = None
        self.last_aux_loss: Tensor = torch.zeros(())
        self.last_intent_read: Tensor | None = None
        self.last_working_read: Tensor | None = None

    def gate_load_balancing_loss(self, gate_logits: Tensor) -> Tensor:
        """Return the load-balancing loss for the intent gate.

        Args:
            gate_logits: Tensor of shape ``(num_queries, intent_tokens)``.

        Returns:
            Scalar tensor.
        """
        num_slots = gate_logits.shape[-1]
        probs = torch.softmax(gate_logits, dim=-1)
        prob_per_slot = probs.sum(dim=0) / gate_logits.shape[0]
        return load_balancing_loss(
            gate_logits, prob_per_slot, num_slots, self.aux_loss_weight
        )

    def forward(
        self,
        intent: Tensor,
        working: Tensor,
        observation: Tensor,
    ) -> Tensor:
        """Generate the next input stream.

        Args:
            intent: Origination bank of shape ``(intent_tokens,
                hidden_size)``.
            working: Working bank of shape ``(working_tokens,
                hidden_size)``.
            observation: Current input stream of shape ``(batch, tokens,
                hidden_size)``, used for the query positions only.

        Returns:
            Tensor of shape ``(batch, tokens, hidden_size)``.

        Raises:
            ValueError: If ``observation`` is not 3D or the hidden sizes
                disagree.
        """
        if observation.dim() != 3:
            raise ValueError(
                f"observation must be 3D (batch, seq, hidden), got "
                f"{tuple(observation.shape)}."
            )
        if intent.dim() != 2 or working.dim() != 2:
            raise ValueError(
                f"intent and working must be 2D (tokens, hidden), got "
                f"{tuple(intent.shape)} and {tuple(working.shape)}."
            )
        if (
            intent.shape[-1] != self.hidden_size
            or working.shape[-1] != self.hidden_size
            or observation.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                f"hidden size mismatch: head={self.hidden_size}, "
                f"intent={intent.shape[-1]}, working={working.shape[-1]}, "
                f"observation={observation.shape[-1]}."
            )
        batch = observation.shape[0]
        num_intent = intent.shape[0]
        queries = self.q_proj(observation)
        intent_context = intent.unsqueeze(0).expand(batch, -1, -1)
        working_context = working.unsqueeze(0).expand(batch, -1, -1)

        # The intent read and the working read get their own softmax. A
        # single softmax over the concatenation would put the few gated
        # intent slots in direct competition with every working slot, so
        # with 2 gated slots against 64 dense ones the origination would
        # receive about 3% of the attention mass and `G` would effectively
        # ignore the bank it is supposed to be driven by.
        intent_scores = torch.matmul(
            queries, self.k_proj(intent_context).transpose(-1, -2)
        ) / (self.hidden_size**0.5)
        keep = top_k_mask(intent_scores, min(self.top_k, num_intent))
        intent_weights = torch.softmax(
            intent_scores.masked_fill(~keep, float("-inf")), dim=-1
        )
        intent_read = torch.matmul(intent_weights, self.v_proj(intent_context))

        working_scores = torch.matmul(
            queries, self.k_proj(working_context).transpose(-1, -2)
        ) / (self.hidden_size**0.5)
        working_read = torch.matmul(
            torch.softmax(working_scores, dim=-1),
            self.v_proj(working_context),
        )

        flat_logits = intent_scores.reshape(-1, num_intent)
        self.last_gate_logits = flat_logits
        self.last_gate_weights = intent_weights
        self.last_gate_mask = keep
        self.last_aux_loss = self.gate_load_balancing_loss(flat_logits)
        self.last_intent_read = intent_read
        self.last_working_read = working_read
        generated: Tensor = self.out_proj(intent_read + working_read)
        return generated


class ProjectionHeads(nn.Module):
    """Bundle of all four projection heads."""

    def __init__(self, config: HeadConfig | None = None) -> None:
        """Initialise the bundle.

        Args:
            config: Optional head configuration. Defaults to
                :class:`HeadConfig` defaults.
        """
        super().__init__()
        if config is None:
            config = HeadConfig()
        self.config = config
        self.language = LanguageHead(config.hidden_size, config.vocab_size)
        self.planning = PlanningHead(config.hidden_size, config.num_plan_tokens)
        self.tool = ToolHead(config.hidden_size, config.num_tools)
        self.memory = MemoryHead(config.hidden_size, config.memory_query_dim)
        # Input-reconstruction head (LeWM-style capacity bottleneck).
        self.input_reconstruct = InputReconstructionHead(
            config.hidden_size, config.reconstruction_dim
        )
        # The origination generator. Deliberately *not* part of
        # ``forward``: it reads the intent bank and the input stream
        # rather than working memory alone, so it cannot share the head
        # contract. The reasoning loop calls it directly.
        self.origination = OriginationHead(
            config.hidden_size,
            top_k=config.origination_top_k,
            aux_loss_weight=config.origination_aux_loss_weight,
        )
        # Refreshes the origination state per iteration so it is not the
        # same constant before every action.
        self.intent_update = IntentUpdate(
            config.hidden_size, scale=config.intent_update_scale
        )

    def forward(self, working_memory: Tensor) -> dict[str, Tensor]:
        """Run every head on ``working_memory``.

        Args:
            working_memory: Tensor of shape
                ``(batch, seq, hidden_size)``.

        Returns:
            Dict mapping head name to its output tensor.
        """
        return {
            "language": self.language(working_memory),
            "planning": self.planning(working_memory),
            "tool": self.tool(working_memory),
            "memory": self.memory(working_memory),
            "input_reconstruct": self.input_reconstruct(working_memory),
        }

    def head_outputs(self, working_memory: Tensor) -> dict[str, Tensor]:
        """Alias for :meth:`forward` matching the spec's vocabulary."""
        outputs: dict[str, Tensor] = self.forward(working_memory)
        return outputs


__all__ = [
    "HeadConfig",
    "InputReconstructionHead",
    "IntentUpdate",
    "LanguageHead",
    "MemoryHead",
    "OriginationHead",
    "PlanningHead",
    "ProjectionHeads",
    "ToolHead",
]


def collect_head_outputs(  # internal: convenient accessor for tests
    heads: ProjectionHeads, working_memory: Tensor
) -> dict[str, Tensor]:
    """internal: run all heads and return their outputs."""
    outputs: dict[str, Tensor] = heads(working_memory)
    return outputs


def head_parameter_count(heads: ProjectionHeads) -> dict[str, int]:
    """internal: per-head parameter count, useful for diagnostics."""
    return {
        "language": sum(p.numel() for p in heads.language.parameters()),
        "planning": sum(p.numel() for p in heads.planning.parameters()),
        "tool": sum(p.numel() for p in heads.tool.parameters()),
        "memory": sum(p.numel() for p in heads.memory.parameters()),
    }
