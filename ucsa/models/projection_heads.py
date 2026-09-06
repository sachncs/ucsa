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
    """

    hidden_size: int = 128
    vocab_size: int = 50257
    num_plan_tokens: int = 64
    num_tools: int = 32
    memory_query_dim: int = 64
    reconstruction_dim: int = 0  # 0 = inherit hidden_size
    origination_top_k: int = 2
    origination_aux_loss_weight: float = 0.01

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
        if self.reconstruction_dim <= 0:
            # ponytail: default the reconstruction dim to hidden_size
            # so reconstructed token embeddings line up with
            # perception.embed_tokens for element-wise loss.
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
        return self.proj(working_memory)


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
        return self.proj(working_memory)


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
        return self.proj(working_memory)


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
        return self.proj(working_memory)


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
        return self.proj(working_memory)


class OriginationHead(nn.Module):
    """Generate the next iteration's input from the origination state.

    This is ``G`` in ``obs_{k+1} = (1 - alpha_k) * G + alpha_k * obs``. It
    is the one place where the signal that *causes* the next state is
    computed, which is what makes that signal addressable: an intent slot
    either contributes here or it does not affect the next input at all.

    The generated stream is a cross-attention read whose keys and values
    come from ``concat(intent, working)`` -- the spec's ``G(intent,
    working)`` -- and whose queries come from the current input stream.
    The query side exists because the generated tensor has to line up with
    a variable-length observation: 16 intent slots cannot by themselves say
    how many input tokens to emit, or in what order. No observation content
    reaches the output except through the attention weights, so the mix in
    :class:`~ucsa.models.reasoning_loop.ReasoningLoop` remains the only
    path by which the real observation survives.

    The intent side of the attention is routed through a **top-k sparse
    gate**, reusing the routing machinery in :mod:`ucsa.models.moe` with
    intent slots in place of experts. The sparsity is the point: with a
    dense read every slot contributes a little to every action and the
    origination cannot be attributed to anything. The working side stays
    dense -- it is context, not origination.
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
        # ponytail: gate diagnostics, overwritten every forward. The
        # attribution and collapse probes read these instead of
        # re-deriving the routing.
        self.last_gate_logits: Tensor | None = None
        self.last_gate_weights: Tensor | None = None
        self.last_gate_mask: Tensor | None = None
        self.last_aux_loss: Tensor = torch.zeros(())

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
        context = torch.cat([intent, working], dim=0)
        context = context.unsqueeze(0).expand(batch, -1, -1)
        queries = self.q_proj(observation)
        keys = self.k_proj(context)
        values = self.v_proj(context)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / (
            self.hidden_size**0.5
        )
        # Sparsify the intent slots only; working memory stays dense.
        intent_scores = scores[..., :num_intent]
        keep = top_k_mask(intent_scores, min(self.top_k, num_intent))
        masked = scores.masked_fill(
            torch.cat(
                [~keep, torch.zeros_like(scores[..., num_intent:], dtype=torch.bool)],
                dim=-1,
            ),
            float("-inf"),
        )
        weights = torch.softmax(masked, dim=-1)
        flat_logits = intent_scores.reshape(-1, num_intent)
        self.last_gate_logits = flat_logits
        self.last_gate_weights = weights[..., :num_intent]
        self.last_gate_mask = keep
        self.last_aux_loss = self.gate_load_balancing_loss(flat_logits)
        generated: Tensor = self.out_proj(torch.matmul(weights, values))
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
        # ponytail: input-reconstruction head (LeWM-style capacity bottleneck).
        self.input_reconstruct = InputReconstructionHead(
            config.hidden_size, config.reconstruction_dim
        )
        # ponytail: the origination generator. Deliberately *not* part of
        # ``forward``: it reads the intent bank and the input stream rather
        # than working memory alone, so it cannot share the head contract.
        # The reasoning loop calls it directly.
        self.origination = OriginationHead(
            config.hidden_size,
            top_k=config.origination_top_k,
            aux_loss_weight=config.origination_aux_loss_weight,
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
        return self.forward(working_memory)


__all__ = [
    "HeadConfig",
    "InputReconstructionHead",
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
    return heads(working_memory)


def head_parameter_count(heads: ProjectionHeads) -> dict[str, int]:
    """internal: per-head parameter count, useful for diagnostics."""
    return {
        "language": sum(p.numel() for p in heads.language.parameters()),
        "planning": sum(p.numel() for p in heads.planning.parameters()),
        "tool": sum(p.numel() for p in heads.tool.parameters()),
        "memory": sum(p.numel() for p in heads.memory.parameters()),
    }
