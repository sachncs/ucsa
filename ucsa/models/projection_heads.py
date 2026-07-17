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
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


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
    """

    hidden_size: int = 128
    vocab_size: int = 50257
    num_plan_tokens: int = 64
    num_tools: int = 32
    memory_query_dim: int = 64
    reconstruction_dim: int = 0  # 0 = inherit hidden_size

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
