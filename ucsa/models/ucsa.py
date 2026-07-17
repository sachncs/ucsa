"""Top-level UCSA model.

:class:`UCSA` composes every subsystem:

- :class:`~ucsa.models.perception.Perception` -- text/code in,
  observation tokens out.
- :class:`~ucsa.models.state.PersistentCognitiveState` -- the
  six-bank persistent state.
- :class:`~ucsa.models.transition_operator.StateTransitionOperator`
  -- the abstract computation engine.
- :class:`~ucsa.models.reasoning_loop.ReasoningLoop` -- N iterations of
  the operator per forward pass.
- :class:`~ucsa.models.projection_heads.ProjectionHeads` -- four
  independent heads reading from working memory.
- :class:`~ucsa.models.memory.Memory` -- hierarchical memory facade.
- :class:`~ucsa.models.memory_service.MemoryService` -- background
  memory worker.
- :class:`~ucsa.models.graph_service.GraphService` -- background graph
  memory.

The forward pass is intentionally minimal: text in, language logits
out. Auxiliary outputs (memory, JEPA predictions) are exposed via
explicit methods.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ucsa.models.graph_service import GraphService
from ucsa.models.head_config import (
    build_head_config_from_cfg,
)
from ucsa.models.memory import Memory
from ucsa.models.memory_service import MemoryService
from ucsa.models.moe import MoEConfig
from ucsa.models.perception import Perception, PerceptionConfig
from ucsa.models.projection_heads import ProjectionHeads
from ucsa.models.reasoning_loop import ReasoningLoop, ReasoningLoopConfig
from ucsa.models.state import PCSConfig, PersistentCognitiveState
from ucsa.models.transformer_operator import (
    TransformerOperator,
    TransformerOperatorConfig,
)
from ucsa.models.transition_operator import StateTransitionOperator
from ucsa.models.verification import HeuristicVerifier, Verifier


@dataclass
class UCSAConfig:
    """Top-level configuration for :class:`UCSA`.

    Attributes:
        hidden_size: Hidden dimensionality of every token.
        num_layers: Transformer layers.
        num_q_heads: Query attention heads.
        num_kv_heads: Key/value heads.
        intermediate_size: FFN intermediate size.
        sliding_window: KV cache sliding window.
        reasoning_iterations: Iterations per forward pass.
        vocab_size: Tokeniser vocabulary size.
        max_seq_len: Maximum observation sequence length.
        num_concepts: Number of graph concepts.
        moe: Optional MoE configuration.
    """

    hidden_size: int = 128
    num_layers: int = 4
    num_q_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 256
    sliding_window: int = 4096
    reasoning_iterations: int = 4
    vocab_size: int = 50257
    max_seq_len: int = 1024
    num_concepts: int = 16
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    ffn_dropout: float = 0.0
    moe: MoEConfig | None = None
    # TC-JEPA text conditioner config; arXiv 2605.03245 (May 2026).
    text_conditioner_top_k: int = 4
    text_conditioner_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {self.hidden_size}."
            )
        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {self.num_layers}."
            )
        if self.reasoning_iterations <= 0:
            raise ValueError(
                f"reasoning_iterations must be positive, "
                f"got {self.reasoning_iterations}."
            )


class UCSA(nn.Module):
    """Top-level UCSA model."""

    def __init__(
        self,
        config: UCSAConfig | None = None,
        operator: StateTransitionOperator | None = None,
        perception: Perception | None = None,
        heads: ProjectionHeads | None = None,
        verifier: Verifier | None = None,
        memory_service: MemoryService | None = None,
        graph_service: GraphService | None = None,
    ) -> None:
        """Initialise the UCSA model.

        Args:
            config: Optional :class:`UCSAConfig`.
            operator: Optional pre-built operator.
            perception: Optional pre-built perception module.
            heads: Optional pre-built projection heads.
            verifier: Optional verifier for memory.
            memory_service: Optional pre-built memory service.
            graph_service: Optional pre-built graph service.
        """
        super().__init__()
        if config is None:
            config = UCSAConfig()
        self.config = config
        self.pcs = PersistentCognitiveState(
            PCSConfig(hidden_size=config.hidden_size)
        )
        self.perception = perception or Perception(
            PerceptionConfig(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )
        )
        if operator is None:
            operator = TransformerOperator(
                TransformerOperatorConfig(
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    num_q_heads=config.num_q_heads,
                    num_kv_heads=config.num_kv_heads,
                    intermediate_size=config.intermediate_size,
                    sliding_window=config.sliding_window,
                    vocab_size=config.vocab_size,
                    moe=config.moe,
                    attention_dropout=config.attention_dropout,
                    residual_dropout=config.residual_dropout,
                    ffn_dropout=config.ffn_dropout,
                )
            )
        self.operator = operator
        self.reasoning_loop = ReasoningLoop(
            operator=self.operator,
            config=ReasoningLoopConfig(
                num_iterations=config.reasoning_iterations,
                capture_intermediates=True,  # ponytail: needed for JEPA aux loss; intermediates are detached clones, no extra graph
            ),
        )
        self.heads = heads or ProjectionHeads(
            build_head_config_from_cfg(config)
        )
        self.memory = Memory(self.pcs)
        self.verifier = verifier or HeuristicVerifier()
        self.memory_service = memory_service or MemoryService(
            self.memory, self.verifier
        )
        self.graph_service = graph_service or GraphService(
            num_concepts=config.num_concepts
        )
        # ponytail: memory_baseline is a non-trainable buffer that the trainer
        # refreshes every N steps. MemoryStabilityLoss uses it as the reference
        # so the loss is meaningful (MSE(long_term, baseline) instead of 0).
        self.register_buffer(
            "memory_baseline",
            torch.zeros(config.hidden_size),
            persistent=False,
        )
        self._baseline_initialised: bool = False
        # TC-JEPA (arXiv 2605.03245, May 2026, Meta): sparse cross-attention
        # text conditioner modulates the JEPA prediction. We treat the
        # model's own input-token embeddings as the conditioning source —
        # natural for a text-only LM, equivalent in spirit to captions.
        self.text_conditioner_top_k: int = max(
            1, getattr(config, "text_conditioner_top_k", 4)
        )
        self.text_conditioner_scale: float = getattr(
            config, "text_conditioner_scale", 0.1
        )
        self._text_q = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self._text_k = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self._text_v = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self._text_o = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )

    def forward(
        self,
        inputs: Tensor,
        modality: int = 0,
    ) -> dict[str, Tensor]:
        """Run a forward pass.

        Args:
            inputs: Token ids of shape ``(batch, seq)``.
            modality: Modality tag.

        Returns:
            Dict with at least ``language`` (logits), plus JEPA/memory/MoE
            aux outputs: ``jepa_predicted``, ``jepa_target``, ``long_term``,
            ``router_logits``. Missing keys indicate aux features weren't
            produced this step (caller should handle None/dummy).
        """
        observation = self.perception.forward_from_ids(
            inputs.to(self.pcs.get_bank("working").device), modality=modality
        )
        new_pcs = self.reasoning_loop(self.pcs, observation)
        heads_out = self.heads(new_pcs.get_bank("working").unsqueeze(0))

        # JEPA aux: predict next working-state from previous.
        # Intermediates are captured clones; pick last two iterations.
        intermediates = self.reasoning_loop.get_intermediates()
        jepa_predicted: Tensor | None = None
        jepa_target: Tensor | None = None
        if len(intermediates) >= 2:
            jepa_predicted = intermediates[-1]
            jepa_target = intermediates[-2]
        elif len(intermediates) == 1:
            jepa_predicted = intermediates[0]
            jepa_target = new_pcs.get_bank("working").detach()

        # TC-JEPA: condition the predicted embedding on input tokens via a
        # sparse (top-k) cross-attention. Queries = JEPA prediction;
        # keys/values = input token embeddings. We keep the scale small
        # (text_conditioner_scale, default 0.1) so the conditioner
        # refines the prediction without dominating it.
        if jepa_predicted is not None:
            token_embeds = self.perception.embed_tokens(
                inputs.to(self.pcs.get_bank("working").device)
            )
            q = self._text_q(jepa_predicted.unsqueeze(0)).unsqueeze(1)
            k = self._text_k(token_embeds).unsqueeze(1)
            v = self._text_v(token_embeds).unsqueeze(1)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (
                token_embeds.shape[-1] ** 0.5
            )
            topk = scores.topk(
                min(self.text_conditioner_top_k, scores.shape[-1]), dim=-1
            )
            attn = torch.zeros_like(scores)
            attn.scatter_(-1, topk.indices, 1.0)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            conditioned = torch.matmul(attn, v).squeeze(1)
            jepa_predicted = jepa_predicted + self.text_conditioner_scale * (
                self._text_o(conditioned).squeeze(1)
            )

        # Long-term memory + MoE router logits are direct reads.
        long_term = new_pcs.get_bank("long_term")
        router_logits = self.operator.last_router_logits

        # Lazy-init the memory baseline the first time we see a tensor.
        if not self._baseline_initialised and long_term.numel() > 0:
            self.memory_baseline = long_term.detach().mean(dim=0).clone()
            self._baseline_initialised = True

        heads_out["jepa_predicted"] = jepa_predicted
        heads_out["jepa_target"] = jepa_target
        heads_out["long_term"] = long_term
        heads_out["router_logits"] = router_logits
        return heads_out

    def start_memory_service(self) -> None:
        """Start the background memory worker."""
        self.memory_service.start()

    def stop_memory_service(self) -> None:
        """Stop the background memory worker."""
        self.memory_service.stop()


__all__ = ["UCSA", "UCSAConfig"]


def build_ucsa(cfg: Any) -> UCSA:
    """Build a :class:`UCSA` model from a Hydra/OmegaConf config.

    Args:
        cfg: Hydra/OmegaConf config, :class:`UCSAConfig`, or a plain dict.

    Returns:
        An initialised :class:`UCSA`.
    """
    if isinstance(cfg, UCSAConfig):
        return UCSA(cfg)
    try:
        from omegaconf import DictConfig, OmegaConf
    except Exception:
        DictConfig = None  # type: ignore[assignment]
    if DictConfig is not None and isinstance(cfg, DictConfig):
        merged = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(merged, dict)
        cfg = merged
    if not isinstance(cfg, Mapping):
        raise TypeError(f"Unsupported config type: {type(cfg)}.")
    moe_cfg: MoEConfig | None = None
    model_section = cfg.get("model", {}) if isinstance(cfg, Mapping) else {}
    moe_section = (
        model_section.get("moe") if isinstance(model_section, Mapping) else None
    )
    if isinstance(moe_section, Mapping):
        moe_cfg = MoEConfig(
            num_experts=int(moe_section.get("num_experts", 4)),
            top_k=int(moe_section.get("top_k", 2)),
            capacity_factor=float(moe_section.get("capacity_factor", 1.25)),
            aux_loss_weight=float(moe_section.get("aux_loss_weight", 0.01)),
        )
    hidden_size = int(model_section.get("hidden_size", 128))
    reasoning_iterations = int(cfg.get("reasoning_iterations", 4))
    vocab_size = int(model_section.get("vocab_size", 50257))
    max_seq_len = int(model_section.get("max_seq_len", 1024))
    num_concepts = int(model_section.get("num_concepts", 16))
    return UCSA(
        UCSAConfig(
            hidden_size=hidden_size,
            num_layers=int(model_section.get("num_layers", 4)),
            num_q_heads=int(model_section.get("num_q_heads", 4)),
            num_kv_heads=int(model_section.get("num_kv_heads", 2)),
            intermediate_size=int(model_section.get("intermediate_size", 256)),
            sliding_window=int(model_section.get("sliding_window", 4096)),
            reasoning_iterations=reasoning_iterations,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_concepts=num_concepts,
            attention_dropout=float(model_section.get("attention_dropout", 0.0)),
            residual_dropout=float(model_section.get("residual_dropout", 0.0)),
            ffn_dropout=float(model_section.get("ffn_dropout", 0.0)),
            moe=moe_cfg,
            text_conditioner_top_k=int(
                model_section.get("text_conditioner_top_k", 4)
            ),
            text_conditioner_scale=float(
                model_section.get("text_conditioner_scale", 0.1)
            ),
        )
    )


def build_ucsa_from_hydra(overrides: list[str] | None = None) -> UCSA:
    """Build a :class:`UCSA` from the default Hydra config with overrides.

    Args:
        overrides: Optional list of Hydra-style CLI overrides.

    Returns:
        An initialised :class:`UCSA`.
    """
    import os

    try:
        from hydra import compose, initialize_config_dir
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Hydra is required for build_ucsa_from_hydra."
        ) from exc
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "configs")
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="default", overrides=overrides or [])
    return build_ucsa(cfg)
