# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding: `pyproject.toml` with project metadata, dependencies,
  and tool configuration for ruff, black, mypy, and pytest.
- README.md with quickstart, layout, and extension guide.
- docs/architecture.md with deep design notes (PCS contract, memory pipeline,
  JEPA cadence, verifier signal flow).
- TODO.md tracking every atomic commit through Phase 10.
- Package skeleton: `ucsa/`, `ucsa/models/`, `ucsa/training/`, `ucsa/utils/`,
  `ucsa/configs/`, `tests/`.
- `ucsa.models.state`: `PersistentCognitiveState`, `PCSConfig`, `BankSpec`,
  `retention_score`, `resolve_bank_sizes`, `build_bank_specs`. Six token
  banks (working/long_term/goal/episode/task/memory_index), retention
  metadata buffers, recycle policy.
- `ucsa.models.transition_operator`: abstract `StateTransitionOperator`
  base class with `forward`, `reset`, `initialize`, and `name` contract.
- `ucsa.models.transformer_operator`: `TransformerOperator` with RMSNorm,
  rotary embeddings, grouped-query attention, sliding-window KV cache,
  `memory_index` cross-attention, gated FFN, and `MoEConfig` (FFN-only
  blocks in this commit; MoE swap-in is added in the next feat commit).
- `ucsa.models.moe`: standalone `MixtureOfExperts`, `Expert`, `MoEConfig`
  modules. Top-k routing with load-balancing auxiliary loss. The operator
  installs MoE on the upper half of layers automatically when configured.
- `ucsa.models.perception`: `Perception`, `PerceptionConfig`,
  `TokenizerWrapper`, `MODALITY_TEXT`, `MODALITY_CODE`. Replaceable
  tokenizer wrapper around Hugging Face `AutoTokenizer`, token embedding,
  modality embedding, and modality projection. Outputs observation tokens
  of shape ``(batch, seq, hidden_size)``.
- `ucsa.models.reasoning_loop`: `ReasoningLoop`, `ReasoningLoopConfig`.
  Runs the operator ``num_iterations`` times per forward pass, with
  optional capture of per-iteration working-memory snapshots for JEPA.
- `ucsa.models.memory`: `Memory`, `MemoryUpdate`. Hierarchical memory
  facade wrapping the PCS: scratch / working / episode / long-term.
  Includes candidate proposal, long-term acceptance with capacity
  enforcement, episode snapshotting, recycle-by-bottom-k, retention-
  score access, FIFO recycle, and threshold-based recycle.
- `ucsa.models.verification`: `Verifier` ABC, `HeuristicVerifier`,
  `LearnedVerifier`. Score-based blend (confidence / novelty / recency
  / usage) and an MLP head trained on the retention signal.
- `ucsa.models.memory_service`: `MemoryService`, `VerificationTask`,
  `PruneTask`, `ServiceStats`. Background asyncio worker with a
  dedicated thread, async queue, sync `submit_*` facades that drop to
  inline processing when no loop is running.
- `ucsa.models.graph_service`: `GraphService`, `CosineClusterer`,
  `Concept`, `GraphEdge`, `GraphMemory`. Pure-torch cosine k-means
  clustering of long-term memory embeddings, co-activation edge
  discovery, concept-token retrieval, and direct injection into PCS
  working memory.
- `ucsa.models.projection_heads`: `ProjectionHeads`, `LanguageHead`,
  `PlanningHead`, `ToolHead`, `MemoryHead`, `HeadConfig`. Four
  independent heads reading only from working memory.
- `ucsa.models.losses`: `UCSACombinedLoss`, `AutoregressiveLoss`,
  `JEPALoss`, `MemoryStabilityLoss`, `RouterLoadBalancingLoss`,
  `LossWeights`. Combined loss = AR + lambda1*JEPA + lambda2*Memory +
  lambda3*Router with per-component reporting.
- `ucsa.training.dataset`: `TextDataset`, `DatasetConfig`,
  `PRIMARY_DATASET`, `FALLBACK_DATASETS`. Hugging Face streaming
  dataset loader with graceful fallback chain (fineweb-edu ->
  openwebtext -> wikitext-103) and fixed-length sequence packing.
- `ucsa.training.curriculum`: `Curriculum`, `CurriculumSchedule`,
  `CurriculumStage`, `CurriculumState`. Four-stage step-gated training
  curriculum with component gating per stage.