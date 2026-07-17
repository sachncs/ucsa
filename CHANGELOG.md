# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `.github/workflows/ci.yml`: GitHub Actions CI on push/PR. Runs
  ruff, black, and pytest on Python 3.11 + 3.12. Integration tests
  excluded.
- `scripts/train.py`: end-to-end training harness on fineweb-edu
  with a 4-stage curriculum, periodic held-out eval, and safetensors
  checkpoints. Default profile is 384-hidden / 6-layer UCSA-small,
  8000 steps, 0.1 dropout.
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
- `ucsa.training.metrics`: `MetricsRegistry`, `MetricState`,
  `DEFAULT_METRIC_NAMES`, helpers for perplexity, expert utilisation,
  attention entropy, throughput, GPU memory, memory utilisation, and
  replacement rate. TensorBoard logging hook.
- `ucsa.training.trainer`: `Trainer`, `TrainerConfig`,
  `CosineWarmupScheduler`, `TrainerState`. AdamW + cosine + warmup +
  AMP + grad clip + checkpoint round-trip + curriculum-aware loss
  gating.
- `ucsa.training.evaluation`: `EvaluationLoop`, `EvaluationState`.
  Periodic no-grad evaluation with running loss and perplexity.
- `ucsa.utils.seed`: `set_seed`, `get_seed`, `seed_context`. Seeds
  Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.
- `ucsa.utils.checkpoint`: `save_checkpoint`, `load_checkpoint`,
  `CheckpointMetadata`. Safetensors-backed weight save/load with
  sidecar JSON metadata.
- `ucsa.utils.logging`: `LoggerConfig`, `LoggerBundle`, `build_logger`,
  `log_metrics`, `close_logger`, `configure_logging`. TensorBoard and
  optional Weights & Biases backends behind a single API.
- `ucsa.models.ucsa`: `UCSA`, `UCSAConfig`, `build_ucsa`,
  `build_ucsa_from_hydra`. Top-level model composing perception, PCS,
  reasoning loop, projection heads, memory, memory service, and graph
  service.
- `ucsa.models.head_config`: `HeadSpec`, `build_head_config`,
  `build_head_config_from_cfg`. Helpers for building
  :class:`HeadConfig` instances.
- `ucsa.train`: training entrypoint with Hydra/OmegaConf loader and
  CLI integration via ``python -m ucsa.train``.
- `ucsa.infer`: inference entrypoint with autoregressive generation.
- `ucsa.configs.default`: YAML configuration with all hyperparameters
  externalized.
- Lint fixes: ruff clean on the entire `ucsa/` and `tests/` tree.

### Changed
- `ucsa.models.ucsa.UCSA.forward`: returns a rich dict that includes
  `jepa_predicted`, `jepa_target`, `long_term`, and `router_logits`
  alongside the head outputs. Aux losses can now be computed on real
  model state instead of dummy tensors.
- `ucsa.models.moe.MoEBlock.forward`: stashes `last_router_logits` so
  the trainer can compute the load-balancing loss from real routing
  distributions, not noise.
- `ucsa.models.transformer_operator.TransformerOperator.forward`:
  aggregates `last_router_logits` across MoE-equipped blocks so a
  single `(num_tokens, num_experts)` tensor reaches the loss.
- `ucsa.models.reasoning_loop`: `capture_intermediates=True` by
  default — the loop now always retains per-iteration working-memory
  clones for the JEPA aux loss.
- `ucsa.models.ucsa.UCSAConfig`: adds `attention_dropout`,
  `residual_dropout`, `ffn_dropout` (default 0.0; plumbed through
  to `TransformerOperatorConfig`). Effective when set non-zero.
- `ucsa.configs.default.yaml`: includes the three dropout fields
  with the project default of 0.0.
- `ucsa.training.trainer.Trainer`: picks `mps:0` (not bare `mps`)
  on Apple silicon so tensor/device equality checks line up. Also
  `compute_loss` now moves inputs/targets to the trainer's device.
  Aux-loss kwargs come from the model's output dict when present
  (with randn fallback for tests / custom models). Refreshes the
  model's `memory_baseline` buffer every 50 steps so
  `MemoryStabilityLoss` references a moving snapshot.
- `tests/test_scripts.test_forward_returns_all_heads`: asserts the
  new aux keys are present (`>=` semantics) — the strict `==` form
  would now be wrong because `UCSA.forward` returns extra keys.

### Fixed
- Trainer on MPS: `device(type='mps', 0) != device(type='mps')` no
  longer breaks `move_batch` and embedding lookups.
- `compute_loss` no longer fails on the latent path where CPU inputs
  hit a device-resident model.

### Removed
- (none)

### Added
- `ucsa.training.eval_harness` + `scripts/eval.py`: standard-LM
  eval harness (HellaSwag, ARC-easy, ARC-challenge, PIQA,
  WinoGrande) following the canonical rank-by-conditional-loglik
  protocol. Loads from a safetensors UCSA checkpoint plus an
  optional baseline JSON for matched-compute tables. Writes a
  structured JSON report to `runs/eval-*.json`. Compatible
  stream-mode dataset loaders keep CI offline-friendly.
- `scripts/train_baseline.py`: matched-compute baseline trainer.
  Vanilla GPT-2-style Transformer (GQA + RoPE + SwiGLU, no PCS
  hooks, no MoE, no JEPA) trained on the exact same fineweb-edu
  stream for the same number of steps; emits structured JSON
  matching UCSA's training output so the two can be tabled
  side-by-side.
- `scripts/run_ablations.py`: ablation matrix runner. Sweeps
  the canonical ablation flags (no-jepa, no-ema, no-recon,
  no-tc-jepa, no-curriculum) over one or more seeds, dumping each
  run to `runs/ablations/`. The summary JSON drives the
  ablation table in `paper/PAPER.md`.
- `paper/PAPER.md`: paper-grade write-up draft with proper
  abstract, method, experiments, results, and appendix sections.
  Numbers in tables are placeholders until the ablation suite
  finishes running.
- Deterministic seed control: `--seed N` on every training /
  eval script, with `utils.seed.set_seed(seed, deterministic=True)`
  wiring Python, NumPy, PyTorch, and CUDA RNGs.
- `scripts/probe_banks.py`: memory-bank probing utility. For a
  trained UCSA, projects each of the six PCS bank tokens back
  into vocabulary space via the tied LM head and reports the
  top-k tokens each bank "remembers", the per-bank L2 norm /
  retention-score statistics, and a 6x6 cosine-similarity
  matrix of bank centroids — evidence of differentiated roles.
  Exports to JSON for plotting.
- `scripts/build_paper_tables.py`: paper-table aggregator. Reads
  every JSON the training / eval / probe scripts emit under
  ``runs/`` and produces ``paper/TABLES.md`` — the four tables
  the paper promises (matched-compute comparison, ablation
  matrix, downstream benchmark accuracy, PCS bank probe) ready
  to paste into ``paper/PAPER.md``.

### Changed
- `scripts/train.py`: refactored around ablation. CLI flags
  `--no-ema`, `--no-lewm`, `--no-recon`, `--no-tc-jepa`,
  `--no-curriculum`, `--ablation NAME`, `--seed N`, and
  `--out-json PATH`. Always writes structured JSON. Default
  behavior (full SOTA stack) is unchanged.
- `scripts/benchmark.py`: adds `--seed` for parity with the rest
  of the suite.
- `README.md`: rewritten to put the paper-grade reproduction
  recipe first, with the original "why" / "layout" moved to
  the bottom and a clear "what's novel / where it lives" map.

### Tests
- 4 new tests in `tests/test_eval_harness.py` covering the
  registry shape, dataclass, empty-loader fallback, and
  fake-loader smoke through the full eval path (offline).
- 4 new tests in `tests/test_probe_banks.py` covering the
  six-bank summary shape, centroid-matrix symmetry, token
  decoding, and finite-statistics invariants.
- 7 new tests in `tests/test_build_paper_tables.py` covering
  per-function builds of all four tables, multi-seed aggregation,
  and JSON loader.
- Total: 443 / 443 tests pass.

### Changed
- `scripts/benchmark.py`: added Kimi K3 (arXiv-style) features as
  opt-in options: `--kda` (Kimi Delta Attention), `--gated-mla`
  (Gated MLA), `--stable-moe` (Stable LatentMoE), `--attn-res`
  (Attention Residuals), `--situ` (Sigmoid-Tanh Unit activation),
  `--per-head-muon`. The default baseline (no flags) is unchanged
  from the previous release. Added an in-script `Muon` import
  via `ucsa.training.optimizer` (new module).
- `scripts/train.py` no longer renamed — reverted on a follow-up
  edit. (The SOTA-comparison scaffold remains under `--baselines`.)
- `ucsa.training.optimizer`: new module providing `Muon`, the
  orthogonalised-momentum SGD variant used by Kimi K2 and DeepSeek
  V4. Newton-Schulz iteration is inline, no new dependencies.
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
- `ucsa.training.metrics`: `MetricsRegistry`, `MetricState`,
  `DEFAULT_METRIC_NAMES`, helpers for perplexity, expert utilisation,
  attention entropy, throughput, GPU memory, memory utilisation, and
  replacement rate. TensorBoard logging hook.
- `ucsa.training.trainer`: `Trainer`, `TrainerConfig`,
  `CosineWarmupScheduler`, `TrainerState`. AdamW + cosine + warmup +
  AMP + grad clip + checkpoint round-trip + curriculum-aware loss
  gating.
- `ucsa.training.evaluation`: `EvaluationLoop`, `EvaluationState`.
  Periodic no-grad evaluation with running loss and perplexity.
- `ucsa.utils.seed`: `set_seed`, `get_seed`, `seed_context`. Seeds
  Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.
- `ucsa.utils.checkpoint`: `save_checkpoint`, `load_checkpoint`,
  `CheckpointMetadata`. Safetensors-backed weight save/load with
  sidecar JSON metadata.
- `ucsa.utils.logging`: `LoggerConfig`, `LoggerBundle`, `build_logger`,
  `log_metrics`, `close_logger`, `configure_logging`. TensorBoard and
  optional Weights & Biases backends behind a single API.
- `ucsa.models.ucsa`: `UCSA`, `UCSAConfig`, `build_ucsa`,
  `build_ucsa_from_hydra`. Top-level model composing perception, PCS,
  reasoning loop, projection heads, memory, memory service, and graph
  service.
- `ucsa.models.head_config`: `HeadSpec`, `build_head_config`,
  `build_head_config_from_cfg`. Helpers for building
  :class:`HeadConfig` instances.
- `ucsa.train`: training entrypoint with Hydra/OmegaConf loader and
  CLI integration via ``python -m ucsa.train``.
- `ucsa.infer`: inference entrypoint with autoregressive generation.
- `ucsa.configs.default`: YAML configuration with all hyperparameters
  externalized.
- Lint fixes: ruff clean on the entire `ucsa/` and `tests/` tree.