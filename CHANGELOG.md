# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ucsa.models.intent_descent`: Phase D. `optimize_intent` runs K steps of
  gradient descent on the `intent` bank alone with every weight frozen,
  against the multi-step JEPA chain as the model's own forward model.
  `K=0` by default, so nothing runs unless asked. Optional early stop on
  the intent gradient norm; `normalize_gradient` (default `True`) makes
  `learning_rate` a step size in bank-norm units, because the raw intent
  gradient is around `1e-7` and no fixed raw step size works across
  models. Every candidate is evaluated from the *same* cognitive state --
  a forward pass rewrites the PCS, and without restoring it the reported
  improvement is drift, not a better origination (measured: the first
  version showed the objective falling 0.0069 to 0.0028 purely from
  drift). `ema_outputs` restores the target encoder's banks too, for the
  same reason. `DescentReport.forward_model_gamed` and
  `outcome_correlation` are the forward-model-hacking check: predicted
  gain with realised loss.
- `ucsa.models.intent_descent.jepa_step_errors`: chain error per step, so
  the "late steps improve more than early ones" signature can be checked
  rather than assumed. Reported by `scripts/probe_origination.py`.
- `ucsa.models.intent_descent`: `grad_norm_relative_threshold`, an early
  stop measured against the input's own first-step gradient, plus
  `intent_grad_norm_relative_threshold` on
  `generate_with_intent_descent`. The absolute threshold cannot adapt:
  intent gradient norms sit in a narrow band across inputs, so across 24
  inputs any absolute value produced step counts of exactly `{1}`. The
  relative criterion produced `{2, 3, 4, 5}`, dropping latency mean from
  463 ms to 202 ms against a fixed `K=8` budget while raising its
  coefficient of variation from 0.040 to 0.243.
- `ucsa/infer.py`: `generate_with_intent_descent` and
  `sample_next_token`; `run_inference(intent_steps=...)` and
  `--intent-steps` / `--intent-learning-rate` on the CLI. `K=0` default.
  Reports carry the forward-pass count so cost travels with any claim.
- `scripts/probe_origination.py`: emits collapse, localisation, and the
  matched-compute descent sweep as one JSON report, mirroring
  `scripts/probe_banks.py`.
- `scripts/train.py`: `--observation-mix`, `--observation-mix-decay`,
  `--origination-top-k`, `--no-origination-balance`,
  `--origination-weight`, `--intent-update-scale`,
  `--stream-intent-bank`. `scripts/run_ablations.py` gains the
  `origination`, `origination-no-balance`, `origination-static-bank`,
  `origination-streamed-intent`, and `origination-dense-gate` arms, all
  at the same step count so the comparison is matched-compute.
- `ucsa.training.metrics`: the intent-collapse diagnostics --
  `intent_state_variance`, `intent_gate_usage`, `intent_gate_entropy`,
  `intent_gate_mutual_info`, `intent_read_share` -- added to
  `DEFAULT_METRIC_NAMES` alongside `jepa_loss`. `Trainer` records them
  every step, computing variance and mutual information over a rolling
  window (`TrainerConfig.intent_window_size`, default 16) because those
  two are only meaningful *across differing inputs* and consecutive
  steps see different batches.
- `ucsa.models.origination.intent_collapse_report`: the combined reading,
  with an explicit `collapsed` verdict and human-readable reasons.
- `ucsa.models.projection_heads.IntentUpdate`: refreshes the origination
  state per iteration from working memory,
  `intent_{k+1} = intent_k + scale * Attn(q=intent_k, kv=working_k)`.
  Residual on purpose so the learned or inference-time-optimised bank
  value survives. `intent_update_scale` (default 0.1) on `HeadConfig`,
  `HeadSpec`, and `UCSAConfig`; `0.0` restores the static bank.
- `ucsa.models.losses.LossWeights.origination` (default 0.01) and the
  `origination_aux_loss` term in `UCSACombinedLoss`, wired by the
  trainer.
- `ucsa.models.moe.load_balancing_loss` and `top_k_mask`: the routing
  machinery extracted as module-level functions so the origination gate
  reuses it with intent slots in place of experts.
  `MixtureOfExperts._load_balancing_loss` now delegates; the arithmetic
  is unchanged.
- `ucsa.models.projection_heads.OriginationHead`: a top-k sparse gate
  over the intent slots. The working side of the attention stays dense
  (it is context, not origination). Exposes `last_gate_logits`,
  `last_gate_weights`, `last_gate_mask`, and `last_aux_loss`.
  `HeadConfig.origination_top_k` / `origination_aux_loss_weight` and the
  matching `UCSAConfig` fields control it. `UCSA.forward` reports
  `origination_aux_loss`; it is **not** summed into the total loss,
  because wiring a regulariser before the Phase C collapse diagnostic
  exists would be tuning toward a number we cannot yet read.
- `ucsa.models.origination`: the localisation probes.
  `intent_attribution` returns a per-slot `grad x activation` map plus
  which slots the sparse gate actually routed to; `intervene_intent`
  ablates or swaps a slot and reports how far the action moved;
  `counterfactual_controllability` combines them into a controllability
  and a specificity rate. `pcs_snapshot` / `pcs_restore` make the probes
  genuinely read-only: the operator rewrites every bank on every
  iteration, so without snapshotting, a baseline and a perturbed forward
  start from different states and the measured difference is mostly that
  drift. Fixing this shrank the observed effect sizes by ~100x.
- `ucsa.models.state`: the `intent` bank (16 tokens, trainable), the
  origination signal from which the reasoning loop generates the next
  iteration's input. Appended last in `BANK_NAMES` so every existing
  bank keeps its token offset and bank id.
- `ucsa.models.projection_heads.OriginationHead`: the generator `G`. A
  cross-attention read with keys/values from `concat(intent, working)`
  and queries from the current input stream, which is what fixes the
  generated stream's token count and order. Attached to
  `ProjectionHeads.origination` but deliberately excluded from
  `ProjectionHeads.forward`, whose contract is working-memory-only.
- `ucsa.models.reasoning_loop`: `observation_mix` (`alpha_0`) and
  `observation_mix_decay` on `ReasoningLoopConfig`, plus
  `observation_weight`, `next_observation`, and `read_bank`. The loop
  feeds `O_{k+1} = (1 - alpha_k) * G + alpha_k * O_0`, records
  `last_observation_weights` and `last_generated_inputs`, and mixes
  against the original observation so the exogenous signal cannot decay
  faster than `alpha_k` says. Defaults keep `alpha_k = 1`, never call
  `G`, and reproduce the previous behaviour exactly.
- `ucsa.models.ucsa.UCSAConfig`: `observation_mix` and
  `observation_mix_decay`, plumbed through `build_ucsa`. The heads are
  now built before the loop because the loop calls `G`.
- `ucsa.utils.checkpoint.adapt_legacy_state_dict` and
  `load_state_dict_compat`: load checkpoints written before a bank
  existed. Adding a bank changes one saved shape, the operator's
  `bank_id_embedding`, which holds one row per bank plus a final
  observation row. Existing bank rows keep their index and the
  observation row is moved to the new last index; copying positionally
  would have put the observation embedding into the intent slot.
  `scripts/eval.py` and `scripts/probe_banks.py` use it and log what
  was adapted. Verified against the real 352 MB `ckpts/ucsa-step1000`.
- `scripts/probe_banks.py`: imports `BANK_NAMES` instead of hardcoding
  six names, so the intent bank is probed like any other.
- `tests`: intent-bank shape/metadata/override coverage, the legacy
  state-dict adapter, `OriginationHead` shapes and validation, the
  `alpha_k` schedule, and the strict-generalisation check that
  `alpha=1` gives the same state as attaching no generator.
- `tests/test_reasoning_loop.TestDifferentiableStateCarry`: covers
  `differentiable_bank`, differentiable JEPA intermediates, carried
  tensors reaching every operator weight, `reset` clearing the carry,
  and the fallback for operators that expose no carried state.
- `tests/test_transformer_operator`: covers the stashed bank tensors,
  the second-call carry, the `differentiable_state_carry=False`
  ablation, `reset` clearing the stash, and a backward through the
  MoE router logits that regresses the in-place cross-attention bug.
- `tests/test_scripts`: the AR loss reaches every operator parameter,
  the JEPA chain keeps predictions differentiable while targets stay
  detached, TC-JEPA preserves shapes, and the severed-carry ablation
  starves the operator.
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
- `ucsa.models.projection_heads.OriginationHead`: the intent read and the
  working read now get their *own* softmax. Sharing one softmax over the
  concatenation put a handful of gated intent slots in direct competition
  with every working slot, so with 2 gated slots against 64 dense ones
  the origination received about 3% of the attention mass and `G`
  effectively ignored the bank it is meant to be driven by. Zeroing one
  intent slot now moves `G`'s output by 39% against 2% for a working
  slot. Also stashes `last_intent_read` / `last_working_read` for the
  diagnostics.
- `ucsa.models.transformer_operator`: new `stream_intent_bank` (default
  `False`). The operator no longer attends over the `intent` bank, so the
  origination generator is the *only* path from the bank to an action.
  With the bank in the attention stream every slot reached every action
  as ordinary context, all 16 slots carried gradient, and the
  origination could not be localised at all. As a side effect the
  streamed layout is identical to the pre-intent one, so legacy
  checkpoints need no bank-id remap in the default configuration.
- `ucsa.models.transformer_operator.TransformerOperator`: stashes the
  differentiable post-transition bank tensors in `last_bank_tensors`
  before the `no_grad` PCS write-back, and consumes them on the next
  call via `carried_bank_tensors`. New
  `TransformerOperatorConfig.differentiable_state_carry` (default
  `True`) and `UCSAConfig.differentiable_state_carry` gate the
  behaviour; `False` reproduces the previous severed graph for the
  ablation. `reset` clears the stash alongside the KV cache.
- `ucsa.models.reasoning_loop.ReasoningLoop`: exposes
  `differentiable_bank(name)` and `capture_working`, so the JEPA
  intermediates carry gradients instead of being detached clones.
  Operators that do not stash carried tensors fall back to the old
  detached PCS reads, so alternative operators need no changes.
- `ucsa.models.ucsa.UCSA.forward`: reads `working` and `long_term`
  from the loop's differentiable tensors (falling back to the PCS
  parameters) so the heads, JEPA, and memory losses all reach the
  operator.
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
- `ucsa.training.eval_harness._load_winogrande` read a non-existent
  `options` key, so the WinoGrande task raised `KeyError` and never ran.
  The dataset exposes `option1`/`option2`. The loader also built its
  choices in reverse while keeping `answer - 1` as the label, which
  inverted every example, so the task would have scored anti-correlated
  had it run at all.
- `Trainer.compute_loss` padded short targets with `0` instead of with the
  loss's ignore index. The working bank is 64 slots, so a short sequence
  left most positions unsupervised, and padding them with zero trained the
  model to emit token id 0 there: **90.4% of the autoregressive loss** for
  a 6-token target came from padding. The real tokens barely mattered,
  which made every claim about what drives the emitted action
  unmeasurable, and it was the root cause of the earlier negative
  origination findings. `Trainer.ignore_index` now supplies the pad value.
- The reasoning loop was not differentiable end to end. Because
  `PersistentCognitiveState.set_bank` copies under `torch.no_grad`,
  the autograd graph was severed at every operator write-back: an AR
  loss reached only `pcs.banks.working`, the language head, and the
  reconstruction head (4 of 50 parameters on a tiny model), and none
  of the 31 operator parameters. Fitting a fixed batch for 100 steps
  went 6.46 -> 2.78 by memorising into the working bank; with the
  differentiable state carry the same run reaches 0.15 and every
  operator parameter receives gradient.
- `TransformerOperator.forward`: the `memory_index` cross-attention
  keys/values were an expanded *view* of the `memory_index`
  parameter. The in-place bank write-back bumped its version counter,
  so any backward that reached the cross-attention raised "one of the
  variables needed for gradient computation has been modified by an
  inplace operation". The tokens are now cloned, matching the
  treatment `pcs_tokens_b` already received. This crashed the
  full-curriculum step whenever MoE was enabled.
- `UCSA._condition_one`: the TC-JEPA conditioner returned `(1, S, H)`
  for an `(S, H)` prediction, so the JEPA loss silently broadcast
  against its `(S, H)` target (`UserWarning` from `smooth_l1_loss`)
  and the LeWM Gaussian regulariser reduced over the wrong axes. The
  conditioner now preserves the prediction's shape.
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