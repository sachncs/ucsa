---
layout: default
title: API reference
permalink: /docs/api-reference/
nav_order: 3
---

# API reference

This page is a module-by-module tour of the `ucsa/` package. For
each module it lists the public surface, the key dataclasses and
protocols, and links back to the source.

The tour follows the dependency graph: leaf utilities first, then
the state, then the operator, then the loop, then the heads, then
the top-level model, then the training utilities.

## `ucsa.utils` — leaf utilities

### `ucsa.utils.seed`

Deterministic seeding across Python, NumPy, and PyTorch.

| Symbol | Purpose |
| --- | --- |
| `set_seed(seed: int)` | Seed Python `random`, NumPy, and PyTorch (CPU and the active GPU backend). |

### `ucsa.utils.checkpoint`

Save and load safetensors checkpoints with a legacy adapter for
pre-Phase-11 state dicts.

| Symbol | Purpose |
| --- | --- |
| `save_state_dict(model, path)` | Save the model state dict to a safetensors file. |
| `load_state_dict_compat(path)` | Load a state dict, remapping legacy bank-name keys via `adapt_legacy_state_dict`. |
| `adapt_legacy_state_dict(state_dict)` | Translate pre-`intent`-bank checkpoints in place. |

### `ucsa.utils.logging`

TensorBoard + Weights & Biases unified logger.

| Symbol | Purpose |
| --- | --- |
| `LoggerConfig` | Dataclass for log directory and backend flags. |
| `LoggerBundle` | Holder for the active writers. |
| `build_logger(config)` | Open the configured writers. |
| `log_metrics(bundle, metrics, step)` | Push a metrics dict to every backend. |
| `close_logger(bundle)` | Flush and close every backend. |

## `ucsa.models.state` — the PCS

The persistent cognitive state and its seven banks.

| Symbol | Purpose |
| --- | --- |
| `BANK_NAMES` | Tuple of the seven default bank names. |
| `INTENT_BANK` | The bank name for the origination signal. |
| `PCSConfig` | Dataclass for hidden size, vocabulary size, and per-bank sizes. |
| `BankSpec` | Static spec for one bank: name and token count. |
| `PersistentCognitiveState` | The PCS module itself. Owns the seven bank tensors and their retention metadata. |
| `retention_score(...)` | Compute the retention score from `importance`, `usage`, and `age`. |

The PCS exposes:

- `get_bank(name)` → `Tensor` of shape `(num_tokens, hidden_size)`
- `get_all_tokens()` → flat `Tensor` of every bank concatenated
- `set_bank(name, tensor)` → write a new tensor to a bank
- `bank_size(name)` → `int` token count
- `recycle_bottom_k(k)` → run the recycle policy on the bottom-k
  long-term memory tokens

## `ucsa.models.transition_operator` — the operator ABC

Abstract base for the state transition operator
`C_{t+1} = F(C_t, O_t)`.

| Symbol | Purpose |
| --- | --- |
| `StateTransitionOperator` | ABC; subclasses implement `forward`, `initialize`, `reset`, and the `name` property. |

The reference implementation is `TransformerOperator`; alternative
implementations (Mamba, RWKV, Hyena, SSMs) plug in by satisfying the
interface and require no changes anywhere else.

## `ucsa.models.transformer_operator` — the reference operator

Pre-norm transformer with grouped-query attention, sliding-window KV
cache, optional `memory_index` cross-attention, and dense or MoE FFN.

| Symbol | Purpose |
| --- | --- |
| `TransformerOperatorConfig` | All operator hyperparameters. |
| `TransformerBlock` | One pre-norm block (self-attn + cross-attn + FFN). |
| `RMSNorm` | Root-mean-square layer norm. |
| `RotaryEmbedding` | RoPE with cached cos/sin tables. |
| `GroupedQueryAttention` | GQA with sliding-window KV cache. |
| `CrossAttention` | Cross-attention from the `memory_index` bank. |
| `FeedForward` | SwiGLU FFN. |
| `TransformerOperator` | The full operator: stacks `num_layers` blocks, runs MoE on the upper half. |

Bank offsets are bound lazily on the first forward, since the sizes
depend on the attached PCS.

## `ucsa.models.moe` — mixture of experts

Top-k routed Mixture of Experts with a load-balancing loss.

| Symbol | Purpose |
| --- | --- |
| `MoEConfig` | Number of experts, top-k, capacity factor, aux loss weight. |
| `Expert` | One expert FFN. |
| `MixtureOfExperts` | The full MoE module. Stashes `last_router_logits` for the trainer. |

## `ucsa.models.perception` — tokenisation and embedding

Tokenizer wrapper and the perception module that turns token ids into
the hidden-size representation consumed by the operator.

| Symbol | Purpose |
| --- | --- |
| `TokenizerWrapper` | Wraps a HuggingFace tokenizer with caching and length handling. |
| `Perception` | Embedding table + projection to the operator's hidden size. |
| `FakeTokenizer` | A char-→-id tokenizer used by the offline test suite. |

## `ucsa.models.reasoning_loop` — N iterations of F

The reasoning loop applies the operator `N` times per forward, mixes
in the observation via `alpha_k` when origination is enabled, and
returns the differentiable bank tensors so the loss can reach the
operator.

| Symbol | Purpose |
| --- | --- |
| `ReasoningLoopConfig` | `num_iterations`, `observation_mix`, `observation_mix_decay`. |
| `ReasoningLoop` | The loop itself. Owns the operator, origination, intent-update, and differentiable carry. |

## `ucsa.models.memory` — hierarchical memory facade

Propose / verify / accept for long-term memory tokens.

| Symbol | Purpose |
| --- | --- |
| `Memory` | Hierarchical facade; snapshot, restore, propose, accept. |

## `ucsa.models.memory_service` — background memory worker

Async memory pipeline that runs verification, consolidation, and
pruning on a background thread.

| Symbol | Purpose |
| --- | --- |
| `MemoryService` | Queue-based async worker. |

## `ucsa.models.graph_service` — concept graph

Cosine k-means clustering of long-term memory embeddings, with
edges for co-activation and retrieval by concept.

| Symbol | Purpose |
| --- | --- |
| `CosineClusterer` | Pure-torch cosine k-means. |
| `GraphService` | Builds concepts, projects relevant nodes back into memory tokens. |

## `ucsa.models.verification` — candidate scoring

Two implementations behind a shared interface.

| Symbol | Purpose |
| --- | --- |
| `Verifier` | ABC. |
| `HeuristicVerifier` | Score blends confidence, novelty, recency, usage. |
| `LearnedVerifier` | Small MLP head trained on the retention signal. |

## `ucsa.models.projection_heads` — four heads + origination

The four projection heads, plus the input-reconstruction head and
the origination generator.

| Symbol | Purpose |
| --- | --- |
| `HeadSpec` / `HeadConfig` | Builders for head hyperparameters. |
| `LanguageHead` | Vocabulary logits from working memory. |
| `PlanningHead` | Discrete plan tokens. |
| `ToolHead` | Discrete tool tokens. |
| `MemoryHead` | Memory query embeddings. |
| `InputReconstructionHead` | LeWM-style capacity bottleneck. |
| `OriginationHead` | The endogenous-origination generator. |
| `IntentUpdate` | Refreshes the `intent` bank per iteration. |
| `ProjectionHeads` | Bundle of all heads; the entry point. |

## `ucsa.models.losses` — the combined loss

| Symbol | Purpose |
| --- | --- |
| `LossWeights` | Per-component weights: `ar`, `jepa`, `memory`, `router`, `reconstruction`, `origination`. |
| `AutoregressiveLoss` | Vanilla cross-entropy on language logits. |
| `JEPALoss` | Cosine similarity + SmoothL1 on the JEPA pairs. Supports single-step (I-JEPA) and multi-step (LeWM) modes. |
| `MemoryStabilityLoss` | MSE against the moving `memory_baseline`. |
| `RouterLoadBalancingLoss` | Switch-Transformer load-balancing loss. |
| `InputReconstructionLoss` | MSE against the input-token embeddings. |
| `UCSACombinedLoss` | Weighted combination of every term above. |

## `ucsa.models.ucsa` — the top-level model

| Symbol | Purpose |
| --- | --- |
| `UCSAConfig` | Top-level config. |
| `UCSA` | The model: builds the PCS, the operator, the loop, the heads, the losses. |
| `build_ucsa(config)` | Build a `UCSA` from a config dict. |

## `ucsa.models.origination` — probes

| Symbol | Purpose |
| --- | --- |
| `intent_attribution(model, inputs)` | `grad × activation` per intent slot. |
| `counterfactual_controllability(model, inputs)` | Ablate-and-swap intervention. |
| `intent_collapse_report(model, inputs)` | Variance, entropy, MI, read share. |

## `ucsa.models.intent_descent` — inference-time optimisation

| Symbol | Purpose |
| --- | --- |
| `optimize_intent(model, inputs, K)` | K-step gradient descent on the `intent` bank alone. |
| `descent_sweep(model, pairs, steps)` | Run a sweep across multiple `K`. |
| `compute_matched_comparison(model, pairs, K)` | Compare to repeat-and-average and more-reasoning controls. |

## `ucsa.training.curriculum` — the 4-stage curriculum

| Symbol | Purpose |
| --- | --- |
| `CurriculumSchedule` | `stage_1_end`, `stage_2_end`, `stage_3_end`. |
| `Curriculum` | Active-component gating by global step. |

The four stages, step-gated:

1. Language modelling only.
2. Language + JEPA.
3. Language + JEPA + memory.
4. Joint training.

## `ucsa.training.ema` — hard-EMA target encoder

| Symbol | Purpose |
| --- | --- |
| `EMATargetEncoder` | Frozen EMA copy of the model. Buffers are not updated on purpose. |

## `ucsa.training.metrics` — metrics registry

| Symbol | Purpose |
| --- | --- |
| `MetricsRegistry` | Time-averaged metrics. |
| `build_default_registry()` | Default registry with the standard UCSA metrics. |
| `intent_state_variance`, `intent_gate_entropy`, `intent_gate_mutual_info`, `intent_gate_usage`, `intent_read_share` | Per-step diagnostics. |
| `perplexity_from_loss` | Cross-entropy → perplexity. |

## `ucsa.training.optimizer` — Muon

| Symbol | Purpose |
| --- | --- |
| `Muon` | Newton-Schulz iteration on the parameter update. |

## `ucsa.training.eval_harness` — standard LM benchmarks

| Symbol | Purpose |
| --- | --- |
| `TaskSpec` | One task: name, loader, `max_examples`, seed. |
| `EvalResult` | Per-task result: `n`, `correct`, `accuracy`, mean log-likelihood, `extras`. |
| `TASK_REGISTRY` | The five tasks: HellaSwag, ARC-easy, ARC-challenge, PIQA, WinoGrande. |
| `evaluate_task(spec, model, tokenizer, device)` | Run one task. |
| `evaluate_all(names, model, tokenizer, device)` | Run a list of tasks. |
| `DEFAULT_EVAL_SEED` | Default seed for deterministic `max_examples` selection. |

## `ucsa.training.evaluation` — the eval loop

| Symbol | Purpose |
| --- | --- |
| `EvaluationLoop` | No-grad loop that produces `(inputs, targets)` batches and runs `compute_loss` for validation. |

## `ucsa.training.trainer` — the training loop

| Symbol | Purpose |
| --- | --- |
| `TrainerConfig` | All trainer hyperparameters, including EMA momentum and intent-window size. |
| `TrainerState` | Mutable training state: `global_step`, `last_loss`, `last_components`, `started_at`. |
| `CosineWarmupScheduler` | Linear warmup followed by cosine decay to `min_lr_ratio` of the peak. |
| `Trainer` | The training loop: `train_step`, `train`, `save_checkpoint`, `load_checkpoint`, `compute_loss`. |

## `ucsa.training.dataset` — streaming fineweb-edu

| Symbol | Purpose |
| --- | --- |
| `DatasetConfig` | Streaming config: sequence length, primary dataset, split, pack. |
| `TextDataset` | Iterable dataset with a `pack_sequences` option. |

## `ucsa` — top-level entrypoints

The `ucsa.train` and `ucsa.infer` modules expose `main` functions
and `build_model` / `build_trainer` / `build_optimizer` helpers
used by `scripts/train.py` and `scripts/infer.py`. They are
intentionally thin wrappers around the lower-level builders; the
scripts are the public surface.
