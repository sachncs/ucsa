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
| `save_checkpoint(model, path)` | Save the model state dict to a safetensors file, with optional metadata. |
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

## `ucsa.models.cognitive` — the PCS

The persistent cognitive state and its seven banks.

| Symbol | Purpose |
| --- | --- |
| `BANK_NAMES` | Tuple of the seven default bank names. |
| `INTENT_BANK` | The bank name for the origination signal. |
| `cognitive.Config` | Dataclass for hidden size, per-bank sizes, and retention-tuning weights. |
| `BankSpec` | Static spec for one bank: name and token count. |
| `cognitive.State` | The PCS module itself. Owns the seven bank tensors and their retention metadata. |
| `retention_score(...)` | Compute the retention score from `importance`, `usage`, and `age`. |

The PCS exposes:

- `get_bank(name)` → `Tensor` of shape `(num_tokens, hidden_size)`
- `get_all_tokens()` → flat `Tensor` of every bank concatenated
- `set_bank(name, tensor)` → write a new tensor to a bank
- `bank_size(name)` → `int` token count
- `recycle_bottom_k(name, k)` → run the recycle policy on the bottom-k
  long-term memory tokens

## `ucsa.models.transition` — the operator ABC

Abstract base for the state transition operator
`C_{t+1} = F(C_t, O_t)`.

| Symbol | Purpose |
| --- | --- |
| `transition.Operator` | ABC; subclasses implement `forward`, `initialize`, `reset`, and the `name` property. |

The reference implementation is `transformer.Operator`; alternative
implementations (Mamba, RWKV, Hyena, SSMs) plug in by satisfying the
interface and require no changes anywhere else.

## `ucsa.models.transformer` — the reference operator

Pre-norm transformer with grouped-query attention, sliding-window KV
cache, optional `memory_index` cross-attention, and dense or MoE FFN.

| Symbol | Purpose |
| --- | --- |
| `transformer.Config` | All operator hyperparameters. |
| `transformer.Block` | One pre-norm block (self-attn + cross-attn + FFN). |
| `RMSNorm` | Root-mean-square layer norm. |
| `RotaryEmbedding` | RoPE with cached cos/sin tables. |
| `GroupedQueryAttention` | GQA with sliding-window KV cache. |
| `CrossAttention` | Cross-attention from the `memory_index` bank. |
| `FeedForward` | SwiGLU FFN. |
| `transformer.Operator` | The full operator: stacks `num_layers` blocks, runs MoE on the upper half. |

Bank offsets are bound lazily on the first forward, since the sizes
depend on the attached PCS.

## `ucsa.models.moe` — mixture of experts

Top-k routed Mixture of Experts with a load-balancing loss.

| Symbol | Purpose |
| --- | --- |
| `moe.Config` | Number of experts, top-k, capacity factor, aux loss weight. |
| `Expert` | One expert FFN. |
| `moe.Mixture` | The full MoE module. Stashes `last_router_logits` for the trainer. |

## `ucsa.models.perception` — tokenisation and embedding

Tokenizer wrapper and the perception module that turns token ids into
the hidden-size representation consumed by the operator.

| Symbol | Purpose |
| --- | --- |
| `perception.Tokenizer` | Wraps a HuggingFace tokenizer with caching and length handling. |
| `Perception` | Embedding table + projection to the operator's hidden size. |
| `FakeTokenizer` | A char-→-id tokenizer used by the offline test suite. |

## `ucsa.models.reasoning` — N iterations of F

The reasoning loop applies the operator `N` times per forward, mixes
in the observation via `alpha_k` when origination is enabled, and
returns the differentiable bank tensors so the loss can reach the
operator.

| Symbol | Purpose |
| --- | --- |
| `reasoning.Config` | `num_iterations`, `observation_mix`, `observation_mix_decay`. |
| `reasoning.Loop` | The loop itself. Owns the operator, origination, intent-update, and differentiable carry. |

## `ucsa.models.tiers` — hierarchical memory facade

Propose / verify / accept for long-term memory tokens.

| Symbol | Purpose |
| --- | --- |
| `Memory` | Hierarchical facade; snapshot, restore, propose, accept. |

## `ucsa.models.curation` — background memory worker

Async memory pipeline that runs verification, consolidation, and
pruning on a background thread.

| Symbol | Purpose |
| --- | --- |
| `curation.Curator` | Queue-based async worker. |

## `ucsa.models.graph` — concept graph

K-means clustering of long-term memory embeddings under a pluggable distance
metric, with edges for co-activation and retrieval by concept.

| Symbol | Purpose |
| --- | --- |
| `Metric` | Abstract distance/similarity/centre definition; subclass to add one. |
| `Cosine`, `Euclidean` | Built-in metrics. |
| `register(name)` | Class decorator that makes a metric selectable by name. |
| `resolve(spec)` | Turns a name or instance into a `Metric`. |
| `Cluster` | Deterministic k-means under any metric; `fit(points)` returns a `Clustering`. |
| `Concept`, `Edge`, `Memory` | Graph data: a cluster, a co-activation link, the whole graph. |
| `Graph` | `build(state)`, `retrieve`, `retrieve_tokens` and `inject` into working memory. |

Adding a metric:

```python
@graph.register("manhattan")
class Manhattan(graph.Metric):
    def distance(self, points, centers): ...
    def similarity(self, a, b): ...
    def center(self, members): ...

graph.Graph(num_concepts=8, metric="manhattan")
```

## `ucsa.models.verification` — candidate scoring

Two implementations behind a shared interface.

| Symbol | Purpose |
| --- | --- |
| `Verifier` | ABC. |
| `verification.Heuristic` | Score blends confidence, novelty, recency, usage. |
| `verification.Learned` | Small MLP head trained on the retention signal. |

## `ucsa.models.projection` — four heads + origination

The four projection heads, plus the input-reconstruction head and
the origination generator.

| Symbol | Purpose |
| --- | --- |
| `head_config.Spec` / `projection.Config` | Builders for head hyperparameters. |
| `projection.Language` | Vocabulary logits from working memory. |
| `projection.Planning` | Discrete plan tokens. |
| `projection.Tool` | Discrete tool tokens. |
| `projection.Memory` | Memory query embeddings. |
| `projection.Reconstruction` | LeWM-style capacity bottleneck. |
| `projection.Origination` | The endogenous-origination generator. |
| `IntentUpdate` | Refreshes the `intent` bank per iteration. |
| `projection.Heads` | Bundle of all heads; the entry point. |

## `ucsa.models.losses` — the combined loss

| Symbol | Purpose |
| --- | --- |
| `losses.Weights` | Per-component weights: `ar`, `jepa`, `memory`, `router`, `reconstruction`, `origination`. |
| `losses.Autoregressive` | Vanilla cross-entropy on language logits. |
| `losses.JEPA` | Cosine similarity + SmoothL1 on the JEPA pairs. Supports single-step (I-JEPA) and multi-step (LeWM) modes. |
| `losses.MemoryStability` | MSE against the moving `memory_baseline`. |
| `losses.RouterBalance` | Switch-Transformer load-balancing loss. |
| `losses.Reconstruction` | MSE against the input-token embeddings. |
| `losses.Combined` | Weighted combination of every term above. |

## `ucsa.models.architecture` — the top-level model

| Symbol | Purpose |
| --- | --- |
| `architecture.Config` | Top-level config. |
| `UCSA` | The model: builds the PCS, the operator, the loop, the heads, the losses. |
| `architecture.build(config)` | Build a `UCSA` from a config dict. |

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
| `descent_sweep(model, pairs, steps)` | Sweep across multiple `K` (defined in `scripts/probe_origination.py`). |
| `compute_matched_comparison(model, pairs, K)` | Compare to repeat-and-average and more-reasoning controls. |

## `ucsa.training.curriculum` — the 4-stage curriculum

| Symbol | Purpose |
| --- | --- |
| `curriculum.Schedule` | `stage_1_end`, `stage_2_end`, `stage_3_end`. |
| `Curriculum` | Active-component gating by global step. |

The four stages, step-gated:

1. Language modelling only.
2. Language + JEPA.
3. Language + JEPA + memory.
4. Joint training.

## `ucsa.training.ema` — hard-EMA target encoder

| Symbol | Purpose |
| --- | --- |
| `ema.TargetEncoder` | Frozen EMA copy of the model. Buffers are not updated on purpose. |

## `ucsa.training.metrics` — metrics registry

| Symbol | Purpose |
| --- | --- |
| `metrics.Registry` | Time-averaged metrics. |
| `build_default_registry()` | Default registry with the standard UCSA metrics. |
| `intent_state_variance`, `intent_gate_entropy`, `intent_gate_mutual_info`, `intent_gate_usage`, `intent_read_share` | Per-step diagnostics. |
| `perplexity_from_loss` | Cross-entropy → perplexity. |

## `ucsa.training.optimizer` — Muon

| Symbol | Purpose |
| --- | --- |
| `Muon` | Newton-Schulz iteration on the parameter update. |

## `ucsa.training.eval_harness` — standard LM benchmarks

Prompts, metrics and splits follow lm-evaluation-harness so results compare
with published numbers.

| Symbol | Purpose |
| --- | --- |
| `TaskSpec` | One task: name, loader, `max_examples` (None = full split), seed, headline `metric`. |
| `EvalResult` | Per-task result; `extras` holds `acc`, `acc_norm` and the binomial standard error. |
| `TASK_REGISTRY` | HellaSwag and ARC-Challenge report `acc_norm`; ARC-Easy, PIQA, WinoGrande report `acc`. |
| `choice_loglik(model, tokenizer, context, choice, device)` | `(sum log-prob, n tokens)` of a choice given its context. A slot model never sees the choice. |
| `evaluate_task`, `evaluate_all` | Run one task or a list. |

## `ucsa.training.evaluation` — the eval loop

| Symbol | Purpose |
| --- | --- |
| `evaluation.Loop` | No-grad loop that produces `(inputs, targets)` batches and runs `compute_loss` for validation. |

## `ucsa.training.trainer` — the training loop

| Symbol | Purpose |
| --- | --- |
| `trainer.Config` | All trainer hyperparameters, including EMA momentum and intent-window size. |
| `trainer.State` | Mutable training state: `global_step`, `last_loss`, `last_components`, `started_at`. |
| `CosineWarmupScheduler` | Linear warmup followed by cosine decay to `min_lr_ratio` of the peak. |
| `Trainer` | The training loop: `train_step`, `train`, `save_checkpoint`, `load_checkpoint`, `compute_loss`. |

## `ucsa.training.dataset` — streaming fineweb-edu

| Symbol | Purpose |
| --- | --- |
| `dataset.Config` | Streaming config: sequence length, primary dataset, split, pack. |
| `dataset.Text` | Iterable dataset with a `pack_sequences` option. |

## UCSA-R and its tooling

See [UCSA-R](ucsa-r.md) for the design. One line per public symbol here.

### `ucsa.models.recurrent`

| Symbol | Purpose |
| --- | --- |
| `Config` | Every architecture switch, validated in order; unknown keys rejected by `from_dict`. |
| `Model` | `forward` (logits for every position), `compute_loss`, `advance` and `next_logits` (streaming), `generate`. |
| `Attention`, `Block`, `StateUpdater` | One attention class serves windowed decoding, chunk encoding and the state write. |
| `window_mask`, `shift_chunks` | The sliding-window visibility rule and the previous-chunk shift. |
| `config_for_params(n)` | Pick width and depth for about `n` parameters. |

### `ucsa.training`

| Module | Symbols |
| --- | --- |
| `engine` | `Config`, `fit` (resumable, one host sync per step), `evaluate`, `WeightEma`, `build_optimizer`, `clip_grad_norm`, `load_model`. |
| `shards` | `Shard` (deterministic, resumable batches), `write_shard` (atomic), `filter_documents` (dedupe), `filter_by_compressibility`, `compression_ratio`. |
| `prefix` | The prefix-only protocol: `split_prefix_targets`, `tail_nll`, continuation log-probs. |
| `compression` | `token_byte_lengths`, `bits_per_byte`, `anchors` (zlib, LZMA). |
| `diagnostics` | `window_nll`, `paired_bootstrap`, `chunk_profile`, `state_rate_distortion`. |
| `scaling` | `fit` and `forecast` of `L(t) = floor + A t^-alpha`, `crossing`, `paired_forecast_gap`. |
| `tuning` | `successive_halving`, `sample_candidates`. |
| `reference` | `load_reference`, `compare`, `format_table` against published results. |

### Top level

| Module | Symbols |
| --- | --- |
| `ucsa.arithmetic` | `compress`, `decompress`, `compress_text`, `decompress_text`, `ModelPredictor`, `ideal_bits`. |
| `ucsa.dryrun` | `run` (dtype audit, memory, speed, health), `Recorder`, `Report`. Never imported by training. |
| `ucsa.utils.precision` | `DTYPE` (the one float type), `configure`, `to_dtype`, `NUMPY_DTYPE`. |

## `ucsa` — top-level entrypoints

The `ucsa.train` and `ucsa.infer` modules expose `main` functions
and `build_model` / `build_trainer` / `build_optimizer` helpers
used by `scripts/train.py` and `scripts/infer.py`. They are
intentionally thin wrappers around the lower-level builders; the
scripts are the public surface.
