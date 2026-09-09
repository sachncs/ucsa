---
layout: default
title: Architecture
permalink: /docs/architecture/
nav_order: 2
---

# Architecture

This page documents the design decisions behind UCSA. It is meant
to be read alongside the source: each section points at the module
that implements it. For a tour of the public surface, see the
[API reference](api-reference.md). For an end-to-end walkthrough,
see the [Tutorials](tutorials.md).

## The persistent cognitive state

UCSA has a single, persistent, differentiable representation: the
PCS. The PCS is a stack of seven learnable tensors, one per bank,
all of shape `(num_tokens, hidden_size)`. Every operator in the
system reads from and writes to the PCS; no other structure stores
knowledge.

| Bank | Default tokens | Role |
| --- | --- | --- |
| `working` | 64 | Scratch space mutated by every reasoning step. |
| `long_term` | 128 | Accepted knowledge, retained across requests. |
| `goal` | 16 | Holds the active objective. |
| `episode` | 32 | Per-request context buffer. |
| `task` | 16 | Long-running task state. |
| `memory_index` | 32 | Retrieval index, cross-attended each block. |
| `intent` | 16 | Origination signal for the next input. |

Implementation: `ucsa/models/state.py`.

Each long-term memory token carries four scalar metadata fields —
`importance`, `usage`, `age`, `retention_score` — that drive the
recycle policy. See `retention_score` in `ucsa/models/state.py` for
the formula.

The `intent` bank is last in `BANK_NAMES` by design. The operator
derives token offsets and bank ids from that order, and
checkpoints encode the ids, so appending a new bank leaves every
existing bank's offset and id untouched.

## The state transition operator

The state transition operator is the only computation engine in
the system. It maps the current PCS and a new observation to a
new PCS:

```
C_{t+1} = F(C_t, O_t)
```

`ucsa.models.transition_operator.StateTransitionOperator` is the
abstract base. The reference implementation is
`ucsa.models.transformer_operator.TransformerOperator`: a pre-norm
transformer with grouped-query attention, sliding-window KV cache,
optional cross-attention from the `memory_index` bank, and a
Mixture of Experts FFN on the upper half of layers.

Alternative implementations (Mamba, RWKV, Hyena, SSMs) plug in by
satisfying the interface. The reasoning loop, memory pipeline, and
projection heads work against the interface and require no changes
when the operator is swapped.

## The reasoning loop

A forward pass through UCSA does three things:

1. Inject the observation into `working`.
2. For `N` iterations (default 4): `C' = F(C, O_k)`.
3. Project `working` through the heads to produce outputs.

Implementation: `ucsa/models/reasoning_loop.py`.

The operator's write-back copies under `torch.no_grad`, so the
loop also carries the operator's *differentiable* bank tensors
from one iteration to the next and out to the heads
(`ReasoningLoop.differentiable_bank`). Without that carry the
autograd graph ends at every transition and no loss can reach the
operator at all. `differentiable_state_carry=False` restores the
older severed behaviour for ablations.

## Endogenous origination

`O_k` need not be the same exogenous observation every iteration.
The origination mix is:

```
O_{k+1} = (1 - alpha_k) * G(intent, working_k) + alpha_k * O_0
alpha_k = observation_mix * observation_mix_decay ** k
```

`G` is the `OriginationHead`: a cross-attention read whose keys and
values come from `concat(intent, working)` and whose queries come
from the current input stream. This makes the signal that
*causes* the next state an explicit, addressable variable rather
than something smeared across the operator's weights — the
analogue of the neural signal that precedes a reach.

`alpha_k` is the brake on the loop running away on its own output.
The defaults (`observation_mix=1.0`, `observation_mix_decay=1.0`)
hold `alpha_k = 1`, never call `G`, and reproduce the previous
behaviour exactly, so origination is opt-in.

Implementation: `ucsa/models/origination.py`,
`ucsa/models/projection_heads.py` (the `OriginationHead` itself).

### Inference-time intent descent

`ucsa.models.intent_descent.optimize_intent` runs `K` steps of
gradient descent on the `intent` bank alone, weights frozen,
against the multi-step JEPA chain with EMA targets. `K=0` by
default. Three properties matter:

- Every candidate is evaluated from the **same** PCS. A forward
  pass rewrites the banks, so speculative rollouts must be
  restored or the measured improvement is state drift.
- **EMA targets are required.** Without them both sides of each
  JEPA pair move with the origination and the objective is
  self-consistency, not prediction.
- `grad_norm_relative_threshold` stops against the input's own
  starting gradient, which is what makes the step count — and
  therefore the cost — vary per input. An absolute threshold does
  not adapt.

## Memory pipeline

```
Working  --(propose)-->  Candidate  --(verify)-->  LongTerm
```

Verification, consolidation, and pruning run in the
`ucsa.models.memory_service.MemoryService` background worker.
Inference never blocks on memory.

### Verifier

Two implementations behind a shared interface
(`ucsa.models.verification.Verifier`):

- **`HeuristicVerifier`** (default). Score blends confidence,
  novelty (1 − cosine similarity to nearest long-term memory),
  recency, and usage.
- **`LearnedVerifier`**. Small MLP head trained on the retention
  signal: whether each candidate was re-accessed in subsequent
  steps.

### Graph memory

Background knowledge organisation. The graph is **never attended**
by the transformer. Concepts are clusters of long-term memory
embeddings (cosine k-means in pure torch). Edges come from
co-activation. Retrieval projects relevant concept nodes back into
memory tokens, which the memory service injects into `working` at
the start of the next reasoning pass.

Implementation: `ucsa/models/graph_service.py`.

## JEPA

JEPA is an auxiliary training objective, not a separate model. Two
heads contribute to `L_JEPA`:

- **Block-level**: every 8th transformer block predicts the
  working-memory output of the next block.
- **Iteration-level**: at the end of each reasoning iteration,
  predict the next iteration's working memory.

Both use cosine similarity + SmoothL1.

When the **hard-EMA target encoder** is active (default
`momentum=0.996`), the trainer swaps the JEPA targets for the EMA
model's intermediates, which keeps the prediction chain aligned
with EMA-tracked latents.

Implementation: `ucsa/models/losses.py` (`JEPALoss`),
`ucsa/training/ema.py` (`EMATargetEncoder`).

## Projection heads

Four independent heads, all reading from `working`:

| Head | Output |
| --- | --- |
| `LanguageHead` | Vocabulary logits. |
| `PlanningHead` | Discrete plan tokens. |
| `ToolHead` | Discrete tool tokens. |
| `MemoryHead` | Memory query embeddings. |

Plus the auxiliary heads:

- **`InputReconstructionHead`** — LeWM-style capacity bottleneck.
  Predicts the input-token embeddings from the working memory;
  the prediction is bounded by `reconstruction_dim` so the model
  cannot trivially copy.
- **`OriginationHead`** — generates the next iteration's input
  from the `intent` bank and the working memory. Held out of
  `forward`; the reasoning loop calls it directly.
- **`IntentUpdate`** — refreshes the `intent` bank per iteration
  so it is not the same constant before every action.

Implementation: `ucsa/models/projection_heads.py`.

## Training losses

```
L = L_AR + 0.1 * L_JEPA + 0.01 * L_MEMORY + 0.01 * L_ROUTER
      + 0.1 * L_RECONSTRUCTION + 0.01 * L_ORIGINATION
```

Every weight is configurable via `LossWeights` in
`ucsa/models/losses.py`. Each ablation flag on
`scripts/train.py` is a one-line override that zeroes the
relevant weight.

## Curriculum

Four stages, step-gated:

1. Language modelling only.
2. Language + JEPA.
3. Language + JEPA + memory.
4. Joint training.

The stage gates are configurable per run; the defaults are
`stage_1_end=2000`, `stage_2_end=4500`, `stage_3_end=6500`.
`--no-curriculum` sets all three to 1, so every loss is on from
step 1 onward.

Implementation: `ucsa/training/curriculum.py`.

## Reproducibility guarantees

- **Deterministic seeding** via `ucsa.utils.seed.set_seed`. Seeds
  Python `random`, NumPy, and PyTorch (CPU and the active GPU
  backend).
- **Safetensors checkpoints** with full config metadata. The
  pre-Phase-11 checkpoint format is auto-adapted on load via
  `adapt_legacy_state_dict`.
- **Hydra / OmegaConf config composition.** Every run is
  reproducible from a config dump; every CLI override is
  recorded in the run JSON.
- **Deterministic eval harness.** Streaming task loaders are
  paired with a fixed-seed shuffle buffer; the `max_examples`
  cap selects a deterministic prefix. The seed is recorded on
  every `EvalResult`.
- **pytest + ruff + black** run in CI on Python 3.11 and 3.12.
  Coverage is enforced at 80%.

## Why a state-centric architecture

The dominant 2024–2026 research line assumes a fixed transformer
architecture and varies only the size and the data. UCSA inverts
that assumption: the structural commitment is to a single,
persistent, differentiable state, and the transformer is one
realisation of an interchangeable transition operator. The state
itself becomes the object of study — retention, recycle, the
intent bank's localisation, the multi-step JEPA chain's
predictive alignment — rather than a side-effect of the
operator's weights.

See [paper/PAPER.md](../paper/PAPER.md) for the full write-up.

## Where to go next

- [API reference →](api-reference.md) for the module tour.
- [Tutorials →](tutorials.md) for end-to-end walkthroughs.
- [Contributing →](contributing.md) for the developer workflow.
