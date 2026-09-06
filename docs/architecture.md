# UCSA — Architecture Notes

## Persistent Cognitive State (PCS)

A single, persistent, differentiable representation consisting of seven token
banks. Every bank is a learnable tensor of shape
``(num_tokens, hidden_size)``:

| Bank            | Tokens | Role                                            |
| --------------- | ------ | ----------------------------------------------- |
| WorkingMemory   | 64     | Scratch space mutated by every reasoning step.  |
| LongTermMemory  | 128    | Accepted knowledge, retained across requests.   |
| Goal            | 16     | Holds the active objective.                     |
| Episode         | 32     | Per-request context buffer.                     |
| Task            | 16     | Long-running task state.                        |
| MemoryIndex     | 32     | Retrieval index, cross-attended each block.     |
| Intent          | 16     | Origination signal for the next input.          |

Banks ship with retention metadata (``importance``, ``usage``, ``age``,
``retention_score``) that drive the long-term memory recycle policy.

``Intent`` is last in ``BANK_NAMES`` on purpose: the operator derives token
offsets and bank ids from that order, and checkpoints encode the ids, so
appending is the only change that leaves existing banks where they were.

## State Transition Operator

Abstractly, every transition computes

```
C_{t+1} = F(C_t, O_t)
```

where ``C_t`` is the current PCS, ``O_t`` is the new observation, and ``F`` is
a :class:`ucsa.models.transition_operator.StateTransitionOperator`.
The reference implementation is the
:class:`ucsa.models.transformer_operator.TransformerOperator`, which is a
standard pre-norm transformer with grouped-query attention and a Mixture of
Experts FFN on the upper half of layers.

## Reasoning Loop

Each forward pass:

1. Inject observation into WorkingMemory.
2. For ``N`` iterations (default 4):
   ``C' = F(C, O_k)``.
3. Project WorkingMemory through the heads to produce outputs.

The operator's write-back copies under ``torch.no_grad``, so the loop also
carries the operator's differentiable bank tensors from one iteration to the
next and out to the heads (``ReasoningLoop.differentiable_bank``). Without
that carry the autograd graph ends at every transition and no loss can reach
the operator at all. ``differentiable_state_carry=False`` restores the old
severed behaviour for ablations.

## Endogenous Origination

``O_k`` need not be the same exogenous observation every iteration:

```
O_{k+1} = (1 - alpha_k) * G(intent, working_k) + alpha_k * O_0
alpha_k = observation_mix * observation_mix_decay ** k
```

``G`` is the ``OriginationHead``: a cross-attention read whose keys and
values come from ``concat(intent, working)`` and whose queries come from the
current input stream (the query side only fixes how many tokens to emit and
in what order). This makes the signal that *causes* the next state an
explicit, addressable variable rather than something smeared across the
operator's weights — the analogue of the neural signal that precedes a
reach.

``alpha_k`` is the brake on the loop running away on its own output. The
defaults (``observation_mix=1.0``, ``observation_mix_decay=1.0``) hold
``alpha_k = 1``, never call ``G``, and reproduce the previous behaviour
exactly, so origination is opt-in.

## Memory Pipeline

```
Working  --(propose)-->  Candidate  --(verify)-->  LongTerm
```

Verification, consolidation, and pruning run in the
:class:`ucsa.models.memory_service.MemoryService` background worker. Inference
never blocks on memory.

## JEPA

JEPA is an auxiliary training objective, not a separate model. Two heads
contribute to ``L_JEPA``:

- **Block-level**: every 8th transformer block predicts the working-memory
  output of the next block.
- **Iteration-level**: at the end of each reasoning iteration, predict the
  next iteration's working memory.

Both use cosine similarity + SmoothL1.

## Verifier

Two implementations behind a shared interface:

- **HeuristicVerifier** (default). Score blends confidence, novelty
  (1 - cosine similarity to nearest long-term memory), recency, and usage.
- **LearnedVerifier**. Small MLP head trained on the retention signal:
  whether each candidate was re-accessed in subsequent steps.

## Graph Memory

Background knowledge organization. The graph is **never attended** by the
transformer. Concepts are clusters of long-term memory embeddings (cosine
k-means in pure torch). Edges come from co-activation. Retrieval projects
relevant concept nodes back into memory tokens, which the memory service
injects into WorkingMemory at the start of the next reasoning pass.

## Projection Heads

Four independent heads, all reading from WorkingMemory only:

- **LanguageHead** — vocab logits.
- **PlanningHead** — discrete plan tokens.
- **ToolHead** — discrete tool tokens.
- **MemoryHead** — memory query embeddings.

## Training Losses

```
L = L_AR + 0.1 * L_JEPA + 0.01 * L_MEMORY + 0.01 * L_ROUTER
```

## Curriculum

Four stages, step-gated:

1. Language modelling only.
2. Language + JEPA.
3. Language + JEPA + memory.
4. Joint training.

## Why this should be reproducible

- Deterministic seeding via :mod:`ucsa.utils.seed`.
- Safetensors checkpoints with full config metadata.
- Hydra config composition; every run is reproducible from a config dump.
- pytest + mypy --strict + ruff run in CI on Python 3.11 and 3.12.