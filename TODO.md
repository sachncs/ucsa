# TODO — Unified Cognitive State Architecture (UCSA)

This document tracks every atomic commit and its verification gate. Tick boxes are
updated in the same commit that completes the work. Every code change must also
update `CHANGELOG.md` `[Unreleased]` and pass `ruff`, `mypy --strict`, and
`pytest` before the commit lands.

## Conventions

- Branch: `master`.
- Commit style: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`,
  `refactor:`, `chore:`).
- No leading-underscore names except dunder. Internal helpers get bare names
  with `"""internal: ..."""` docstrings.
- Python 3.11+. Type annotations everywhere. `from __future__ import annotations`
  on every module.
- Google Python style. 4-space indent, 80-column lines.

## Phase 1 — Scaffolding

- [x] chore: initialise pyproject.toml with deps and tool configs
- [x] chore: add ruff + black + mypy + pytest config files
- [x] docs: add README.md skeleton and docs/architecture.md skeleton
- [x] docs: add CHANGELOG.md and TODO.md with phase breakdown
- [x] chore: create ucsa/ directory tree and .gitignore

## Phase 2 — PCS + operator

- [x] feat(pcs): add PersistentCognitiveState with six banks and retention
      buffers
- [x] test(pcs): cover bank shapes, parameter registration, retention math
- [x] feat(operator): add StateTransitionOperator ABC with reset/initialize
      contract
- [x] test(operator): cover ABC contract via dummy implementation
- [x] feat(operator): add TransformerOperator with LN, GQA, FF, residual
- [x] test(operator): cover forward shapes, KV cache wiring, gradient flow
- [x] feat(operator): add MoE block on upper-half layers with top-2 routing
- [x] test(operator): cover MoE routing, load-balancing aux loss, expert
      collapse guard

## Phase 3 — Reasoning + perception

- [x] feat(perception): add tokenizer + embedding + positional + modality
      projection
- [x] test(perception): cover tokenize -> embed -> project pipeline, modality
      alignment
- [x] feat(loop): add ReasoningLoop with N-iteration operator calls
- [x] test(loop): cover iteration count, observation injection, state isolation

## Phase 4 — Memory

- [x] feat(memory): add Memory hierarchy (scratch/working/episode/long-term)
- [x] test(memory): cover bank mutation, capacity enforcement
- [x] feat(memory): add retention scoring and recycle-by-bottom-k
- [x] test(memory): cover retention math, FIFO recycle, capacity invariants

## Phase 5 — Memory service + verification

- [x] feat(verify): add Verifier ABC
- [x] feat(verify): add HeuristicVerifier (confidence, novelty, recency, usage)
- [x] test(verify): cover HeuristicVerifier score components and acceptance
      thresholds
- [x] feat(verify): add LearnedVerifier MLP head, trained on retention signal
- [x] test(verify): cover LearnedVerifier forward, training step, signal flow
- [x] feat(service): add MemoryService with asyncio Queue and worker task
- [x] test(service): cover non-blocking enqueue, FIFO processing, error
      isolation

## Phase 6 — Graph service

- [x] feat(graph): add cosine clustering for concept extraction (pure torch)
- [x] test(graph): cover cluster assignment, concept deduplication
- [x] feat(graph): add co-activation edge discovery
- [x] test(graph): cover edge weight, threshold, decay
- [x] feat(graph): add retrieval -> memory-token injection into PCS
- [x] test(graph): cover end-to-end retrieval and injection shape correctness

## Phase 7 — Projection heads

- [x] feat(heads): add Language/Planning/Tool/Memory heads over working memory
- [x] test(heads): cover independent forward, gradient isolation, output shapes

## Phase 8 — Training pipeline

- [x] feat(losses): add AR + JEPA (block+iter) + MemoryStability + RouterLB
      losses
- [x] test(losses): cover loss finiteness, gradient flow, weighted sum
- [x] feat(data): add HF streaming dataset (fineweb-edu -> OWT -> WikiText-103
      fallback)
- [x] test(data): cover tokenization, packing, fallback chain
- [x] feat(curriculum): add four-stage curriculum with step-gated activation
- [x] test(curriculum): cover stage transitions, loss-component gating
- [x] feat(metrics): add all 11 metrics from spec
- [x] test(metrics): cover metric accumulation, reset, tensorboard logging
- [x] feat(trainer): add Accelerate trainer with AMP, grad-ckpt, torch.compile
- [x] test(trainer): cover training step, checkpoint round-trip, scheduler step
- [x] feat(eval): add evaluation loop
- [x] test(eval): cover no-grad path, periodic eval trigger

## Phase 9 — Plumbing

- [x] feat(utils): add seed, checkpoint (safetensors), logging (TB + W&B)
- [x] test(utils): cover deterministic seed, checkpoint round-trip, logger init
- [x] feat(scripts): add train.py and infer.py with Hydra entrypoints
- [x] test(scripts): cover CLI arg parsing, config override, smoke step
- [x] feat(config): add configs/default.yaml with all hyperparameters
      externalized

## Phase 10 — Verification

- [x] chore: run pytest, mypy, ruff; fix any lints
- [x] chore: smoke run 100 steps on tiny model, confirm no NaN
- [x] chore: validate multi-epoch fineweb-edu config compiles

## Phase 11 — Endogenous origination (intent bank)

Motivation: before a hand reaches for an object there is a signal that
*originates* the action, and in a brain we cannot localise it. UCSA had no
origination variable at all — the reasoning loop fed the *same* observation to
every iteration, so nothing generated the input to the next state. This phase
makes that signal explicit, localisable, and optimisable.

### Phase 11.0 — differentiable reasoning loop (prerequisite)

- [x] fix(operator): clone the `memory_index` cross-attention tokens so the
      in-place bank write-back cannot invalidate saved K/V tensors
- [x] fix(jepa): `_condition_one` preserves the prediction's shape so the JEPA
      loss stops broadcasting against its target
- [x] feat(loop): carry the operator's differentiable bank tensors across
      iterations and out to the heads, with a
      `differentiable_state_carry=False` ablation for the old severed graph
- [x] test: AR loss reaches every operator parameter (was 0 of 31); JEPA
      predictions differentiable with detached targets; 100-step smoke run on a
      fixed batch, no NaN, 6.46 -> 0.15 (severed: 6.46 -> 2.78)

### Phase 11.A — origination exists

- [x] feat(pcs): add the `intent` bank (16 tokens, trainable, last in
      `BANK_NAMES` so existing bank offsets and bank ids are unchanged)
- [x] fix(ckpt): `adapt_legacy_state_dict` loads pre-intent checkpoints by
      remapping the `bank_id_embedding` observation row; verified against the
      real `ckpts/ucsa-step1000.safetensors`
- [x] feat(heads): add the origination generator `G` (`OriginationHead`)
- [x] feat(loop): mix `obs_{k+1} = (1 - alpha_k) * G + alpha_k * observation`,
      defaults reducing exactly to the current behaviour
- [x] test: `alpha=1` matches attaching no generator; generated stream tracks
      the observation's token count; the language loss trains `G` and the
      intent bank; 100-step smoke on four `alpha` schedules, no NaN, and the
      generated stream's RMS stays at or below the observation's rather than
      growing (drift measured, not assumed)

### Phase 11.B — origination is localisable

- [x] feat(heads): route `G` through a top-k sparse gate over intent slots,
      reusing `load_balancing_loss` / `top_k_mask` extracted from `moe.py`
- [x] feat(probe): `ucsa/models/origination.py` -- per-slot
      `grad x activation` attribution, ablate/swap intervention, and a
      combined controllability + specificity rate
- [x] fix(probe): snapshot and restore the whole PCS around every probe. The
      operator rewrites every bank each iteration, so back-to-back forwards
      start from different states; without this the measured effect was
      ~100x too large and was mostly state drift.
- [ ] **open finding, do not close**: measured controllability is `0.000` in
      every configuration after a 100-step fixed-batch run. Ablating any
      intent slot moves the emitted action by `<= 1.3e-4` relative, and
      gated slots are not distinguishable from ungated ones. Two candidate
      causes, both needing a decision before Phase 11.D:
      1. The intent bank reaches the action by two paths -- through `G`
         (sparse, attributable) and as ordinary PCS context the operator
         attends over (dense, not attributable). The second path dilutes
         and un-localises the first.
      2. No loss term rewards using intent, so there is a path but no
         incentive. 100 steps on one batch is also not training.

### Phase 11.C — collapse diagnostic

- [ ] feat(metrics): intent variance / mutual information across inputs

### Phase 11.D — origination is optimised

- [ ] feat(infer): K steps of gradient descent on the `intent` bank only,
      default `K=0`

## Exit criteria

- All checkboxes above are ticked.
- `ruff check` clean.
- `mypy --strict` clean on `ucsa/`.
- `pytest -q` all green.
- `python -m ucsa.train` smoke mode finishes 100 steps without NaN.