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

- [ ] feat(graph): add cosine clustering for concept extraction (pure torch)
- [ ] test(graph): cover cluster assignment, concept deduplication
- [ ] feat(graph): add co-activation edge discovery
- [ ] test(graph): cover edge weight, threshold, decay
- [ ] feat(graph): add retrieval -> memory-token injection into PCS
- [ ] test(graph): cover end-to-end retrieval and injection shape correctness

## Phase 7 — Projection heads

- [ ] feat(heads): add Language/Planning/Tool/Memory heads over working memory
- [ ] test(heads): cover independent forward, gradient isolation, output shapes

## Phase 8 — Training pipeline

- [ ] feat(losses): add AR + JEPA (block+iter) + MemoryStability + RouterLB
      losses
- [ ] test(losses): cover loss finiteness, gradient flow, weighted sum
- [ ] feat(data): add HF streaming dataset (fineweb-edu -> OWT -> WikiText-103
      fallback)
- [ ] test(data): cover tokenization, packing, fallback chain
- [ ] feat(curriculum): add four-stage curriculum with step-gated activation
- [ ] test(curriculum): cover stage transitions, loss-component gating
- [ ] feat(metrics): add all 11 metrics from spec
- [ ] test(metrics): cover metric accumulation, reset, tensorboard logging
- [ ] feat(trainer): add Accelerate trainer with AMP, grad-ckpt, torch.compile
- [ ] test(trainer): cover training step, checkpoint round-trip, scheduler step
- [ ] feat(eval): add evaluation loop
- [ ] test(eval): cover no-grad path, periodic eval trigger

## Phase 9 — Plumbing

- [ ] feat(utils): add seed, checkpoint (safetensors), logging (TB + W&B)
- [ ] test(utils): cover deterministic seed, checkpoint round-trip, logger init
- [ ] feat(scripts): add train.py and infer.py with Hydra entrypoints
- [ ] test(scripts): cover CLI arg parsing, config override, smoke step
- [ ] feat(config): add configs/default.yaml with all hyperparameters
      externalized

## Phase 10 — Verification

- [ ] chore: run pytest, mypy, ruff; fix any lints
- [ ] chore: smoke run 100 steps on tiny model, confirm no NaN
- [ ] chore: validate multi-epoch fineweb-edu config compiles

## Exit criteria

- All checkboxes above are ticked.
- `ruff check` clean.
- `mypy --strict` clean on `ucsa/`.
- `pytest -q` all green.
- `python -m ucsa.train` smoke mode finishes 100 steps without NaN.