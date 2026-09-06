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

### Phase 11.0 — measurement prerequisites

- [x] fix(trainer): pad short targets with the loss's ignore index, not with
      zero. The working bank is 64 slots, so a short sequence left most
      positions unsupervised and padding them with `0` trained the model to
      emit token id 0 there: **90.4% of the autoregressive loss** for a
      6-token target. The real tokens barely mattered, which made every
      claim about what drives the emitted action unmeasurable. This was the
      root cause of all three earlier negative findings.

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
- [x] fix(heads): give the intent read and the working read separate
      softmaxes. Sharing one gave the gated intent slots ~3% of the
      attention mass, so `G` ignored the bank. Zeroing one intent slot now
      moves `G` by 39% vs 2% for a working slot.
- [x] fix(operator): `stream_intent_bank=False` -- the operator no longer
      attends over the intent bank, so `G` is the only path from bank to
      action. This is what made attribution well posed: grad-active slots
      went from 16/16 to 2/16.
- [x] **localisation claim achieved.** After a 600-step run on a structured
      copy task: 2 of 16 intent slots carry gradient, and ablating an
      ungated slot changes the action by exactly 0.0 while a gated slot
      changes it by 6e-5. Absolute effect sizes are still small, so the
      `1e-3` controllability threshold still reports 0.00 -- reported as
      measured rather than by lowering the threshold.

### Phase 11.C — collapse diagnostic


- [x] feat(metrics): `intent_state_variance`, `intent_gate_entropy`,
      `intent_gate_mutual_info`, `intent_read_share` in
      `ucsa/training/metrics.py`, in `DEFAULT_METRIC_NAMES`, recorded by the
      trainer over a rolling window
- [x] feat(probe): `intent_collapse_report` with an explicit verdict
- [x] fix(heads): `IntentUpdate`. The diagnostic immediately showed the bank
      was a *static parameter*: identical for every input, variance zero by
      construction. A signal that is the same before every reach is not an
      origination signal. The state is now refreshed residually from working
      memory each iteration.
- [x] fix(losses): wire the gate's load-balancing loss into the total. The
      diagnostic caught gate collapse -- the gate settled on the same two
      slots whatever it was shown, mutual information exactly 0.0000.
- [x] **diagnostic is green in the intended configuration** and correctly red
      in the degenerate ones. 600 steps, structured copy task:

      | config | collapsed | var | MI | H/max | read share | slots |
      | --- | --- | --- | --- | --- | --- | --- |
      | origination on, balanced | no | 1.2e-7 | 0.440 | 2.10/2.77 | 0.418 | 16/16 |
      | origination on, no balance | yes | 1.0e-7 | 0.000 | 0.69/2.77 | 0.423 | 2/16 |
      | static bank (scale 0) | yes | 5.6e-19 | 0.000 | 0.69/2.77 | 0.038 | 2/16 |
      | origination off (alpha=1) | yes | 0.0 | 0.000 | 0.00/2.77 | 0.000 | 0/16 |

- [ ] **open finding**: load balancing and localisation pull against each
      other. Balancing the gate lifts mutual information from 0.000 to 0.440
      but widens the per-action gated set from 2 slots to 9, and the
      gated-vs-ungated effect ratio falls from ~5.7e7 to ~0.9. Both ends of
      that trade-off are reported; no weight has been tuned to flatter it.
- [ ] **open finding**: final loss is 1.354 in *every* configuration,
      including origination fully off. On this task the origination path
      buys nothing measurable. Phase 11.D is where it is supposed to pay.

### Phase 11.D — origination is optimised

- [x] feat(models): `ucsa/models/intent_descent.py` -- K steps of gradient
      descent on the `intent` bank only, weights frozen, against the
      multi-step JEPA chain, with optional early stop on the intent gradient
      norm and `K=0` by default
- [x] feat(infer): `generate_with_intent_descent` plus `--intent-steps` /
      `--intent-learning-rate`, `K=0` default
- [x] fix(descent): evaluate every candidate from the same cognitive state.
      A forward pass rewrites the PCS, so back-to-back rollouts drift; the
      first version reported the objective falling 0.0069 -> 0.0028 purely
      from drift while the bank barely moved. Same fix needed for the EMA
      encoder, whose own forward also rewrote its banks.
- [x] fix(descent): use EMA targets. Without them both sides of every JEPA
      pair come from the same forward pass, so moving intent moves the
      prediction and its target together; the gradient measured ~3e-7 and
      the objective refused to fall even when the bank moved 2.2x its norm.
- [x] feat(scripts): `scripts/probe_origination.py` emits collapse,
      localisation and the matched-compute descent sweep as JSON; ablation
      arms added to `scripts/run_ablations.py`
- [x] **forward-model hacking is real and detected.** 400 steps, EMA
      momentum 0.99, 8 probe inputs:

      | K | passes/input | predicted better | realised better | gamed | corr(pred,real) |
      | --- | --- | --- | --- | --- | --- |
      | 0 | 2 | 0/8 | 0/8 | 0/8 | +0.000 |
      | 1 | 3 | 8/8 | 5/8 | 3/8 | +0.005 |
      | 3 | 5 | 8/8 | 6/8 | 2/8 | -0.313 |
      | 5 | 7 | 8/8 | 7/8 | 2/8 | -0.126 |

- [x] feat(models): `jepa_step_errors` reports the chain error per step, so
      the spec's "late steps should improve more than early ones" signature
      can actually be checked; surfaced in `scripts/probe_origination.py`
- [x] **all Phase 11.B/C/D numbers re-measured after the padding fix, on a
      task the model actually learns** (vocab 32, repeat-the-first-half,
      1200 steps, final ppl 1.05-1.13 against a chance ppl of 31):

      | config | ppl | collapsed | grad-active | gated vs ungated effect | controllability | specificity | descent corr |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | origination on, balanced gate | 1.132 | no | 4/16 | 0.02827 vs 0.00186 (15.2x) | **1.00** | **1.00** | +0.008 |
      | origination on, no load balance | 1.132 | yes (gate MI 0) | 2/16 | 0.04035 vs 0.00000 | **1.00** | **1.00** | **+0.755** |
      | origination off (alpha=1) | 1.049 | yes (inert) | 0/16 | 0.0 vs 0.0 | 0.00 | 1.00 | +0.000 |

- [x] **headline localisation claim achieved.** 2-4 of 16 intent slots carry
      the gradient; every attributed slot moves the emitted action when
      ablated (controllability 1.00) and no unattributed slot does
      (specificity 1.00). With the gate unbalanced the unattributed slots
      move it by *exactly* 0.0.
- [x] **perplexity flat-to-slightly-worse, as the spec predicted**: 1.049
      without origination, 1.132 with. The sparse gate costs capacity.
- [x] **forward-model hacking check survives optimisation in the unbalanced
      configuration**: predicted-vs-realised correlation +0.755, 7/8 inputs
      improved on both, 1/8 gamed.
- [x] fix(eval): `_load_winogrande` read a non-existent `options` key
      (`KeyError`, task never ran) and inverted its labels. All four named
      tasks now run.
- [x] **HellaSwag / ARC / PIQA / WinoGrande are flat, as the spec
      predicted.** Real `ckpts/ucsa-step3000.safetensors` (384-hidden,
      6-layer, fineweb-edu), 40 examples per task, origination off vs on:

      | task | acc off | acc on | delta | mean ll off | mean ll on | s/example off -> on |
      | --- | --- | --- | --- | --- | --- | --- |
      | piqa | 0.5250 | 0.5250 | +0.0000 | -14.40213 | -14.40103 | 0.393 -> 0.406 |
      | arc_easy | 0.2750 | 0.2750 | +0.0000 | -58.04808 | -58.05323 | 0.612 -> 0.675 |
      | hellaswag | 0.2250 | 0.2250 | +0.0000 | -21.01121 | -21.00932 | 0.668 -> 0.663 |
      | winogrande | 0.5250 | 0.5250 | +0.0000 | -8.58527 | -8.58428 | 0.374 -> 0.393 |

      Accuracy is *exactly* unchanged while the mean log-likelihood shifts
      by 1e-3 to 5e-3. So origination does perturb the forward pass, just
      far below the granularity that reorders a multiple-choice ranking.
      That is the same finding as "no intervention flips the arg-max
      token", seen from the evaluation side.
- [ ] **honest caveats, left open.**
      1. No intervention flips the arg-max token (0/16 in every config). The
         action *logits* move measurably and specifically, but at ppl 1.13
         the model is confident enough that the decision does not change.
         Controllability is measured in logit space, not decision space.
      2. Load balancing and descent quality pull against each other:
         balancing keeps the gate conditioning on the input (MI 0.041 vs
         0.000) but drops the descent correlation from +0.755 to +0.008.
         Both ends reported; no weight tuned to flatter either.
      3. The JEPA chain error *rises* during training when origination is on
         (k0 -0.029) and barely moves when it is off (k0 -0.001). It rises
         less at late steps than early ones, which is the ordering the spec
         predicts, but the sign is not an improvement.

## Exit criteria

- All checkboxes above are ticked.
- `ruff check` clean.
- `mypy --strict` clean on `ucsa/`.
- `pytest -q` all green.
- `python -m ucsa.train` smoke mode finishes 100 steps without NaN.