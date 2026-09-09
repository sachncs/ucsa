---
layout: home
title: UCSA — Unified Cognitive State Architecture
description: A research-grade foundation model whose computation orbits a single persistent differentiable cognitive state. Apache-2.0, open source, 600+ tests, paper draft in progress.
permalink: /
---

<!-- =====================================================================
     Section 1 · Hero
     ===================================================================== -->
<section class="ucsa-hero">
  <div class="ucsa-container">
    <div class="ucsa-hero__grid">

      <div class="ucsa-hero__copy">
        <span class="ucsa-hero__eyebrow">
          <span class="ucsa-hero__eyebrow-dot"></span>
          research platform · v0.1
        </span>
        <h1 class="ucsa-hero__title">
          A foundation model built around one
          <em>persistent cognitive state.</em>
        </h1>
        <p class="ucsa-hero__sub">
          <strong>UCSA</strong> is a research-grade architecture whose
          entire computation — language logits, JEPA predictions,
          memory, planning, tool — reads from or writes to the same
          seven-bank differentiable state. A Transformer is one
          realisation of the operator; the state is the thesis.
        </p>
        <div class="ucsa-hero__ctas">
          <a class="ucsa-btn ucsa-btn--primary" href="{{ '/docs/getting-started/' | relative_url }}">
            Get started <span class="ucsa-btn__arrow">→</span>
          </a>
          <a class="ucsa-btn ucsa-btn--ghost" href="{{ '/docs/architecture/' | relative_url }}">
            Read architecture
          </a>
          <a class="ucsa-btn ucsa-btn--ghost" href="{{ '/paper/PAPER/' | relative_url }}">
            View paper draft
          </a>
        </div>
        <div class="ucsa-hero__signals">
          <div class="ucsa-hero__signal">
            <span class="ucsa-hero__signal-dot"></span>
            <span class="ucsa-hero__signal-value">607</span>
            <span>tests passing</span>
          </div>
          <div class="ucsa-hero__signal">
            <span class="ucsa-hero__signal-dot" style="background: var(--accent)"></span>
            <span class="ucsa-hero__signal-value">80%</span>
            <span>coverage gate enforced</span>
          </div>
          <div class="ucsa-hero__signal">
            <span class="ucsa-hero__signal-dot" style="background: var(--bank-goal)"></span>
            <span class="ucsa-hero__signal-value">3.11 · 3.12</span>
            <span>Python supported</span>
          </div>
          <div class="ucsa-hero__signal">
            <span class="ucsa-hero__signal-dot" style="background: var(--bank-long-term)"></span>
            <span class="ucsa-hero__signal-value">Apache-2.0</span>
            <span>license</span>
          </div>
        </div>
      </div>

      <div class="ucsa-hero__visual">
        <div class="ucsa-hero__visual-frame">
          <div class="ucsa-hero__visual-head">
            <span class="ucsa-hero__visual-title">persistent cognitive state</span>
            <span class="ucsa-hero__visual-tag">C<sub>t+1</sub> = F(C<sub>t</sub>, O<sub>t</sub>)</span>
          </div>
          <img class="ucsa-diagram" src="{{ '/assets/img/pcs-hero.svg' | relative_url }}" alt="PCS diagram: operator stream with six banks plus a held-out intent bank, with the JEPA multi-step chain below">
        </div>
      </div>

    </div>
  </div>
</section>

<!-- =====================================================================
     Section 2 · Core concept
     ===================================================================== -->
<section class="ucsa-section" id="core-concept">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">core concept</span>
      <h2 class="ucsa-section-head__title">One state. Seven banks. One chain.</h2>
      <p class="ucsa-section-head__lede">
        UCSA replaces the conventional "stack more layers, hope it
        generalises" pattern with a single, persistent,
        differentiable state that every projection in the model
        reads from or writes to.
      </p>
    </div>

    <div class="ucsa-concept">
      <div class="ucsa-concept__copy">
        <p class="ucsa-concept__lede">
          The Persistent Cognitive State (PCS) is a stack of seven
          learnable tensors — one per bank — all of shape
          <code>(num_tokens, hidden_size)</code>. Every projection in
          the model reads from or writes to this state; no other
          structure stores knowledge.
        </p>
        <p>
          The first six banks — <code>working</code>,
          <code>long_term</code>, <code>goal</code>,
          <code>episode</code>, <code>task</code>,
          <code>memory_index</code> — flow through the operator's
          attention stream. The seventh, <code>intent</code>, is
          held out of the stream on purpose: it is the origination
          signal, and making the origination generator the *only*
          path from intent to behaviour is what makes per-slot
          attribution well posed.
        </p>
        <p>
          A forward pass runs the operator <code>N</code> times
          (default 4), each time writing a new PCS. A multi-step
          JEPA prediction chain then pairs consecutive working
          latents into <em>(predicted, target)</em> pairs and adds
          them to the loss.
        </p>
        <ul class="ucsa-concept__list">
          <li>Everything is anchored to one state.</li>
          <li>Memory, goals, and intent are first-class tensors.</li>
          <li>The operator is interchangeable; the state is not.</li>
        </ul>
      </div>

      <div class="ucsa-concept__pull">
        <p class="ucsa-concept__pull-title">the central equation</p>
        <p class="ucsa-concept__equation">
          <em>C</em><sub>t+1</sub> = <em>F</em>(<em>C</em><sub>t</sub>, <em>O</em><sub>t</sub>)
        </p>
        <p style="font-size: var(--type-sm); color: var(--ink-500); margin: var(--s-3) 0 0 0;">
          A single transition operator updates one persistent state.
          Mamba, RWKV, and Hyena are alternative realisations of
          the same <em>F</em>; the rest of the system is unchanged.
        </p>
      </div>
    </div>
  </div>
</section>

<!-- =====================================================================
     Section 3 · Architecture overview
     ===================================================================== -->
<section class="ucsa-section ucsa-section--alt" id="architecture">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">architecture overview</span>
      <h2 class="ucsa-section-head__title">A structured map, not prose.</h2>
      <p class="ucsa-section-head__lede">
        Seven subsystems, one shared state. Each subsystem has a
        single responsibility and a single integration point: the
        PCS.
      </p>
    </div>

    <div class="ucsa-arch-grid">
      <div class="ucsa-arch-card ucsa-arch-card--wide">
        <span class="ucsa-arch-card__index">01</span>
        <h3 class="ucsa-arch-card__title">PCS — Persistent Cognitive State</h3>
        <p class="ucsa-arch-card__desc">
          Seven differentiable token banks. Every read and write
          in the system is a slice of this state.
        </p>
        <span class="ucsa-arch-card__tag">ucsa/models/state.py</span>
      </div>
      <div class="ucsa-arch-card ucsa-arch-card--wide">
        <span class="ucsa-arch-card__index">02</span>
        <h3 class="ucsa-arch-card__title">Operator — state transition F</h3>
        <p class="ucsa-arch-card__desc">
          The only computation engine. Maps
          <code>(C_t, O_t) → C_{t+1}</code>. Reference impl is a
          pre-norm Transformer with GQA, MoE, and memory-index
          cross-attention.
        </p>
        <span class="ucsa-arch-card__tag">ucsa/models/transformer_operator.py</span>
      </div>

      <div class="ucsa-arch-card ucsa-arch-card--half">
        <span class="ucsa-arch-card__index">03</span>
        <h3 class="ucsa-arch-card__title">Reasoning loop</h3>
        <p class="ucsa-arch-card__desc">
          Runs <code>F</code> <em>N</em> times per forward.
          Carries differentiable bank tensors between iterations
          so the loss can reach the operator.
        </p>
        <span class="ucsa-arch-card__tag">reasoning_loop.py</span>
      </div>
      <div class="ucsa-arch-card ucsa-arch-card--half">
        <span class="ucsa-arch-card__index">04</span>
        <h3 class="ucsa-arch-card__title">Memory service</h3>
        <p class="ucsa-arch-card__desc">
          Background worker. Verifies, consolidates, and prunes
          long-term memory. Inference never blocks on memory.
        </p>
        <span class="ucsa-arch-card__tag">memory_service.py</span>
      </div>
      <div class="ucsa-arch-card ucsa-arch-card--half">
        <span class="ucsa-arch-card__index">05</span>
        <h3 class="ucsa-arch-card__title">Projection heads</h3>
        <p class="ucsa-arch-card__desc">
          Four heads — language, planning, tool, memory — plus the
          input-reconstruction and origination heads.
        </p>
        <span class="ucsa-arch-card__tag">projection_heads.py</span>
      </div>
      <div class="ucsa-arch-card ucsa-arch-card--half">
        <span class="ucsa-arch-card__index">06</span>
        <h3 class="ucsa-arch-card__title">EMA target encoder</h3>
        <p class="ucsa-arch-card__desc">
          Frozen EMA copy of the model. Provides the JEPA chain's
          targets so the prediction chain stays stable.
        </p>
        <span class="ucsa-arch-card__tag">training/ema.py</span>
      </div>
      <div class="ucsa-arch-card ucsa-arch-card--half">
        <span class="ucsa-arch-card__index">07</span>
        <h3 class="ucsa-arch-card__title">Auxiliary losses</h3>
        <p class="ucsa-arch-card__desc">
          JEPA chain, input-reconstruction capacity bottleneck,
          memory stability, and MoE load-balancing.
        </p>
        <span class="ucsa-arch-card__tag">models/losses.py</span>
      </div>
      <div class="ucsa-arch-card ucsa-arch-card--half">
        <span class="ucsa-arch-card__index">08</span>
        <h3 class="ucsa-arch-card__title">Eval harness</h3>
        <p class="ucsa-arch-card__desc">
          Seed-deterministic rank-by-log-likelihood on HellaSwag,
          ARC, PIQA, and WinoGrande.
        </p>
        <span class="ucsa-arch-card__tag">training/eval_harness.py</span>
      </div>

      <div class="ucsa-arch-card ucsa-arch-card--full">
        <img class="ucsa-diagram" src="{{ '/assets/img/architecture.svg' | relative_url }}" alt="Architecture overview: PCS at center, surrounded by perception, reasoning loop, memory service, projection heads, auxiliary losses, eval harness">
      </div>
    </div>
  </div>
</section>

<!-- =====================================================================
     Section 4 · How UCSA works
     ===================================================================== -->
<section class="ucsa-section" id="how-it-works">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">how UCSA works</span>
      <h2 class="ucsa-section-head__title">Eight steps. One computation.</h2>
      <p class="ucsa-section-head__lede">
        The full forward pass, end to end, in the order the
        computation actually executes.
      </p>
    </div>

    <div class="ucsa-timeline">

      <div class="ucsa-timeline__step">
        <div class="ucsa-timeline__step-icon">01</div>
        <h3 class="ucsa-timeline__step-title">Input arrives</h3>
        <p class="ucsa-timeline__step-desc">
          A tokenised sequence lands in the trainer. Targets are
          the next-token ids with <code>ignore_index</code> on
          padding.
        </p>
      </div>

      <div class="ucsa-timeline__step">
        <div class="ucsa-timeline__step-icon">02</div>
        <h3 class="ucsa-timeline__step-title">Perception &amp; routing</h3>
        <p class="ucsa-timeline__step-desc">
          Tokens are embedded and projected to the operator's
          hidden size. MoE router logits are computed if MoE is on.
        </p>
      </div>

      <div class="ucsa-timeline__step">
        <div class="ucsa-timeline__step-icon">03</div>
        <h3 class="ucsa-timeline__step-title">PCS banks update</h3>
        <p class="ucsa-timeline__step-desc">
          The observation is injected into the working bank. Long-
          term, goal, episode, task, and memory-index banks are
          concatenated into the operator stream.
        </p>
      </div>

      <div class="ucsa-timeline__step">
        <div class="ucsa-timeline__step-icon">04</div>
        <h3 class="ucsa-timeline__step-title">Reasoning loop iterates</h3>
        <p class="ucsa-timeline__step-desc">
          <code>F</code> runs <em>N</em>=4 times. Each iteration
          carries the previous iteration's differentiable banks so
          the loss can reach the operator.
        </p>
      </div>

      <div class="ucsa-timeline__step">
        <div class="ucsa-timeline__step-icon">05</div>
        <h3 class="ucsa-timeline__step-title">Memory service syncs</h3>
        <p class="ucsa-timeline__step-desc">
          A background worker verifies, consolidates, and prunes
          long-term memory. Inference never blocks.
        </p>
      </div>

      <div class="ucsa-timeline__step">
        <div class="ucsa-timeline__step-icon">06</div>
        <h3 class="ucsa-timeline__step-title">Heads read the state</h3>
        <p class="ucsa-timeline__step-desc">
          Language, planning, tool, and memory heads project
          working memory to their respective outputs.
        </p>
      </div>

      <div class="ucsa-timeline__step">
        <div class="ucsa-timeline__step-icon">07</div>
        <h3 class="ucsa-timeline__step-title">JEPA chain predicts</h3>
        <p class="ucsa-timeline__step-desc">
          Consecutive working latents are paired into
          <em>(predicted, target)</em> tuples. Targets come from the
          EMA encoder for stability.
        </p>
      </div>

      <div class="ucsa-timeline__step">
        <div class="ucsa-timeline__step-icon">08</div>
        <h3 class="ucsa-timeline__step-title">Outputs are generated</h3>
        <p class="ucsa-timeline__step-desc">
          The combined loss flows back through every projection
          that touched the PCS — including the operator.
        </p>
      </div>

    </div>

    <div style="margin-top: var(--s-6);">
      <img class="ucsa-diagram" src="{{ '/assets/img/reasoning-loop.svg' | relative_url }}" alt="Reasoning loop timeline: input → four iterations of F → logits, with JEPA chain back-edges">
    </div>
  </div>
</section>

<!-- =====================================================================
     Section 5 · PCS bank visualization
     ===================================================================== -->
<section class="ucsa-section ucsa-section--alt" id="pcs-banks">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">the seven banks</span>
      <h2 class="ucsa-section-head__title">Anatomy of the PCS.</h2>
      <p class="ucsa-section-head__lede">
        Six banks flow through the operator's attention stream.
        The seventh — <code>intent</code> — is held out so the
        origination generator is the only path from intent to
        behaviour.
      </p>
    </div>

    <div class="ucsa-banks">

      <article class="ucsa-bank" data-bank="working">
        <header class="ucsa-bank__head">
          <span class="ucsa-bank__name">working</span>
          <span class="ucsa-bank__tokens">64 tok</span>
        </header>
        <p class="ucsa-bank__role">
          Scratch space mutated by every reasoning step. Every
          head reads from this bank.
        </p>
        <div class="ucsa-bank__meta">
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">role</span>
            <span class="ucsa-bank__meta-val">scratch</span>
          </div>
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">path</span>
            <span><span class="ucsa-bank__pill ucsa-bank__pill--stream">in stream</span></span>
          </div>
        </div>
      </article>

      <article class="ucsa-bank" data-bank="long_term">
        <header class="ucsa-bank__head">
          <span class="ucsa-bank__name">long_term</span>
          <span class="ucsa-bank__tokens">128 tok</span>
        </header>
        <p class="ucsa-bank__role">
          Accepted knowledge, retained across requests. Each token
          carries retention metadata that drives the recycle
          policy.
        </p>
        <div class="ucsa-bank__meta">
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">role</span>
            <span class="ucsa-bank__meta-val">memory</span>
          </div>
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">path</span>
            <span><span class="ucsa-bank__pill ucsa-bank__pill--stream">in stream</span></span>
          </div>
        </div>
      </article>

      <article class="ucsa-bank" data-bank="goal">
        <header class="ucsa-bank__head">
          <span class="ucsa-bank__name">goal</span>
          <span class="ucsa-bank__tokens">16 tok</span>
        </header>
        <p class="ucsa-bank__role">
          Holds the active objective. Mutated when the active goal
          changes; otherwise a stable anchor for the reasoning
          loop.
        </p>
        <div class="ucsa-bank__meta">
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">role</span>
            <span class="ucsa-bank__meta-val">objective</span>
          </div>
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">path</span>
            <span><span class="ucsa-bank__pill ucsa-bank__pill--stream">in stream</span></span>
          </div>
        </div>
      </article>

      <article class="ucsa-bank" data-bank="episode">
        <header class="ucsa-bank__head">
          <span class="ucsa-bank__name">episode</span>
          <span class="ucsa-bank__tokens">32 tok</span>
        </header>
        <p class="ucsa-bank__role">
          Per-request context buffer. Holds the immediate working
          memory between the start and end of a single
          request.
        </p>
        <div class="ucsa-bank__meta">
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">role</span>
            <span class="ucsa-bank__meta-val">context</span>
          </div>
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">path</span>
            <span><span class="ucsa-bank__pill ucsa-bank__pill--stream">in stream</span></span>
          </div>
        </div>
      </article>

      <article class="ucsa-bank" data-bank="task">
        <header class="ucsa-bank__head">
          <span class="ucsa-bank__name">task</span>
          <span class="ucsa-bank__tokens">16 tok</span>
        </header>
        <p class="ucsa-bank__role">
          Long-running task state. Persists across episodes when
          a multi-request task is in flight.
        </p>
        <div class="ucsa-bank__meta">
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">role</span>
            <span class="ucsa-bank__meta-val">task state</span>
          </div>
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">path</span>
            <span><span class="ucsa-bank__pill ucsa-bank__pill--stream">in stream</span></span>
          </div>
        </div>
      </article>

      <article class="ucsa-bank" data-bank="memory_index">
        <header class="ucsa-bank__head">
          <span class="ucsa-bank__name">memory_index</span>
          <span class="ucsa-bank__tokens">32 tok</span>
        </header>
        <p class="ucsa-bank__role">
          Retrieval index, cross-attended by every transformer
          block. Holds the keys and values for retrieval.
        </p>
        <div class="ucsa-bank__meta">
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">role</span>
            <span class="ucsa-bank__meta-val">retrieval</span>
          </div>
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">path</span>
            <span><span class="ucsa-bank__pill ucsa-bank__pill--stream">cross-attn</span></span>
          </div>
        </div>
      </article>

      <article class="ucsa-bank" data-bank="intent" style="grid-column: span 2;">
        <header class="ucsa-bank__head">
          <span class="ucsa-bank__name">intent</span>
          <span class="ucsa-bank__tokens">16 tok</span>
        </header>
        <p class="ucsa-bank__role">
          Origination signal. Held out of the operator stream by
          design: the <code>OriginationHead</code> reads the
          intent bank and the working memory, and produces the
          next iteration's input. Per-slot attribution is well
          posed because there is one path, not many.
        </p>
        <div class="ucsa-bank__meta">
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">role</span>
            <span class="ucsa-bank__meta-val">origination</span>
          </div>
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">path</span>
            <span><span class="ucsa-bank__pill ucsa-bank__pill--held">held out · origination</span></span>
          </div>
          <div class="ucsa-bank__meta-row">
            <span class="ucsa-bank__meta-key">service</span>
            <span><span class="ucsa-bank__pill ucsa-bank__pill--service">OriginationHead + IntentUpdate</span></span>
          </div>
        </div>
      </article>

    </div>
  </div>
</section>

<!-- =====================================================================
     Section 6 · Proof / trust
     ===================================================================== -->
<section class="ucsa-section" id="proof">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">proof</span>
      <h2 class="ucsa-section-head__title">Evidence, not marketing.</h2>
      <p class="ucsa-section-head__lede">
        The project earns trust by shipping reproducible tests,
        deterministic evals, and a working paper draft — not by
        claiming benchmark wins it has not measured.
      </p>
    </div>

    <div class="ucsa-evidence">
      <div class="ucsa-evidence__card">
        <div class="ucsa-evidence__icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M5 10l3 3 7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div class="ucsa-evidence__metric">607<span class="ucsa-evidence__metric-suffix">tests</span></div>
        <h3 class="ucsa-evidence__title">Test suite</h3>
        <p class="ucsa-evidence__desc">
          Unit tests across every subsystem: PCS, operator,
          reasoning loop, projection heads, JEPA chain, EMA,
          curriculum, trainer, eval harness, and intent descent.
        </p>
      </div>

      <div class="ucsa-evidence__card">
        <div class="ucsa-evidence__icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <rect x="3" y="3" width="14" height="14" rx="2" stroke="currentColor" stroke-width="1.6"/>
            <path d="M7 10l2 2 4-4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div class="ucsa-evidence__metric">80<span class="ucsa-evidence__metric-suffix">%</span></div>
        <h3 class="ucsa-evidence__title">Coverage gate</h3>
        <p class="ucsa-evidence__desc">
          CI fails on any coverage drop below 80%. Coverage is
          measured per branch and uploaded to Codecov on every
          push.
        </p>
      </div>

      <div class="ucsa-evidence__card">
        <div class="ucsa-evidence__icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M10 3l6 3v4c0 4-2.5 7-6 7s-6-3-6-7V6l6-3z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
          </svg>
        </div>
        <div class="ucsa-evidence__metric">3.11<span class="ucsa-evidence__metric-suffix">·</span>3.12</div>
        <h3 class="ucsa-evidence__title">Python support</h3>
        <p class="ucsa-evidence__desc">
          CI matrix runs on Python 3.11 and 3.12. PyTorch 2.1+,
          MPS, CUDA, and CPU are all in the supported matrix.
        </p>
      </div>

      <div class="ucsa-evidence__card">
        <div class="ucsa-evidence__icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="1.6"/>
            <path d="M7 10l2 2 4-4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div class="ucsa-evidence__metric">Apache<span class="ucsa-evidence__metric-suffix">-2.0</span></div>
        <h3 class="ucsa-evidence__title">License</h3>
        <p class="ucsa-evidence__desc">
          Apache-2.0 from day one. The full LICENSE file is at the
          repo root and matches the declaration in
          <code>pyproject.toml</code>.
        </p>
      </div>

      <div class="ucsa-evidence__card">
        <div class="ucsa-evidence__icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <rect x="4" y="3" width="12" height="14" rx="2" stroke="currentColor" stroke-width="1.6"/>
            <path d="M7 8h6M7 11h6M7 14h4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="ucsa-evidence__metric">Paper<span class="ucsa-evidence__metric-suffix">draft</span></div>
        <h3 class="ucsa-evidence__title">Working paper</h3>
        <p class="ucsa-evidence__desc">
          A full paper draft lives in <code>paper/PAPER.md</code>.
          It records what the system measures, what is in the
          repo, and what is left to run at paper-grade scale.
        </p>
      </div>

      <div class="ucsa-evidence__card">
        <div class="ucsa-evidence__icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M3 17l5-5 4 4 5-5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M3 10h14" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="ucsa-evidence__metric">matched<span class="ucsa-evidence__metric-suffix">compute</span></div>
        <h3 class="ucsa-evidence__title">Matched-compute baselines</h3>
        <p class="ucsa-evidence__desc">
          Every UCSA result is paired with a vanilla-Transformer
          baseline of identical parameter count, dataset, and
          step budget. The protocol catches negative results.
        </p>
      </div>

      <div class="ucsa-evidence__card">
        <div class="ucsa-evidence__icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <rect x="3" y="4" width="14" height="12" rx="2" stroke="currentColor" stroke-width="1.6"/>
            <path d="M3 8h14" stroke="currentColor" stroke-width="1.6"/>
          </svg>
        </div>
        <div class="ucsa-evidence__metric">5<span class="ucsa-evidence__metric-suffix">tasks</span></div>
        <h3 class="ucsa-evidence__title">Standard eval harness</h3>
        <p class="ucsa-evidence__desc">
          HellaSwag, ARC-easy, ARC-challenge, PIQA, and WinoGrande
          via rank-by-log-likelihood. Deterministic by seed.
        </p>
      </div>

      <div class="ucsa-evidence__card">
        <div class="ucsa-evidence__icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M5 4h10v12H5z" stroke="currentColor" stroke-width="1.6"/>
            <path d="M7 8l2 2 4-4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div class="ucsa-evidence__metric">Hydra<span class="ucsa-evidence__metric-suffix">/OmegaConf</span></div>
        <h3 class="ucsa-evidence__title">Reproducible configs</h3>
        <p class="ucsa-evidence__desc">
          Every run is reproducible from a config dump. CLI
          overrides are recorded in the JSON output.
        </p>
      </div>
    </div>
  </div>
</section>

<!-- =====================================================================
     Section 7 · Quickstart
     ===================================================================== -->
<section class="ucsa-section ucsa-section--alt" id="quickstart">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">quickstart</span>
      <h2 class="ucsa-section-head__title">From clone to trained model in five minutes.</h2>
      <p class="ucsa-section-head__lede">
        The smoke command exits in under a minute on CPU and proves
        the install wires up correctly. The full reproduction runs
        for hours on GPU.
      </p>
    </div>

    <div class="ucsa-quickstart">

      <ol class="ucsa-quickstart__steps">
        <li class="ucsa-quickstart__step">
          <div>
            <h3 class="ucsa-quickstart__step-title">Install</h3>
            <p class="ucsa-quickstart__step-desc">
              Clone, create a virtualenv, and install with the
              <code>[dev]</code> extra. Pulls in pytest, ruff,
              black, mypy, and pre-commit.
            </p>
          </div>
        </li>
        <li class="ucsa-quickstart__step">
          <div>
            <h3 class="ucsa-quickstart__step-title">Smoke test</h3>
            <p class="ucsa-quickstart__step-desc">
              Five training steps, no checkpoints, no eval, no
              baseline comparison. Exits in under a minute.
              Confirms every subsystem wires up.
            </p>
          </div>
        </li>
        <li class="ucsa-quickstart__step">
          <div>
            <h3 class="ucsa-quickstart__step-title">Full reproduction</h3>
            <p class="ucsa-quickstart__step-desc">
              Twelve thousand steps on fineweb-edu with the
              multi-step JEPA chain, hard-EMA target encoder,
              input-reconstruction head, and TC-JEPA conditioner
              all on.
            </p>
          </div>
        </li>
        <li class="ucsa-quickstart__step">
          <div>
            <h3 class="ucsa-quickstart__step-title">Evaluate</h3>
            <p class="ucsa-quickstart__step-desc">
              Seed-deterministic rank-by-log-likelihood on
              HellaSwag, ARC, PIQA, and WinoGrande. Compare
              against the matched-compute baseline.
            </p>
          </div>
        </li>
        <li class="ucsa-quickstart__step">
          <div>
            <h3 class="ucsa-quickstart__step-title">Probe</h3>
            <p class="ucsa-quickstart__step-desc">
              Inspect what each PCS bank has learned (top tokens,
              centroid cosine similarity) and where the intent
              bank is localising.
            </p>
          </div>
        </li>
      </ol>

      <div class="ucsa-terminal">
        <div class="ucsa-terminal__head">
          <div class="ucsa-terminal__dots">
            <span class="ucsa-terminal__dot ucsa-terminal__dot--r"></span>
            <span class="ucsa-terminal__dot ucsa-terminal__dot--y"></span>
            <span class="ucsa-terminal__dot ucsa-terminal__dot--g"></span>
          </div>
          <span class="ucsa-terminal__label">~/ucsa — zsh</span>
          <button class="ucsa-terminal__copy" type="button">copy</button>
        </div>
        <div class="ucsa-terminal__body">

<div class="ucsa-terminal__line"><span class="ucsa-terminal__comment"># 1. install</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__prompt">$</span> <span class="ucsa-terminal__cmd">git clone https://github.com/sachncs/ucsa.git</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__prompt">$</span> <span class="ucsa-terminal__cmd">cd ucsa && python -m venv .venv && source .venv/bin/activate</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__prompt">$</span> <span class="ucsa-terminal__cmd">pip install -e ".[dev]"</span></div>
<div class="ucsa-terminal__line">&nbsp;</div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__comment"># 2. smoke test (5 steps, ~1 minute)</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__prompt">$</span> <span class="ucsa-terminal__cmd">.venv/bin/python scripts/train.py \</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__accent">    --max-steps 5 \</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__accent">    --ckpt-every 0 --eval-every 0 \</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__accent">    --skip-baselines --seed 42</span></div>
<div class="ucsa-terminal__line">&nbsp;</div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__output">  UCSA-small params: 63,000,000</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__output">  Device: mps</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__output">  Stack: JEPA=lewm(multi-step) + EMA=0.996 + recon(w=0.1) ...</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__output">  Final: loss=10.4123 val_ppl=33102 (best=33102@5)</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__output">  Wrote runs/ucsa-full-seed42.json</span></div>
<div class="ucsa-terminal__line">&nbsp;</div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__comment"># 3. full reproduction (~hours on GPU)</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__prompt">$</span> <span class="ucsa-terminal__cmd">.venv/bin/python scripts/train.py --seed 42</span></div>
<div class="ucsa-terminal__line">&nbsp;</div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__comment"># 4. evaluate</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__prompt">$</span> <span class="ucsa-terminal__cmd">.venv/bin/python scripts/eval.py \</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__accent">    --ucsa-ckpt ckpts/ucsa-final.safetensors \</span></div>
<div class="ucsa-terminal__line"><span class="ucsa-terminal__accent">    --out-json runs/eval-ucsa-small.json</span></div>

        </div>
      </div>

    </div>
  </div>
</section>

<!-- =====================================================================
     Section 8 · Docs entry
     ===================================================================== -->
<section class="ucsa-section" id="docs">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">documentation</span>
      <h2 class="ucsa-section-head__title">Pick your entry point.</h2>
      <p class="ucsa-section-head__lede">
        The docs are organised by role. Implementers go straight to
        the API reference; researchers start at the architecture
        doc and the paper; evaluators jump to the eval harness;
        contributors go through the dev setup first.
      </p>
    </div>

    <div class="ucsa-roles">

      <article class="ucsa-role">
        <div class="ucsa-role__icon">IM</div>
        <h3 class="ucsa-role__title">Implementers</h3>
        <p class="ucsa-role__desc">
          Build UCSA, swap the operator, add a bank, wire a custom
          loss. You want the API tour and the tutorials.
        </p>
        <div class="ucsa-role__links">
          <a class="ucsa-role__link" href="{{ '/docs/api-reference/' | relative_url }}">→ API reference</a>
          <a class="ucsa-role__link" href="{{ '/docs/tutorials/' | relative_url }}">→ Tutorials</a>
        </div>
      </article>

      <article class="ucsa-role">
        <div class="ucsa-role__icon">RS</div>
        <h3 class="ucsa-role__title">Researchers</h3>
        <p class="ucsa-role__desc">
          Understand the thesis, follow the math, scrutinise the
          matched-compute protocol. You want the architecture and
          the paper.
        </p>
        <div class="ucsa-role__links">
          <a class="ucsa-role__link" href="{{ '/docs/architecture/' | relative_url }}">→ Architecture</a>
          <a class="ucsa-role__link" href="{{ '/paper/PAPER/' | relative_url }}">→ Paper draft</a>
        </div>
      </article>

      <article class="ucsa-role">
        <div class="ucsa-role__icon">EV</div>
        <h3 class="ucsa-role__title">Evaluators</h3>
        <p class="ucsa-role__desc">
          Rerun the numbers. The harness is deterministic and the
          protocol catches negative results.
        </p>
        <div class="ucsa-role__links">
          <a class="ucsa-role__link" href="{{ '/paper/TABLES/' | relative_url }}">→ Paper tables</a>
          <a class="ucsa-role__link" href="{{ '/docs/getting-started/' | relative_url }}">→ Get started</a>
        </div>
      </article>

      <article class="ucsa-role">
        <div class="ucsa-role__icon">CT</div>
        <h3 class="ucsa-role__title">Contributors</h3>
        <p class="ucsa-role__desc">
          Open a PR. The dev setup is short, the test suite is
          fast, and the issue templates are short.
        </p>
        <div class="ucsa-role__links">
          <a class="ucsa-role__link" href="{{ '/docs/contributing/' | relative_url }}">→ Contributing</a>
          <a class="ucsa-role__link" href="https://github.com/sachncs/ucsa/issues" rel="noopener">→ Issues ↗</a>
        </div>
      </article>

    </div>

    <div style="margin-top: var(--s-8);">
      <div class="ucsa-section-head" style="margin-bottom: var(--s-5);">
        <h3 style="font-size: var(--type-xl);">All documentation</h3>
      </div>

      <div class="ucsa-doc-grid">
        <a class="ucsa-doc-tile" href="{{ '/docs/getting-started/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">onboarding</span>
          <h4 class="ucsa-doc-tile__title">Getting started</h4>
          <p class="ucsa-doc-tile__desc">
            Install, smoke test, full reproduction, ablation flags.
          </p>
        </a>
        <a class="ucsa-doc-tile" href="{{ '/docs/architecture/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">design</span>
          <h4 class="ucsa-doc-tile__title">Architecture</h4>
          <p class="ucsa-doc-tile__desc">
            Deep design notes for every subsystem, with the maths.
          </p>
        </a>
        <a class="ucsa-doc-tile" href="{{ '/docs/api-reference/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">reference</span>
          <h4 class="ucsa-doc-tile__title">API reference</h4>
          <p class="ucsa-doc-tile__desc">
            Module-by-module tour of <code>ucsa/</code>, organised
            by responsibility.
          </p>
        </a>
        <a class="ucsa-doc-tile" href="{{ '/docs/tutorials/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">walkthroughs</span>
          <h4 class="ucsa-doc-tile__title">Tutorials</h4>
          <p class="ucsa-doc-tile__desc">
            Five end-to-end examples: build, customise, ablate,
            measure, probe.
          </p>
        </a>
        <a class="ucsa-doc-tile" href="{{ '/docs/contributing/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">dev workflow</span>
          <h4 class="ucsa-doc-tile__title">Contributing</h4>
          <p class="ucsa-doc-tile__desc">
            Setup, lint, type-check, tests, PR flow, code review
            checklist.
          </p>
        </a>
        <a class="ucsa-doc-tile" href="{{ '/CHANGELOG/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">release notes</span>
          <h4 class="ucsa-doc-tile__title">Changelog</h4>
          <p class="ucsa-doc-tile__desc">
            Per-phase history of the project, with the matching
            tests added.
          </p>
        </a>
        <a class="ucsa-doc-tile" href="{{ '/paper/PAPER/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">research</span>
          <h4 class="ucsa-doc-tile__title">Paper draft</h4>
          <p class="ucsa-doc-tile__desc">
            The full paper draft, including the negative-results
            section.
          </p>
        </a>
        <a class="ucsa-doc-tile" href="{{ '/paper/TABLES/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">results</span>
          <h4 class="ucsa-doc-tile__title">Paper tables</h4>
          <p class="ucsa-doc-tile__desc">
            Generated from <code>runs/*.json</code>. The numbers
            the paper claims.
          </p>
        </a>
        <a class="ucsa-doc-tile" href="{{ '/SECURITY/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">policy</span>
          <h4 class="ucsa-doc-tile__title">Security</h4>
          <p class="ucsa-doc-tile__desc">
            Supported versions matrix and the private-disclosure
            SLA.
          </p>
        </a>
        <a class="ucsa-doc-tile" href="{{ '/SUPPORT/' | relative_url }}">
          <span class="ucsa-doc-tile__eyebrow">help</span>
          <h4 class="ucsa-doc-tile__title">Support</h4>
          <p class="ucsa-doc-tile__desc">
            Where to ask questions: issues for bugs, discussions
            for design.
          </p>
        </a>
      </div>
    </div>
  </div>
</section>

<!-- =====================================================================
     Section 9 · Research and paper pathway
     ===================================================================== -->
<section class="ucsa-section ucsa-section--alt" id="research">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">research</span>
      <h2 class="ucsa-section-head__title">A research companion, not a hidden appendix.</h2>
      <p class="ucsa-section-head__lede">
        The paper draft, the evaluation tables, the experiment
        framing, and the reproducibility notes live here — at the
        same level as the code.
      </p>
    </div>

    <div class="ucsa-paper">
      <div class="ucsa-paper__copy">
        <p>
          The working paper draft
          (<a href="{{ '/paper/PAPER/' | relative_url }}">paper/PAPER.md</a>) targets
          ICML / NeurIPS and records exactly what the system
          measures, what is in the repo, and what remains to run
          at paper-grade scale. The matched-compute protocol is the
          spine: every UCSA result is paired with a
          vanilla-Transformer baseline of identical parameter
          count, dataset, and step budget.
        </p>
        <p class="ucsa-paper__pull">
          "The contribution is the design — a state-centric
          inductive bias — and the protocol that can say which is
          which. The matched-compute baseline is what makes that
          claim falsifiable."
        </p>
        <p>
          The headline result for Phase 11 (endogenous origination)
          is a <em>negative result</em> at the 65M-on-fineweb
          scale: the matched-compute perplexity gap is real but
          inside one pooled seed sd. The paper says so. The
          matched-compute protocol is what lets it say so.
        </p>
        <p>
          A reproduction recipe, hyperparameter appendix, and PCS
          retention-score derivation live in the paper's appendices.
          Every table reads from <code>runs/*.json</code> so the
          numbers can be regenerated by anyone who can run the
          scripts.
        </p>
        <div style="margin-top: var(--s-5); display: flex; gap: var(--s-3); flex-wrap: wrap;">
          <a class="ucsa-btn ucsa-btn--primary" href="{{ '/paper/PAPER/' | relative_url }}">
            Read the paper <span class="ucsa-btn__arrow">→</span>
          </a>
          <a class="ucsa-btn ucsa-btn--ghost" href="{{ '/paper/TABLES/' | relative_url }}">
            See the tables
          </a>
        </div>
      </div>

      <div>
        <div class="ucsa-paper__table">
          <div class="ucsa-paper__table-head">
            <span class="ucsa-paper__table-title">table 1 — matched compute (illustrative)</span>
            <span class="ucsa-paper__table-tag">runs/ucsa-full-seed42.json</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Params</th>
                <th>Val PPL</th>
                <th>Δ vs baseline</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>vanilla-transformer</code></td>
                <td class="ucsa-paper__table--num">63M</td>
                <td class="ucsa-paper__table--num">1.0413 ± 0.0053</td>
                <td>—</td>
              </tr>
              <tr>
                <td><code>ucsa-full</code></td>
                <td class="ucsa-paper__table--num">63M</td>
                <td class="ucsa-paper__table--num">1.0610 ± 0.0119</td>
                <td>+0.0197</td>
              </tr>
              <tr>
                <td><code>ucsa-no-jepa</code></td>
                <td class="ucsa-paper__table--num">63M</td>
                <td class="ucsa-paper__table--num">—</td>
                <td>ablation</td>
              </tr>
              <tr>
                <td><code>ucsa-no-ema</code></td>
                <td class="ucsa-paper__table--num">63M</td>
                <td class="ucsa-paper__table--num">—</td>
                <td>ablation</td>
              </tr>
              <tr>
                <td><code>ucsa-no-recon</code></td>
                <td class="ucsa-paper__table--num">63M</td>
                <td class="ucsa-paper__table--num">—</td>
                <td>ablation</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p style="font-size: var(--type-sm); color: var(--ink-500); margin: var(--s-3) 0 0 0;">
          Numbers above are illustrative placeholders. The live
          table is regenerated from <code>runs/*.json</code> by
          <a href="{{ '/paper/TABLES/' | relative_url }}">paper/TABLES.md</a>.
        </p>
      </div>
    </div>
  </div>
</section>

<!-- =====================================================================
     Section 10 · Limitations and honesty
     ===================================================================== -->
<section class="ucsa-section" id="limitations">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">limitations &amp; honesty</span>
      <h2 class="ucsa-section-head__title">What the architecture does not claim.</h2>
      <p class="ucsa-section-head__lede">
        This section is deliberately visible. Trust comes from
        stating the limits clearly, not from omitting them.
      </p>
    </div>

    <div class="ucsa-stack-lg">

      <div class="ucsa-callout">
        <div class="ucsa-callout__head">
          <div class="ucsa-callout__icon">!</div>
          <span class="ucsa-callout__title">scale-bound negative result</span>
        </div>
        <div class="ucsa-callout__body">
          <p>
            At the 65M-on-fineweb scale the matched-compute
            perplexity gap between UCSA and the vanilla
            Transformer is real (+0.0197) but inside one pooled
            seed sd (2.14 sd away from zero, not 0 sd). The paper
            says so. Closing this requires running the full
            12k-step recipe at larger scale; that is in progress
            and not yet a positive result.
          </p>
        </div>
      </div>

      <div class="ucsa-callout ucsa-callout--accent">
        <div class="ucsa-callout__head">
          <div class="ucsa-callout__icon">→</div>
          <span class="ucsa-callout__title">what needs validation</span>
        </div>
        <div class="ucsa-callout__body">
          <ul class="ucsa-callout__list">
            <li>
              <strong>Origination is localisable at the small scale.</strong>
              The headline claim (2–4 of 16 intent slots carry
              gradient, ablation of an attributed slot moves the
              action, ablation of an unattributed slot does not)
              has been measured on a 64-hidden 4-layer model
              trained on a copy task. Generalising to the
              full-scale UCSA requires re-running the same probe.
            </li>
            <li>
              <strong>Intent descent does not yet beat matched controls.</strong>
              At the small scale, descent helps when the critic
              and the realised outcome agree, and it hurts when
              they disagree. A learned verifier trained on a
              genuine outcome signal is required for the descent
              to be useful at scale.
            </li>
            <li>
              <strong>Memory service runs in-process.</strong>
              For paper-grade runs, the verification,
              consolidation, and pruning workers should be
              separated into a separate process; today they run on
              a background thread inside the trainer.
            </li>
          </ul>
        </div>
      </div>

      <div class="ucsa-callout ucsa-callout--stable">
        <div class="ucsa-callout__head">
          <div class="ucsa-callout__icon">✓</div>
          <span class="ucsa-callout__title">what is stable</span>
        </div>
        <div class="ucsa-callout__body">
          <ul class="ucsa-callout__list">
            <li>
              <strong>Architecture.</strong> The seven-bank PCS,
              the reasoning loop with differentiable carry, the
              multi-step JEPA chain with EMA targets, and the
              held-out intent bank are first-class.
            </li>
            <li>
              <strong>Reproducibility.</strong> Deterministic
              seeding, Hydra/OmegaConf config composition, seed-
              deterministic eval harness, safetensors checkpoints
              with full metadata, and a CI matrix that fails on
              coverage regression.
            </li>
            <li>
              <strong>Protocol.</strong> Matched-compute baselines
              and a standard LM eval harness catch negative
              results and let the paper say so.
            </li>
          </ul>
        </div>
      </div>

    </div>
  </div>
</section>

<!-- =====================================================================
     Section 11 · Contribution model
     ===================================================================== -->
<section class="ucsa-section ucsa-section--alt" id="contributing">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">contributing</span>
      <h2 class="ucsa-section-head__title">A clean developer onboarding path.</h2>
      <p class="ucsa-section-head__lede">
        The contributor workflow is short, structured, and
        welcoming. Open an issue, send a PR, watch the CI run.
      </p>
    </div>

    <div class="ucsa-contrib">

      <div class="ucsa-contrib__copy">
        <p>
          UCSA is research code with a software-quality bar. The
          contributor workflow mirrors that: dev setup, lint, type
          check, tests, PR template, and a code review checklist.
          The maintainers answer issues on a best-effort basis.
        </p>
        <p>
          For substantive changes — new banks, new loss terms,
          operator swaps — open an issue first so the design
          discussion happens before the code review. For fixes and
          small improvements, a PR with tests and a docs update is
          enough.
        </p>
        <p>
          The full guide lives at
          <a href="{{ '/docs/contributing/' | relative_url }}">docs/contributing.md</a>.
          It covers every command a contributor will need, the
          conventional-commits style, and the merge criteria.
        </p>
        <div style="margin-top: var(--s-5); display: flex; gap: var(--s-3); flex-wrap: wrap;">
          <a class="ucsa-btn ucsa-btn--primary" href="{{ '/docs/contributing/' | relative_url }}">
            Read the contributor guide <span class="ucsa-btn__arrow">→</span>
          </a>
          <a class="ucsa-btn ucsa-btn--ghost" href="https://github.com/sachncs/ucsa/issues" rel="noopener">
            Open an issue ↗
          </a>
        </div>
      </div>

      <div class="ucsa-contrib__flow">

        <div class="ucsa-contrib-step">
          <span class="ucsa-contrib-step__num">step 01</span>
          <h3 class="ucsa-contrib-step__title">Dev setup</h3>
          <p class="ucsa-contrib-step__desc">
            Clone, create a venv, install with the
            <code>[dev]</code> extra.
          </p>
          <code class="ucsa-contrib-step__cmd">pip install -e ".[dev]"</code>
        </div>

        <div class="ucsa-contrib-step">
          <span class="ucsa-contrib-step__num">step 02</span>
          <h3 class="ucsa-contrib-step__title">Lint &amp; format</h3>
          <p class="ucsa-contrib-step__desc">
            Ruff for lint, black for format. Runs in CI; failures
            block the PR.
          </p>
          <code class="ucsa-contrib-step__cmd">ruff check ucsa tests scripts</code>
        </div>

        <div class="ucsa-contrib-step">
          <span class="ucsa-contrib-step__num">step 03</span>
          <h3 class="ucsa-contrib-step__title">Type-check</h3>
          <p class="ucsa-contrib-step__desc">
            Mypy on <code>ucsa/</code>. CI on Python 3.11 and
            3.12.
          </p>
          <code class="ucsa-contrib-step__cmd">mypy ucsa</code>
        </div>

        <div class="ucsa-contrib-step">
          <span class="ucsa-contrib-step__num">step 04</span>
          <h3 class="ucsa-contrib-step__title">Tests</h3>
          <p class="ucsa-contrib-step__desc">
            600 fast tests (607 total). The slow marker runs in a separate
            job. New behaviour must ship with a test.
          </p>
          <code class="ucsa-contrib-step__cmd">pytest -q -m "not slow"</code>
        </div>

        <div class="ucsa-contrib-step">
          <span class="ucsa-contrib-step__num">step 05</span>
          <h3 class="ucsa-contrib-step__title">Smoke test</h3>
          <p class="ucsa-contrib-step__desc">
            Five training steps. Exits in under a minute on CPU.
            Catches wiring regressions.
          </p>
          <code class="ucsa-contrib-step__cmd">scripts/train.py --max-steps 5 --skip-baselines</code>
        </div>

        <div class="ucsa-contrib-step">
          <span class="ucsa-contrib-step__num">step 06</span>
          <h3 class="ucsa-contrib-step__title">PR &amp; review</h3>
          <p class="ucsa-contrib-step__desc">
            Conventional Commits, filled-in PR template, code
            review checklist in the contributor guide.
          </p>
          <code class="ucsa-contrib-step__cmd">git commit -m "feat: ..."</code>
        </div>

      </div>

    </div>
  </div>
</section>

<!-- =====================================================================
     Section 12 · Closing note
     ===================================================================== -->
<section class="ucsa-section ucsa-section--ink" id="closing">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow" style="color: var(--accent-soft);">closing note</span>
      <h2 class="ucsa-section-head__title">A flagship research system.</h2>
      <p class="ucsa-section-head__lede">
        A clear architectural thesis. A navigable documentation
        surface. A trustworthy release process. A research
        direction worth following.
      </p>
    </div>

    <div style="display: flex; flex-wrap: wrap; gap: var(--s-3); margin-top: var(--s-6);">
      <a class="ucsa-btn ucsa-btn--accent" href="{{ '/docs/getting-started/' | relative_url }}">
        Get started <span class="ucsa-btn__arrow">→</span>
      </a>
      <a class="ucsa-btn" style="background: transparent; color: #fff; border-color: rgba(255,255,255,0.4);" href="{{ '/paper/PAPER/' | relative_url }}">
        Read the paper
      </a>
      <a class="ucsa-btn" style="background: transparent; color: #fff; border-color: rgba(255,255,255,0.4);" href="https://github.com/sachncs/ucsa" rel="noopener">
        Source on GitHub ↗
      </a>
    </div>
  </div>
</section>
