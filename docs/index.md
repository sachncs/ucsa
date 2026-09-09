---
layout: home
title: Documentation
description: Documentation for UCSA — installation, architecture, API reference, tutorials, contributing, paper draft.
permalink: /docs/
---

<!-- =====================================================================
     Section 1 · Docs landing header
     ===================================================================== -->
<section class="ucsa-hero" style="padding-top: clamp(var(--s-8), 6vw, var(--s-12)); padding-bottom: clamp(var(--s-6), 5vw, var(--s-10));">
  <div class="ucsa-container">
    <div class="ucsa-hero__grid">
      <div class="ucsa-hero__copy">
        <span class="ucsa-hero__eyebrow">
          <span class="ucsa-hero__eyebrow-dot"></span>
          documentation
        </span>
        <h1 class="ucsa-hero__title">
          Documentation, <em>organised by role.</em>
        </h1>
        <p class="ucsa-hero__sub">
          The UCSA documentation surface is built for four
          audiences: <strong>implementers</strong> swapping the
          operator, <strong>researchers</strong> scrutinising the
          thesis, <strong>evaluators</strong> rerunning the
          numbers, and <strong>contributors</strong> opening PRs.
          Pick a role below; every link is curated.
        </p>
        <div class="ucsa-hero__ctas">
          <a class="ucsa-btn ucsa-btn--primary" href="/ucsa/docs/getting-started/">
            Get started <span class="ucsa-btn__arrow">→</span>
          </a>
          <a class="ucsa-btn ucsa-btn--ghost" href="/ucsa/docs/architecture/">
            Read architecture
          </a>
          <a class="ucsa-btn ucsa-btn--ghost" href="/ucsa/paper/PAPER/">
            Paper draft
          </a>
        </div>
      </div>
      <div class="ucsa-hero__visual">
        <div class="ucsa-hero__visual-frame">
          <div class="ucsa-hero__visual-head">
            <span class="ucsa-hero__visual-title">documentation surface</span>
            <span class="ucsa-hero__visual-tag">10 pages · 1 sidebar</span>
          </div>
          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--s-3);">
            <a href="/ucsa/docs/getting-started/" style="display: block; padding: var(--s-3); background: var(--paper-2); border: 1px solid var(--line); border-radius: var(--r-2); color: var(--ink-900); font-weight: 600; font-size: var(--type-sm);">Getting started →</a>
            <a href="/ucsa/docs/architecture/" style="display: block; padding: var(--s-3); background: var(--paper-2); border: 1px solid var(--line); border-radius: var(--r-2); color: var(--ink-900); font-weight: 600; font-size: var(--type-sm);">Architecture →</a>
            <a href="/ucsa/docs/api-reference/" style="display: block; padding: var(--s-3); background: var(--paper-2); border: 1px solid var(--line); border-radius: var(--r-2); color: var(--ink-900); font-weight: 600; font-size: var(--type-sm);">API reference →</a>
            <a href="/ucsa/docs/tutorials/" style="display: block; padding: var(--s-3); background: var(--paper-2); border: 1px solid var(--line); border-radius: var(--r-2); color: var(--ink-900); font-weight: 600; font-size: var(--type-sm);">Tutorials →</a>
            <a href="/ucsa/docs/contributing/" style="display: block; padding: var(--s-3); background: var(--paper-2); border: 1px solid var(--line); border-radius: var(--r-2); color: var(--ink-900); font-weight: 600; font-size: var(--type-sm);">Contributing →</a>
            <a href="/ucsa/paper/PAPER/" style="display: block; padding: var(--s-3); background: var(--paper-2); border: 1px solid var(--line); border-radius: var(--r-2); color: var(--ink-900); font-weight: 600; font-size: var(--type-sm);">Paper draft →</a>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- =====================================================================
     Section 2 · Role-based entry
     ===================================================================== -->
<section class="ucsa-section ucsa-section--tight">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">pick your role</span>
      <h2 class="ucsa-section-head__title">Four ways in.</h2>
      <p class="ucsa-section-head__lede">
        Each role has a curated entry path. The links below take
        you to the pages that matter most for your work — not to
        a generic getting-started.
      </p>
    </div>

    <div class="ucsa-roles">

      <article class="ucsa-role">
        <div class="ucsa-role__icon">IM</div>
        <h3 class="ucsa-role__title">Implementers</h3>
        <p class="ucsa-role__desc">
          You want to build UCSA, swap the operator, add a bank,
          wire a custom loss, or plug UCSA into your own training
          pipeline.
        </p>
        <div class="ucsa-role__links">
          <a class="ucsa-role__link" href="/ucsa/docs/api-reference/">→ API reference</a>
          <a class="ucsa-role__link" href="/ucsa/docs/tutorials/">→ Tutorials</a>
          <a class="ucsa-role__link" href="/ucsa/docs/getting-started/">→ Getting started</a>
        </div>
      </article>

      <article class="ucsa-role">
        <div class="ucsa-role__icon">RS</div>
        <h3 class="ucsa-role__title">Researchers</h3>
        <p class="ucsa-role__desc">
          You want to understand the architectural thesis, follow
          the math, scrutinise the matched-compute protocol, and
          read the working paper draft.
        </p>
        <div class="ucsa-role__links">
          <a class="ucsa-role__link" href="/ucsa/docs/architecture/">→ Architecture</a>
          <a class="ucsa-role__link" href="/ucsa/paper/PAPER/">→ Paper draft</a>
          <a class="ucsa-role__link" href="/ucsa/paper/TABLES/">→ Paper tables</a>
        </div>
      </article>

      <article class="ucsa-role">
        <div class="ucsa-role__icon">EV</div>
        <h3 class="ucsa-role__title">Evaluators</h3>
        <p class="ucsa-role__desc">
          You want to rerun the numbers: smoke test, full
          reproduction, ablation sweep, eval harness, paper tables.
        </p>
        <div class="ucsa-role__links">
          <a class="ucsa-role__link" href="/ucsa/docs/getting-started/">→ Getting started</a>
          <a class="ucsa-role__link" href="/ucsa/paper/TABLES/">→ Paper tables</a>
        </div>
      </article>

      <article class="ucsa-role">
        <div class="ucsa-role__icon">CT</div>
        <h3 class="ucsa-role__title">Contributors</h3>
        <p class="ucsa-role__desc">
          You want to open a PR: dev setup, lint, type-check,
          tests, smoke test, and the code review checklist.
        </p>
        <div class="ucsa-role__links">
          <a class="ucsa-role__link" href="/ucsa/docs/contributing/">→ Contributing</a>
          <a class="ucsa-role__link" href="https://github.com/sachncs/ucsa/issues" rel="noopener">→ Issues ↗</a>
        </div>
      </article>

    </div>
  </div>
</section>

<!-- =====================================================================
     Section 3 · Full doc index
     ===================================================================== -->
<section class="ucsa-section ucsa-section--alt">
  <div class="ucsa-container">
    <div class="ucsa-section-head">
      <span class="ucsa-section-head__eyebrow">all pages</span>
      <h2 class="ucsa-section-head__title">Every page, in one place.</h2>
      <p class="ucsa-section-head__lede">
        Ten documentation pages plus the paper draft, paper
        tables, changelog, security, and support. Each one has a
        single purpose.
      </p>
    </div>

    <div class="ucsa-doc-grid">
      <a class="ucsa-doc-tile" href="/ucsa/docs/getting-started/">
        <span class="ucsa-doc-tile__eyebrow">onboarding</span>
        <h4 class="ucsa-doc-tile__title">Getting started</h4>
        <p class="ucsa-doc-tile__desc">
          Install, smoke test, full reproduction, ablation flags.
        </p>
      </a>
      <a class="ucsa-doc-tile" href="/ucsa/docs/architecture/">
        <span class="ucsa-doc-tile__eyebrow">design</span>
        <h4 class="ucsa-doc-tile__title">Architecture</h4>
        <p class="ucsa-doc-tile__desc">
          Deep design notes for every subsystem, with the maths.
        </p>
      </a>
      <a class="ucsa-doc-tile" href="/ucsa/docs/api-reference/">
        <span class="ucsa-doc-tile__eyebrow">reference</span>
        <h4 class="ucsa-doc-tile__title">API reference</h4>
        <p class="ucsa-doc-tile__desc">
          Module-by-module tour of <code>ucsa/</code>, organised
          by responsibility.
        </p>
      </a>
      <a class="ucsa-doc-tile" href="/ucsa/docs/tutorials/">
        <span class="ucsa-doc-tile__eyebrow">walkthroughs</span>
        <h4 class="ucsa-doc-tile__title">Tutorials</h4>
        <p class="ucsa-doc-tile__desc">
          Five end-to-end examples: build, customise, ablate,
          measure, probe.
        </p>
      </a>
      <a class="ucsa-doc-tile" href="/ucsa/docs/contributing/">
        <span class="ucsa-doc-tile__eyebrow">dev workflow</span>
        <h4 class="ucsa-doc-tile__title">Contributing</h4>
        <p class="ucsa-doc-tile__desc">
          Setup, lint, type-check, tests, PR flow, code review
          checklist.
        </p>
      </a>
      <a class="ucsa-doc-tile" href="/ucsa/CHANGELOG/">
        <span class="ucsa-doc-tile__eyebrow">release notes</span>
        <h4 class="ucsa-doc-tile__title">Changelog</h4>
        <p class="ucsa-doc-tile__desc">
          Per-phase history of the project, with the matching
          tests added.
        </p>
      </a>
      <a class="ucsa-doc-tile" href="/ucsa/paper/PAPER/">
        <span class="ucsa-doc-tile__eyebrow">research</span>
        <h4 class="ucsa-doc-tile__title">Paper draft</h4>
        <p class="ucsa-doc-tile__desc">
          The full paper draft, including the negative-results
          section.
        </p>
      </a>
      <a class="ucsa-doc-tile" href="/ucsa/paper/TABLES/">
        <span class="ucsa-doc-tile__eyebrow">results</span>
        <h4 class="ucsa-doc-tile__title">Paper tables</h4>
        <p class="ucsa-doc-tile__desc">
          Generated from <code>runs/*.json</code>. The numbers
          the paper claims.
        </p>
      </a>
      <a class="ucsa-doc-tile" href="/ucsa/SECURITY/">
        <span class="ucsa-doc-tile__eyebrow">policy</span>
        <h4 class="ucsa-doc-tile__title">Security</h4>
        <p class="ucsa-doc-tile__desc">
          Supported versions matrix and the private-disclosure
          SLA.
        </p>
      </a>
      <a class="ucsa-doc-tile" href="/ucsa/SUPPORT/">
        <span class="ucsa-doc-tile__eyebrow">help</span>
        <h4 class="ucsa-doc-tile__title">Support</h4>
        <p class="ucsa-doc-tile__desc">
          Where to ask questions: issues for bugs, discussions
          for design.
        </p>
      </a>
    </div>
  </div>
</section>
