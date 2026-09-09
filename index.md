---
layout: default
title: UCSA — Unified Cognitive State Architecture
description: "UCSA is a research-grade foundation model whose computation orbits a single persistent differentiable cognitive state. This site is the project documentation hub: getting started, architecture, API reference, tutorials, and the working paper draft."
logo: /assets/logo.svg
show_downloads: false
permalink: /
---

# UCSA — Unified Cognitive State Architecture

> A research-grade foundation model whose entire computation orbits a
> single, persistent, differentiable cognitive state (PCS).

UCSA is two ideas at once:

1. **Architecturally**, every projection — language logits, JEPA
   predictions, input reconstruction, memory, planning, tool — reads
   from or writes to the same seven-bank PCS.
2. **Training-wise**, UCSA introduces a **multi-step JEPA prediction
   chain** across the reasoning loop's intermediates, with a hard-EMA
   target encoder tracking latents across the chain.

The repository contains the implementation, the matched-compute
baselines, the eval harness, and the paper draft. This site is the
documentation hub for developers who want to tinker with or build on
UCSA.

---

## Quick links

| I want to… | Start here |
| --- | --- |
| Install UCSA and run a smoke test | [Getting started →](docs/getting-started.md) |
| Understand the architecture | [Architecture →](docs/architecture.md) |
| Look up a module or class | [API reference →](docs/api-reference.md) |
| Walk through an end-to-end example | [Tutorials →](docs/tutorials.md) |
| Add a feature, fix a bug, run the tests | [Contributing →](docs/contributing.md) |
| Read the paper draft | [PAPER.md →](paper/PAPER.md) |
| Track the project roadmap | [TODO.md →](TODO.md) |

---

## What the documentation covers

### Getting started

The fastest path to a running UCSA on your machine. Includes the
5-step smoke command, the full 12k-step reproduction recipe, and the
configuration knobs you will need most often.

### Architecture

Deep design notes for the seven-bank persistent cognitive state, the
reasoning loop, the multi-step JEPA chain, the memory pipeline, the
projection heads, and the loss weights. Includes a one-paragraph
summary, a section on each subsystem, and the mathematical
specifications.

### API reference

Module-by-module tour of `ucsa/`, organised by responsibility. Each
entry lists the public surface, the key dataclasses and protocols,
and a link to the source.

### Tutorials

End-to-end walkthroughs:

- Building a UCSA from scratch on a toy task
- Wiring a custom PCS bank
- Designing your own ablation
- Running matched-compute experiments
- Probing the intent bank

### Contributing

Developer setup, lint and type-check, the test suite, the PR template,
the issue templates, and the code review checklist.

---

## The PCS in one diagram

```
                  ┌──────────── Persistent Cognitive State ────────────┐
                  │                                                   │
                  │   Working     LongTerm       Goal     Episode     │
 inputs ─► Percep ─► Memory Bank  Bank  ...      Bank     Bank  ... ─► heads
                  │       │           │            │       │
                  │       └────────┐  │            │       │
                  │                ▼  ▼            │       │
                  │   Reasoning loop              Memory Service
                  │   (operator F, N=4 iters)      (background)
                  │       │           │            │       │
                  │       └─────────┴─────────────┴───────┘
                  │   ────────────────  intent  ────────────────
                  │        (origination signal; not in stream)
                  └───────────────┬─────────────────────────┘
                                  ▼
                  jepa_multi_step pairs + aux losses
```

The seven banks are `working`, `long_term`, `goal`, `episode`,
`task`, `memory_index`, and `intent`. The first six flow through the
operator's attention stream; `intent` is held out of the stream so
that the origination generator is the only path from intent to
behaviour.

---

## Project status

- 600+ tests passing on Python 3.11 and 3.12.
- Coverage enforced at 80% in CI.
- Apache-2.0 license.
- Citation pending — see [paper/PAPER.md](paper/PAPER.md) for the
  working draft.

Source: [github.com/sachncs/ucsa](https://github.com/sachncs/ucsa).
Issues: [github.com/sachncs/ucsa/issues](https://github.com/sachncs/ucsa/issues).
Discussions: [github.com/sachncs/ucsa/discussions](https://github.com/sachncs/ucsa/discussions).
