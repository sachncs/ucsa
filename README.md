# UCSA — Unified Cognitive State Architecture

[![CI](https://github.com/sachncs/ucsa/actions/workflows/ci.yml/badge.svg)](https://github.com/sachncs/ucsa/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-870%20passing-brightgreen.svg)](https://github.com/sachncs/ucsa)
[![Coverage](https://img.shields.io/badge/coverage-80%25%20enforced-brightgreen.svg)](https://github.com/sachncs/ucsa)
[![Docs](https://img.shields.io/badge/docs-site-blue.svg)](https://sachncs.github.io/ucsa/)

**A research-grade foundation model whose entire computation orbits a single,
persistent, differentiable cognitive state (PCS).**

UCSA is two ideas at once:

1. **Architecturally**, every projection — language logits, JEPA
   predictions, input reconstruction, memory, planning, tool — reads from
   or writes to the same seven-bank PCS. The Transformer decoder is one
   realisation of an interchangeable transition operator; Mamba, RWKV,
   and friends are also possible implementations of the same abstraction.
2. **Training-wise**, UCSA introduces a **multi-step JEPA prediction chain**
   across the reasoning loop's intermediates, with a hard-EMA target
   encoder tracking latents across the chain. This is a single-parameter
   (EMA momentum) replacement for the multi-term loss juggling common in
   I-JEPA / LeWM-style setups.

This repository has two models. **UCSA** is the original slot-based
architecture. **UCSA-R** is its causal, sliding-window successor: it keeps the
persistent multi-bank state, makes every logit an exact next-token prediction
(the original could copy the token it was scored on), and can therefore be run
as a lossless compressor. Results are compared with *published* numbers, not
with a baseline trained here. See [docs/ucsa-r.md](docs/ucsa-r.md) for the
design and [paper/PAPER.md](paper/PAPER.md) for the write-up.

## Table of contents

- [Quickstart](#quickstart)
- [What's novel](#whats-novel)
- [Repository layout](#repository-layout)
- [The PCS in one diagram](#the-pcs-in-one-diagram)
- [Configuration](#configuration)
- [Tests](#tests)
- [Documentation](#documentation)
- [Project roadmap](#project-roadmap)
- [Citation](#citation)
- [License](#license)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.14
pip install -e ".[dev]"
pytest -q                                            # ~870 tests

# One-time: tokenise a corpus into local shards (dedupes, no network later)
.venv/bin/python scripts/prepare_data.py --out data

# Pre-flight: every tensor in one dtype, memory, speed. Gates a long run.
.venv/bin/python scripts/dry_run.py --preset small

# Train UCSA-R (resumable: add --resume after an interruption)
.venv/bin/python scripts/train_r.py --preset small --out-dir ckpts/r-small

# Zero-shot benchmarks vs published models, perplexity, bits per byte
.venv/bin/python scripts/eval.py --recurrent-ckpt ckpts/r-small/final.pt

# Real lossless compression with the model, verified by decoding
.venv/bin/python scripts/compress.py --ckpt ckpts/r-small/final.pt

# What does the state buy? (needs a --set model.use_state=false control)
.venv/bin/python scripts/probe_state.py --ckpt ckpts/r-small/final.pt \
    --control ckpts/r-nostate/final.pt
```

Choosing a design that holds at the full-run length, not just after a few
hundred steps:

```bash
.venv/bin/python scripts/ablate.py --steps 600 --out runs/ablate-window
.venv/bin/python scripts/ladder.py --arms base no-state --out runs/ladder
.venv/bin/python scripts/ladder.py --analyze --target 12000 --out runs/ladder
```

The original UCSA trains with `scripts/train.py`. Every hyperparameter of both
models is in the single file [`ucsa/config.yaml`](ucsa/config.yaml); override
UCSA-R fields with `--set model.window=64 train.lr=3e-4`. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [docs/](docs/).

## What's novel

| Contribution | Citation slot | Where it lives |
| --- | --- | --- |
| Seven-bank PCS with retention scoring + recycle policy | §3.1 | `ucsa/models/cognitive.py` |
| Multi-step JEPA chain + EMA-tracked targets | §3.3 | `ucsa/models/architecture.py` (`jepa_multi_step`) + `ucsa/training/trainer.py` |
| Hard-EMA target encoder inside UCSA | §3.5 | `ucsa/training/ema.py` |
| Input-reconstruction capacity bottleneck | §3.4 | `ucsa/models/projection.py` |
| Endogenous origination: `intent` bank + per-slot attribution | §3.6 | `ucsa/models/origination.py`, `ucsa/models/intent_descent.py` |
| Causal sliding-window model with a persistent state (UCSA-R) | §3.7 | `ucsa/models/recurrent.py` |
| Surprise-gated writes: store what the JEPA predictor missed | §3.7 | `ucsa/models/recurrent.py` |
| Lossless compression with the model; round trip proves causality | §4 | `ucsa/arithmetic.py` |
| Bits per byte, zlib/xz anchors, state probes, rate-distortion | §4 | `ucsa/training/compression.py`, `diagnostics.py` |
| Learning-curve forecasts that carry short runs to the full length | §4 | `ucsa/training/scaling.py`, `scripts/ladder.py` |
| Lm-eval-harness-faithful benchmarks vs published models | §4 | `ucsa/training/eval_harness.py`, `reference.py` |

## Repository layout

```
ucsa/
├── config.yaml      Hydra/OmegaConf configuration (one file, both models)
├── arithmetic.py    lossless arithmetic coder driven by the model
├── dryrun.py        pre-flight dtype audit, memory and speed (never in training)
├── models/          cognitive (state), tiers (memory), curation, graph,
│                    architecture (UCSA), recurrent (UCSA-R), losses, ...
├── training/        engine, shards, prefix, eval_harness, reference,
│                    compression, diagnostics, scaling, tuning, trainer, ...
├── utils/           precision (the one dtype), seed, checkpoint, logging
├── train.py         original-model training entrypoint
└── infer.py         inference entrypoint
scripts/
├── prepare_data.py  tokenise a corpus into shards
├── dry_run.py       pre-flight check that gates a long run
├── train_r.py       train UCSA-R from presets and --set overrides
├── eval.py          benchmarks vs published results, perplexity
├── compress.py      real lossless compression, verified by decoding
├── probe_state.py   what the state contributes, in compression terms
├── ablate.py        short-budget ablation matrix with paired intervals
├── ladder.py        forecast designs at the full-run length
├── tune.py          successive-halving search
├── profile_r.py     throughput and memory
└── train.py, run_ablations.py, probe_banks.py, probe_origination.py,
    build_paper_tables.py   original-model tooling
paper/               PAPER.md, RESULTS.md, artifacts/, reference_results.json
docs/
├── index.md         Jekyll landing page (GitHub Pages)
├── architecture.md  deep design notes
├── getting-started.md install, configure, smoke-test
├── api-reference.md  module-by-module API tour
├── tutorials.md     end-to-end walkthroughs
└── contributing.md  developer setup, lint, test, PR flow
```

See [paper/PAPER.md](paper/PAPER.md) for the full write-up,
[docs/architecture.md](docs/architecture.md) for the deep design notes,
and the [project site](https://sachncs.github.io/ucsa/) for the rendered
documentation.

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

The seven banks are: `working`, `long_term`, `goal`, `episode`, `task`,
`memory_index`, and `intent`. The first six flow through the operator's
attention stream; `intent` is held out of the stream so that the
origination generator is the *only* path from intent to behaviour. That
separation is what makes per-slot attribution well posed.

## Configuration

All hyperparameters live in [`ucsa/config.yaml`](ucsa/config.yaml).
Override on the CLI:

```bash
.venv/bin/python scripts/train.py \
    model.hidden_size=128 \
    training.learning_rate=1e-3 \
    --max-steps 2000
```

Ablation flags accepted by `scripts/train.py`:

- `--no-ema` / `--ema-momentum N` — disable or set the EMA decay.
- `--no-lewm` / `--lewm-gaussian-reg N` — drop the multi-step JEPA
  chain or tune its Gaussian regulariser weight.
- `--no-recon` / `--reconstruction-weight N` — drop the
  capacity-bottleneck input-reconstruction loss.
- `--no-tc-jepa` / `--text-conditioner-scale N` — drop or scale the
  sparse text conditioner.
- `--no-curriculum` — disable the four-stage curriculum.
- `--observation-mix N` / `--observation-mix-decay N` — enable
  endogenous origination with the given decay schedule.
- `--stream-intent-bank` — ablation that puts `intent` back in the
  operator stream (breaks per-slot localisation by design).
- `--seed N` — deterministic seed for Python / NumPy / PyTorch.
- `--max-steps N` — step budget.

## Tests

```bash
pytest -q               # 607 tests total
pytest -q -m "not slow" # 600 fast tests, runs in ~2 minutes
pytest -q -m slow       # 7 slow tests (the localisation-claim run)
ruff check ucsa tests scripts  # lint
ruff check --fix ucsa tests scripts  # autofix safe violations
black --check ucsa tests scripts  # format check
```

Coverage is enforced at 80% in CI
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Documentation

- [Project site](https://sachncs.github.io/ucsa/) — rendered from
  [`docs/`](docs/) on every push to `master`.
- [`docs/getting-started.md`](docs/getting-started.md) — install,
  configure, smoke-test, full reproduction.
- [`docs/architecture.md`](docs/architecture.md) — deep design notes for
  the PCS, reasoning loop, JEPA chain, memory, projection heads.
- [`docs/api-reference.md`](docs/api-reference.md) — module-by-module
  API tour.
- [`docs/tutorials.md`](docs/tutorials.md) — end-to-end walkthroughs:
  building a UCSA from scratch, wiring a custom bank, designing your
  own ablation, running matched-compute experiments.
- [`docs/contributing.md`](docs/contributing.md) — developer setup,
  lint, test, PR flow.
- [`paper/PAPER.md`](paper/PAPER.md) — the working paper draft.

## Project roadmap

See [TODO.md](TODO.md) for the full atomic-commit ledger through the
current phase (Phase 11 — Endogenous Origination). The document is the
source of truth for what is in flight, what is shipped, and what
remains to be measured.

## Citation

Pending — see [paper/PAPER.md](paper/PAPER.md) for the working draft.

## License

Apache-2.0. See [LICENSE](LICENSE).
