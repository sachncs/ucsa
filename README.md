# UCSA — Unified Cognitive State Architecture

[![CI](https://github.com/sachncs/ucsa/actions/workflows/ci.yml/badge.svg)](https://github.com/sachncs/ucsa/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-603%20passing-brightgreen.svg)](https://github.com/sachncs/ucsa)
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

This repository contains the implementation, the matched-compute baselines
used for the paper, and the eval harness. The full architecture write-up
lives in [paper/PAPER.md](paper/PAPER.md); deep design notes and the
mathematical specifications live in [docs/](docs/).

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

For a smoke test (exits in under a minute on CPU):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 5 training steps, no checkpoints, no eval, no baseline comparison.
.venv/bin/python scripts/train.py \
    --max-steps 5 \
    --ckpt-every 0 \
    --eval-every 0 \
    --skip-baselines \
    --seed 42
```

For the full reproduction of paper numbers (~12,000 steps, ~hours):

```bash
# Train UCSA-small on fineweb-edu (12k steps, all four losses, EMA on)
.venv/bin/python scripts/train.py --seed 42

# Train a matched-compute vanilla Transformer baseline
.venv/bin/python scripts/train_baseline.py --seed 42 \
    --out-json runs/baseline.json

# Evaluate UCSA on HellaSwag / ARC / PIQA / WinoGrande
.venv/bin/python scripts/eval.py \
    --ucsa-ckpt ckpts/ucsa-final.safetensors \
    --baseline-results runs/baseline.json \
    --out-json runs/eval-ucsa-small.json

# Probe what each PCS bank has learned
.venv/bin/python scripts/probe_banks.py \
    --ckpt ckpts/ucsa-final.safetensors \
    --out-json runs/bank-probe.json

# Run the ablation sweep (5 seeds × 6 ablations is typical for the paper)
.venv/bin/python scripts/run_ablations.py \
    --max-steps 4000 --seeds 42 43 44

# Aggregate the runs/*.json files into paper-ready tables
.venv/bin/python scripts/build_paper_tables.py \
    --runs-dir runs --out-md paper/TABLES.md
```

If you only want to verify the codebase runs end-to-end without a long
training run, the smoke command above is the path. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the developer setup recipe and
[docs/](docs/) for the architecture, API reference, and tutorials.

## What's novel

| Contribution | Citation slot | Where it lives |
| --- | --- | --- |
| Seven-bank PCS with retention scoring + recycle policy | §3.1 | `ucsa/models/state.py` |
| Multi-step JEPA chain + EMA-tracked targets | §3.3 | `ucsa/models/ucsa.py` (`jepa_multi_step`) + `ucsa/training/trainer.py` |
| Hard-EMA target encoder inside UCSA | §3.5 | `ucsa/training/ema.py` |
| Input-reconstruction capacity bottleneck | §3.4 | `ucsa/models/projection_heads.py` |
| Endogenous origination: `intent` bank + per-slot attribution | §3.6 | `ucsa/models/origination.py`, `ucsa/models/intent_descent.py` |
| Matched-compute baseline + standard-LM eval harness | §4 | `scripts/train_baseline.py`, `scripts/eval.py` |

## Repository layout

```
ucsa/
├── configs/         Hydra/OmegaConf YAML configuration
├── models/          PCS, operators, memory, heads, losses, top-level UCSA
├── training/        trainer, dataset, curriculum, metrics,
│                    evaluation, EMA, Muon optimiser, eval harness
├── utils/           seed, checkpoint, logging
├── tests/           603 tests, run with `pytest -q`
├── train.py         training entrypoint
├── infer.py         inference entrypoint
└── paper/           paper draft (PAPER.md, TABLES.md)
scripts/
├── train.py         UCSA training + SOTA stack + benchmark comparison
├── train_baseline.py matched-compute vanilla-Transformer baseline
├── eval.py          HellaSwag / ARC / PIQA / WinoGrande evaluation
├── probe_banks.py   PCS bank probe (top tokens per bank, centroid sim)
├── probe_origination.py intent-bank localisation, collapse, and descent probes
├── run_ablations.py ablation matrix driver
├── build_paper_tables.py reads runs/*.json, writes paper/TABLES.md
└── benchmark.py     one-file showcase against modern-LM baselines
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

All hyperparameters live in [`ucsa/configs/default.yaml`](ucsa/configs/default.yaml).
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
pytest -q           # 603 tests, no integration marker
pytest -q -m slow   # 603 tests including the slow localisation-claim run
ruff check          # lint
ruff check --fix    # autofix safe violations
black --check ucsa tests
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
