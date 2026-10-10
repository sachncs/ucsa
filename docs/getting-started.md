---
layout: default
title: Getting started
permalink: /docs/getting-started/
nav_order: 1
---

# Getting started

This page takes you from a clean checkout to a trained, evaluated model. The
checks take a few minutes; the full 12,000-step run takes hours on a single
laptop GPU.

## Prerequisites

- Python 3.14
- A virtual environment tool (`venv`, `uv`, `conda`)
- PyTorch with a working MPS, CUDA or CPU backend
- Network access once, to download the corpus and the benchmark datasets

## Install and verify

```bash
git clone https://github.com/sachncs/ucsa.git && cd ucsa
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q -m "not slow"                  # ~1,000 tests, a few minutes
ruff check ucsa tests scripts            # lint
```

## 1. Prepare data (once)

```bash
.venv/bin/python scripts/prepare_data.py --out data
```

Streams FineWeb-Edu, drops short and duplicate documents, tokenises with the
GPT-2 vocabulary and writes `data/train.bin`, `data/val.bin` and
`data/wikitext_test.bin`. Training reads these files, so it never touches the
network and is exactly reproducible and resumable. Optional
`--zratio LOW HIGH --tag zfilter` also writes a compressibility-filtered
training shard.

## 2. Dry run (before every long run)

```bash
.venv/bin/python scripts/dry_run.py --preset small
```

Builds the real model and batch shape, audits that every tensor is in the
library dtype, and measures memory and throughput. It exits non-zero if
anything is off. This instrumentation lives here so that training itself
carries none.

## 3. Train

```bash
.venv/bin/python scripts/train_r.py --preset small --out-dir ckpts/r-small
.venv/bin/python scripts/train_r.py --preset small --out-dir ckpts/r-small --resume
```

Presets and every field live in `ucsa/config.yaml`; override with
`--set model.window=64 train.lr=3e-4` (unknown keys are rejected, so a typo
cannot silently change nothing). `--params N` auto-sizes the model. A crashed
run resumes from its last checkpoint and continues the data stream exactly.

## 4. Evaluate

```bash
.venv/bin/python scripts/eval.py --ckpt ckpts/r-small/final.pt
.venv/bin/python scripts/compress.py --ckpt ckpts/r-small/final.pt
.venv/bin/python scripts/probe_state.py --ckpt ckpts/r-small/final.pt \
    --control ckpts/r-nostate/final.pt
.venv/bin/python scripts/report.py            # writes paper/RESULTS.md
```

Everything in one command: `scripts/reproduce.sh` (set `STEPS=300` for a
quick end-to-end check).

## Choosing a design that holds at full length

```bash
.venv/bin/python scripts/ablate.py --steps 600 --out runs/ablate-window
.venv/bin/python scripts/ladder.py --arms base no-state --out runs/ladder
.venv/bin/python scripts/ladder.py --analyze --target 12000 --out runs/ladder
```

See [UCSA-R](ucsa-r.md) for what each arm switches and how the forecast
decides.

## The original model

```bash
.venv/bin/python scripts/train.py --max-steps 5 --ckpt-every 0 --eval-every 0
```

trains the original UCSA for five steps as a smoke test. Its flags:

| Flag | Effect |
| --- | --- |
| `--no-ema` | Disable the hard-EMA target encoder. |
| `--ema-momentum N` | Set the EMA decay (default 0.996). |
| `--no-lewm` | Drop the multi-step JEPA chain. |
| `--lewm-gaussian-reg N` | Tune the multi-step Gaussian regulariser. |
| `--no-recon` | Drop the input-reconstruction loss. |
| `--reconstruction-weight N` | Set the reconstruction loss weight. |
| `--no-tc-jepa` | Drop the sparse text conditioner. |
| `--text-conditioner-scale N` | Set the text conditioner scale. |
| `--no-curriculum` | Disable the four-stage curriculum. |
| `--observation-mix N` | Enable endogenous origination (`< 1.0`). |
| `--observation-mix-decay N` | Set the per-iteration mix decay. |
| `--origination-top-k N` | Set the top-k gate sparsity. |
| `--stream-intent-bank` | Re-include `intent` in the operator stream (ablation). |
| `--seed N` | Deterministic seed. |
| `--max-steps N` | Step budget. |

## Where to go next

- [UCSA-R →](ucsa-r.md) for the design, guarantees and workflow.
- [Architecture →](architecture.md) for the design notes.
- [API reference →](api-reference.md) for the module tour.
- [Tutorials →](tutorials.md) for end-to-end walkthroughs.
- [Contributing →](contributing.md) for the developer workflow.
