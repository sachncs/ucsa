---
layout: default
title: Getting started
permalink: /docs/getting-started/
nav_order: 1
---

# Getting started

This page walks you from a clean checkout to a working UCSA training
run. The smoke test takes about a minute on a CPU; the full paper
reproduction takes hours on a GPU.

## Prerequisites

- Python 3.11 or 3.12
- A virtual environment tool (`venv`, `uv`, `conda`)
- PyTorch 2.1 or newer with a working CUDA, MPS, or CPU backend
- An optional HuggingFace token for downloading the GPT-2 baselines
  used by `scripts/eval.py`

## Install

```bash
git clone https://github.com/sachncs/ucsa.git
cd ucsa

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

The `[dev]` extra pulls in `pytest`, `pytest-cov`, `ruff`, `black`,
`mypy`, and `pre-commit`.

## Verify the install

```bash
pytest -q -m "not slow"        # ~600 tests, runs in ~2 minutes
ruff check ucsa tests scripts  # lint
```

## Smoke test (5 steps, ~1 minute)

The smoke test trains for five steps on the streaming fineweb-edu
dataset, writes no checkpoints, runs no eval, and skips the GPT-2
baseline comparison. It exists to confirm that the install, the
operator, the PCS, the JEPA chain, the projection heads, the EMA
target encoder, and the trainer all wire up correctly.

```bash
.venv/bin/python scripts/train.py \
    --max-steps 5 \
    --ckpt-every 0 \
    --eval-every 0 \
    --skip-baselines \
    --seed 42
```

Expected output: a JSON report at `runs/ucsa-<tag>-seed42.json` with
a `final_val_loss` between 8 and 12 (random) and a `n_ucsa_params`
count close to 63 million.

## Full reproduction (~hours on GPU)

```bash
.venv/bin/python scripts/train.py --seed 42
.venv/bin/python scripts/train_baseline.py --seed 42 --out-json runs/baseline.json
.venv/bin/python scripts/eval.py \
    --ucsa-ckpt ckpts/ucsa-final.safetensors \
    --baseline-results runs/baseline.json \
    --out-json runs/eval-ucsa-small.json
.venv/bin/python scripts/probe_banks.py \
    --ckpt ckpts/ucsa-final.safetensors \
    --out-json runs/bank-probe.json
.venv/bin/python scripts/run_ablations.py \
    --max-steps 4000 --seeds 42 43 44
.venv/bin/python scripts/build_paper_tables.py \
    --runs-dir runs --out-md paper/TABLES.md
```

The first command writes `ckpts/ucsa-final.safetensors` and
`runs/ucsa-<tag>-seed42.json`. Each subsequent command reads from the
previous artefacts and writes the next.

## Configuration

Every hyperparameter lives in `ucsa/configs/default.yaml`. Override
on the CLI:

```bash
.venv/bin/python scripts/train.py \
    model.hidden_size=128 \
    training.learning_rate=1e-3 \
    --max-steps 2000
```

The override syntax is Hydra / OmegaConf. Dotted keys navigate into
nested sections; the leading `--` introduces a non-config CLI flag
(handled by `argparse`).

## Ablation flags

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

- [Architecture →](architecture.md) for the design notes.
- [API reference →](api-reference.md) for the module tour.
- [Tutorials →](tutorials.md) for end-to-end walkthroughs.
- [Contributing →](contributing.md) for the developer workflow.
