# UCSA — Unified Cognitive State Architecture

**A research-grade foundation model whose entire computation
orbits a single, persistent, differentiable cognitive state (PCS).**

UCSA is two ideas at once:

1. **Architecturally**, every projection — language logits, JEPA
   predictions, input reconstruction, memory, planning, tool —
   reads from or writes to the same six-bank PCS. The Transformer
   decoder is one realisation of an interchangeable transition
   operator; Mamba, RWKV, and friends are also possible
   implementations of the same abstraction.
2. **Training-wise**, UCSA introduces a **multi-step JEPA prediction
   chain** across the reasoning loop's intermediates, with a
   hard-EMA target encoder tracking latents across the chain.
   This is a single-parameter (EMA momentum) replacement for the
   multi-term loss juggling common in I-JEPA / LeWM-style setups.

This repository contains the implementation, the matched-compute
baselines used for the paper, and the eval harness.

## What's in this paper, what's novel

| Contribution | Citation slot | Where it lives |
| --- | --- | --- |
| PCS with retention scoring + recycle policy | §3.1 | `ucsa/models/state.py` |
| Multi-step JEPA chain + EMA-tracked targets | §3.3 | `ucsa/models/ucsa.py` (`jepa_multi_step`) + `ucsa/training/trainer.py` |
| Hard-EMA target encoder inside UCSA | §3.5 | `ucsa/training/ema.py` |
| Input-reconstruction capacity bottleneck | §3.4 | `ucsa/models/projection_heads.py` |
| Matched-compute baseline + standard-LM eval harness | §4 | `scripts/train_baseline.py`, `scripts/eval.py` |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

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
# 4. Probe what each PCS bank has learned
.venv/bin/python scripts/probe_banks.py \
    --ckpt ckpts/ucsa-final.safetensors \
    --out-json runs/bank-probe.json
# 5. Run the ablation sweep (5 seeds × 6 ablations is typical for the paper)
.venv/bin/python scripts/run_ablations.py \
    --max-steps 4000 --seeds 42 43 44
```

## Repository layout

```
ucsa/
├── configs/         Hydra/OmegaConf YAML configuration
├── models/          PCS, operators, memory, heads, losses, top-level UCSA
├── training/        trainer, dataset, curriculum, metrics,
│                    evaluation, EMA, Muon optimiser, eval harness
├── utils/           seed, checkpoint, logging
├── tests/           436 tests, run with `pytest -q`
├── train.py         training entrypoint
├── infer.py         inference entrypoint
└── paper/           paper draft (PAPER.md)
scripts/
├── train.py         UCSA training + SOTA stack + benchmark comparison
├── train_baseline.py matched-compute vanilla-Transformer baseline
├── eval.py          HellaSwag / ARC / PIQA / WinoGrande evaluation
├── probe_banks.py   PCS bank probe (top tokens per bank, centroid sim)
├── run_ablations.py ablation matrix driver
└── benchmark.py     one-file showcase against modern-LM baselines
```

See `paper/PAPER.md` for the full write-up and
`docs/architecture.md` for the deep design notes.

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
                 └───────────────┬─────────────────────────┘
                                 ▼
                 jepa_multi_step pairs + aux losses
```

## Reproducing every paper number

```bash
# 1. UCSA-small (full SOTA stack)
.venv/bin/python scripts/train.py --seed 42   # → ckpts/ucsa-final.safetensors
# 2. Vanilla-Transformer (matched compute)
.venv/bin/python scripts/train_baseline.py --seed 42 --out-json runs/baseline.json
# 3. Downstream accuracy on standard LM benchmarks
.venv/bin/python scripts/eval.py \
    --ucsa-ckpt ckpts/ucsa-final.safetensors \
    --baseline-results runs/baseline.json \
    --out-json runs/eval-ucsa-small.json
# 4. Ablation sweep (custom)
for ablation in full no-jepa no-ema no-recon no-tc-jepa no-curriculum; do
    .venv/bin/python scripts/train.py --seed 42 \
        "$@" \
        --out-json "runs/ucsa-${ablation}.json"
done
```

All runs emit structured JSON to `runs/`. The paper's tables
(`paper/PAPER.md`) read from those JSONs.

## Configuration

All hyperparameters live in `ucsa/configs/default.yaml`. Override
on the CLI:

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
- `--seed N` — deterministic seed for Python / NumPy / PyTorch.
- `--max-steps N` — step budget.

## Tests

```bash
pytest -q        # 428 tests
ruff check       # lint
```

## Citation

Pending — see `paper/PAPER.md` for the working draft.

## License

Apache-2.0.
