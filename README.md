# UCSA — Unified Cognitive State Architecture

A research-grade foundation model whose entire computation revolves around a
single persistent differentiable cognitive state (PCS). Language, planning,
tool use, and the JEPA predictive objective are projections of the same state.
The transformer is one realization of the state-transition operator; future
operators (Mamba, RWKV, Hyena, SSM) plug in without changing the cognitive
architecture.

## Why

Most neural architectures scatter knowledge across parameters, activations,
KV caches, and external stores. UCSA consolidates it: there is exactly one
representation, and everything else is an operator on it. This makes the
memory hierarchy, retrieval, and reasoning loop first-class rather than
incidental.

## Layout

```
ucsa/
├── configs/         Hydra/OmegaConf YAML configuration
├── models/          PCS, operators, memory, heads, losses, top-level UCSA model
├── training/        trainer, dataset, curriculum, metrics, evaluation
├── utils/           seed, checkpoint, logging
├── tests/           comprehensive pytest suite
├── train.py         training entrypoint
└── infer.py         inference entrypoint
```

See [docs/architecture.md](docs/architecture.md) for the deep design.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

Smoke training (100 steps, tiny model):

```bash
python -m ucsa.train training.smoke=true
```

Full multi-epoch training on fineweb-edu:

```bash
python -m ucsa.train
```

Inference with KV cache:

```bash
python -m ucsa.infer prompt="Once upon a time"
```

## Configuration

All hyperparameters live in [`configs/default.yaml`](ucsa/configs/default.yaml).
Override any value via the CLI:

```bash
python -m ucsa.train model.hidden_size=256 training.batch_size=16
```

## Swapping the transition operator

Implement :class:`ucsa.models.transition_operator.StateTransitionOperator` and
register the implementation under a name in
[`configs/default.yaml`](ucsa/configs/default.yaml) under
``operator.name``. The reasoning loop, memory pipeline, and projection heads
work against the interface and require no changes.

## Tests

```bash
ruff check
mypy --strict ucsa
pytest -q
```

## License

Apache-2.0.