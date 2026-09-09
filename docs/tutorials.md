---
layout: default
title: Tutorials
permalink: /docs/tutorials/
nav_order: 4
---

# Tutorials

End-to-end walkthroughs. Each tutorial is a self-contained Python
script (or a sequence of them) that you can copy into a `.py` file
and run from the repository root after `pip install -e ".[dev]"`.

## Tutorial 1 — Building a UCSA from scratch on a toy task

This is the smallest end-to-end UCSA you can write. It builds the
PCS, the operator, the reasoning loop, the projection heads, and
the trainer, then trains for a few hundred steps on a synthetic
copy task.

```python
"""Minimal end-to-end UCSA on a copy task."""

from __future__ import annotations

import torch

from ucsa.models.perception import Perception, TokenizerWrapper
from ucsa.models.state import PCSConfig, PersistentCognitiveState
from ucsa.models.transformer_operator import (
    TransformerOperator,
    TransformerOperatorConfig,
)
from ucsa.models.reasoning_loop import (
    ReasoningLoop,
    ReasoningLoopConfig,
)
from ucsa.models.projection_heads import (
    HeadConfig,
    ProjectionHeads,
)
from ucsa.models.ucsa import UCSA, UCSAConfig
from ucsa.models.losses import UCSACombinedLoss
from ucsa.training.trainer import Trainer, TrainerConfig
from ucsa.training.curriculum import Curriculum, CurriculumSchedule
from ucsa.training.dataset import DatasetConfig, TextDataset
from ucsa.utils.seed import set_seed


def main() -> None:
    set_seed(0)

    vocab_size = 64
    hidden_size = 64

    pcs = PersistentCognitiveState(PCSConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    ))

    operator = TransformerOperator(TransformerOperatorConfig(
        hidden_size=hidden_size,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        vocab_size=vocab_size,
    ))

    heads = ProjectionHeads(HeadConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_plan_tokens=8,
        num_tools=8,
        memory_query_dim=hidden_size,
        reconstruction_dim=hidden_size,
        origination_top_k=2,
        origination_aux_loss_weight=0.01,
        intent_update_scale=0.1,
    ))

    loop = ReasoningLoop(
        operator=operator,
        config=ReasoningLoopConfig(num_iterations=4),
        origination=heads.origination,
        intent_update=heads.intent_update,
    )

    tokenizer = TokenizerWrapper(
        tokenizer_name="gpt2", max_seq_len=16
    )
    perception = Perception(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
    )

    model = UCSA(
        UCSAConfig(
            hidden_size=hidden_size,
            vocab_size=tokenizer.vocab_size,
            num_layers=2,
        ),
        pcs=pcs,
        perception=perception,
        reasoning_loop=loop,
        heads=heads,
        memory=None,
        memory_service=None,
        graph_service=None,
        verifier=None,
    )

    loss_fn = UCSACombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=TrainerConfig(
            learning_rate=3e-4,
            max_steps=200,
            warmup_steps=20,
            amp_dtype=torch.float32,
        ),
        curriculum=Curriculum(
            CurriculumSchedule(
                stage_1_end=1, stage_2_end=2, stage_3_end=3
            )
        ),
    )

    # Toy data: random token ids.
    inputs = torch.randint(1, vocab_size, (32, 16))
    targets = torch.roll(inputs, -1, dims=1)
    ds = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=True
    )

    history = trainer.train(loader)
    print(
        "final loss:",
        history[-1]["training_loss"],
    )


if __name__ == "__main__":
    main()
```

Run it with:

```bash
.venv/bin/python tutorial_1_minimal_ucsa.py
```

You should see the loss decreasing from ~4.2 (uniform over 64) to
~3.0 within 200 steps.

## Tutorial 2 — Wiring a custom PCS bank

The default PCS has seven banks. If you want to add an eighth bank
(for example, a `world_state` bank the operator reads but never
writes), extend `PCSConfig` and override the bank sizes.

```python
"""UCSA with a custom 8th PCS bank."""

from __future__ import annotations

from ucsa.models.state import (
    BANK_NAMES, PCSConfig, PersistentCognitiveState,
)


def main() -> None:
    # Add 'world_state' as an 8th bank.
    sizes = {name: 32 for name in BANK_NAMES}
    sizes["world_state"] = 16

    pcs = PersistentCognitiveState(PCSConfig(
        hidden_size=64,
        vocab_size=64,
        bank_sizes=sizes,
    ))

    print("Banks:", pcs.bank_names())
    print("Sizes:", {n: pcs.bank_size(n) for n in pcs.bank_names()})


if __name__ == "__main__":
    main()
```

`PCSConfig.bank_sizes` overrides the defaults for any bank you
specify; the rest fall back to the standard sizes.

## Tutorial 3 — Designing your own ablation

Every ablation flag on `scripts/train.py` is a one-line override
that zeroes out one loss weight. If you want to design a new one
(for example, disabling only the JEPA chain's Gaussian
regulariser), edit `ucsa/configs/default.yaml` and add a new
argument to the argparse block.

```python
"""Ablation: JEPA chain on, Gaussian regulariser off."""

from __future__ import annotations

from ucsa.models.losses import LossWeights, UCSACombinedLoss


def main() -> None:
    loss_fn = UCSACombinedLoss(
        weights=LossWeights(
            jepa=1.0,        # JEPA chain contributes
            ar=1.0,
            memory=0.01,
            router=0.01,
            reconstruction=0.1,
            origination=0.0,  # origination disabled
        ),
        jepa_mode="lewm",
        gaussian_reg_weight=0.0,  # disable the Gaussian regulariser
    )
    # ... wire up the rest of the trainer as in Tutorial 1.


if __name__ == "__main__":
    main()
```

The same approach works for any loss term: set its weight to 0.0 to
ablate it, leave it at its default to keep it on.

## Tutorial 4 — Running matched-compute experiments

A matched-compute run compares UCSA against a vanilla Transformer
with the same parameter count, the same dataset, the same step
budget, and the same optimiser. The script does the grid search
over `(hidden, num_layers, ffn_mult)` for you.

```bash
.venv/bin/python scripts/train_baseline.py \
    --seed 42 \
    --out-json runs/baseline.json
```

The output JSON includes the chosen config, the per-step history,
the final validation perplexity, and the best validation perplexity.
Compare it against `runs/ucsa-<tag>-seed42.json` to see whether
UCSA's extra structure pays for itself.

For a multi-seed sweep:

```bash
for seed in 42 43 44; do
    .venv/bin/python scripts/train_baseline.py \
        --seed $seed \
        --out-json runs/baseline-seed${seed}.json
done
```

Then build the table:

```bash
.venv/bin/python scripts/build_paper_tables.py \
    --runs-dir runs \
    --out-md paper/TABLES.md
```

## Tutorial 5 — Probing the intent bank

The intent bank is the origination signal. To check whether the
gating is doing what you think it is, run the per-slot attribution
probe:

```bash
.venv/bin/python scripts/probe_origination.py \
    --ckpt ckpts/ucsa-final.safetensors \
    --num-inputs 8 \
    --intent-steps 0 1 3 5 \
    --out-json runs/origination-probe.json
```

The output JSON reports:

- **Collapse**: variance, entropy, MI, read share for the configured
  origination setup.
- **Localisation**: per-slot attribution, controllability, and
  specificity rates.
- **Descent**: realised improvement after K steps of gradient
  descent on the `intent` bank, against matched controls.

Three readings per K:

- `predicted_improved`: did the JEPA critic predict improvement?
- `realized_improved`: did the realised loss actually drop?
- `forward_model_gamed`: did the predictor game the verifier?
- `outcome_correlation`: correlation between predicted and realised
  improvement.

A negative `outcome_correlation` is the headline result: descent
helps when the critic and the realised outcome agree, and it hurts
when they disagree. See `paper/PAPER.md` §5.1.2 for the matched-
compute table.

## Where to go next

- [Architecture →](architecture.md) for the design notes behind
  every tutorial.
- [API reference →](api-reference.md) for the full module tour.
- [Contributing →](contributing.md) for the PR workflow.
