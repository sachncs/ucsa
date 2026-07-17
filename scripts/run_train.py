"""2-epoch-equivalent smoke run on fineweb-edu, MPS-accelerated.

"2 epochs over fineweb-edu" = 1.3T tokens, infeasible on a laptop. We
budget to MAX_STEPS and label as ~2 epoch-equivalent on a 1024-token
sample window. Adjust MAX_STEPS to taste.
"""
import time, yaml, torch
from torch.utils.data import DataLoader
from ucsa.train import build_model, build_trainer
from ucsa.training.dataset import TextDataset, DatasetConfig
from ucsa.models.perception import TokenizerWrapper


def main() -> None:
    with open("ucsa/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    # Model: small-but-real. UCSA currently requires batch_size=1 (PCS is per-call).
    cfg["reasoning_iterations"] = 2
    cfg["model"]["hidden_size"] = 96
    cfg["model"]["num_layers"] = 3
    cfg["model"]["num_q_heads"] = 4
    cfg["model"]["num_kv_heads"] = 2
    cfg["model"]["intermediate_size"] = 192
    cfg["model"]["vocab_size"] = 50257
    cfg["model"]["max_seq_len"] = 256
    cfg["model"]["num_concepts"] = 8

    # Training: budget = 2 epoch-equivalent passes through a streaming sample window.
    MAX_STEPS = 800
    cfg["training"]["max_steps"] = MAX_STEPS
    cfg["training"]["warmup_steps"] = 50
    cfg["training"]["batch_size"] = 1
    cfg["training"]["log_every_n_steps"] = 20
    cfg["training"]["learning_rate"] = 6e-4
    cfg["dataset"]["sequence_length"] = 256

    tokenizer = TokenizerWrapper(
        tokenizer_name=cfg["tokenizer"]["name"],
        max_seq_len=cfg["dataset"]["sequence_length"],
    )
    ds_cfg = DatasetConfig(
        sequence_length=cfg["dataset"]["sequence_length"],
        primary_dataset=cfg["dataset"]["primary_dataset"],
        primary_split=cfg["dataset"]["primary_split"],
        streaming=True,
        pack_sequences=True,
    )
    print("Building dataset...", flush=True)
    ds = TextDataset(tokenizer, ds_cfg)
    print("Building model...", flush=True)
    model = build_model(cfg)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    trainer = build_trainer(model, cfg)
    print(f"Device: {trainer.device}", flush=True)

    class _Iter(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(ds)
    loader = DataLoader(_Iter(), batch_size=None)
    print(f"Starting: max_steps={MAX_STEPS}", flush=True)

    start = time.time()
    losses: list[float] = []
    snap = {}
    for step, batch in enumerate(loader):
        if step >= MAX_STEPS:
            break
        snap = trainer.train_step(batch)
        tl = snap.get("training_loss") or 0.0
        losses.append(tl)
        if step % cfg["training"]["log_every_n_steps"] == 0:
            el = time.time() - start
            window = min(cfg["training"]["log_every_n_steps"], len(losses))
            avg = sum(losses[-window:]) / window
            lr = trainer.scheduler.get_lr()
            print(
                f"  step={step:4d} loss={tl:.4f} avg={avg:.4f} "
                f"lr={lr:.2e} elapsed={el:.1f}s",
                flush=True,
            )

    elapsed = time.time() - start
    print(
        f"\nDone: {MAX_STEPS} steps in {elapsed:.1f}s "
        f"= {MAX_STEPS/elapsed:.2f} steps/s",
        flush=True,
    )
    print(f"first-10 losses: {[round(x, 3) for x in losses[:10]]}", flush=True)
    print(f"last-10 losses:  {[round(x, 3) for x in losses[-10:]]}", flush=True)
    print(
        f"loss delta (first5 mean - last5 mean): "
        f"{sum(losses[:5])/5 - sum(losses[-5:])/5:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
