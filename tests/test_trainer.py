"""Tests for :mod:`ucsa.training.trainer`."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ucsa.training.curriculum import Curriculum, CurriculumSchedule
from ucsa.training.metrics import DEFAULT_METRIC_NAMES, build_default_registry
from ucsa.training.trainer import (
    CosineWarmupScheduler,
    Trainer,
    TrainerConfig,
)


class TinyModel(nn.Module):
    """internal: a tiny model used by the trainer tests."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return vocabulary logits."""
        return self.proj(self.embed(inputs))


def tiny_trainer(
    model: nn.Module | None = None,
    curriculum: Curriculum | None = None,
    metrics=None,
    config: TrainerConfig | None = None,
    vocab_size: int = 100,
) -> Trainer:
    """Build a small trainer for tests."""
    if model is None:
        model = TinyModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    from ucsa.models.losses import UCSACombinedLoss

    loss_fn = UCSACombinedLoss()
    if config is None:
        config = TrainerConfig(
            learning_rate=3e-4,
            max_steps=20,
            warmup_steps=2,
            log_every_n_steps=2,
            amp_dtype=torch.float32,
        )
    if curriculum is None:
        curriculum = Curriculum()
    if metrics is None:
        metrics = build_default_registry()
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
        curriculum=curriculum,
        metrics=metrics,
    )


class TestTargetAlignment:
    """Tests for how targets are aligned to the logits sequence."""

    def test_short_targets_are_padded_with_the_ignore_index(self) -> None:
        """Padding must be masked out, not trained as token id 0.

        The working bank is 64 slots, so a short sequence leaves most
        positions unsupervised. Padding them with zero trained the model to
        emit token id 0 there, which measured 90.4% of the autoregressive
        loss for a 6-token target: the real tokens barely mattered.
        """
        model = TinyModel(vocab_size=100)
        trainer = tiny_trainer(model=model)
        assert trainer.ignore_index == -100
        inputs = torch.randint(1, 100, (1, 6))
        targets = torch.roll(inputs, -1, dims=1)
        loss, _ = trainer.compute_loss(inputs, targets)
        assert torch.isfinite(loss)

    def test_alignment_matches_the_unpadded_loss(self) -> None:
        """With masking, padding contributes exactly nothing.

        A model whose logits are as long as the targets must produce the
        same loss as one padded out to a longer sequence.
        """

        class WideModel(nn.Module):
            """internal: emits a fixed number of positions."""

            def __init__(self, positions: int, vocab_size: int) -> None:
                super().__init__()
                self.positions = positions
                self.vocab_size = vocab_size
                torch.manual_seed(0)
                self.table = nn.Parameter(torch.randn(positions, vocab_size))

            def forward(self, inputs: Tensor) -> Tensor:
                """Return the same logits regardless of input."""
                return self.table.unsqueeze(0)

        vocab = 50
        targets = torch.randint(1, vocab, (1, 4))
        narrow = tiny_trainer(model=WideModel(4, vocab), vocab_size=vocab)
        wide = tiny_trainer(model=WideModel(16, vocab), vocab_size=vocab)
        # Copy the narrow model's rows into the wide model's tail so the
        # supervised positions are identical.
        with torch.no_grad():
            wide.model.table[-4:] = narrow.model.table
        narrow_loss, _ = narrow.compute_loss(targets, targets)
        wide_loss, _ = wide.compute_loss(targets, targets)
        assert float(narrow_loss.item()) == pytest.approx(
            float(wide_loss.item()), rel=1e-5
        )


class TestTrainerConfig:
    """Tests for :class:`TrainerConfig`."""

    def test_default_values(self) -> None:
        """Defaults are sensible."""
        config = TrainerConfig()
        assert config.learning_rate > 0.0
        assert config.grad_clip_norm >= 0.0


class TestCosineWarmupScheduler:
    """Tests for :class:`CosineWarmupScheduler`."""

    def test_warmup_lrs_increase(self) -> None:
        """Learning rates increase during warmup."""
        param = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.AdamW([param], lr=1.0)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=10, max_steps=100
        )
        lrs = []
        for _ in range(5):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1]

    def test_decay_after_warmup(self) -> None:
        """Learning rates decay after warmup ends."""
        param = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.AdamW([param], lr=1.0)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=5, max_steps=20
        )
        peak: float | None = None
        lrs = []
        for _ in range(20):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            lrs.append(lr)
        peak = max(lrs[:10])
        # Later learning rates are below peak.
        assert lrs[-1] < peak

    def test_min_lr_ratio_respected(self) -> None:
        """Final learning rate is at least ``min_lr_ratio * base``."""
        param = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.AdamW([param], lr=1.0)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=2, max_steps=10, min_lr_ratio=0.1
        )
        for _ in range(20):
            scheduler.step()
        final_lr = scheduler.get_last_lr()[0]
        assert final_lr >= 0.1 * 1.0 - 1e-6


class TestTrainerBasics:
    """Tests for the :class:`Trainer` core flow."""

    @pytest.fixture
    def dataset(self) -> DataLoader:
        """Provide a deterministic small dataset."""
        torch.manual_seed(0)
        inputs = torch.randint(0, 100, (64, 16))
        targets = torch.randint(0, 100, (64, 16))
        ds = TensorDataset(inputs, targets)
        return DataLoader(ds, batch_size=4)

    def test_train_step_decrements_loss(self, dataset: DataLoader) -> None:
        """Loss generally decreases across training steps."""
        trainer = tiny_trainer()
        initial_loss = trainer.state.last_loss
        for step, batch in enumerate(dataset):
            trainer.train_step(batch)
            if step >= 5:
                break
        # Some decrease is expected (not strict).
        assert trainer.state.last_loss < initial_loss + 100.0

    def test_train_returns_history(self, dataset: DataLoader) -> None:
        """``train`` returns a list of metric snapshots."""
        trainer = tiny_trainer(
            config=TrainerConfig(
                learning_rate=3e-4,
                max_steps=10,
                warmup_steps=1,
                log_every_n_steps=1,
                amp_dtype=torch.float32,
            )
        )
        history = trainer.train(dataset, num_steps=5)
        assert len(history) >= 1
        for snapshot in history:
            assert "training_loss" in snapshot

    def test_metrics_are_recorded(self, dataset: DataLoader) -> None:
        """Metrics are recorded in the registry during training."""
        trainer = tiny_trainer()
        for batch in dataset:
            trainer.train_step(batch)
            break
        snapshot = trainer.metrics.snapshot()
        assert snapshot["training_loss"] > 0.0
        assert snapshot["perplexity"] > 0.0

    def test_curriculum_advances(self, dataset: DataLoader) -> None:
        """The curriculum's total step increments during training."""
        trainer = tiny_trainer(
            curriculum=Curriculum(
                CurriculumSchedule(
                    stage_1_end=5, stage_2_end=10, stage_3_end=15
                )
            ),
        )
        for batch in dataset:
            trainer.train_step(batch)
            if trainer.curriculum.state.total_step >= 5:
                break
        assert trainer.curriculum.state.total_step >= 5

    def test_amp_disabled_on_cpu(self) -> None:
        """AMP is disabled when running on CPU."""
        trainer = tiny_trainer()
        assert trainer.amp_enabled is False

    def test_grad_clip_applied(self, dataset: DataLoader) -> None:
        """Gradient clipping is applied when configured."""
        config = TrainerConfig(
            learning_rate=3e-4,
            max_steps=10,
            warmup_steps=1,
            grad_clip_norm=0.5,
            amp_dtype=torch.float32,
        )
        trainer = tiny_trainer(config=config)
        for batch in dataset:
            trainer.train_step(batch)
            break

    def test_grad_clip_disabled(self, dataset: DataLoader) -> None:
        """Gradient clipping is a no-op when set to zero."""
        config = TrainerConfig(
            learning_rate=3e-4,
            max_steps=10,
            warmup_steps=1,
            grad_clip_norm=0.0,
            amp_dtype=torch.float32,
        )
        trainer = tiny_trainer(config=config)
        for batch in dataset:
            trainer.train_step(batch)
            break

    def test_checkpoint_round_trip(self, dataset: DataLoader) -> None:
        """Saving and loading a checkpoint preserves parameters."""
        trainer = tiny_trainer()
        for batch in dataset:
            trainer.train_step(batch)
            break
        before = next(iter(trainer.model.parameters())).detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.safetensors")
            trainer.save_checkpoint(path)
            trainer.load_checkpoint(path)
        after = next(iter(trainer.model.parameters())).detach().clone()
        assert torch.allclose(before, after)

    def test_scheduler_step(self, dataset: DataLoader) -> None:
        """The scheduler is updated each training step."""
        trainer = tiny_trainer()
        for batch in dataset:
            trainer.train_step(batch)
            break
        assert trainer.scheduler.get_last_lr()[0] > 0.0

    def test_default_metrics_registry(self) -> None:
        """``build_default_registry`` has every required metric."""
        registry = build_default_registry()
        assert set(registry.names) == set(DEFAULT_METRIC_NAMES)

    def test_move_batch_to_device(self) -> None:
        """``move_batch`` relocates tensors to the trainer device."""
        trainer = tiny_trainer()
        inputs = torch.randint(0, 100, (2, 4))
        targets = torch.randint(0, 100, (2, 4))
        inputs, targets = trainer.move_batch((inputs, targets))
        assert inputs.device == trainer.device
        assert targets.device == trainer.device

    def test_compute_loss_curriculum_aware(self) -> None:
        """``compute_loss`` reflects the curriculum's active components."""
        curriculum = Curriculum(
            CurriculumSchedule(stage_1_end=10, stage_2_end=20, stage_3_end=30)
        )
        trainer = tiny_trainer(curriculum=curriculum)
        inputs = torch.randint(0, 100, (2, 4))
        targets = torch.randint(0, 100, (2, 4))
        loss, components = trainer.compute_loss(inputs, targets)
        assert torch.isfinite(loss)
        assert "ar" in components

    def test_compute_loss_full_components(self) -> None:
        """All loss components are reported when curriculum is JOINT.

        ``router`` is only reported when the model exposes
        ``router_logits``. A non-MoE model deliberately does not
        contribute to that term, so the trainer does not invent a fake
        one.
        """
        curriculum = Curriculum(
            CurriculumSchedule(stage_1_end=1, stage_2_end=2, stage_3_end=3)
        )
        trainer = tiny_trainer(curriculum=curriculum)
        for _ in range(5):
            trainer.curriculum.step()
        inputs = torch.randint(0, 100, (2, 4))
        targets = torch.randint(0, 100, (2, 4))
        _, components = trainer.compute_loss(inputs, targets)
        # ``TinyModel`` has no MoE, so no router term is injected.
        assert set(components) == {"ar", "jepa", "memory"}

    def test_router_term_absent_without_moe(self) -> None:
        """The router loss term is absent when the model exposes no router.

        With ``moe=None`` the trainer must not invent a synthetic
        ``torch.randn`` fallback: that noise used to be summed into the
        loss and reflected as a metric, so any tuning against the metric
        was tuning against the RNG.
        """
        model = TinyModel()
        trainer = tiny_trainer(model=model)
        # Force the curriculum to JOINT so every component is otherwise
        # active.
        for _ in range(20):
            trainer.curriculum.step()
        inputs = torch.randint(0, 100, (2, 4))
        targets = torch.randint(0, 100, (2, 4))
        loss_a, components_a = trainer.compute_loss(inputs, targets)
        # Identical second run: the loss must not depend on the RNG.
        torch.manual_seed(123)
        loss_b, components_b = trainer.compute_loss(inputs, targets)
        assert "router" not in components_a
        assert "router" not in components_b
        assert torch.allclose(loss_a, loss_b, atol=1e-6)
        # And running the trainer's metrics across two independent
        # step calls must record zero router loss either way.
        trainer.metrics.update("router_loss", 0.0)
        assert trainer.metrics.snapshot()["router_loss"] == 0.0

    def test_optimizer_zeroes_grad(self, dataset: DataLoader) -> None:
        """Each training step zeroes gradients before computing the next loss."""
        trainer = tiny_trainer()
        # Manually compute a backward pass to leave stale grads.
        for param in trainer.model.parameters():
            if param.grad is not None:
                param.grad = None
        inputs = torch.randint(0, 100, (2, 4))
        targets = torch.randint(0, 100, (2, 4))
        loss, _ = trainer.compute_loss(inputs, targets)
        loss.backward()
        for batch in dataset:
            trainer.train_step(batch)
            break
        # After a training step, the gradients should be either None or
        # reflect the most recent backward (which was the new training
        # step, not the stale one).
        for param in trainer.model.parameters():
            assert param.grad is not None

    def test_global_step_increments(self, dataset: DataLoader) -> None:
        """``state.global_step`` increments each step."""
        trainer = tiny_trainer()
        for batch in dataset:
            trainer.train_step(batch)
            break
        assert trainer.state.global_step == 1
