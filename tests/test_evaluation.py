"""Tests for :mod:`ucsa.training.evaluation`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ucsa.training.evaluation import EvaluationLoop


class ToyModel(nn.Module):
    """internal: a tiny model for evaluation tests."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return vocabulary logits."""
        return self.proj(self.embed(inputs))


class ARLoss(nn.Module):
    """internal: AR-only loss function for evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self, inputs: Tensor, targets: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute cross-entropy loss on toy model logits."""
        model: nn.Module = self.model_reference  # type: ignore[attr-defined]
        logits = model(inputs)
        batch, seq, vocab = logits.shape
        flat_logits = logits.reshape(batch * seq, vocab)
        flat_targets = targets.reshape(batch * seq)
        loss = self.ce(flat_logits, flat_targets)
        return loss, {"ar": float(loss.item())}


class BoundARLoss(ARLoss):
    """internal: AR loss with a model reference bound at construction."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model_reference = model


@pytest.fixture
def model() -> ToyModel:
    """Provide a toy model."""
    return ToyModel()


@pytest.fixture
def dataloader() -> DataLoader:
    """Provide a deterministic dataset."""
    torch.manual_seed(0)
    inputs = torch.randint(0, 100, (32, 8))
    targets = torch.randint(0, 100, (32, 8))
    return DataLoader(TensorDataset(inputs, targets), batch_size=4)


class TestEvaluationState:
    """Tests for :class:`EvaluationState`."""

    def test_defaults(self) -> None:
        """Default state is zeroed."""
        from ucsa.training.evaluation import EvaluationState

        state = EvaluationState()
        assert state.steps == 0
        assert state.last_loss == 0.0


class TestEvaluationLoopConstruction:
    """Tests for :class:`EvaluationLoop` construction."""

    def test_default_device_is_cuda_or_cpu(self, model: ToyModel) -> None:
        """Default device is CUDA when available, else CPU."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        assert loop.device.type in ("cuda", "cpu")


class TestEvaluationLoopEvaluate:
    """Tests for :meth:`EvaluationLoop.evaluate`."""

    def test_evaluate_returns_loss_and_perplexity(
        self, model: ToyModel, dataloader: DataLoader
    ) -> None:
        """``evaluate`` returns ``loss`` and ``perplexity`` keys."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        result = loop.evaluate(dataloader, max_batches=3)
        assert "loss" in result
        assert "perplexity" in result
        assert result["loss"] > 0.0
        assert result["perplexity"] > 0.0

    def test_evaluate_no_grad(
        self, model: ToyModel, dataloader: DataLoader
    ) -> None:
        """No gradient is retained after evaluation."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        for param in model.parameters():
            param.grad = None
        loop.evaluate(dataloader, max_batches=2)
        for param in model.parameters():
            assert param.grad is None

    def test_evaluate_max_batches(
        self, model: ToyModel, dataloader: DataLoader
    ) -> None:
        """``max_batches`` caps the number of evaluated batches."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        loop.evaluate(dataloader, max_batches=2)
        assert loop.state.steps == 2

    def test_evaluate_state_updated(
        self, model: ToyModel, dataloader: DataLoader
    ) -> None:
        """The loop's state is updated after evaluation."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        loop.evaluate(dataloader, max_batches=2)
        assert loop.state.steps == 2
        assert loop.state.last_loss > 0.0
        assert loop.state.last_perplexity > 0.0

    def test_evaluate_resets_train_mode(
        self, model: ToyModel, dataloader: DataLoader
    ) -> None:
        """The model returns to ``train`` mode after evaluation."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        model.train()
        loop.evaluate(dataloader, max_batches=1)
        assert model.training

    def test_evaluate_empty_dataloader(self, model: ToyModel) -> None:
        """An empty dataloader leaves the state unchanged."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        empty_loader = DataLoader(
            TensorDataset(torch.empty(0, 4), torch.empty(0, 4))
        )
        result = loop.evaluate(empty_loader)
        assert result["loss"] == 0.0


class TestShouldEvaluate:
    """Tests for :meth:`EvaluationLoop.should_evaluate`."""

    def test_evaluates_at_interval(self, model: ToyModel) -> None:
        """``should_evaluate`` returns ``True`` at multiples of the interval."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        assert loop.should_evaluate(100, every_n_steps=10)
        assert loop.should_evaluate(0, every_n_steps=10) is False

    def test_zero_interval_disables(self, model: ToyModel) -> None:
        """``every_n_steps=0`` disables evaluation."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        assert loop.should_evaluate(100, every_n_steps=0) is False

    def test_non_multiple_does_not_evaluate(self, model: ToyModel) -> None:
        """A non-multiple step does not trigger evaluation."""
        loss_fn = BoundARLoss(model)
        loop = EvaluationLoop(model, loss_fn)
        assert loop.should_evaluate(7, every_n_steps=10) is False
