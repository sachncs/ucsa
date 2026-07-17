"""Tests for :mod:`ucsa.utils`."""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
from torch import nn

from ucsa.utils.checkpoint import (
    CheckpointMetadata,
    load_checkpoint,
    metadata_from_dict,
    save_checkpoint,
)
from ucsa.utils.logging import (
    LoggerBundle,
    LoggerConfig,
    build_logger,
    close_logger,
    configure_logging,
    log_metrics,
    smoke_log_check,
)
from ucsa.utils.seed import (
    environment_fingerprint,
    get_seed,
    seed_context,
    set_seed,
)


class TestSeed:
    """Tests for :mod:`ucsa.utils.seed`."""

    def test_set_seed_returns_consistent_value(self) -> None:
        """``set_seed`` sets every RNG to a known state."""
        set_seed(123)
        first = torch.rand(1)
        set_seed(123)
        second = torch.rand(1)
        assert torch.allclose(first, second)

    def test_set_seed_changes_value(self) -> None:
        """Different seeds produce different values."""
        set_seed(1)
        a = torch.rand(1)
        set_seed(2)
        b = torch.rand(1)
        assert not torch.allclose(a, b)

    def test_get_seed(self) -> None:
        """``get_seed`` returns the current torch seed."""
        set_seed(99)
        assert get_seed() == 99

    def test_seed_context_restores(self) -> None:
        """``seed_context`` restores the previous RNG state."""
        set_seed(10)
        before = torch.rand(1).clone()
        with seed_context(99):
            _ = torch.rand(1)
        set_seed(10)
        after = torch.rand(1)
        assert torch.allclose(before, after)

    def test_environment_fingerprint_keys(self) -> None:
        """``environment_fingerprint`` includes core keys."""
        info = environment_fingerprint()
        assert "python" in info
        assert "torch" in info
        assert "cuda_available" in info


class TestCheckpoint:
    """Tests for :mod:`ucsa.utils.checkpoint`."""

    @pytest.fixture
    def tiny_model(self) -> nn.Module:
        """Provide a tiny model."""
        torch.manual_seed(0)
        return nn.Linear(8, 4)

    def test_save_load_round_trip(self, tiny_model: nn.Module) -> None:
        """Saving and loading reproduces parameters."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.safetensors")
            save_checkpoint(tiny_model, path)
            new_model = nn.Linear(8, 4)
            load_checkpoint(new_model, path)
            for (n1, p1), (n2, p2) in zip(
                tiny_model.named_parameters(),
                new_model.named_parameters(),
                strict=False,
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)

    def test_metadata_round_trip(self, tiny_model: nn.Module) -> None:
        """Metadata round-trips through the sidecar JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.safetensors")
            meta = CheckpointMetadata(
                step=100,
                epoch=2,
                config={"hidden_size": 8},
                metrics={"loss": 1.5},
                extras={"foo": "bar"},
            )
            save_checkpoint(tiny_model, path, metadata=meta)
            new_model = nn.Linear(8, 4)
            loaded = load_checkpoint(new_model, path)
            assert loaded.step == 100
            assert loaded.epoch == 2
            assert loaded.metrics == {"loss": 1.5}
            assert loaded.config == {"hidden_size": 8}
            assert loaded.extras == {"foo": "bar"}

    def test_metadata_from_dict(self) -> None:
        """``metadata_from_dict`` builds a metadata object."""
        meta = metadata_from_dict(
            {"step": 5, "epoch": 1, "config": None, "metrics": None, "extras": {}}
        )
        assert meta.step == 5
        assert meta.epoch == 1

    def test_no_metadata_file_returns_default(self, tiny_model: nn.Module) -> None:
        """Missing metadata file returns an empty :class:`CheckpointMetadata`."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.safetensors")
            save_checkpoint(tiny_model, path)
            new_model = nn.Linear(8, 4)
            meta = load_checkpoint(new_model, path)
            assert meta.step == 0
            assert meta.extras == {}

    def test_save_writes_meta_json(self, tiny_model: nn.Module) -> None:
        """The save call writes a ``.meta.json`` file when metadata exists."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.safetensors")
            save_checkpoint(
                tiny_model, path, CheckpointMetadata(step=1)
            )
            meta_path = path + ".meta.json"
            assert os.path.exists(meta_path)
            with open(meta_path, encoding="utf-8") as fp:
                payload = json.load(fp)
            assert payload["step"] == 1


class TestLogging:
    """Tests for :mod:`ucsa.utils.logging`."""

    def test_configure_logging_does_not_raise(self) -> None:
        """``configure_logging`` runs without error."""
        configure_logging(level=logging_level_from_int(20))

    def test_build_logger_with_tensorboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TensorBoard writer is constructed when enabled."""
        fake_writer = _FakeSummaryWriter()

        class FakeSummaryWriter:
            def __init__(self, log_dir: str) -> None:
                fake_writer.init_log_dir = log_dir
                self.scalars: list[tuple[str, float, int]] = []

            def add_scalar(
                self, name: str, value: float, step: int
            ) -> None:
                self.scalars.append((name, value, step))

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        import sys

        sys.modules.setdefault(
            "torch.utils.tensorboard", __import__("types").ModuleType(
                "torch.utils.tensorboard"
            )
        )
        sys.modules["torch.utils.tensorboard"].SummaryWriter = (
            FakeSummaryWriter
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = LoggerConfig(
                log_dir=tmp,
                use_tensorboard=True,
                use_wandb=False,
            )
            bundle = build_logger(config)
            try:
                assert bundle.tensorboard_writer is not None
            finally:
                close_logger(bundle)

    def test_build_logger_disables_when_requested(self) -> None:
        """``use_tensorboard=False`` skips the writer."""
        config = LoggerConfig(
            use_tensorboard=False,
            use_wandb=False,
        )
        bundle = build_logger(config)
        try:
            assert bundle.tensorboard_writer is None
            assert bundle.wandb_run is None
        finally:
            close_logger(bundle)

    def test_log_metrics_is_no_op_with_no_writers(self) -> None:
        """``log_metrics`` is a no-op when no writer is active."""
        bundle = LoggerBundle(
            tensorboard_writer=None,
            wandb_run=None,
            config=LoggerConfig(use_tensorboard=False, use_wandb=False),
        )
        log_metrics(bundle, {"a": 1.0}, step=0)

    def test_smoke_log_check(self) -> None:
        """``smoke_log_check`` returns ``True``."""
        assert smoke_log_check() is True

    def test_close_logger_handles_no_writers(self) -> None:
        """``close_logger`` is safe when no writers are configured."""
        bundle = LoggerBundle()
        close_logger(bundle)


class _FakeSummaryWriter:
    """internal: capture for TensorBoard fake writer."""

    init_log_dir: str | None = None
    scalars: list[tuple[str, float, int]] = []


def logging_level_from_int(value: int) -> int:
    """internal: lift an int into the logging level namespace."""
    import logging

    return value if value in logging._nameToLevel.values() else logging.INFO


class TestSeedCudaFallback:
    """Tests for the CUDA fallbacks in :mod:`ucsa.utils.seed`."""

    def test_seed_context_without_cuda(self) -> None:
        """``seed_context`` works without CUDA."""
        with seed_context(7):
            assert get_seed() == 7
