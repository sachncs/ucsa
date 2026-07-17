"""Training entrypoint.

Run with::

    python -m ucsa.train [overrides...]

Hydra/OmegaConf loads ``ucsa/configs/default.yaml`` and applies CLI
overrides. The script builds the model, dataset, loss, and trainer,
then runs the configured number of training steps.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ucsa.models.losses import UCSACombinedLoss
from ucsa.training.curriculum import Curriculum, CurriculumSchedule
from ucsa.training.dataset import DatasetConfig, TextDataset
from ucsa.training.metrics import build_default_registry
from ucsa.training.trainer import Trainer, TrainerConfig
from ucsa.utils.logging import (
    LoggerConfig,
    build_logger,
    close_logger,
    configure_logging,
    log_metrics,
)
from ucsa.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def build_model(cfg: Any) -> torch.nn.Module:
    """Construct the UCSA top-level model from the config.

    The full UCSA model lives in :mod:`ucsa.models.ucsa`. For the smoke
    path we expose a tiny stand-in via :func:`ucsa.models.ucsa.build_ucsa`.
    """
    from ucsa.models.ucsa import build_ucsa

    try:
        from omegaconf import DictConfig, OmegaConf
    except Exception:  # pragma: no cover - environment-dependent
        DictConfig = None  # type: ignore[assignment]
    if DictConfig is not None and isinstance(cfg, DictConfig):
        merged = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(merged, dict)
        return build_ucsa(merged)
    return build_ucsa(cfg)


def build_dataset(cfg: Any) -> TextDataset:
    """Construct the training dataset from the config."""
    from ucsa.models.perception import TokenizerWrapper

    cfg_dict = _config_to_dict(cfg)
    tokenizer = TokenizerWrapper(
        tokenizer_name=cfg_dict["tokenizer"]["name"],
        max_seq_len=cfg_dict["dataset"]["sequence_length"],
    )
    dataset_config = DatasetConfig(
        sequence_length=cfg_dict["dataset"]["sequence_length"],
        primary_dataset=cfg_dict["dataset"]["primary_dataset"],
        primary_split=cfg_dict["dataset"]["primary_split"],
        streaming=cfg_dict["dataset"]["streaming"],
        pack_sequences=cfg_dict["dataset"]["pack_sequences"],
    )
    return TextDataset(tokenizer, dataset_config)


def _config_to_dict(cfg: Any) -> dict[str, Any]:
    """internal: convert a Hydra/OmegaConf config to a plain dict."""
    try:
        from omegaconf import DictConfig, OmegaConf
    except Exception:  # pragma: no cover - environment-dependent
        DictConfig = None  # type: ignore[assignment]
    if DictConfig is not None and isinstance(cfg, DictConfig):
        merged = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(merged, dict)
        return merged
    if not isinstance(cfg, dict):
        raise TypeError(f"Unsupported config type: {type(cfg)}.")
    return cfg


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Construct the AdamW optimiser."""
    cfg_dict = _config_to_dict(cfg)
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg_dict["training"]["learning_rate"],
        weight_decay=cfg_dict["training"]["weight_decay"],
        betas=(cfg_dict["training"]["beta1"], cfg_dict["training"]["beta2"]),
    )


def build_trainer(
    model: torch.nn.Module,
    cfg: Any,
) -> Trainer:
    """Construct the trainer from the config."""
    cfg_dict = _config_to_dict(cfg)
    # ponytail: pull JEPA-loss config out of the model section if
    # present. Defaults preserve the original I-JEPA-style loss.
    model_section = cfg_dict.get("model", {})
    jepa_mode = model_section.get("jepa_mode", "ijepa")
    jepa_alpha = float(model_section.get("jepa_alpha", 0.5))
    gaussian_reg_weight = float(
        model_section.get("gaussian_reg_weight", 0.1)
    )
    loss_fn = UCSACombinedLoss(
        jepa_mode=jepa_mode,
        jepa_alpha=jepa_alpha,
        gaussian_reg_weight=gaussian_reg_weight,
    )
    optimizer = build_optimizer(model, cfg)
    training = cfg_dict["training"]
    trainer_config = TrainerConfig(
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        beta1=training["beta1"],
        beta2=training["beta2"],
        grad_clip_norm=training["grad_clip_norm"],
        warmup_steps=training["warmup_steps"],
        max_steps=training["max_steps"],
        amp_dtype=_resolve_dtype(training["amp_dtype"]),
        log_every_n_steps=training["log_every_n_steps"],
        checkpoint_every_n_steps=training["checkpoint_every_n_steps"],
        gradient_checkpointing=training["gradient_checkpointing"],
        compile_model=training["compile_model"],
    )
    curriculum_dict = cfg_dict["curriculum"]
    curriculum = Curriculum(
        CurriculumSchedule(
            stage_1_end=curriculum_dict["stage_1_end"],
            stage_2_end=curriculum_dict["stage_2_end"],
            stage_3_end=curriculum_dict["stage_3_end"],
        )
    )
    metrics = build_default_registry()
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=trainer_config,
        curriculum=curriculum,
        metrics=metrics,
    )


def _resolve_dtype(name: str) -> torch.dtype:
    """Resolve a dtype name string to a :class:`torch.dtype`."""
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported amp_dtype '{name}'.")
    return mapping[name]


def run_training(cfg: Any) -> dict[str, Any]:
    """Run the configured training loop.

    Args:
        cfg: The Hydra configuration.

    Returns:
        The final metrics snapshot.
    """
    cfg_dict = _config_to_dict(cfg)
    set_seed(int(cfg_dict["seed"]))
    model = build_model(cfg)
    dataset = build_dataset(cfg)
    trainer = build_trainer(model, cfg)
    LOGGER.info(
        "Starting training for %d steps",
        cfg_dict["training"]["max_steps"],
    )

    dataloader = _make_dataloader(dataset, cfg)
    history = trainer.train(
        dataloader, num_steps=int(cfg_dict["training"]["max_steps"])
    )
    final_snapshot = history[-1] if history else trainer.metrics.snapshot()
    LOGGER.info("Training complete. Final metrics: %s", final_snapshot)
    return final_snapshot


def _make_dataloader(dataset: TextDataset, cfg: Any) -> Any:
    """Wrap the dataset in a lightweight DataLoader."""
    from torch.utils.data import DataLoader

    cfg_dict = _config_to_dict(cfg)
    batch_size = int(cfg_dict["training"]["batch_size"])

    class IterableDatasetAdapter(torch.utils.data.IterableDataset):
        def __init__(self_inner) -> None:  # type: ignore[override]
            super().__init__()
            self_inner.dataset = dataset

        def __iter__(self_inner):  # type: ignore[override]
            return iter(dataset)

    return DataLoader(IterableDatasetAdapter(), batch_size=batch_size)


def main() -> None:
    """Entry point for ``python -m ucsa.train``."""
    import os

    try:
        from hydra import compose, initialize_config_dir
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Hydra is required for the ucsa.train CLI."
        ) from exc
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "configs")
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="default")
        configure_logging()
        logger_bundle = build_logger(
            LoggerConfig(
                log_dir=str(cfg.logging.log_dir),
                use_tensorboard=bool(cfg.logging.use_tensorboard),
                use_wandb=bool(cfg.logging.use_wandb),
                wandb_project=str(cfg.logging.wandb_project),
            )
        )
        try:
            snapshot = run_training(cfg)
            log_metrics(logger_bundle, snapshot, step=snapshot.get("step", 0))
        finally:
            close_logger(logger_bundle)


__all__ = [
    "build_dataset",
    "build_model",
    "build_optimizer",
    "build_trainer",
    "main",
    "run_training",
]


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
