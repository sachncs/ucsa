"""Logging utilities.

Provides a unified logger that writes to TensorBoard and optionally
Weights & Biases. Falls back to a no-op writer when neither backend
is available.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class LoggerConfig:
    """Configuration for the unified logger.

    Attributes:
        log_dir: Directory for TensorBoard logs.
        use_tensorboard: Whether to enable TensorBoard logging.
        use_wandb: Whether to enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_entity: Optional W&B entity (user/team).
        wandb_run_name: Optional W&B run name.
    """

    log_dir: str = "runs/ucsa"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "ucsa"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


@dataclass
class LoggerBundle:
    """internal: holds the active loggers and supporting state."""

    tensorboard_writer: Any | None = None
    wandb_run: Any | None = None
    config: LoggerConfig = field(default_factory=LoggerConfig)


def configure_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """Configure the root Python logger.

    Args:
        level: Logging level.
        log_file: Optional path to a log file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def build_logger(config: LoggerConfig | None = None) -> LoggerBundle:
    """Build a unified logger bundle.

    Args:
        config: Optional logger configuration.

    Returns:
        A :class:`LoggerBundle` with the active writers attached.
    """
    if config is None:
        config = LoggerConfig()
    bundle = LoggerBundle(config=config)
    if config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(config.log_dir, exist_ok=True)
            bundle.tensorboard_writer = SummaryWriter(log_dir=config.log_dir)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("TensorBoard logging disabled: %s", exc)
            bundle.tensorboard_writer = None
    if config.use_wandb:
        try:
            import wandb

            bundle.wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("W&B logging disabled: %s", exc)
            bundle.wandb_run = None
    return bundle


def log_metrics(
    bundle: LoggerBundle,
    metrics: dict[str, float],
    step: int,
) -> None:
    """Log a metrics dict to every active backend.

    Args:
        bundle: The logger bundle.
        metrics: Mapping of metric name to value.
        step: Global step.
    """
    if bundle.tensorboard_writer is not None:
        for name, value in metrics.items():
            bundle.tensorboard_writer.add_scalar(name, float(value), step)
    if bundle.wandb_run is not None:
        bundle.wandb_run.log({**metrics, "step": step}, step=step)


def close_logger(bundle: LoggerBundle) -> None:
    """Close any active writers.

    Args:
        bundle: The logger bundle.
    """
    if bundle.tensorboard_writer is not None:
        try:
            bundle.tensorboard_writer.flush()
            bundle.tensorboard_writer.close()
        except Exception:  # pragma: no cover - defensive
            pass
    if bundle.wandb_run is not None:
        try:
            bundle.wandb_run.finish()
        except Exception:  # pragma: no cover - defensive
            pass


__all__ = [
    "LoggerBundle",
    "LoggerConfig",
    "build_logger",
    "close_logger",
    "configure_logging",
    "log_metrics",
]


def smoke_log_check() -> bool:
    """internal: quick smoke check for the logger configuration."""
    bundle = build_logger(LoggerConfig(use_tensorboard=False, use_wandb=False))
    log_metrics(bundle, {"a": 1.0}, step=0)
    close_logger(bundle)
    return True