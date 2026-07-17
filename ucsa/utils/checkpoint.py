"""Checkpoint save and load.

Wraps :mod:`safetensors` with metadata support. Provides a single
``save_checkpoint`` / ``load_checkpoint`` API for the trainer and
inference script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn

from safetensors.torch import load_file, save_file


@dataclass
class CheckpointMetadata:
    """Sidecar metadata stored alongside model weights.

    Attributes:
        step: Global training step.
        epoch: Epoch number.
        config: Optional Hydra/OmegaConf dump.
        metrics: Optional metrics snapshot.
        extras: Free-form key/value pairs.
    """

    step: int = 0
    epoch: int = 0
    config: dict[str, Any] | None = None
    metrics: dict[str, float] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    model: nn.Module,
    path: str,
    metadata: CheckpointMetadata | None = None,
) -> None:
    """Save model weights and metadata.

    Args:
        model: The model whose ``state_dict`` is saved.
        path: Output path (``.safetensors``).
        metadata: Optional metadata payload. When provided, a companion
            ``<path>.meta.json`` file is written.
    """
    state_dict = {
        name: tensor.detach().contiguous().cpu()
        for name, tensor in model.state_dict().items()
    }
    save_file(state_dict, path)
    if metadata is not None:
        meta_dict = {
            "step": metadata.step,
            "epoch": metadata.epoch,
            "config": metadata.config,
            "metrics": metadata.metrics,
            "extras": metadata.extras,
        }
        meta_path = path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(meta_dict, fp, indent=2, default=str)


def load_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = True,
) -> CheckpointMetadata:
    """Load model weights and metadata.

    Args:
        model: The model into which weights are loaded.
        path: Path to a ``.safetensors`` file.
        strict: Whether to require an exact match.

    Returns:
        The loaded :class:`CheckpointMetadata` (empty if no metadata file
        is found).
    """
    state_dict = load_file(path)
    model.load_state_dict(state_dict, strict=strict)
    meta_path = path + ".meta.json"
    if not __import__("os").path.exists(meta_path):
        return CheckpointMetadata()
    with open(meta_path, "r", encoding="utf-8") as fp:
        meta_dict = json.load(fp)
    return CheckpointMetadata(
        step=int(meta_dict.get("step", 0)),
        epoch=int(meta_dict.get("epoch", 0)),
        config=meta_dict.get("config"),
        metrics=meta_dict.get("metrics"),
        extras=meta_dict.get("extras", {}),
    )


def metadata_from_dict(payload: Mapping[str, Any]) -> CheckpointMetadata:
    """internal: build a :class:`CheckpointMetadata` from a dict."""
    return CheckpointMetadata(
        step=int(payload.get("step", 0)),
        epoch=int(payload.get("epoch", 0)),
        config=payload.get("config"),
        metrics=payload.get("metrics"),
        extras=dict(payload.get("extras", {})),
    )


__all__ = [
    "CheckpointMetadata",
    "load_checkpoint",
    "save_checkpoint",
]