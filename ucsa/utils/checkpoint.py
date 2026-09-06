"""Checkpoint save and load.

Wraps :mod:`safetensors` with metadata support. Provides a single
``save_checkpoint`` / ``load_checkpoint`` API for the trainer and
inference script.

Checkpoints written before a PCS bank was added are still loadable:
:func:`adapt_legacy_state_dict` resizes the entries whose shape depends on
the number of banks. See its docstring for the row remapping.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from safetensors.torch import load_file, save_file
from torch import Tensor, nn

BANK_ID_EMBEDDING_SUFFIX = "bank_id_embedding.weight"


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


def adapt_legacy_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], list[str]]:
    """Resize checkpoint entries whose shape depends on the bank count.

    Adding a PCS bank changes exactly one saved shape: the operator's
    ``bank_id_embedding``, which holds one row per bank plus a final row
    for observation tokens. New banks are appended to
    :data:`ucsa.models.state.BANK_NAMES`, so existing bank rows keep their
    index, but the observation row moves to the new last index. Copying the
    old rows positionally would put the observation embedding into the new
    bank's slot, so the rows are remapped:

    - ``new[:n_old_banks] = old[:n_old_banks]`` -- existing banks.
    - ``new[-1] = old[-1]`` -- the observation row.
    - the rows in between keep their fresh initialisation.

    Entries for banks absent from the checkpoint are simply left out, so a
    non-strict load keeps their initialised values.

    Args:
        model: The model the state dict will be loaded into.
        state_dict: The checkpoint's state dict.

    Returns:
        Tuple ``(adapted_state_dict, notes)`` where ``notes`` describes
        every adaptation performed, for logging.
    """
    adapted = dict(state_dict)
    notes: list[str] = []
    model_state = model.state_dict()
    for name, tensor in state_dict.items():
        if not name.endswith(BANK_ID_EMBEDDING_SUFFIX):
            continue
        target = model_state.get(name)
        if target is None or tuple(target.shape) == tuple(tensor.shape):
            continue
        if tensor.dim() != 2 or target.dim() != 2:
            continue
        if tensor.shape[1] != target.shape[1]:
            continue
        if tensor.shape[0] >= target.shape[0]:
            continue
        merged = target.detach().clone()
        n_old_banks = tensor.shape[0] - 1
        merged[:n_old_banks] = tensor[:n_old_banks].to(
            device=merged.device, dtype=merged.dtype
        )
        merged[-1] = tensor[-1].to(device=merged.device, dtype=merged.dtype)
        adapted[name] = merged
        notes.append(
            f"{name}: grew {tuple(tensor.shape)} -> {tuple(target.shape)}; "
            f"kept {n_old_banks} bank rows and moved the observation row to "
            f"index {merged.shape[0] - 1}"
        )
    return adapted, notes


def load_state_dict_compat(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    strict: bool = False,
) -> list[str]:
    """Load a possibly-legacy state dict into ``model``.

    Args:
        model: The model to load into.
        state_dict: The checkpoint's state dict.
        strict: Passed through to ``load_state_dict`` after adaptation.
            Missing keys for newly added banks make a strict load fail, so
            the default is non-strict.

    Returns:
        The adaptation notes from :func:`adapt_legacy_state_dict`.
    """
    adapted, notes = adapt_legacy_state_dict(model, state_dict)
    model.load_state_dict(adapted, strict=strict)
    return notes


def load_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = True,
) -> CheckpointMetadata:
    """Load model weights and metadata.

    Args:
        model: The model into which weights are loaded.
        path: Path to a ``.safetensors`` file.
        strict: Whether to require an exact match. Shapes that changed
            because a PCS bank was added are adapted first, so a strict
            load of a legacy checkpoint fails only on genuinely missing
            entries.

    Returns:
        The loaded :class:`CheckpointMetadata` (empty if no metadata file
        is found).
    """
    state_dict = load_file(path)
    adapted, _ = adapt_legacy_state_dict(model, state_dict)
    model.load_state_dict(adapted, strict=strict)
    meta_path = path + ".meta.json"
    if not __import__("os").path.exists(meta_path):
        return CheckpointMetadata()
    with open(meta_path, encoding="utf-8") as fp:
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
    "BANK_ID_EMBEDDING_SUFFIX",
    "CheckpointMetadata",
    "adapt_legacy_state_dict",
    "load_checkpoint",
    "load_state_dict_compat",
    "save_checkpoint",
]
