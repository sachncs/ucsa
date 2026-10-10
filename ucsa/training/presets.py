"""Named configurations: presets, overrides and the batch sources they imply.

`ucsa/config.yaml` is the single configuration file. It holds named presets,
each with a `model` section (fields of `recurrent.Config`) and a `train`
section (fields of `engine.Config`). A run is a preset plus `section.key=value`
overrides; every field is validated by the config classes, so a typo fails
before any compute is spent.
"""

import copy
import json
import os
from collections.abc import Iterable
from typing import Any

import yaml

from ucsa.models import recurrent
from ucsa.training import engine, shards

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
SECTIONS = ("model", "train")
SIZED_FIELDS = ("hidden", "layers", "heads", "ffn_dim")


def names(path: str = CONFIG_PATH) -> list[str]:
    """Returns the preset names defined in the configuration file."""
    with open(path) as f:
        return sorted(yaml.safe_load(f)["presets"])


def load(name: str, path: str = CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """Returns a deep copy of one preset.

    Args:
      name: Preset name.
      path: Configuration file.

    Returns:
      A dict with `model` and `train` sections.

    Raises:
      ValueError: If the preset does not exist.
    """
    with open(path) as f:
        presets = yaml.safe_load(f)["presets"]
    if name not in presets:
        raise ValueError(f"unknown preset {name!r}: {sorted(presets)}")
    preset = copy.deepcopy(presets[name])
    return {section: dict(preset.get(section, {})) for section in SECTIONS}


def parse_value(raw: str) -> Any:
    """Parses an override value: JSON first, then YAML, then the raw text.

    JSON first because YAML 1.1 reads `3e-4` (no dot) as a string.

    Args:
      raw: The text after `=`.

    Returns:
      The parsed value.
    """
    try:
        return json.loads(raw)
    except ValueError:
        return yaml.safe_load(raw)


def apply_overrides(
    config: dict[str, dict[str, Any]], overrides: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Applies `section.key=value` overrides in place.

    Args:
      config: Dict with `model` and `train` sections.
      overrides: Strings such as `model.chunk_size=64`; values are parsed
        so `false`, `3e-4` and `[1, 2]` mean what they look like.

    Returns:
      The updated `config`.

    Raises:
      ValueError: If an override is malformed or names an unknown section.
    """
    for item in overrides:
        path, sep, raw = item.partition("=")
        section, dot, key = path.partition(".")
        if not (sep and dot and key):
            raise ValueError(f"bad override {item!r}; use section.key=value")
        if section not in SECTIONS:
            raise ValueError(f"unknown section {section!r} in {item!r}")
        config[section][key] = parse_value(raw)
    return config


def build_configs(
    preset: str,
    overrides: Iterable[str] = (),
    params: int = 0,
    seed: int = 42,
    out_dir: str = "ckpts/r",
) -> tuple[recurrent.Config, engine.Config]:
    """Builds validated model and training configs.

    Args:
      preset: Preset name.
      overrides: `section.key=value` overrides; they win over everything,
        including `params`.
      params: If positive, size width, depth, heads and FFN width for about
        this many parameters.
      seed: Training seed.
      out_dir: Output directory for checkpoints and the run record.

    Returns:
      The model and training configs.

    Raises:
      ValueError: If the preset, an override or any field is invalid.
    """
    config = load(preset)
    if params:
        sized = recurrent.config_for_params(params).to_dict()
        config["model"].update({key: sized[key] for key in SIZED_FIELDS})
    apply_overrides(config, overrides)
    config["train"].update(seed=seed, out_dir=out_dir)
    return (
        recurrent.Config.from_dict(config["model"]),
        engine.Config.from_dict(config["train"]),
    )


def batch_factories(
    train_config: engine.Config, data_dir: str
) -> tuple[engine.Batches, engine.Batches]:
    """Builds the train and validation batch sources from local shards.

    Args:
      train_config: Training configuration.
      data_dir: Directory with the training shard and `val.bin`.

    Returns:
      `(train, val)`; each maps batches already consumed to an iterator.
    """
    train_shard = shards.Shard(os.path.join(data_dir, train_config.train_shard))
    val_shard = shards.Shard(os.path.join(data_dir, "val.bin"))

    def train(skip: int) -> engine.BatchIterator:
        return train_shard.batches(
            train_config.batch_size,
            train_config.seq_len,
            skip=skip,
            seed=train_config.seed,
        )

    def val(skip: int) -> engine.BatchIterator:
        return val_shard.batches(
            train_config.batch_size,
            train_config.seq_len,
            skip=skip,
            seed=None,
            loop=False,
        )

    return train, val
