"""Trains UCSA-R (causal chunked state recurrence) on local token shards.

Examples:
    python scripts/prepare_data.py --out data
    python scripts/train_r.py --preset small --out-dir ckpts/r-small
    python scripts/train_r.py --preset small --set model.use_state=false \
        --out-dir ckpts/r-nostate          # chunk-local control
    python scripts/train_r.py --preset small --resume --out-dir ckpts/r-small
    python scripts/train_r.py --preset small --params 60000000
"""

import argparse
import dataclasses
import json
import os
from typing import Any

import transformers
import yaml

from ucsa.models import recurrent
from ucsa.training import compression, engine, shards

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "ucsa", "config.yaml"
)


def apply_overrides(
    config: dict[str, Any], overrides: list[str]
) -> dict[str, Any]:
    """Applies `section.key=value` overrides in place.

    Args:
      config: Dict with `model` and `train` sections.
      overrides: Strings such as `model.chunk_size=64`; values are parsed
        as YAML.

    Returns:
      The updated `config`.

    Raises:
      SystemExit: If an override is malformed or names an unknown section.
    """
    for item in overrides:
        path, sep, raw = item.partition("=")
        section, dot, key = path.partition(".")
        if not (sep and dot):
            raise SystemExit(f"bad --set {item!r}; use section.key=value")
        if section not in ("model", "train"):
            raise SystemExit(f"unknown section {section!r} in {item!r}")
        config.setdefault(section, {})[key] = yaml.safe_load(raw)
    return config


def build_configs(
    args: argparse.Namespace,
) -> tuple[recurrent.Config, engine.Config]:
    """Resolves preset, then auto-sizing (`--params`), then `--set`.

    Explicit `--set` always wins, so a sweep can pin any single field.

    Args:
      args: Parsed command-line arguments.

    Returns:
      The model and training configs.

    Raises:
      SystemExit: If the preset is unknown.
    """
    with open(CONFIG_PATH) as f:
        presets = yaml.safe_load(f)["recurrent"]["presets"]
    if args.preset not in presets:
        raise SystemExit(f"unknown preset {args.preset!r}: {sorted(presets)}")
    preset = json.loads(json.dumps(presets[args.preset]))
    model_cfg = dict(preset.get("model", {}))
    if args.params:
        sized = recurrent.config_for_params(args.params).to_dict()
        for key in ("hidden", "layers", "heads", "ffn_dim"):
            model_cfg[key] = sized[key]
    merged = apply_overrides(
        {"model": model_cfg, "train": dict(preset.get("train", {}))},
        args.set or [],
    )
    merged["train"].update(seed=args.seed, out_dir=args.out_dir)
    return (
        recurrent.Config.from_dict(merged["model"]),
        engine.Config.from_dict(merged["train"]),
    )


def batch_factories(
    train_config: engine.Config, data_dir: str
) -> tuple[engine.Batches, engine.Batches]:
    """Builds the train and validation batch sources from local shards.

    Args:
      train_config: Training configuration.
      data_dir: Directory with `train.bin` and `val.bin`.

    Returns:
      `(train, val)`; each maps batches already consumed to an iterator.
    """
    train_shard = shards.Shard(os.path.join(data_dir, "train.bin"))
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


def main() -> None:
    """Parses arguments, trains, and prints the final metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="small")
    parser.add_argument("--set", nargs="*", help="section.key=value")
    parser.add_argument("--params", type=int, default=0, help="auto-size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="ckpts/r")
    parser.add_argument("--data", default="data")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_config, train_config = build_configs(args)
    model = recurrent.Model(model_config)
    count = recurrent.count_parameters(model)
    print(f"UCSA-R {args.preset}: {count:,} params", flush=True)
    print(json.dumps(model_config.to_dict()), flush=True)
    print(json.dumps(dataclasses.asdict(train_config)), flush=True)
    if args.dry_run:
        return
    train, val = batch_factories(train_config, args.data)
    byte_lengths = compression.token_byte_lengths(
        transformers.AutoTokenizer.from_pretrained("gpt2")
    )
    record = engine.fit(
        model,
        train_config,
        train,
        val,
        resume=args.resume,
        byte_lengths=byte_lengths,
    )
    print(f"final: {record['final']}", flush=True)


if __name__ == "__main__":
    main()
