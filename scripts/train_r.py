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
import sys

import transformers

from ucsa.models import recurrent
from ucsa.training import compression, engine, presets


def main() -> None:
    """Parses arguments, trains, and prints the final metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="small")
    parser.add_argument(
        "--set", nargs="*", default=[], help="section.key=value"
    )
    parser.add_argument("--params", type=int, default=0, help="auto-size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="ckpts/r")
    parser.add_argument("--data", default="data")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        model_config, train_config = presets.build_configs(
            args.preset, args.set, args.params, args.seed, args.out_dir
        )
    except ValueError as error:
        sys.exit(str(error))
    model = recurrent.Model(model_config)
    count = recurrent.count_parameters(model)
    print(f"UCSA-R {args.preset}: {count:,} params", flush=True)
    print(json.dumps(model_config.to_dict()), flush=True)
    print(json.dumps(dataclasses.asdict(train_config)), flush=True)
    if args.dry_run:
        return
    train, val = presets.batch_factories(train_config, args.data)
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
