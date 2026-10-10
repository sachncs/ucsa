"""Compresses held-out text losslessly with a trained model.

    python scripts/compress.py --ckpt ckpts/r-small/final.pt --tokens 4096

Takes tokens from a shard, codes them with the model and an arithmetic coder,
decodes them again to prove the round trip, and compares the size with zlib
and LZMA on the same raw bytes. The result is the model's bits per byte as an
actual file size rather than a loss value.
"""

import argparse
import json
import os
import time

import numpy as np
import transformers

from ucsa import arithmetic
from ucsa.training import compression, engine, shards


def main() -> None:
    """Parses arguments, compresses, verifies and reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--shard", default="val.bin")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-json", default="runs/compress.json")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = engine.load_model(args.ckpt, engine.pick_device()).cpu()
    shard = shards.Shard(os.path.join(args.data, args.shard))
    tokens = [
        int(t)
        for t in np.asarray(
            shard.tokens[args.offset : args.offset + args.tokens]
        )
    ]
    raw = tokenizer.decode(tokens).encode("utf-8")
    byte_lengths = compression.token_byte_lengths(tokenizer)
    n_bytes = int(byte_lengths[tokens].sum())

    start = time.time()
    data = arithmetic.compress(
        arithmetic.ModelPredictor(model, shards.EOS_ID), tokens
    )
    encode_seconds = time.time() - start
    decoded = arithmetic.decompress(
        arithmetic.ModelPredictor(model, shards.EOS_ID), data
    )
    if decoded != tokens:
        raise SystemExit("round trip FAILED: decoded tokens differ")

    payload = len(data) - arithmetic.HEADER.size
    report = {
        "tokens": len(tokens),
        "raw_bytes": n_bytes,
        "compressed_bytes": payload,
        "ucsa_bits_per_byte": 8.0 * payload / n_bytes,
        "anchors_bits_per_byte": compression.anchors(raw),
        "lossless_round_trip": True,
        "encode_seconds": encode_seconds,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
