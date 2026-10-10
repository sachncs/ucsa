"""Tokenises corpora once into local shards.

    python scripts/prepare_data.py --out data --train-tokens 120000000

Writes `train.bin` and `val.bin` from a streamed corpus (validation comes
first in the stream, so it never overlaps training) and `wikitext_test.bin`
as a second, out-of-domain held-out set. Training, tuning and evaluation then
read the shards and never touch the network.
"""

import argparse
import os
import time
from collections.abc import Iterator

import datasets
import transformers

from ucsa.training import shards


def tokenized_documents(
    texts: Iterator[str], tokenizer: transformers.PreTrainedTokenizerBase
) -> Iterator[list[int]]:
    """Yields token-id lists for non-empty texts, tokenised in batches.

    Args:
      texts: Document texts.
      tokenizer: Tokenizer to apply.

    Yields:
      One list of token ids per non-empty document.
    """
    batch: list[str] = []
    for text in texts:
        if text:
            batch.append(text)
        if len(batch) == 256:
            yield from tokenizer(batch, add_special_tokens=False)["input_ids"]
            batch = []
    if batch:
        yield from tokenizer(batch, add_special_tokens=False)["input_ids"]


def main() -> None:
    """Parses arguments and writes the shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--train-tokens", type=int, default=120_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument(
        "--no-dedup", action="store_true", help="Keep duplicate documents."
    )
    parser.add_argument(
        "--zratio",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="keep documents whose zlib ratio lies in [LOW, HIGH]",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for the train shard, e.g. 'zfilter' -> train-zfilter.bin",
    )
    parser.add_argument("--skip-wikitext", action="store_true")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    os.makedirs(args.out, exist_ok=True)
    stream = datasets.load_dataset(args.dataset, split="train", streaming=True)
    raw = iter(row["text"] for row in stream)
    stats: dict[str, int] = {}
    zstats: dict[str, int] = {}
    seen: set[bytes] = set()  # shared, so no document lands in two shards

    def documents(texts: Iterator[str]) -> Iterator[list[int]]:
        docs = tokenized_documents(texts, tokenizer)
        if args.no_dedup:
            return docs
        return shards.filter_documents(
            docs, args.min_tokens, stats=stats, seen=seen
        )

    started = time.time()
    plan = (("val", args.val_tokens, False), ("train", args.train_tokens, True))
    for name, limit, filtered in plan:
        texts = raw
        if filtered and args.zratio:  # only training data is filtered
            texts = shards.filter_by_compressibility(
                raw, args.zratio[0], args.zratio[1], zstats
            )
        suffix = f"-{args.tag}" if args.tag and name == "train" else ""
        path = os.path.join(args.out, f"{name}{suffix}.bin")
        written = shards.write_shard(documents(texts), path, limit)
        print(
            f"{name}: {written:,} tokens -> {path} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
    if stats:
        print(f"document filter: {stats}", flush=True)
    if zstats:
        print(f"compressibility filter: {zstats}", flush=True)
    if not args.skip_wikitext:
        wiki = datasets.load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split="test"
        )
        text = "".join(row["text"] for row in wiki)
        path = os.path.join(args.out, "wikitext_test.bin")
        written = shards.write_shard(
            [tokenizer(text, add_special_tokens=False)["input_ids"]], path
        )
        print(f"wikitext_test: {written:,} tokens -> {path}", flush=True)


if __name__ == "__main__":
    main()
