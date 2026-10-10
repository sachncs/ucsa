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
    parser.add_argument("--skip-wikitext", action="store_true")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    os.makedirs(args.out, exist_ok=True)
    stream = datasets.load_dataset(args.dataset, split="train", streaming=True)
    docs = tokenized_documents((row["text"] for row in stream), tokenizer)

    started = time.time()
    for name, limit in (("val", args.val_tokens), ("train", args.train_tokens)):
        path = os.path.join(args.out, f"{name}.bin")
        written = shards.write_shard(docs, path, limit)
        print(
            f"{name}: {written:,} tokens -> {path} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
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
