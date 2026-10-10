"""Tokenises the BabyLM corpus once into local shards.

    python scripts/prepare_data.py --out data

BabyLM (Warstadt et al., 2023) is a 100-million-word corpus of child-directed
speech, dialogue, subtitles, stories and encyclopedia text, built so that
small models can be trained and compared in hours. Writes `train.bin` (the
100M-word training split), `val.bin` (the official dev split) and `test.bin`
(the official test split), plus `wikitext_test.bin` as an out-of-domain
held-out set. Training, tuning and evaluation then read the shards and never
touch the network.
"""

import argparse
import os
import time
from collections.abc import Iterator

import datasets
import huggingface_hub
import transformers

from ucsa.training import shards

REPO = "cambridge-climb/BabyLM"
SOURCES = (
    "aochildes",
    "bnc_spoken",
    "cbt",
    "children_stories",
    "gutenberg",
    "open_subtitles",
    "qed",
    "simple_wikipedia",
    "switchboard",
    "wikipedia",
)
SPLITS = {"train": "100M", "val": "dev", "test": "test"}
LINES_PER_DOCUMENT = 32


def documents(path: str) -> Iterator[str]:
    """Yields pseudo-documents: runs of consecutive non-empty lines.

    The corpus is a stream of utterances and sentences without document
    boundaries, so consecutive lines are grouped; the end-of-text token then
    separates groups, not sentences.

    Args:
      path: A corpus text file.

    Yields:
      Up to `LINES_PER_DOCUMENT` lines joined by newlines.
    """
    block: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            block.append(line)
            if len(block) == LINES_PER_DOCUMENT:
                yield "\n".join(block)
                block = []
    if block:
        yield "\n".join(block)


def tokenized(
    texts: Iterator[str], tokenizer: transformers.PreTrainedTokenizerBase
) -> Iterator[list[int]]:
    """Yields token-id lists for texts, tokenised in batches of 256.

    Args:
      texts: Document texts.
      tokenizer: Tokenizer to apply.

    Yields:
      One list of token ids per text.
    """
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == 256:
            yield from tokenizer(batch, add_special_tokens=False)["input_ids"]
            batch = []
    if batch:
        yield from tokenizer(batch, add_special_tokens=False)["input_ids"]


def corpus_files(split: str) -> Iterator[str]:
    """Downloads (or finds in the cache) the files of one corpus split.

    Args:
      split: Folder under `clean/`: `100M`, `dev` or `test`.

    Yields:
      Local paths, one per source.
    """
    for source in SOURCES:
        yield huggingface_hub.hf_hub_download(
            REPO, f"clean/{split}/{source}.txt", repo_type="dataset"
        )


def main() -> None:
    """Parses arguments and writes the shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data")
    parser.add_argument("--skip-wikitext", action="store_true")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    for name, split in SPLITS.items():

        def texts(split: str = split) -> Iterator[str]:
            for path in corpus_files(split):
                yield from documents(path)

        path = os.path.join(args.out, f"{name}.bin")
        written = shards.write_shard(tokenized(texts(), tokenizer), path)
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
