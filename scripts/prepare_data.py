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
import concurrent.futures
import os
import time

import datasets
import huggingface_hub
import transformers

from ucsa.training import corpus, shards

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


def download(split: str) -> list[str]:
    """Downloads (or finds in the cache) the files of one corpus split.

    Args:
      split: Folder under `clean/`: `100M`, `dev` or `test`.

    Returns:
      Local paths, one per source, in `SOURCES` order.
    """

    def fetch(source: str) -> str:
        return huggingface_hub.hf_hub_download(
            REPO, f"clean/{split}/{source}.txt", repo_type="dataset"
        )

    with concurrent.futures.ThreadPoolExecutor(len(SOURCES)) as pool:
        return list(pool.map(fetch, SOURCES))


def main() -> None:
    """Parses arguments and writes the shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-wikitext", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    for name, split in SPLITS.items():
        path = os.path.join(args.out, f"{name}.bin")
        tokens, built = corpus.build(
            download(split), path, workers=args.workers, force=args.force
        )
        print(
            f"{name}: {tokens:,} tokens -> {path} "
            f"({'built' if built else 'already current'}, "
            f"{time.time() - started:.0f}s)",
            flush=True,
        )
    if not args.skip_wikitext:
        wiki = datasets.load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split="test"
        )
        text = "".join(row["text"] for row in wiki)
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        path = os.path.join(args.out, "wikitext_test.bin")
        written = shards.write_shard(
            [tokenizer(text, add_special_tokens=False)["input_ids"]], path
        )
        print(f"wikitext_test: {written:,} tokens -> {path}", flush=True)


if __name__ == "__main__":
    main()
