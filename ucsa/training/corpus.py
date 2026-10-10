"""Turns text files into token shards, in parallel, and only when needed.

Tokenising is the slowest step between a raw corpus and a training run, and
it parallelises perfectly across source files. Each file is tokenised in its
own process; the pieces are joined in file order, so the shard is identical
whatever the worker count. A manifest next to the shard records what it was
built from (tokenizer, and the name and size of every source), so rebuilding
an unchanged corpus is a no-op.
"""

import concurrent.futures
import json
import os
from collections.abc import Iterator

import numpy as np
import transformers

from ucsa.training import shards

LINES_PER_DOCUMENT = 32
BATCH = 256
MANIFEST_VERSION = 1


def documents(path: str) -> Iterator[str]:
    """Yields pseudo-documents: runs of consecutive non-empty lines.

    A corpus of utterances or sentences has no document boundaries, so
    consecutive lines are grouped; the end-of-text token then separates
    groups, not sentences.

    Args:
      path: A text file.

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


def tokenize_file(path: str, tokenizer_name: str = "gpt2") -> np.ndarray:
    """Tokenises one file into ids, with an end-of-text id after each document.

    Args:
      path: A text file.
      tokenizer_name: Hugging Face tokenizer name.

    Returns:
      A `uint16` array of token ids.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # one process per file
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    pieces: list[np.ndarray] = []

    def flush(batch: list[str]) -> None:
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            pieces.append(np.asarray([*ids, shards.EOS_ID], dtype=shards.DTYPE))

    batch: list[str] = []
    for text in documents(path):
        batch.append(text)
        if len(batch) == BATCH:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    if not pieces:
        return np.zeros(0, dtype=shards.DTYPE)
    return np.concatenate(pieces)


def manifest_path(out: str) -> str:
    """Returns the manifest file that belongs to a shard."""
    return out + ".json"


def describe(sources: list[str], tokenizer_name: str) -> dict:
    """Describes a build's inputs: what must match for a shard to be current.

    Args:
      sources: Source files, in shard order.
      tokenizer_name: Hugging Face tokenizer name.

    Returns:
      A JSON-serialisable dict.
    """
    return {
        "version": MANIFEST_VERSION,
        "tokenizer": tokenizer_name,
        "sources": [
            {"name": os.path.basename(p), "bytes": os.path.getsize(p)}
            for p in sources
        ],
    }


def is_current(out: str, sources: list[str], tokenizer_name: str) -> bool:
    """Whether `out` was built from exactly these sources and tokenizer.

    Args:
      out: Shard path.
      sources: Source files, in shard order.
      tokenizer_name: Hugging Face tokenizer name.

    Returns:
      True if the shard and its manifest exist, agree with the inputs, and
      the shard has the size the manifest promises.
    """
    if not (os.path.exists(out) and os.path.exists(manifest_path(out))):
        return False
    try:
        with open(manifest_path(out)) as f:
            recorded = json.load(f)
    except ValueError:
        return False
    want = describe(sources, tokenizer_name)
    same = all(recorded.get(k) == v for k, v in want.items())
    size = recorded.get("tokens", -1) * np.dtype(shards.DTYPE).itemsize
    return same and os.path.getsize(out) == size


def build(
    sources: list[str],
    out: str,
    tokenizer_name: str = "gpt2",
    workers: int | None = None,
    force: bool = False,
) -> tuple[int, bool]:
    """Builds a shard from text files unless it is already current.

    Args:
      sources: Source files, in shard order.
      out: Shard path.
      tokenizer_name: Hugging Face tokenizer name.
      workers: Processes to use; None means one per core, up to the number of
        files. One runs in this process.
      force: Rebuild even if the shard is current.

    Returns:
      `(tokens, built)`: the shard's token count and whether it was rebuilt.
    """
    if not force and is_current(out, sources, tokenizer_name):
        with open(manifest_path(out)) as f:
            return int(json.load(f)["tokens"]), False
    count = min(len(sources), workers or os.cpu_count() or 1)
    if count <= 1:
        parts = [tokenize_file(p, tokenizer_name) for p in sources]
    else:
        with concurrent.futures.ProcessPoolExecutor(count) as pool:
            parts = list(
                pool.map(
                    tokenize_file, sources, [tokenizer_name] * len(sources)
                )
            )
    tokens = shards.write_array(
        np.concatenate(parts) if parts else np.zeros(0), out
    )
    manifest = {**describe(sources, tokenizer_name), "tokens": tokens}
    with shards.atomic(manifest_path(out)) as f:
        f.write(json.dumps(manifest, indent=2).encode())
    return tokens, True
