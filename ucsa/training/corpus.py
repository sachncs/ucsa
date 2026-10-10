"""Turns text files into token shards, in parallel, and only when needed.

Tokenising is the slowest step between a raw corpus and a training run, and
it parallelises perfectly. Files are cut into pieces of about `PIECE_BYTES` on
line boundaries, so one large file cannot serialise the build; each piece is
tokenised in its own process and the pieces are joined in order. The cut
points depend on the data only, so the shard is identical whatever the worker
count. A manifest next to the shard records what it was
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
PIECE_BYTES = 8 * 2**20
MANIFEST_VERSION = 2


def pieces(path: str, size: int = PIECE_BYTES) -> list[tuple[str, int, int]]:
    """Cuts a file into byte ranges that end on line boundaries.

    Args:
      path: A text file.
      size: Target bytes per piece.

    Returns:
      `(path, start, end)` ranges that tile the file, in order.
    """
    total = os.path.getsize(path)
    out: list[tuple[str, int, int]] = []
    start = 0
    with open(path, "rb") as f:
        while start < total:
            f.seek(min(start + size, total))
            f.readline()  # extend to the end of the line
            end = min(f.tell(), total)
            out.append((path, start, end))
            start = end
    return out


def documents(text: str) -> Iterator[str]:
    """Yields pseudo-documents: runs of consecutive non-empty lines.

    A corpus of utterances or sentences has no document boundaries, so
    consecutive lines are grouped; the end-of-text token then separates
    groups, not sentences.

    Args:
      text: Text to group.

    Yields:
      Up to `LINES_PER_DOCUMENT` lines joined by newlines.
    """
    block: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        block.append(line)
        if len(block) == LINES_PER_DOCUMENT:
            yield "\n".join(block)
            block = []
    if block:
        yield "\n".join(block)


def tokenize_piece(
    piece: tuple[str, int, int], tokenizer_name: str = "gpt2"
) -> np.ndarray:
    """Tokenises one byte range, with an end-of-text id after each document.

    Args:
      piece: `(path, start, end)` as returned by `pieces`.
      tokenizer_name: Hugging Face tokenizer name.

    Returns:
      A `uint16` array of token ids.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # one process per piece
    path, start, end = piece
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    parts: list[np.ndarray] = []

    def flush(batch: list[str]) -> None:
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            parts.append(np.asarray([*ids, shards.EOS_ID], dtype=shards.DTYPE))

    batch: list[str] = []
    for doc in documents(text):
        batch.append(doc)
        if len(batch) == BATCH:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=shards.DTYPE)


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
        pieces. One runs in this process.
      force: Rebuild even if the shard is current.

    Returns:
      `(tokens, built)`: the shard's token count and whether it was rebuilt.
    """
    if not force and is_current(out, sources, tokenizer_name):
        with open(manifest_path(out)) as f:
            return int(json.load(f)["tokens"]), False
    work = [piece for path in sources for piece in pieces(path)]
    count = min(len(work), workers or os.cpu_count() or 1)
    if count <= 1:
        parts = [tokenize_piece(piece, tokenizer_name) for piece in work]
    else:
        with concurrent.futures.ProcessPoolExecutor(count) as pool:
            parts = list(
                pool.map(
                    tokenize_piece,
                    work,
                    [tokenizer_name] * len(work),
                    chunksize=1,
                )
            )
    tokens = shards.write_array(
        np.concatenate(parts) if parts else np.zeros(0), out
    )
    manifest = {**describe(sources, tokenizer_name), "tokens": tokens}
    with shards.atomic(manifest_path(out)) as f:
        f.write(json.dumps(manifest, indent=2).encode())
    return tokens, True
