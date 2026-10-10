"""Pre-tokenised token shards: deterministic, resumable, network-free input.

Streaming a corpus while training ties step time to the network and makes an
exact resume impossible. A shard is a flat `uint16` file of GPT-2 token ids
(documents separated by the end-of-text id). Reading it is a memory map, a
batch is a pure function of `(seed, step)`, and a resumed run continues the
data stream exactly.
"""

import hashlib
import os
import zlib
from collections.abc import Iterable, Iterator

import numpy as np
import torch

EOS_ID = 50256
DTYPE = np.uint16


def filter_documents(
    token_lists: Iterable[list[int]],
    min_tokens: int = 32,
    prefix_len: int = 128,
    stats: dict[str, int] | None = None,
    seen: set[bytes] | None = None,
) -> Iterator[list[int]]:
    """Drops short and duplicate documents.

    A document is a duplicate when its first `prefix_len` tokens were already
    seen, whatever follows. That removes repeated boilerplate, mirrored pages
    and templated series, which otherwise leak between training and
    validation and inflate the apparent quality of memorisation. It also
    drops a distinct document that happens to share a long common opening.

    Args:
      token_lists: Token-id lists, one per document.
      min_tokens: Documents shorter than this are dropped.
      prefix_len: Number of leading tokens that identify a document.
      stats: If given, updated in place with `seen`, `kept`, `short` and
        `duplicate` counts.
      seen: Fingerprints already seen. Pass the same set to several calls so
        a document written to one shard is never written to another.

    Yields:
      The documents that survive both filters, in order.
    """
    counts = stats if stats is not None else {}
    for key in ("seen", "kept", "short", "duplicate"):
        counts.setdefault(key, 0)
    seen_hashes = set() if seen is None else seen
    for ids in token_lists:
        counts["seen"] += 1
        if len(ids) < min_tokens:
            counts["short"] += 1
            continue
        head = np.asarray(ids[:prefix_len], dtype=DTYPE).tobytes()
        digest = hashlib.blake2b(head, digest_size=8).digest()
        if digest in seen_hashes:
            counts["duplicate"] += 1
            continue
        seen_hashes.add(digest)
        counts["kept"] += 1
        yield ids


def compression_ratio(text: str) -> float:
    """Returns compressed size over raw size under zlib level 6.

    Natural prose sits near 0.4. Boilerplate and repeated lists compress far
    below that; garbled or machine-generated text compresses far less.

    Args:
      text: Document text.

    Returns:
      `len(zlib(text)) / len(text)` in bytes, or 1.0 for empty text.
    """
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 6)) / len(raw)


def filter_by_compressibility(
    texts: Iterable[str],
    low: float,
    high: float,
    stats: dict[str, int] | None = None,
) -> Iterator[str]:
    """Keeps documents whose compression ratio lies in `[low, high]`.

    Args:
      texts: Document texts.
      low: Minimum ratio; below it a document is too repetitive.
      high: Maximum ratio; above it a document is too random.
      stats: If given, updated in place with `seen`, `kept`, `too_repetitive`
        and `too_random` counts.

    Yields:
      The documents that pass.
    """
    counts = stats if stats is not None else {}
    for key in ("seen", "kept", "too_repetitive", "too_random"):
        counts.setdefault(key, 0)
    for text in texts:
        counts["seen"] += 1
        ratio = compression_ratio(text)
        if ratio < low:
            counts["too_repetitive"] += 1
        elif ratio > high:
            counts["too_random"] += 1
        else:
            counts["kept"] += 1
            yield text


def write_shard(
    token_lists: Iterable[list[int]], path: str, limit: int | None = None
) -> int:
    """Writes documents to a shard, separated by the end-of-text id.

    The file appears all at once: tokens go to a temporary file that replaces
    `path` only when complete. A reader that has `path` memory-mapped keeps
    its old, intact copy instead of seeing a truncated file, and a crash while
    writing leaves the previous shard untouched.

    Args:
      token_lists: Iterable of token-id lists, one per document.
      path: Output file.
      limit: Stop after at least this many tokens; None writes everything.

    Returns:
      The number of tokens written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = 0
    temporary = path + ".tmp"
    try:
        with open(temporary, "wb") as out:
            for ids in token_lists:
                block = np.asarray([*ids, EOS_ID], dtype=DTYPE)
                out.write(block.tobytes())
                written += block.size
                if limit is not None and written >= limit:
                    break
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return written


class Shard:
    """A memory-mapped token file that yields language-model batches."""

    def __init__(self, path: str) -> None:
        """Opens a shard.

        Args:
          path: File written by `write_shard`.

        Raises:
          FileNotFoundError: If `path` does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; run scripts/prepare_data.py first"
            )
        self.path = path
        self.tokens = np.memmap(path, dtype=DTYPE, mode="r")

    def __len__(self) -> int:
        """Returns the number of tokens in the shard."""
        return int(self.tokens.size)

    def windows(self, seq_len: int) -> int:
        """Returns how many non-overlapping `seq_len` windows fit."""
        return (len(self) - 1) // seq_len

    def batch(
        self, starts: np.ndarray, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds one `(inputs, targets)` batch from window start offsets."""
        rows = np.stack(
            [self.tokens[s : s + seq_len + 1].astype(np.int64) for s in starts]
        )
        data = torch.from_numpy(rows)
        return data[:, :-1].contiguous(), data[:, 1:].contiguous()

    def batches(
        self,
        batch_size: int,
        seq_len: int,
        skip: int = 0,
        seed: int | None = 0,
        loop: bool = True,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yields batches in a deterministic order.

        Args:
          batch_size: Sequences per batch.
          seq_len: Tokens per sequence.
          skip: Batches to skip, so a resumed run continues the stream.
          seed: Shuffles windows per epoch with `seed + epoch`; None keeps
            file order (used for validation).
          loop: Whether to start another epoch after the last window.

        Yields:
          `(inputs, targets)` int64 tensors of shape `(batch_size, seq_len)`.

        Raises:
          ValueError: If the shard is shorter than one window.
        """
        count = self.windows(seq_len)
        if count == 0:
            raise ValueError(f"shard shorter than seq_len={seq_len}")
        index = skip * batch_size
        order_epoch, order = -1, None
        while loop or index + batch_size <= count:
            picks = []
            for k in range(index, index + batch_size):
                epoch, pos = divmod(k, count)
                if epoch != order_epoch:
                    order_epoch = epoch
                    order = (
                        np.arange(count)
                        if seed is None
                        else np.random.default_rng(seed + epoch).permutation(
                            count
                        )
                    )
                picks.append(order[pos] * seq_len)
            yield self.batch(np.asarray(picks), seq_len)
            index += batch_size
