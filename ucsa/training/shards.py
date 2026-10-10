"""Pre-tokenised token shards: deterministic, resumable, network-free input.

Streaming a corpus while training ties step time to the network and makes an
exact resume impossible. A shard is a flat `uint16` file of GPT-2 token ids
(documents separated by the end-of-text id). Reading it is a memory map, a
batch is a pure function of `(seed, step)`, and a resumed run continues the
data stream exactly.
"""

import hashlib
import os
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
) -> Iterator[list[int]]:
    """Drops short and duplicate documents.

    A document is a duplicate when its first `prefix_len` tokens (and its
    length class) were already seen. That removes repeated boilerplate and
    mirrored pages, which otherwise leak between training and validation and
    inflate the apparent quality of memorisation.

    Args:
      token_lists: Token-id lists, one per document.
      min_tokens: Documents shorter than this are dropped.
      prefix_len: Number of leading tokens that identify a document.
      stats: If given, updated in place with `seen`, `kept`, `short` and
        `duplicate` counts.

    Yields:
      The documents that survive both filters, in order.
    """
    counts = stats if stats is not None else {}
    for key in ("seen", "kept", "short", "duplicate"):
        counts.setdefault(key, 0)
    seen_hashes: set[bytes] = set()
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


def write_shard(
    token_lists: Iterable[list[int]], path: str, limit: int | None = None
) -> int:
    """Writes documents to a shard, separated by the end-of-text id.

    Args:
      token_lists: Iterable of token-id lists, one per document.
      path: Output file.
      limit: Stop after at least this many tokens; None writes everything.

    Returns:
      The number of tokens written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = 0
    with open(path, "wb") as out:
        for ids in token_lists:
            block = np.asarray([*ids, EOS_ID], dtype=DTYPE)
            out.write(block.tobytes())
            written += block.size
            if limit is not None and written >= limit:
                break
    return written


class TokenShard:
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
