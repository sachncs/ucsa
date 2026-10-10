"""Compression-based evaluation: bits per byte and classical anchors.

A language model is a compressor: a probability for every next token is a
code length for the text. Bits per byte (BPB) states that code length per
byte of raw text, so it is independent of the tokenizer and comparable
across models and with general-purpose compressors. It is also a far less
noisy measure of quality than benchmark accuracy for small models, whose
accuracy sits near chance.

The anchors (zlib, LZMA) compress the same bytes and show what a model must
beat to have learned anything about language beyond local repetition.
"""

import lzma
import math
import zlib
from typing import Any

import torch

# Bytes a special token such as <|endoftext|> stands for (a document break).
SPECIAL_TOKEN_BYTES = 1


def byte_to_unicode() -> dict[int, str]:
    """Returns GPT-2's reversible map from each byte to a printable char.

    Printable Latin-1 bytes map to themselves; the remaining bytes map to
    code points from 256 upward, so every byte has a visible, unique symbol.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    extra = 0
    table = {}
    for byte in range(256):
        if byte in printable:
            table[byte] = chr(byte)
        else:
            table[byte] = chr(256 + extra)
            extra += 1
    return table


def token_byte_lengths(tokenizer: Any) -> torch.Tensor:
    """Returns the number of raw bytes each token id decodes to.

    Args:
      tokenizer: A GPT-2 style byte-level BPE tokenizer.

    Returns:
      A long tensor of shape `(vocab,)`.
    """
    unicode_to_byte = {char: byte for byte, char in byte_to_unicode().items()}
    special = set(tokenizer.all_special_ids)
    lengths = []
    for index, token in enumerate(
        tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    ):
        if index in special or any(c not in unicode_to_byte for c in token):
            lengths.append(SPECIAL_TOKEN_BYTES)
        else:
            lengths.append(len(token))
    return torch.tensor(lengths, dtype=torch.long)


def bits_per_byte(total_nll_nats: float, total_bytes: int) -> float:
    """Converts a summed negative log-likelihood into bits per byte.

    Args:
      total_nll_nats: Summed NLL in nats over the scored tokens.
      total_bytes: Raw bytes those tokens stand for.

    Returns:
      Bits per byte, or infinity when there are no bytes.
    """
    if total_bytes <= 0:
        return math.inf
    return total_nll_nats / math.log(2.0) / total_bytes


def anchors(data: bytes) -> dict[str, float]:
    """Compresses `data` with general-purpose compressors.

    Args:
      data: Raw bytes.

    Returns:
      Bits per byte under zlib (level 9) and LZMA (preset 9).
    """
    if not data:
        return {"zlib": math.inf, "lzma": math.inf}
    return {
        "zlib": 8.0 * len(zlib.compress(data, 9)) / len(data),
        "lzma": 8.0 * len(lzma.compress(data, preset=9)) / len(data),
    }


def ideal_code_bits(logprobs: torch.Tensor) -> float:
    """Returns the Shannon code length, in bits, of a sequence.

    Args:
      logprobs: Natural-log probabilities of the tokens that occurred.

    Returns:
      `-sum(logprobs) / ln 2`.
    """
    return float(-logprobs.sum().item() / math.log(2.0))
