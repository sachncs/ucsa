"""Lossless compression with a language model and arithmetic coding.

A model that assigns a probability to every next token defines a code: an
arithmetic coder turns those probabilities into a bit stream whose length is
the model's negative log-likelihood, to within a few bytes. Decoding runs the
same model on the tokens recovered so far and inverts the coding step.

That makes compression an end-to-end test of the model as well as a result:

* A decoder only has the tokens before the one it is decoding. If the model's
  probabilities depended on the future (a leak), the decoder could not
  reproduce the encoder's probabilities and the round trip would fail.
* The compressed size is the bits-per-byte actually achieved, not an
  estimate.

The coder is the integer range coder of Witten, Neal and Cleary with Python
integers, so there is no overflow at any precision. Probabilities are
quantised to integer counts from float64 softmax, identically on both sides.
"""

import math
import struct
from typing import Protocol

import torch
from torch.nn import functional

from ucsa.models import recurrent

PRECISION = 62
WHOLE = 1 << PRECISION
HALF = WHOLE >> 1
QUARTER = WHOLE >> 2
THREE_QUARTERS = HALF + QUARTER
# Total of all symbol counts; must exceed the vocabulary so each symbol keeps
# a non-zero count, and stay far below 2**PRECISION so ranges never collapse.
SCALE = 1 << 28
HEADER = struct.Struct(">Q")


class Predictor(Protocol):
    """A source of next-token logits that both sides advance identically."""

    def logits(self) -> torch.Tensor:
        """Returns logits of shape `(vocab,)` for the next token."""

    def update(self, token: int) -> None:
        """Consumes the token that actually occurred."""


def cumulative_counts(logits: torch.Tensor) -> torch.Tensor:
    """Quantises logits into cumulative integer counts.

    Every symbol receives at least one count so any token can be coded.

    Args:
      logits: Tensor of shape `(vocab,)`.

    Returns:
      Cumulative counts of shape `(vocab + 1,)` starting at zero.

    Raises:
      ValueError: If the vocabulary is larger than `SCALE`.
    """
    vocab = logits.shape[-1]
    if vocab >= SCALE:
        raise ValueError("vocabulary too large for the count scale")
    probs = functional.softmax(logits.detach().cpu(), dim=-1)
    counts = (probs * (SCALE - vocab)).floor().long() + 1
    return torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])


class Encoder:
    """Arithmetic encoder producing a bit stream."""

    def __init__(self) -> None:
        """Starts with the full interval."""
        self.low = 0
        self.high = WHOLE - 1
        self.pending = 0
        self.bits: list[int] = []

    def emit(self, bit: int) -> None:
        """Appends `bit` followed by the bits deferred by underflow."""
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0

    def encode(self, low_count: int, high_count: int, total: int) -> None:
        """Narrows the interval to a symbol's slice.

        Args:
          low_count: Cumulative count below the symbol.
          high_count: Cumulative count through the symbol.
          total: Sum of all counts.
        """
        span = self.high - self.low + 1
        self.high = self.low + span * high_count // total - 1
        self.low = self.low + span * low_count // total
        while True:
            if self.high < HALF:
                self.emit(0)
            elif self.low >= HALF:
                self.emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low *= 2
            self.high = self.high * 2 + 1

    def finish(self) -> bytes:
        """Flushes the final interval and returns the packed bytes."""
        self.pending += 1
        self.emit(0 if self.low < QUARTER else 1)
        padded = self.bits + [0] * (-len(self.bits) % 8)
        out = bytearray()
        for i in range(0, len(padded), 8):
            byte = 0
            for bit in padded[i : i + 8]:
                byte = byte << 1 | bit
            out.append(byte)
        return bytes(out)


class Decoder:
    """Arithmetic decoder reading a bit stream."""

    def __init__(self, data: bytes) -> None:
        """Primes the decoder with the first `PRECISION` bits.

        Args:
          data: Bytes produced by `Encoder.finish`.
        """
        self.data = data
        self.position = 0
        self.low = 0
        self.high = WHOLE - 1
        self.value = 0
        for _ in range(PRECISION):
            self.value = self.value << 1 | self.read_bit()

    def read_bit(self) -> int:
        """Returns the next bit; reads zeros past the end of the data."""
        byte, offset = divmod(self.position, 8)
        self.position += 1
        if byte >= len(self.data):
            return 0
        return (self.data[byte] >> (7 - offset)) & 1

    def decode(self, cumulative: torch.Tensor) -> int:
        """Decodes one symbol.

        Args:
          cumulative: Output of `cumulative_counts` for this step.

        Returns:
          The symbol index.
        """
        total = int(cumulative[-1])
        span = self.high - self.low + 1
        scaled = ((self.value - self.low + 1) * total - 1) // span
        symbol = int(
            torch.searchsorted(cumulative, torch.tensor(scaled), right=True) - 1
        )
        self.high = self.low + span * int(cumulative[symbol + 1]) // total - 1
        self.low = self.low + span * int(cumulative[symbol]) // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.value -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low *= 2
            self.high = self.high * 2 + 1
            self.value = self.value * 2 + self.read_bit()
        return symbol


class ModelPredictor:
    """Streams a `recurrent.Model` one token at a time, on the CPU.

    Encoder and decoder each build one of these, so both see bit-identical
    logits: the same streaming code path, device and operation order.
    """

    def __init__(self, model: recurrent.Model, start_token: int) -> None:
        """Prepares the stream.

        Args:
          model: A trained model; it is moved to the CPU and set to eval mode.
          start_token: Token that stands before the first coded token (the
            end-of-text id), so the first real token has a context.
        """
        self.model = model.cpu().eval()
        self.size = model.config.chunk_size
        self.state = self.model.initial_state(1)
        self.previous = None  # cache of the chunk before the current one
        self.cache = None  # cache of the current chunk, filled by `logits`
        self.current = [start_token]

    @torch.no_grad()
    def logits(self) -> torch.Tensor:
        """Returns next-token logits given everything consumed so far."""
        current = torch.tensor([self.current], dtype=torch.long)
        logits, self.cache = self.model.next_logits(
            self.state, current, self.previous
        )
        return logits[0]

    @torch.no_grad()
    def update(self, token: int) -> None:
        """Appends `token`, advancing the state when a chunk completes."""
        self.current.append(token)
        if len(self.current) > self.size:
            chunk = torch.tensor([self.current[: self.size]], dtype=torch.long)
            self.state = self.model.advance(self.state, chunk)
            self.previous = self.cache  # computed when the chunk was full
            self.current = self.current[self.size :]


def compress(predictor: Predictor, tokens: list[int]) -> bytes:
    """Compresses a token sequence.

    Args:
      predictor: Fresh predictor positioned before the first token.
      tokens: Token ids to code.

    Returns:
      An 8-byte length header followed by the arithmetic-coded payload.
    """
    encoder = Encoder()
    for token in tokens:
        cumulative = cumulative_counts(predictor.logits())
        encoder.encode(
            int(cumulative[token]),
            int(cumulative[token + 1]),
            int(cumulative[-1]),
        )
        predictor.update(token)
    return HEADER.pack(len(tokens)) + encoder.finish()


def decompress(predictor: Predictor, data: bytes) -> list[int]:
    """Recovers the token sequence from `compress` output.

    Args:
      predictor: Fresh predictor positioned before the first token, built the
        same way as the encoder's.
      data: Bytes returned by `compress`.

    Returns:
      The decoded token ids.

    Raises:
      ValueError: If `data` is too short to hold the header.
    """
    if len(data) < HEADER.size:
        raise ValueError("data too short to contain a header")
    (count,) = HEADER.unpack_from(data)
    decoder = Decoder(data[HEADER.size :])
    tokens = []
    for _ in range(count):
        token = decoder.decode(cumulative_counts(predictor.logits()))
        tokens.append(token)
        predictor.update(token)
    return tokens


def ideal_bits(predictor: Predictor, tokens: list[int]) -> float:
    """Returns the Shannon code length of `tokens` under the quantised model.

    Args:
      predictor: Fresh predictor positioned before the first token.
      tokens: Token ids.

    Returns:
      The sum of `-log2(count / total)` over the tokens.
    """
    bits = 0.0
    for token in tokens:
        cumulative = cumulative_counts(predictor.logits())
        count = int(cumulative[token + 1] - cumulative[token])
        bits -= math.log2(count / int(cumulative[-1]))
        predictor.update(token)
    return bits
