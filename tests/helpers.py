"""Fixtures shared by the test modules."""

import torch

from ucsa.models import recurrent
from ucsa.utils import precision


def tiny_model(**overrides) -> recurrent.Model:
    """A 2-layer model small enough to run in milliseconds."""
    precision.configure()
    torch.manual_seed(0)
    fields = {
        "vocab_size": 64,
        "hidden": 32,
        "layers": 2,
        "heads": 2,
        "ffn_dim": 64,
        "chunk_size": 8,
        "banks": (("working", 4), ("long_term", 2)),
        "bank_write_bias": (("working", 0.0), ("long_term", -2.0)),
    }
    fields.update(overrides)
    return recurrent.Model(recurrent.Config(**fields))


def token_batch(
    batch_size: int = 2, seq_len: int = 32, vocab: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed `(inputs, targets)` batch: a fixture, not result data."""
    ids = torch.arange(batch_size * (seq_len + 1)) % vocab
    ids = ids.view(batch_size, seq_len + 1)
    return ids[:, :-1], ids[:, 1:]


class Tokenizer:
    """A character tokenizer with the Hugging Face interface."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + ord(c) % 60 for c in text]
