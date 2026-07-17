"""Dataset loader.

Streams text from Hugging Face datasets with a graceful fallback chain:

1. ``HuggingFaceFW/fineweb-edu`` (default)
2. ``Skylion007/openwebtext``
3. ``wikitext`` (configuration ``wikitext-103-raw-v1``)

The loader packs tokenised text into fixed-length sequences suitable for
training. Streaming keeps memory pressure low for any dataset size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import Tensor

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from ucsa.models.perception import TokenizerWrapper


PRIMARY_DATASET: str = "HuggingFaceFW/fineweb-edu"
FALLBACK_DATASETS: tuple[tuple[str, str], ...] = (
    ("Skylion007/openwebtext", "text"),
    ("wikitext", "wikitext-103-raw-v1"),
)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for the streaming dataset loader.

    Attributes:
        sequence_length: Number of tokens per training sequence.
        primary_dataset: First-choice dataset identifier.
        primary_split: Split name for the primary dataset.
        primary_text_field: Text field name for the primary dataset.
        fallback_chain: Sequence of ``(dataset_name, split)`` tuples tried
            in order when the primary fails.
        streaming: Whether to stream the dataset (``True``) or download.
        pack_sequences: Whether to concatenate tokenised text into
            fixed-length sequences (``True``) or yield examples of
            arbitrary length.
    """

    sequence_length: int = 1024
    primary_dataset: str = PRIMARY_DATASET
    primary_split: str = "train"
    primary_text_field: str = "text"
    fallback_chain: tuple[tuple[str, str], ...] = FALLBACK_DATASETS
    streaming: bool = True
    pack_sequences: bool = True


class TextDataset:
    """Streaming text dataset producing fixed-length token sequences."""

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        config: DatasetConfig | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            tokenizer: Tokenizer wrapper used for tokenisation.
            config: Optional dataset configuration.
        """
        self.tokenizer = tokenizer
        if config is None:
            config = DatasetConfig()
        self.config = config
        self.dataset = self._initialise_dataset()
        self.text_field = self._detect_text_field()

    def _initialise_dataset(self) -> object:
        """Try to load the primary dataset; fall back on failure.

        Returns:
            A Hugging Face :class:`datasets.Dataset` (or streaming iterable).
        """
        try:
            return load_dataset(
                self.config.primary_dataset,
                split=self.config.primary_split,
                streaming=self.config.streaming,
            )
        except Exception:
            for dataset_name, split in self.config.fallback_chain:
                try:
                    return load_dataset(
                        dataset_name,
                        split=split,
                        streaming=self.config.streaming,
                    )
                except Exception:
                    continue
            raise RuntimeError(
                "Unable to load any dataset from the configured chain."
            )

    def _detect_text_field(self) -> str:
        """Return the dataset's text field name."""
        primary = self.config.primary_text_field
        sample = next(iter(self.dataset))
        if primary in sample:
            return primary
        for candidate in ("text", "content", "article"):
            if candidate in sample:
                return candidate
        raise RuntimeError(
            f"Could not detect a text field in dataset sample: "
            f"{sorted(sample.keys())}."
        )

    def pack_token_ids(
        self, token_buffer: list[int]
    ) -> Iterator[Tensor]:
        """Pack tokenised text into fixed-length tensors.

        Args:
            token_buffer: A growing buffer of token ids.

        Yields:
            Tensors of shape ``(sequence_length,)``.
        """
        seq_len = self.config.sequence_length
        while len(token_buffer) >= seq_len + 1:
            chunk = token_buffer[: seq_len + 1]
            del token_buffer[: seq_len + 1]
            yield torch.tensor(chunk, dtype=torch.long)

    def stream_batches(
        self,
        batch_size: int = 1,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield ``(inputs, targets)`` batches.

        Args:
            batch_size: Number of sequences per batch.

        Yields:
            Tuples ``(input_ids, target_ids)`` of shape
            ``(batch_size, sequence_length)``.
        """
        seq_len = self.config.sequence_length
        token_buffer: list[int] = []
        for example in self.dataset:
            text = example[self.text_field]
            if not text:
                continue
            ids = self.tokenizer.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=seq_len * 4,
            )
            token_buffer.extend(ids)
            if self.config.pack_sequences:
                packed: list[Tensor] = []
                for chunk in self.pack_token_ids(token_buffer):
                    packed.append(chunk)
                while len(packed) >= batch_size:
                    batch = torch.stack(packed[:batch_size], dim=0)
                    packed = packed[batch_size:]
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]
                    yield inputs, targets
            else:
                if len(token_buffer) >= seq_len + 1:
                    chunk = torch.tensor(
                        token_buffer[: seq_len + 1], dtype=torch.long
                    )
                    token_buffer = token_buffer[seq_len + 1:]
                    inputs = chunk[:-1].unsqueeze(0)
                    targets = chunk[1:].unsqueeze(0)
                    yield inputs, targets

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate over batches with default batch size 1."""
        return self.stream_batches(batch_size=1)


__all__ = [
    "FALLBACK_DATASETS",
    "DatasetConfig",
    "PRIMARY_DATASET",
    "TextDataset",
]


def supported_datasets() -> tuple[str, ...]:
    """internal: return the canonical dataset chain as a flat tuple."""
    return (PRIMARY_DATASET, *tuple(name for name, _ in FALLBACK_DATASETS))


def dataset_tokenizer(  # internal: helper for tests
    tokenizer: PreTrainedTokenizerBase, sequence_length: int = 64
) -> TextDataset:
    """internal: build a TextDataset against an in-memory fake dataset.

    Used by unit tests so we never hit the network.
    """
    from datasets import Dataset

    wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
    wrapper.tokenizer = tokenizer
    wrapper.max_seq_len = sequence_length
    wrapper.vocab_size = int(tokenizer.vocab_size)
    config = DatasetConfig(
        sequence_length=sequence_length,
        streaming=True,
        pack_sequences=False,
    )

    class FakeDataset(TextDataset):
        def _initialise_dataset(self_inner) -> object:  # type: ignore[override]
            data = {
                "text": [
                    "hello world " * 10,
                    "another example sentence " * 8,
                    "yet more tokens for the loader " * 12,
                ]
            }
            return Dataset.from_dict(data)

        def _detect_text_field(self_inner) -> str:  # type: ignore[override]
            return "text"

    return FakeDataset(wrapper, config)