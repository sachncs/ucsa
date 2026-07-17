"""Tests for :mod:`ucsa.training.dataset`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from datasets import Dataset

from ucsa.models.perception import TokenizerWrapper
from ucsa.training.dataset import (
    FALLBACK_DATASETS,
    PRIMARY_DATASET,
    DatasetConfig,
    TextDataset,
    dataset_tokenizer,
    supported_datasets,
)


class FakeTokenizer:
    """internal: deterministic tokenizer for unit tests."""

    def __init__(self, vocab_size: int = 1024, pad_token_id: int = 0) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = 1

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        del add_special_tokens, truncation, max_length
        return [(ord(c) % self.vocab_size) for c in text]


def tiny_wrapper() -> TokenizerWrapper:
    """Return a TokenizerWrapper backed by FakeTokenizer."""
    fake = FakeTokenizer()
    wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
    wrapper.tokenizer = fake
    wrapper.max_seq_len = 64
    wrapper.vocab_size = fake.vocab_size
    return wrapper


def tiny_config(**overrides: object) -> DatasetConfig:
    """Return a tiny dataset config."""
    defaults: dict[str, object] = {
        "sequence_length": 32,
        "streaming": True,
        "pack_sequences": False,
    }
    defaults.update(overrides)
    return DatasetConfig(**defaults)  # type: ignore[arg-type]


def make_fake_dataset(
    texts: list[str] | None = None,
    config: DatasetConfig | None = None,
) -> TextDataset:
    """Build a TextDataset against an in-memory fake dataset."""

    class FakeBackedDataset(TextDataset):
        def _initialise_dataset(self_inner) -> object:  # type: ignore[override]
            data = {
                "text": texts
                if texts is not None
                else [
                    "hello world " * 10,
                    "another example sentence " * 8,
                    "yet more tokens for the loader " * 12,
                ]
            }
            return Dataset.from_dict(data)

        def _detect_text_field(self_inner) -> str:  # type: ignore[override]
            return "text"

    return FakeBackedDataset(tiny_wrapper(), config or tiny_config())


class TestDatasetConfig:
    """Tests for :class:`DatasetConfig`."""

    def test_default_config_valid(self) -> None:
        """Defaults construct without error."""
        config = DatasetConfig()
        assert config.sequence_length > 0
        assert config.primary_dataset == PRIMARY_DATASET
        assert config.fallback_chain == FALLBACK_DATASETS


class TestTextDataset:
    """Tests for :class:`TextDataset`."""

    def test_construction_with_fake_data(self) -> None:
        """A fake-backed dataset constructs without hitting the network."""
        ds = make_fake_dataset()
        assert ds.text_field == "text"

    def test_text_field_detection(self) -> None:
        """Text-field detection finds a known field name."""

        class CustomFieldDataset(TextDataset):
            def _initialise_dataset(self_inner) -> object:  # type: ignore[override]
                return Dataset.from_dict({"content": ["sample text"]})

            def _detect_text_field(self_inner) -> str:  # type: ignore[override]
                return "content"

        ds = CustomFieldDataset(tiny_wrapper(), tiny_config())
        assert ds.text_field == "content"

    def test_stream_batches_yields_input_target_pairs(self) -> None:
        """``stream_batches`` yields ``(inputs, targets)`` pairs."""
        ds = make_fake_dataset()
        for inputs, targets in ds.stream_batches(batch_size=2):
            assert inputs.dim() == 2
            assert targets.dim() == 2
            assert inputs.shape == targets.shape
            assert inputs.shape[1] == ds.config.sequence_length
            break

    def test_targets_are_shifted_inputs(self) -> None:
        """``targets`` equal ``inputs`` shifted by one token."""
        ds = make_fake_dataset()
        for inputs, targets in ds.stream_batches(batch_size=1):
            # Token at position i in targets == token at position i+1 in inputs.
            assert torch.all(targets[0, :-1] == inputs[0, 1:])
            break

    def test_pack_sequences(self) -> None:
        """Packed sequences produce chunks of the configured length."""
        config = tiny_config(sequence_length=10, pack_sequences=True)
        ds = make_fake_dataset(
            texts=["abcdefghij" * 50],
            config=config,
        )
        for inputs, targets in ds.stream_batches(batch_size=2):
            assert inputs.shape == (2, config.sequence_length)
            assert targets.shape == (2, config.sequence_length)
            break

    def test_pack_token_ids_yields_chunks(self) -> None:
        """``pack_token_ids`` yields tensors of the configured length."""
        ds = make_fake_dataset()
        buffer = list(range(100))
        chunks = list(ds.pack_token_ids(buffer))
        for chunk in chunks:
            assert chunk.shape == (ds.config.sequence_length + 1,)

    def test_pack_token_ids_drains_buffer(self) -> None:
        """Pack leaves less than ``sequence_length+1`` tokens in the buffer."""
        ds = make_fake_dataset()
        buffer = list(range(50))
        list(ds.pack_token_ids(buffer))
        assert len(buffer) < ds.config.sequence_length + 1

    def test_iter_yields_default_batch(self) -> None:
        """``__iter__`` yields batches of size 1 by default."""
        ds = make_fake_dataset()
        for inputs, targets in ds:
            assert inputs.shape[0] == 1
            break

    def test_no_text_field_raises(self) -> None:
        """Dataset raises when no recognisable text field is found."""

        class NoTextFieldDataset(TextDataset):
            def _initialise_dataset(self_inner) -> object:  # type: ignore[override]
                return Dataset.from_dict({"other": ["x"]})

            def _detect_text_field(self_inner) -> str:  # type: ignore[override]
                sample = next(iter(self_inner.dataset))
                for candidate in ("text", "content", "article"):
                    if candidate in sample:
                        return candidate
                raise RuntimeError(
                    f"Could not detect a text field in dataset sample: "
                    f"{sorted(sample.keys())}."
                )

        with pytest.raises(RuntimeError):
            NoTextFieldDataset(tiny_wrapper(), tiny_config())


class TestSupportedDatasets:
    """Tests for the supported dataset helpers."""

    def test_supported_datasets_includes_primary(self) -> None:
        """``supported_datasets`` includes the primary dataset name."""
        datasets = supported_datasets()
        assert PRIMARY_DATASET in datasets

    def test_supported_datasets_includes_fallbacks(self) -> None:
        """``supported_datasets`` includes the fallback names."""
        datasets = supported_datasets()
        for fallback_name, _ in FALLBACK_DATASETS:
            assert fallback_name in datasets


class TestDatasetTokenizerHelper:
    """Tests for the :func:`dataset_tokenizer` test helper."""

    def test_helper_returns_dataset(self) -> None:
        """``dataset_tokenizer`` builds a TextDataset."""
        fake = FakeTokenizer()
        ds = dataset_tokenizer(fake, sequence_length=8)
        for inputs, _ in ds:
            assert inputs.shape == (1, 8)
            break