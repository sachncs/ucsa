"""Tests for :mod:`ucsa.models.perception`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ucsa.models.perception import (
    MODALITY_CODE,
    MODALITY_TEXT,
    Perception,
    PerceptionConfig,
    TokenizerWrapper,
)


def tiny_config(**overrides: object) -> PerceptionConfig:
    """Return a tiny perception config for tests."""
    defaults: dict[str, object] = {
        "hidden_size": 32,
        "vocab_size": 1024,
        "max_seq_len": 64,
        "tokenizer_name": "gpt2",
    }
    defaults.update(overrides)
    return PerceptionConfig(**defaults)  # type: ignore[arg-type]


class FakeTokenizer:
    """internal: deterministic tokenizer for unit tests."""

    def __init__(self, vocab_size: int = 1024, pad_token_id: int = 0) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = 1
        self.calls: list[list[str]] = []

    def encode(
        self, text: str, add_special_tokens: bool = True, **kwargs: object
    ) -> list[int]:
        del add_special_tokens, kwargs
        return [(ord(c) % self.vocab_size) for c in text[:32]]

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, Tensor]:
        self.calls.append(texts)
        ids_list = [self.encode(t) for t in texts]
        max_len = max(len(ids) for ids in ids_list)
        padded = [
            ids + [self.pad_token_id] * (max_len - len(ids)) for ids in ids_list
        ]
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}


class TestPerceptionConfig:
    """Tests for :class:`PerceptionConfig`."""

    def test_default_config_valid(self) -> None:
        """Defaults construct without error."""
        config = PerceptionConfig()
        assert config.hidden_size > 0
        assert config.tokenizer_name == "gpt2"

    def test_zero_hidden_size_rejected(self) -> None:
        """``hidden_size`` of zero or less raises."""
        with pytest.raises(ValueError):
            PerceptionConfig(hidden_size=0)

    def test_zero_vocab_size_rejected(self) -> None:
        """``vocab_size`` of zero or less raises."""
        with pytest.raises(ValueError):
            PerceptionConfig(vocab_size=0)

    def test_empty_modalities_rejected(self) -> None:
        """``modalities`` must contain at least one entry."""
        with pytest.raises(ValueError):
            PerceptionConfig(modalities=())


class TestTokenizerWrapper:
    """Tests for :class:`TokenizerWrapper`."""

    def test_constructs_with_injected_tokenizer(self) -> None:
        """A pre-built tokenizer can be injected."""
        fake = FakeTokenizer()
        wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
        wrapper.tokenizer = fake
        wrapper.max_seq_len = 64
        wrapper.vocab_size = fake.vocab_size
        assert wrapper.vocab_size == 1024
        assert wrapper.pad_id == 0

    def test_encode_returns_long_tensor(self) -> None:
        """``encode`` returns a Long tensor."""
        fake = FakeTokenizer()
        wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
        wrapper.tokenizer = fake
        wrapper.max_seq_len = 64
        wrapper.vocab_size = fake.vocab_size
        out = wrapper.encode("hello")
        assert isinstance(out, Tensor)
        assert out.dtype == torch.long

    def test_batch_encode_pads(self) -> None:
        """``batch_encode`` pads shorter sequences to the longest."""
        fake = FakeTokenizer()
        wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
        wrapper.tokenizer = fake
        wrapper.max_seq_len = 64
        wrapper.vocab_size = fake.vocab_size
        out = wrapper.batch_encode(["hi", "longer text here"])
        assert out.dim() == 2
        assert out.shape[0] == 2
        assert out.shape[1] >= max(len("hi"), len("longer text here"))

    def test_pad_id_property(self) -> None:
        """``pad_id`` returns the tokenizer's pad token id."""
        fake = FakeTokenizer(pad_token_id=7)
        wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
        wrapper.tokenizer = fake
        wrapper.max_seq_len = 64
        wrapper.vocab_size = fake.vocab_size
        assert wrapper.pad_id == 7


class TestPerception:
    """Tests for :class:`Perception`."""

    @pytest.fixture
    def perception(self) -> Perception:
        """Provide a perception instance with a fake tokenizer."""
        config = tiny_config()
        fake = FakeTokenizer(
            vocab_size=config.vocab_size, pad_token_id=config.pad_token_id
        )
        wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
        wrapper.tokenizer = fake
        wrapper.max_seq_len = config.max_seq_len
        wrapper.vocab_size = fake.vocab_size
        return Perception(config=config, tokenizer=wrapper)

    def test_embed_tokens_shape(self, perception: Perception) -> None:
        """Token embedding produces ``(batch, seq, hidden_size)`` outputs."""
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        out = perception.embed_tokens(ids)
        assert out.shape == (2, 3, 32)

    def test_project_shape(self, perception: Perception) -> None:
        """Projection preserves shape."""
        x = torch.randn(2, 4, 32)
        out = perception.project(x, modality=MODALITY_TEXT)
        assert out.shape == x.shape

    def test_project_modality_changes_output(
        self, perception: Perception
    ) -> None:
        """Different modalities produce different projected outputs."""
        x = torch.randn(1, 4, 32)
        out_text = perception.project(x, modality=MODALITY_TEXT)
        out_code = perception.project(x, modality=MODALITY_CODE)
        assert not torch.allclose(out_text, out_code)

    def test_project_unknown_modality_raises(
        self, perception: Perception
    ) -> None:
        """An unknown modality raises ``ValueError``."""
        with pytest.raises(ValueError):
            perception.project(torch.randn(1, 4, 32), modality=999)

    def test_forward_from_ids(self, perception: Perception) -> None:
        """``forward_from_ids`` produces observation tokens."""
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = perception.forward_from_ids(ids, modality=MODALITY_TEXT)
        assert out.shape == (1, 3, 32)

    def test_forward_string_input(self, perception: Perception) -> None:
        """``forward`` accepts a single string."""
        out = perception.forward("hello world", modality=MODALITY_TEXT)
        assert out.dim() == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 32

    def test_forward_list_input(self, perception: Perception) -> None:
        """``forward`` accepts a list of strings."""
        out = perception.forward(["a", "bb", "ccc"], modality=MODALITY_CODE)
        assert out.dim() == 3
        assert out.shape[0] == 3
        assert out.shape[2] == 32

    def test_forward_different_modalities_produce_different_outputs(
        self, perception: Perception
    ) -> None:
        """Same text in different modalities yields different outputs."""
        text_text = perception.forward("def f():", modality=MODALITY_TEXT)
        text_code = perception.forward("def f():", modality=MODALITY_CODE)
        assert not torch.allclose(text_text, text_code)

    def test_gradient_flow(self, perception: Perception) -> None:
        """Loss on observation tokens flows gradients to embeddings."""
        out = perception.forward("hello", modality=MODALITY_TEXT)
        out.sum().backward()
        assert perception.token_embedding.weight.grad is not None

    def test_perception_without_modality_embedding(self) -> None:
        """Modality embedding is optional."""
        config = tiny_config(add_modality_embedding=False)
        fake = FakeTokenizer(
            vocab_size=config.vocab_size, pad_token_id=config.pad_token_id
        )
        wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
        wrapper.tokenizer = fake
        wrapper.max_seq_len = config.max_seq_len
        wrapper.vocab_size = fake.vocab_size
        perception = Perception(config=config, tokenizer=wrapper)
        assert perception.modality_embedding is None
        out = perception.forward("hi", modality=MODALITY_TEXT)
        assert out.shape == (1, 2, 32)

    def test_perception_with_real_gpt2_tokenizer(self) -> None:
        """End-to-end with the real GPT-2 tokenizer produces a valid tensor."""
        config = tiny_config()
        perception = Perception(config=config)
        out = perception.forward("Hello world", modality=MODALITY_TEXT)
        assert out.shape[2] == 32
        assert out.shape[0] == 1

    def test_pad_token_zero_grad(self, perception: Perception) -> None:
        """Pad tokens contribute zero gradient when padding_idx is set."""
        ids = torch.tensor(
            [[1, 2, perception.config.pad_token_id]], dtype=torch.long
        )
        out = perception.embed_tokens(ids)
        out.sum().backward()
        grad = perception.token_embedding.weight.grad
        assert grad is not None
        assert grad[perception.config.pad_token_id].abs().sum() == 0

    def test_perception_does_not_store_state(
        self, perception: Perception
    ) -> None:
        """Perception holds no PCS-like state."""
        state_dict = perception.state_dict()
        assert "token_embedding.weight" in state_dict
        # No bank buffers.
        for key in state_dict:
            assert "meta_" not in key
