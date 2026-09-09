"""Perception layer.

Perception converts raw input (text, code) into observation tokens suitable
for injection into the Persistent Cognitive State. It is *stateless* in the
PCS sense: it never stores memory, never performs reasoning. Its outputs are
projections of raw input only.

The pipeline is::

    raw_input -> Tokenizer -> Embedding -> ModalityProjection -> ObservationTokens

The output has shape ``(batch, seq, hidden_size)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODALITY_TEXT: int = 0
MODALITY_CODE: int = 1


@dataclass(frozen=True)
class PerceptionConfig:
    """Configuration for :class:`Perception`.

    Attributes:
        hidden_size: Hidden dimensionality of observation tokens.
        vocab_size: Vocabulary size of the tokenizer.
        max_seq_len: Maximum sequence length of any single observation.
        tokenizer_name: Hugging Face tokenizer name or path. Defaults to
            the GPT-2 tokenizer per project specification.
        modalities: Tuple of supported modality identifiers. Tokenised text
            is tagged with the first modality by default.
        pad_token_id: Padding token id. Defaults to ``0``.
        add_modality_embedding: If ``True`` a learned per-modality embedding
            is added to each token.
    """

    hidden_size: int = 128
    vocab_size: int = 50257
    max_seq_len: int = 1024
    tokenizer_name: str = "gpt2"
    modalities: tuple[int, ...] = (MODALITY_TEXT, MODALITY_CODE)
    pad_token_id: int = 0
    add_modality_embedding: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {self.hidden_size}."
            )
        if self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {self.vocab_size}."
            )
        if self.max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be positive, got {self.max_seq_len}."
            )
        if not self.modalities:
            raise ValueError("modalities must contain at least one entry.")


class TokenizerWrapper:
    """A thin, replaceable wrapper around a Hugging Face tokenizer.

    The wrapper is intentionally minimal: it exposes the tokenizer plus a
    couple of convenience methods (:meth:`encode`, :meth:`batch_encode`) that
    handle truncation and padding consistently. If the chosen tokenizer has
    no pad token, the EOS token is used as the pad token.
    """

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        max_seq_len: int = 1024,
        pad_token_id: int | None = None,
    ) -> None:
        """Load and configure the underlying tokenizer.

        Args:
            tokenizer_name: Hugging Face tokenizer name or local path.
            max_seq_len: Maximum length of any single encoded sequence.
            pad_token_id: Override for the pad token id. Defaults to EOS
                when the tokenizer lacks a pad token.
        """
        self.tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                tokenizer_name
            )
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    f"Tokenizer '{tokenizer_name}' has neither pad nor EOS "
                    f"token; cannot default pad token id."
                )
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is not None:
            self.tokenizer.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.vocab_size = int(self.tokenizer.vocab_size)

    @property
    def pad_id(self) -> int:
        """Return the pad token id."""
        return int(self.tokenizer.pad_token_id)

    def encode(self, text: str, modality: int = MODALITY_TEXT) -> Tensor:
        """Encode a single string into token ids.

        Args:
            text: Raw input string.
            modality: Modality tag, kept for future use.

        Returns:
            Long tensor of shape ``(seq,)``.
        """
        del modality  # internal: modality is reserved for future token tagging
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(
        self,
        texts: Sequence[str],
        modality: int = MODALITY_TEXT,
    ) -> Tensor:
        """Encode a batch of strings.

        Args:
            texts: Raw input strings.
            modality: Modality tag for the entire batch.

        Returns:
            Long tensor of shape ``(batch, seq)``.
        """
        del modality
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids: Tensor = encoded["input_ids"]
        return input_ids


class Perception(nn.Module):
    """Stateless perception layer.

    Converts token ids into observation tokens of shape
    ``(batch, seq, hidden_size)``. The module contains a token embedding, an
    optional per-modality embedding, and a small modality projection that
    keeps the embedding dimensionality aligned with the operator's
    ``hidden_size``.
    """

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        tokenizer: TokenizerWrapper | None = None,
    ) -> None:
        """Initialise perception.

        Args:
            config: Optional perception configuration.
            tokenizer: Optional pre-built tokenizer. If ``None``, one is
                loaded using ``config.tokenizer_name``.
        """
        super().__init__()
        if config is None:
            config = PerceptionConfig()
        self.config = config
        if tokenizer is None:
            tokenizer = TokenizerWrapper(
                tokenizer_name=config.tokenizer_name,
                max_seq_len=config.max_seq_len,
                pad_token_id=config.pad_token_id,
            )
        self.tokenizer = tokenizer
        embedding_vocab_size = max(config.vocab_size, tokenizer.vocab_size)
        self.token_embedding = nn.Embedding(
            embedding_vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        if config.add_modality_embedding:
            self.modality_embedding: nn.Embedding | None = nn.Embedding(
                len(config.modalities), config.hidden_size
            )
        else:
            self.modality_embedding = None
        self.modality_projection = nn.Linear(
            config.hidden_size, config.hidden_size
        )
        self.modality_to_index: dict[int, int] = {
            modality: index for index, modality in enumerate(config.modalities)
        }

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        """Embed raw token ids into ``hidden_size``-dimensional vectors.

        Args:
            token_ids: Long tensor of shape ``(batch, seq)``.

        Returns:
            Tensor of shape ``(batch, seq, hidden_size)``.
        """
        embedded: Tensor = self.token_embedding(token_ids)
        return embedded

    def project(
        self,
        embeddings: Tensor,
        modality: int = MODALITY_TEXT,
    ) -> Tensor:
        """Apply modality projection to embedded tokens.

        Args:
            embeddings: Tensor of shape ``(batch, seq, hidden_size)``.
            modality: Modality tag.

        Returns:
            Tensor of the same shape as ``embeddings``.
        """
        projected = self.modality_projection(embeddings)
        if self.modality_embedding is not None:
            if modality not in self.modality_to_index:
                raise ValueError(
                    f"Unknown modality '{modality}'. Supported: "
                    f"{sorted(self.modality_to_index)}."
                )
            index = self.modality_to_index[modality]
            modality_ids = torch.full(
                (embeddings.shape[0], embeddings.shape[1]),
                index,
                device=embeddings.device,
                dtype=torch.long,
            )
            projected = projected + self.modality_embedding(modality_ids)
        result: Tensor = projected
        return result

    def forward(
        self,
        texts: Sequence[str] | str,
        modality: int = MODALITY_TEXT,
    ) -> Tensor:
        """Convert raw input into observation tokens.

        Args:
            texts: A single string or a sequence of strings.
            modality: Modality tag for the input.

        Returns:
            Observation tokens of shape ``(batch, seq, hidden_size)``.
        """
        batch = [texts] if isinstance(texts, str) else list(texts)
        token_ids = self.tokenizer.batch_encode(batch, modality=modality)
        token_ids = token_ids.to(self.token_embedding.weight.device)
        embeddings = self.embed_tokens(token_ids)
        return self.project(embeddings, modality=modality)

    def forward_from_ids(
        self,
        token_ids: Tensor,
        modality: int = MODALITY_TEXT,
    ) -> Tensor:
        """Convert pre-tokenised ids into observation tokens.

        Args:
            token_ids: Long tensor of shape ``(batch, seq)``.
            modality: Modality tag.

        Returns:
            Observation tokens of shape ``(batch, seq, hidden_size)``.
        """
        embeddings = self.embed_tokens(token_ids)
        return self.project(embeddings, modality=modality)


__all__ = [
    "MODALITY_CODE",
    "MODALITY_TEXT",
    "Perception",
    "PerceptionConfig",
    "TokenizerWrapper",
]
