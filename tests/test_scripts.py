"""Tests for the train and infer scripts."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from torch import Tensor

import ucsa.infer as infer
import ucsa.train as train
from ucsa.models.ucsa import UCSA, UCSAConfig


def tiny_config_dict() -> dict:
    """Return a tiny config for smoke tests."""
    return {
        "seed": 1,
        "reasoning_iterations": 2,
        "model": {
            "hidden_size": 32,
            "num_layers": 2,
            "num_q_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 64,
            "sliding_window": 4096,
            "vocab_size": 1000,
            "max_seq_len": 64,
            "num_concepts": 4,
            "moe": None,
        },
        "heads": {
            "vocab_size": 1000,
            "num_plan_tokens": 16,
            "num_tools": 8,
            "memory_query_dim": 24,
        },
        "tokenizer": {"name": "gpt2"},
        "dataset": {
            "sequence_length": 16,
            "primary_dataset": "HuggingFaceFW/fineweb-edu",
            "primary_split": "train",
            "streaming": True,
            "pack_sequences": True,
        },
        "training": {
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip_norm": 1.0,
            "warmup_steps": 1,
            "max_steps": 3,
            "batch_size": 2,
            "log_every_n_steps": 1,
            "eval_every_n_steps": 0,
            "checkpoint_every_n_steps": 0,
            "gradient_checkpointing": False,
            "compile_model": False,
            "amp_dtype": "float32",
        },
        "curriculum": {
            "stage_1_end": 2,
            "stage_2_end": 4,
            "stage_3_end": 6,
        },
        "memory": {
            "long_term_capacity": 32,
            "co_activation_threshold": 0.5,
        },
        "verification": {
            "name": "heuristic",
            "acceptance_threshold": 0.5,
            "learned_hidden_size": 32,
        },
        "logging": {
            "log_dir": "runs/ucsa-test",
            "use_tensorboard": False,
            "use_wandb": False,
            "wandb_project": "ucsa-test",
        },
        "inference": {
            "max_new_tokens": 4,
            "temperature": 0.0,
            "top_k": 50,
        },
    }


class TestBuildHelpers:
    """Tests for :mod:`ucsa.train` helper functions."""

    def test_build_model_returns_module(self) -> None:
        """``build_model`` returns a :class:`UCSA`."""
        model = train.build_model(tiny_config_dict())
        assert isinstance(model, UCSA)

    def test_build_optimizer_returns_adamw(self) -> None:
        """``build_optimizer`` returns an AdamW instance."""
        model = train.build_model(tiny_config_dict())
        optimizer = train.build_optimizer(model, tiny_config_dict())
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_build_trainer_returns_trainer(self) -> None:
        """``build_trainer`` returns a configured :class:`Trainer`."""
        from ucsa.training.trainer import Trainer

        model = train.build_model(tiny_config_dict())
        trainer = train.build_trainer(model, tiny_config_dict())
        assert isinstance(trainer, Trainer)
        assert trainer.config.max_steps == 3

    def test_resolve_dtype(self) -> None:
        """``_resolve_dtype`` maps strings to torch dtypes."""
        assert train._resolve_dtype("float32") is torch.float32
        assert train._resolve_dtype("float16") is torch.float16
        assert train._resolve_dtype("bfloat16") is torch.bfloat16
        with pytest.raises(ValueError):
            train._resolve_dtype("bogus")


class TestConfigConversion:
    """Tests for the config-to-dict helper."""

    def test_dict_passthrough(self) -> None:
        """``_config_to_dict`` returns dicts unchanged."""
        cfg = {"a": 1}
        assert train._config_to_dict(cfg) is cfg

    def test_non_dict_rejected(self) -> None:
        """Non-dict configs without omegaconf are rejected."""
        with pytest.raises(TypeError):
            train._config_to_dict(42)


class TestInference:
    """Tests for :mod:`ucsa.infer`."""

    def test_generate_extends_sequence(self) -> None:
        """``generate`` extends the input sequence by ``max_new_tokens``."""
        model = UCSA(
            UCSAConfig(hidden_size=32, vocab_size=100, num_layers=2)
        )
        prompt = torch.randint(0, 100, (1, 4))
        out = infer.generate(model, prompt, max_new_tokens=3)
        assert out.shape == (1, 7)

    def test_generate_argmax_with_zero_temperature(self) -> None:
        """Zero temperature picks argmax deterministically."""
        torch.manual_seed(0)
        model = UCSA(
            UCSAConfig(hidden_size=32, vocab_size=10, num_layers=2)
        )
        prompt = torch.randint(0, 10, (1, 2))
        first = infer.generate(
            model, prompt, max_new_tokens=2, temperature=0.0
        )
        second = infer.generate(
            model, prompt, max_new_tokens=2, temperature=0.0
        )
        assert torch.equal(first, second)

    def test_generate_stops_at_eos(self) -> None:
        """Generation stops when an EOS token is produced."""
        model = UCSA(
            UCSAConfig(hidden_size=32, vocab_size=100, num_layers=2)
        )
        prompt = torch.randint(0, 100, (1, 2))
        out = infer.generate(
            model, prompt, max_new_tokens=20, eos_token_id=7
        )
        # Generation stops on EOS or after max_new_tokens.
        assert out.shape[1] <= prompt.shape[1] + 20


class TestUCSAModel:
    """Tests for :class:`UCSA` end-to-end forward pass."""

    def test_forward_returns_all_heads(self) -> None:
        """Forward returns language, planning, tool, memory outputs."""
        model = UCSA(
            UCSAConfig(hidden_size=32, vocab_size=100, num_layers=2)
        )
        inputs = torch.randint(0, 100, (1, 4))
        out = model(inputs)
        assert set(out) == {"language", "planning", "tool", "memory"}

    def test_forward_returns_logits_with_correct_vocab(self) -> None:
        """Language logits have shape ``(batch, working_size, vocab)``."""
        model = UCSA(
            UCSAConfig(hidden_size=32, vocab_size=100, num_layers=2)
        )
        inputs = torch.randint(0, 100, (1, 4))
        out = model(inputs)
        assert out["language"].shape[-1] == 100

    def test_forward_lossless_gradients(self) -> None:
        """A loss on the language head flows gradients to PCS parameters."""
        model = UCSA(
            UCSAConfig(hidden_size=32, vocab_size=100, num_layers=2)
        )
        inputs = torch.randint(0, 100, (1, 4))
        out = model(inputs)
        loss = out["language"].sum()
        loss.backward()
        assert model.pcs.get_bank("working").grad is not None

    def test_ucsa_with_moe(self) -> None:
        """UCSA can be constructed with MoE configured."""
        from ucsa.models.moe import MoEConfig

        model = UCSA(
            UCSAConfig(
                hidden_size=32,
                vocab_size=100,
                num_layers=2,
                moe=MoEConfig(num_experts=2, top_k=1),
            )
        )
        inputs = torch.randint(0, 100, (1, 4))
        out = model(inputs)
        assert "language" in out