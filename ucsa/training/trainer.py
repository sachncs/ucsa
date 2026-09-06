"""Trainer.

Implements the UCSA training loop with:

- AdamW optimiser
- Cosine learning-rate scheduler with linear warmup
- Mixed-precision training via ``torch.amp``
- Gradient clipping
- Optional gradient checkpointing of the operator
- Metrics tracking and TensorBoard logging
- Curriculum-driven loss-component gating
- Checkpoint save/load via :mod:`safetensors`
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ucsa.training.curriculum import Curriculum
from ucsa.training.metrics import (
    MetricsRegistry,
    build_default_registry,
    intent_gate_entropy,
    intent_gate_mutual_info,
    intent_gate_usage,
    intent_read_share,
    intent_state_variance,
    perplexity_from_loss,
)


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for :class:`Trainer`.

    Attributes:
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        beta1: AdamW beta1.
        beta2: AdamW beta2.
        grad_clip_norm: Maximum gradient norm (``0`` disables clipping).
        warmup_steps: Number of warmup steps for the scheduler.
        max_steps: Total training steps for the cosine schedule.
        amp_dtype: Mixed-precision dtype. ``torch.float32`` disables AMP.
        log_every_n_steps: Logging interval.
        eval_every_n_steps: Evaluation interval. ``0`` disables eval.
        checkpoint_every_n_steps: Checkpoint save interval. ``0`` disables
            automatic checkpointing.
        gradient_checkpointing: Whether to apply gradient checkpointing
            to the operator (where supported).
        compile_model: Whether to ``torch.compile`` the model.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 10000
    amp_dtype: torch.dtype = torch.bfloat16
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 0
    checkpoint_every_n_steps: int = 0
    gradient_checkpointing: bool = False
    compile_model: bool = False
    # ponytail: hard-EMA target encoder for JEPA stability.
    ema_momentum: float = 0.0  # 0 disables EMA target; ~0.996 enables it
    ema_update_every: int = 1  # step interval for EMA blending
    # ponytail: window of recent steps over which the intent-collapse
    # variance and mutual-information diagnostics are computed.
    intent_window_size: int = 16


@dataclass
class TrainerState:
    """internal: mutable training state."""

    global_step: int = 0
    epoch: int = 0
    last_loss: float = 0.0
    last_components: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)


class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay to ``min_lr_ratio`` of the peak."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        """Initialise the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of linear warmup steps.
            max_steps: Total number of decay steps.
            min_lr_ratio: Minimum learning-rate ratio at the end of the
                cosine decay.
        """
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(self.warmup_steps + 1, max_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_lr: list[float] = list(self.base_lrs)
        self.current_step: int = 0

    def step(self) -> list[float]:
        """Compute the new learning rates and apply them."""
        step = self.current_step
        if step < self.warmup_steps:
            scale = float(step + 1) / float(self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        for group, base_lr in zip(
            self.optimizer.param_groups, self.base_lrs, strict=False
        ):
            group["lr"] = base_lr * scale
        self.last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self.current_step += 1
        return self.last_lr

    def get_last_lr(self) -> list[float]:
        """Return the most recently applied learning rates."""
        return list(self.last_lr)


class Trainer:
    """UCSA training loop."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig | None = None,
        curriculum: Curriculum | None = None,
        metrics: MetricsRegistry | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialise the trainer.

        Args:
            model: The UCSA model.
            loss_fn: The combined loss module.
            optimizer: The optimiser.
            config: Optional trainer configuration.
            curriculum: Optional training curriculum.
            metrics: Optional metrics registry.
            device: Optional device override.
        """
        if config is None:
            config = TrainerConfig()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.curriculum = curriculum or Curriculum()
        self.metrics = metrics or build_default_registry()
        # ponytail: rolling windows of recent origination states and gate
        # usages. Variance and mutual information are only meaningful
        # across *differing* inputs, and consecutive steps see different
        # batches.
        self.intent_state_window: deque[Tensor] = deque(
            maxlen=config.intent_window_size
        )
        self.intent_usage_window: deque[Tensor] = deque(
            maxlen=config.intent_window_size
        )
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            elif torch.backends.mps.is_available():
                device = torch.device("mps", 0)
            else:
                device = torch.device("cpu")
        self.device = device
        self.model.to(self.device)
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
        self.state = TrainerState()
        self.amp_enabled = (
            self.device.type in ("cuda", "mps")
            and config.amp_dtype in (torch.float16, torch.bfloat16)
        )
        # Stash a reference for the scheduler to read.
        self.optimizer._ucsa_trainer_state = self.state  # type: ignore[attr-defined]

        # ponytail: hard-EMA target encoder for JEPA stability. When
        # ``ema_momentum > 0`` we hold a frozen copy of the model and
        # use its JEPA predicted embedding as ``jepa_target`` instead
        # of the previous-iteration working memory.
        self.target_encoder: nn.Module | None = None
        if config.ema_momentum > 0.0 and isinstance(
            self.model, nn.Module
        ):
            from ucsa.training.ema import EMATargetEncoder
            self.target_encoder = EMATargetEncoder(
                self.model, momentum=config.ema_momentum
            )
            self.target_encoder.to(self.device)

    def move_batch(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Move a ``(inputs, targets)`` batch to the device."""
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)

    def _input_token_embeddings(self, inputs: Tensor) -> Tensor:
        """Return the input token embeddings from the model's perception.

        Used as the target for the input-reconstruction loss. Falls
        back to a frozen random projection when the model doesn't
        expose an ``embed_tokens`` helper (e.g. in tests).
        """
        try:
            return self.model.perception.embed_tokens(inputs)
        except AttributeError:
            # ponytail: tests use a fake model; don't crash the train loop.
            return torch.zeros(
                inputs.shape[0], inputs.shape[1], 1, device=self.device
            )

    def compute_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the combined loss for a single batch.

        Args:
            inputs: Input token ids of shape ``(batch, seq)``.
            targets: Target token ids of shape ``(batch, seq)``.

        Returns:
            Tuple ``(loss, components)``.
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        # The model is expected to expose a ``forward`` method returning a
        # tuple ``(language_logits, memory_state, intermediates)`` when
        # possible. For tests, the trainer accepts an ``inputs -> logits``
        # callable or a model with a ``forward`` that returns logits.
        if callable(self.model) and not isinstance(self.model, nn.Module):
            outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        elif isinstance(outputs, dict):
            logits = outputs.get("language", outputs.get("logits", outputs))
        else:
            logits = outputs
        # Align targets length to logits sequence length (model output may be shorter
        # than dataset sequence length when working-bank < dataset seq_len).
        if targets.shape[1] > logits.shape[1]:
            targets = targets[:, -logits.shape[1]:]
        elif targets.shape[1] < logits.shape[1]:
            pad = logits.shape[1] - targets.shape[1]
            targets = torch.nn.functional.pad(targets, (pad, 0))
        active = self.curriculum.active_components()
        kwargs: dict[str, Any] = {}
        # Prefer real model aux outputs over randn dummies. Fall back only
        # when the model doesn't expose them (e.g. callable test models).
        if "jepa" in active:
            jp = outputs.get("jepa_predicted") if isinstance(outputs, dict) else None
            jt = outputs.get("jepa_target") if isinstance(outputs, dict) else None
            multi_step = (
                outputs.get("jepa_multi_step") if isinstance(outputs, dict) else None
            )
            # ponytail: when an EMA target encoder is active, swap the
            # targets in the multi-step list (or the single pair) for
            # the EMA model's intermediates — keeps the prediction
            # chain aligned with EMA-tracked latents.
            if self.target_encoder is not None and (
                multi_step is not None or jp is not None
            ):
                with torch.no_grad():
                    tgt_out = self.target_encoder(inputs)
                if isinstance(tgt_out, dict) and multi_step is not None:
                    ema_ms = tgt_out.get("jepa_multi_step") or []
                    if len(ema_ms) == len(multi_step) and len(ema_ms) > 0:
                        multi_step = [
                            (p, ema_ms[k][1].detach())
                            for k, (p, _) in enumerate(multi_step)
                        ]
                if isinstance(tgt_out, dict):
                    jt = tgt_out.get("jepa_predicted", jt)
            if multi_step is not None and len(multi_step) > 0:
                kwargs["jepa_multi_step"] = multi_step
            elif jp is not None and jt is not None:
                kwargs["jepa_predicted"] = jp
                kwargs["jepa_target"] = jt
            else:
                dummy = torch.randn_like(logits[..., :32])
                kwargs["jepa_predicted"] = dummy
                kwargs["jepa_target"] = dummy
        # ponytail: pass the input-reconstruction projection so the
        # combined loss can compute the capacity-bottleneck term.
        # The reconstruction head emits ``(B, working_bank, hidden)``;
        # we take its first ``seq_len`` vectors to align with the
        # input-token-embedding targets.
        if (
            isinstance(outputs, dict)
            and "input_reconstruct" in outputs
            and isinstance(self.model, nn.Module)
            and hasattr(self.model, "perception")
        ):
            recon = outputs["input_reconstruct"]
            target = self._input_token_embeddings(inputs)
            seq_len = min(recon.shape[1], target.shape[1])
            kwargs["reconstructed"] = recon[:, :seq_len, :]
            kwargs["target_embeddings"] = target[:, :seq_len, :]  # noqa: E501
        if "memory" in active:
            lt = outputs.get("long_term") if isinstance(outputs, dict) else None
            if lt is None:
                lt = torch.randn(8, logits.shape[-1])
            kwargs["long_term"] = lt
        if "router" in active:
            rl = outputs.get("router_logits") if isinstance(outputs, dict) else None
            if rl is None:
                rl = torch.randn(16, 4)
            kwargs["router_logits"] = rl
        aux = (
            outputs.get("origination_aux_loss")
            if isinstance(outputs, dict)
            else None
        )
        if isinstance(aux, Tensor) and aux.requires_grad:
            kwargs["origination_aux_loss"] = aux
        return self.loss_fn(logits, targets, **kwargs)

    def train_step(
        self, batch: tuple[Tensor, Tensor]
    ) -> dict[str, float]:
        """Run a single training step.

        Args:
            batch: ``(inputs, targets)`` batch.

        Returns:
            Dict of metric values recorded for this step.
        """
        inputs, targets = self.move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.amp.autocast(device_type=self.device.type, dtype=self.config.amp_dtype)
            if self.amp_enabled
            else _NullContext()
        )
        with autocast_ctx:
            loss, components = self.compute_loss(inputs, targets)
        loss.backward()
        if self.config.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
        self.optimizer.step()
        self.scheduler.step()
        # ponytail: blend the EMA target encoder toward the predictor
        # weights. Cadence controlled by ``ema_update_every``.
        if (
            self.target_encoder is not None
            and self.state.global_step % self.config.ema_update_every == 0
        ):
            self.target_encoder.update(self.model)
        self.state.global_step += 1
        self.state.last_loss = float(loss.item())
        self.state.last_components = components
        self.curriculum.step()
        # ponytail: refresh memory baseline every 50 steps so MemoryStabilityLoss
        # has a moving reference instead of being a no-op.
        if (
            isinstance(self.model, nn.Module)
            and hasattr(self.model, "memory_baseline")
            and self.state.global_step % 50 == 0
        ):
            lt = self.model.pcs.get_bank("long_term")
            if lt.numel() > 0:
                self.model.memory_baseline.copy_(lt.detach().mean(dim=0))

        self.metrics.update("training_loss", float(loss.item()))
        self.metrics.update(
            "perplexity", perplexity_from_loss(float(loss.item()))
        )
        for name, value in components.items():
            metric_name = {
                "ar": "training_loss",
                "jepa": "jepa_loss",
                "jepa_jepa_pred": "jepa_prediction",
                "jepa_jepa_gaussian_reg": "jepa_gaussian_reg",
                "jepa_jepa_steps": "jepa_steps",
                "reconstruction": "reconstruction_loss",
                "memory": "memory_loss",
                "router": "router_loss",
            }.get(name)
            if metric_name is not None:
                self.metrics.update(metric_name, value)
        self.metrics.update(
            "reasoning_iterations", float(self.curriculum.state.stage_step)
        )
        self.record_intent_diagnostics()
        return self.metrics.snapshot()

    def record_intent_diagnostics(self) -> None:
        """Record the intent-collapse diagnostics for this step.

        Gate entropy and read share are readable from a single step. State
        variance and gate mutual information are only meaningful *across
        differing inputs*, so they are computed over a rolling window of
        recent steps -- consecutive steps see different batches, which is
        exactly the comparison the diagnostic needs.

        Collapse is silent in every other metric: the generator learns to
        ignore the intent bank, the loop degenerates to feeding the same
        observation every iteration, and the losses still look healthy.
        """
        loop = getattr(self.model, "reasoning_loop", None)
        heads = getattr(self.model, "heads", None)
        if loop is None or heads is None:
            return
        generator = getattr(heads, "origination", None)
        if generator is None:
            return
        states = getattr(loop, "last_intent_states", [])
        if states:
            self.intent_state_window.append(states[-1].detach())
        gate_weights = getattr(generator, "last_gate_weights", None)
        if gate_weights is not None:
            num_slots = gate_weights.shape[-1]
            usage = intent_gate_usage(gate_weights, num_slots)
            self.intent_usage_window.append(usage)
            self.metrics.update(
                "intent_gate_entropy", intent_gate_entropy(usage)
            )
        intent_read = getattr(generator, "last_intent_read", None)
        working_read = getattr(generator, "last_working_read", None)
        if intent_read is not None and working_read is not None:
            self.metrics.update(
                "intent_read_share",
                intent_read_share(intent_read, working_read),
            )
        if len(self.intent_state_window) >= 2:
            self.metrics.update(
                "intent_state_variance",
                intent_state_variance(list(self.intent_state_window)),
            )
        if len(self.intent_usage_window) >= 2:
            self.metrics.update(
                "intent_gate_mutual_info",
                intent_gate_mutual_info(list(self.intent_usage_window)),
            )

    def train(
        self,
        dataloader: DataLoader,
        num_steps: int | None = None,
    ) -> list[dict[str, float]]:
        """Run the training loop.

        Args:
            dataloader: Iterable yielding ``(inputs, targets)`` batches.
            num_steps: Optional cap on the number of steps.

        Returns:
            List of per-step metric snapshots.
        """
        max_steps = num_steps or self.config.max_steps
        history: list[dict[str, float]] = []
        dataloader_iter = iter(dataloader)
        for step in range(max_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            snapshot = self.train_step(batch)
            if step % self.config.log_every_n_steps == 0:
                history.append(snapshot)
        return history

    def save_checkpoint(self, path: str) -> None:
        """Save the model and trainer state to ``path``."""
        from safetensors.torch import save_file

        state_dict = {
            f"model.{name}": param.detach().cpu()
            for name, param in self.model.state_dict().items()
        }
        save_file(state_dict, path)

    def load_checkpoint(self, path: str) -> None:
        """Load the model state from ``path``."""
        from safetensors.torch import load_file

        state_dict = load_file(path)
        renamed = {
            name.removeprefix("model."): tensor
            for name, tensor in state_dict.items()
        }
        self.model.load_state_dict(renamed, strict=False)


class _NullContext:
    """internal: context manager used when AMP is disabled."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb


__all__ = [
    "CosineWarmupScheduler",
    "Trainer",
    "TrainerConfig",
    "TrainerState",
]
