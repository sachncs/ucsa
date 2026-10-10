"""Pre-flight checks that run once, before training, never during it.

Real training carries no instrumentation. Everything worth measuring is
measured here, on a few real optimisation steps of the actual model and
batch shape:

* dtype purity: every floating tensor any operator produces, in the forward,
  backward and optimiser step, must be `precision.DTYPE`;
* memory: peak accelerator memory of a step;
* time: forward, backward and optimiser seconds, and tokens per second;
* health: finite loss and gradient norm.

The audit pass wraps the step in a dispatch hook that sees every operator,
which is slow and allocates, so it is run on its own; the timing passes run
without any hook and report what training will actually cost.
"""

import dataclasses
import time
from typing import Any

import torch
from torch import nn
from torch.utils import _python_dispatch

from ucsa.models import recurrent
from ucsa.training import engine
from ucsa.utils import precision


def off_dtype_parameters(module: nn.Module) -> list[str]:
    """Lists parameters, buffers and gradients not stored in `DTYPE`.

    Args:
      module: Module to inspect.

    Returns:
      Names of the offending tensors; empty when the module is uniform.
    """
    bad = []
    for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
        if tensor.is_floating_point() and tensor.dtype != precision.DTYPE:
            bad.append(f"{name}: {tensor.dtype}")
        grad = getattr(tensor, "grad", None)
        if grad is not None and grad.dtype != precision.DTYPE:
            bad.append(f"{name}.grad: {grad.dtype}")
    return bad


class Recorder(_python_dispatch.TorchDispatchMode):
    """Records the dtype of every floating-point tensor an operator makes.

    Used as a context manager around a step. `seen` then holds every distinct
    floating dtype any operator produced, which must be `{DTYPE}` or empty.
    """

    def __init__(self, ignore_scalars: bool = False) -> None:
        """Starts with nothing recorded.

        Args:
          ignore_scalars: Skip 0-dimensional results. PyTorch's AdamW keeps
            its step counter as a float32 scalar whatever the parameter
            dtype; the counter holds an integer value, so it cannot carry
            rounding error, but it is a second dtype. Strict mode (the
            default) includes scalars.
        """
        super().__init__()
        self.ignore_scalars = ignore_scalars
        self.seen: set[torch.dtype] = set()
        self.offenders: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        """Runs the operator and records the dtypes of its outputs."""
        out = func(*args, **(kwargs or {}))
        flat = out if isinstance(out, tuple | list) else (out,)
        for item in flat:
            if not isinstance(item, torch.Tensor):
                continue
            if not item.is_floating_point():
                continue
            if self.ignore_scalars and item.dim() == 0:
                continue
            self.seen.add(item.dtype)
            if item.dtype != precision.DTYPE:
                self.offenders.append(
                    f"{func}: {item.dtype} {tuple(item.shape)}"
                )
        return out


@dataclasses.dataclass
class Report:
    """What a dry run found.

    Attributes:
      dtype: The library dtype under test.
      dtypes_seen: Every floating dtype an operator produced.
      offenders: Operators that produced another dtype (should be empty).
      off_dtype_tensors: Parameters, buffers or grads in another dtype.
      parameters: Trainable parameter count.
      peak_memory_mb: Accelerator memory in use after the timed steps.
      forward_seconds: Median seconds of the forward pass.
      backward_seconds: Median seconds of the backward pass.
      optimizer_seconds: Median seconds of clip plus optimiser step.
      step_seconds: Median seconds of a whole step.
      tokens_per_second: Throughput implied by `step_seconds`.
      loss_finite: Whether the loss was finite on every step.
      grad_norm_finite: Whether the gradient norm was finite on every step.
    """

    dtype: str
    dtypes_seen: list[str]
    offenders: list[str]
    off_dtype_tensors: list[str]
    parameters: int
    peak_memory_mb: float
    forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    step_seconds: float
    tokens_per_second: float
    loss_finite: bool
    grad_norm_finite: bool

    @property
    def clean(self) -> bool:
        """True when the model is uniform in dtype and numerically healthy."""
        return (
            not self.offenders
            and not self.off_dtype_tensors
            and self.loss_finite
            and self.grad_norm_finite
        )


def synchronize(device: torch.device) -> None:
    """Blocks until queued device work finishes."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def memory_mb(device: torch.device) -> float:
    """Returns accelerator memory in use, in MiB (0 on the CPU)."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    return 0.0


def median(values: list[float]) -> float:
    """Returns the median of a non-empty list."""
    return sorted(values)[len(values) // 2]


def run(
    model: recurrent.Model,
    config: engine.Config,
    batch: tuple[torch.Tensor, torch.Tensor],
    steps: int = 5,
    ignore_scalars: bool = False,
) -> Report:
    """Dry-runs `steps` real optimisation steps and reports.

    The model's storage is checked first: if any parameter or buffer is not in
    the library dtype, the report says so and no step is run. Otherwise one
    step runs under the dtype `Recorder`, then the timed steps run with no
    hook at all.

    Args:
      model: Model to test; it is moved to the best device and left trained
        for a few steps, so build a throwaway instance.
      config: Training configuration; its batch size and sequence length set
        the shape that is measured.
      batch: A real `(inputs, targets)` batch from the training shard, of
        shape `(config.batch_size, config.seq_len)`; every step reuses it.
      steps: Timed steps (one warm-up step is added).
      ignore_scalars: See `Recorder`.

    Returns:
      The `Report`.
    """
    precision.configure()
    device = engine.pick_device()
    stray = off_dtype_parameters(model)
    if stray:
        # Fail fast: running mixed-dtype kernels can abort the GPU driver, so
        # an inconsistent model is reported from its storage alone.
        return Report(
            dtype=str(precision.DTYPE),
            dtypes_seen=[],
            offenders=[],
            off_dtype_tensors=stray,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_memory_mb=0.0,
            forward_seconds=0.0,
            backward_seconds=0.0,
            optimizer_seconds=0.0,
            step_seconds=0.0,
            tokens_per_second=0.0,
            loss_finite=False,
            grad_norm_finite=False,
        )
    model.to(device).train()
    optimizer = engine.build_optimizer(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    x, y = (t.to(device) for t in batch)

    def step() -> tuple[float, float, float, bool, bool]:
        synchronize(device)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(x, y)
        synchronize(device)
        t1 = time.time()
        loss.backward()
        synchronize(device)
        t2 = time.time()
        norm = engine.clip_grad_norm(params, config.grad_clip)
        optimizer.step()
        model.update_ema()
        synchronize(device)
        t3 = time.time()
        return (
            t1 - t0,
            t2 - t1,
            t3 - t2,
            bool(torch.isfinite(loss)),
            bool(torch.isfinite(norm)),
        )

    with Recorder(ignore_scalars) as recorder:
        step()  # The audited step: slow, and kept out of the timings.
    off = off_dtype_parameters(model)

    step()  # Warm-up: kernel compilation and allocator growth.
    rows = [step() for _ in range(steps)]
    forward, backward, update = (median([r[i] for r in rows]) for i in range(3))
    total = forward + backward + update
    return Report(
        dtype=str(precision.DTYPE),
        dtypes_seen=sorted(str(d) for d in recorder.seen),
        offenders=sorted(set(recorder.offenders)),
        off_dtype_tensors=off,
        parameters=sum(p.numel() for p in params),
        peak_memory_mb=memory_mb(device),
        forward_seconds=forward,
        backward_seconds=backward,
        optimizer_seconds=update,
        step_seconds=total,
        tokens_per_second=config.batch_size * config.seq_len / total,
        loss_finite=all(r[3] for r in rows),
        grad_norm_finite=all(r[4] for r in rows),
    )
