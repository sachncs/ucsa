"""Persistent Cognitive State (PCS).

The PCS is the single persistent differentiable representation in UCSA.
It consists of seven token banks, each a learnable tensor of shape
``(num_tokens, hidden_size)``. Every operator in the system reads from and
writes to the PCS; no other structure stores knowledge.

Banks
-----

Every PCS has the following banks by default. Sizes and roles are configurable
via :class:`PCSConfig`.

================== ============== ============================================
Bank               Default tokens Role
================== ============== ============================================
``working``        64             Scratch space mutated by every reasoning step.
``long_term``      128            Accepted knowledge, retained across requests.
``goal``           16             Holds the active objective.
``episode``        32             Per-request context buffer.
``task``           16             Long-running task state.
``memory_index``   32             Retrieval index, cross-attended each block.
``intent``         16             Origination signal for the next input.
================== ============== ============================================

``intent`` is the origination bank: the reasoning loop generates iteration
``k + 1``'s input from it, so it is the one place where the signal that
*causes* the next state lives. It is deliberately last in
:data:`BANK_NAMES` so that adding it leaves every other bank's offset and
bank id inside the operator's token stream unchanged.

Retention metadata
------------------

Every long-term memory token carries four scalar metadata fields:

- ``importance``     -- how significant the token is.
- ``usage``          -- how often it has been accessed.
- ``age``            -- how many requests have elapsed since last write.
- ``retention_score``-- composite score driving the recycle policy.

The recycle policy lives in :func:`retention_score` and
:meth:`PersistentCognitiveState.recycle_bottom_k`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

INTENT_BANK = "intent"

METADATA_FIELDS: tuple[str, ...] = (
    "importance",
    "usage",
    "age",
    "retention",
)

BANK_NAMES: tuple[str, ...] = (
    "working",
    "long_term",
    "goal",
    "episode",
    "task",
    "memory_index",
    "intent",
)


@dataclass(frozen=True)
class BankSpec:
    """Static specification for one PCS bank.

    Attributes:
        name: Bank identifier. Must be one of :data:`BANK_NAMES` for default
            banks; custom names are allowed but must be unique.
        num_tokens: Number of token slots in this bank.
        trainable: If ``True`` the bank tensor is an ``nn.Parameter``
            (default). If ``False`` the bank is a non-trainable buffer.
    """

    name: str
    num_tokens: int
    trainable: bool = True

    def __post_init__(self) -> None:
        if self.num_tokens <= 0:
            raise ValueError(
                f"Bank '{self.name}' num_tokens must be positive, "
                f"got {self.num_tokens}."
            )
        if not self.name:
            raise ValueError("Bank name must be non-empty.")


@dataclass(frozen=True)
class PCSConfig:
    """Configuration for :class:`PersistentCognitiveState`.

    Attributes:
        hidden_size: Hidden dimensionality of every token in every bank.
        bank_sizes: Mapping from bank name to number of tokens. Missing keys
            fall back to defaults defined in :data:`DEFAULT_BANK_SIZES`.
        retention_importance_weight: Weight for importance in
            :func:`retention_score`.
        retention_usage_weight: Weight for usage in :func:`retention_score`.
        retention_age_weight: Weight for age (with negative sign) in
            :func:`retention_score`.
        retention_floor: Minimum retention score; tokens below this are
            candidates for recycling regardless of other factors.
        init_std: Standard deviation of the bank initialisation.
    """

    hidden_size: int = 128
    bank_sizes: Mapping[str, int] = field(default_factory=dict)
    retention_importance_weight: float = 0.5
    retention_usage_weight: float = 0.3
    retention_age_weight: float = 0.2
    retention_floor: float = 0.01
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {self.hidden_size}."
            )
        if self.init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {self.init_std}.")


DEFAULT_BANK_SIZES: dict[str, int] = {
    "working": 64,
    "long_term": 128,
    "goal": 16,
    "episode": 32,
    "task": 16,
    "memory_index": 32,
    "intent": 16,
}


def resolve_bank_sizes(
    overrides: Mapping[str, int] | None,
) -> dict[str, int]:
    """Return the resolved per-bank token counts.

    Args:
        overrides: User-supplied size overrides keyed by bank name.

    Returns:
        A dictionary covering every default bank with possibly overridden
        sizes.
    """
    resolved: dict[str, int] = dict(DEFAULT_BANK_SIZES)
    if overrides:
        for name, size in overrides.items():
            if size <= 0:
                raise ValueError(
                    f"Bank '{name}' override size must be positive, "
                    f"got {size}."
                )
            resolved[name] = size
    return resolved


def build_bank_specs(
    config: PCSConfig,
) -> list[BankSpec]:
    """Build the :class:`BankSpec` list for the configured banks.

    Args:
        config: The PCS configuration.

    Returns:
        Ordered list of bank specifications.
    """
    sizes = resolve_bank_sizes(config.bank_sizes)
    return [BankSpec(name=name, num_tokens=sizes[name]) for name in BANK_NAMES]


def retention_score(
    importance: Tensor,
    usage: Tensor,
    age: Tensor,
    weights: PCSConfig,
) -> Tensor:
    r"""Compute retention score in ``[0, 1]``.

    The score is a weighted sum clipped to ``[0, 1]``:

    .. math::

        r = \mathrm{clip}\bigl(
            w_i \cdot \sigma(i)
            + w_u \cdot \sigma(u)
            - w_a \cdot a / (a + 1),
            0, 1\bigr)

    Args:
        importance: Importance values, arbitrary positive scale.
        usage: Usage counts, non-negative integers castable to float.
        age: Age in requests since last write, non-negative.
        weights: PCS config providing the three weights and the floor.

    Returns:
        Tensor of retention scores with the broadcasted shape of inputs.
    """
    if importance.shape != usage.shape or importance.shape != age.shape:
        raise ValueError(
            "importance, usage, and age must share shape; got "
            f"{tuple(importance.shape)}, {tuple(usage.shape)}, "
            f"{tuple(age.shape)}."
        )
    if torch.any(importance < 0):
        raise ValueError("importance must be non-negative.")
    if torch.any(usage < 0):
        raise ValueError("usage must be non-negative.")
    if torch.any(age < 0):
        raise ValueError("age must be non-negative.")

    importance_term = weights.retention_importance_weight * torch.sigmoid(importance)
    usage_term = weights.retention_usage_weight * torch.sigmoid(usage)
    age_term = weights.retention_age_weight * (age / (age + 1.0))
    raw = importance_term + usage_term - age_term
    return torch.clamp(raw, min=0.0, max=1.0)


class PersistentCognitiveState(nn.Module):
    """The single persistent differentiable cognitive state.

    The PCS holds its token banks as :class:`torch.nn.ParameterDict` entries
    plus retention metadata as registered buffers. It exposes a uniform API
    for operators to read all tokens as a single concatenated tensor and for
    the memory subsystem to mutate individual banks.

    The PCS is *the* state of the model. Loss of PCS state is loss of model
    state.
    """

    def __init__(self, config: PCSConfig | None = None) -> None:
        """Initialise the PCS.

        Args:
            config: Optional PCS configuration. Defaults to
                :class:`PCSConfig` defaults.
        """
        super().__init__()
        if config is None:
            config = PCSConfig()
        self.config = config

        bank_parameter_dict: nn.ParameterDict = nn.ParameterDict()
        bank_specs = build_bank_specs(config)
        for spec in bank_specs:
            tensor = torch.empty(spec.num_tokens, config.hidden_size)
            nn.init.normal_(tensor, mean=0.0, std=config.init_std)
            if spec.trainable:
                bank_parameter_dict[spec.name] = nn.Parameter(tensor)
            else:
                self.register_buffer(
                    f"bank_{spec.name}", tensor, persistent=True
                )

        self.banks = bank_parameter_dict
        self.bank_specs = {spec.name: spec for spec in bank_specs}
        self.bank_order: tuple[str, ...] = tuple(spec.name for spec in bank_specs)

        for spec in bank_specs:
            num_tokens = spec.num_tokens
            self.register_buffer(
                f"meta_importance_{spec.name}",
                torch.zeros(num_tokens),
                persistent=True,
            )
            self.register_buffer(
                f"meta_usage_{spec.name}",
                torch.zeros(num_tokens),
                persistent=True,
            )
            self.register_buffer(
                f"meta_age_{spec.name}",
                torch.zeros(num_tokens, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                f"meta_retention_{spec.name}",
                torch.zeros(num_tokens),
                persistent=True,
            )

    @property
    def hidden_size(self) -> int:
        """Return the hidden dimensionality of every bank."""
        return self.config.hidden_size

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens across all banks."""
        return sum(spec.num_tokens for spec in self.bank_specs.values())

    def bank_size(self, name: str) -> int:
        """Return the number of tokens in ``name``.

        Args:
            name: Bank identifier.

        Returns:
            Token count.

        Raises:
            KeyError: If the bank does not exist.
        """
        if name not in self.bank_specs:
            raise KeyError(f"Unknown bank '{name}'.")
        return self.bank_specs[name].num_tokens

    def get_bank(self, name: str) -> Tensor:
        """Return the bank tensor for ``name``.

        Args:
            name: Bank identifier.

        Returns:
            Tensor of shape ``(num_tokens, hidden_size)``.

        Raises:
            KeyError: If the bank does not exist.
        """
        if name not in self.bank_specs:
            raise KeyError(f"Unknown bank '{name}'.")
        if name in self.banks:
            parameter = self.banks[name]
            assert isinstance(parameter, Tensor)
            return parameter
        buffer = getattr(self, f"bank_{name}")
        assert isinstance(buffer, Tensor)
        return buffer

    def metadata(self, name: str, field: str) -> Tensor:
        """Return one metadata buffer for ``name``, typed as a tensor.

        Buffers are reached through ``nn.Module.__getattr__``, which is
        typed as ``Tensor | Module``. Callers need the tensor, so the cast
        happens here once instead of at every call site.

        Args:
            name: Bank identifier.
            field: One of :data:`METADATA_FIELDS`.

        Returns:
            The metadata buffer of shape ``(num_tokens,)``.

        Raises:
            KeyError: If the bank or the field does not exist.
        """
        if name not in self.bank_specs:
            raise KeyError(f"Unknown bank '{name}'.")
        if field not in METADATA_FIELDS:
            raise KeyError(
                f"Unknown metadata field '{field}'; expected one of "
                f"{METADATA_FIELDS}."
            )
        buffer = getattr(self, f"meta_{field}_{name}")
        assert isinstance(buffer, Tensor)
        return buffer

    def set_bank(self, name: str, tensor: Tensor) -> None:
        """Replace the contents of ``name`` with ``tensor`` in-place.

        The tensor is copied into the existing parameter or buffer to keep
        the optimiser state consistent.

        Args:
            name: Bank identifier.
            tensor: Replacement tensor of shape
                ``(num_tokens, hidden_size)``.

        Raises:
            KeyError: If the bank does not exist.
            ValueError: If the tensor has the wrong shape or dtype.
        """
        spec = self.bank_specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown bank '{name}'.")
        expected_shape = (spec.num_tokens, self.config.hidden_size)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Bank '{name}' expected shape {expected_shape}, got "
                f"{tuple(tensor.shape)}."
            )
        target = self.get_bank(name)
        with torch.no_grad():
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))

    def get_all_tokens(self) -> Tensor:
        """Concatenate every bank along the token dimension.

        Returns:
            Tensor of shape ``(total_tokens, hidden_size)``.
        """
        tokens = [self.get_bank(name) for name in self.bank_order]
        return torch.cat(tokens, dim=0)

    def get_all_metadata(self) -> dict[str, dict[str, Tensor]]:
        """Return a snapshot of every bank's metadata.

        Returns:
            Nested dict ``{bank_name: {field: tensor, ...}, ...}``.
        """
        snapshot: dict[str, dict[str, Tensor]] = {}
        for name in self.bank_order:
            snapshot[name] = {
                name_field: self.metadata(name, name_field).clone()
                for name_field in METADATA_FIELDS
            }
        return snapshot

    def reset_metadata(self) -> None:
        """Zero every metadata field for every bank."""
        for name in self.bank_order:
            for name_field in METADATA_FIELDS:
                self.metadata(name, name_field).zero_()

    def step_age(self) -> None:
        """Increment the age of every token in every bank by one request."""
        for name in self.bank_order:
            self.metadata(name, "age").add_(1)

    def record_usage(self, name: str, indices: Tensor, increment: float = 1.0) -> None:
        """Increment the usage counter for specific slots in ``name``.

        Args:
            name: Bank identifier.
            indices: Long tensor of slot indices.
            increment: Amount to add (default 1.0).
        """
        if name not in self.bank_specs:
            raise KeyError(f"Unknown bank '{name}'.")
        usage_buffer = self.metadata(name, "usage")
        usage_buffer[indices] = usage_buffer[indices] + increment
        self.metadata(name, "age")[indices] = 0

    def update_retention(self) -> None:
        """Recompute retention scores for every bank."""
        weights = self.config
        for name in self.bank_order:
            score = retention_score(
                self.metadata(name, "importance"),
                self.metadata(name, "usage"),
                self.metadata(name, "age").float(),
                weights,
            )
            self.metadata(name, "retention").copy_(score)

    def recycle_bottom_k(
        self,
        name: str,
        k: int,
        replacement: Tensor | None = None,
    ) -> Tensor:
        """Recycle the ``k`` lowest-retention slots in ``name``.

        Recycled slots are zeroed (or replaced with ``replacement``) and their
        metadata reset.

        Args:
            name: Bank identifier. Should usually be ``"long_term"``.
            k: Number of slots to recycle.
            replacement: Optional replacement tensor of shape
                ``(k, hidden_size)``. If ``None``, slots are zeroed.

        Returns:
            Long tensor of recycled slot indices.

        Raises:
            KeyError: If the bank does not exist.
            ValueError: If ``replacement`` has the wrong shape.
        """
        if name not in self.bank_specs:
            raise KeyError(f"Unknown bank '{name}'.")
        spec = self.bank_specs[name]
        retention = self.metadata(name, "retention")
        k_eff = min(k, spec.num_tokens)
        if k_eff <= 0:
            return torch.empty(0, dtype=torch.long)
        _, indices = torch.topk(retention, k_eff, largest=False)

        target = self.get_bank(name)
        if replacement is None:
            replacement_tensor = torch.zeros(
                k_eff, self.config.hidden_size, device=target.device, dtype=target.dtype
            )
        else:
            expected_shape = (k_eff, self.config.hidden_size)
            if tuple(replacement.shape) != expected_shape:
                raise ValueError(
                    f"Replacement shape {tuple(replacement.shape)} does not "
                    f"match expected {expected_shape}."
                )
            replacement_tensor = replacement.to(
                device=target.device, dtype=target.dtype
            )
        with torch.no_grad():
            target[indices] = replacement_tensor

        for name_field in METADATA_FIELDS:
            buffer = self.metadata(name, name_field)
            buffer[indices] = 0.0 if name_field != "age" else 0
        return indices

    def extra_repr(self) -> str:
        """Return a compact string representation of the PCS."""
        return (
            f"hidden_size={self.config.hidden_size}, "
            f"banks={self.bank_order}, "
            f"total_tokens={self.total_tokens}"
        )


__all__ = [
    "BANK_NAMES",
    "DEFAULT_BANK_SIZES",
    "INTENT_BANK",
    "METADATA_FIELDS",
    "BankSpec",
    "PCSConfig",
    "PersistentCognitiveState",
    "build_bank_specs",
    "resolve_bank_sizes",
    "retention_score",
]
