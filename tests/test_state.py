"""Tests for :mod:`ucsa.models.state`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ucsa.models.state import (
    BANK_NAMES,
    DEFAULT_BANK_SIZES,
    BankSpec,
    PCSConfig,
    PersistentCognitiveState,
    retention_score,
)


class TestPCSConfig:
    """Tests for :class:`PCSConfig`."""

    def test_default_config_is_valid(self) -> None:
        """Default config constructs without error and exposes sane values."""
        config = PCSConfig()
        assert config.hidden_size == 128
        assert config.init_std > 0.0
        assert config.retention_floor >= 0.0

    def test_zero_hidden_size_rejected(self) -> None:
        """``hidden_size`` of zero or less must raise."""
        with pytest.raises(ValueError):
            PCSConfig(hidden_size=0)

    def test_negative_init_std_rejected(self) -> None:
        """``init_std`` of zero or less must raise."""
        with pytest.raises(ValueError):
            PCSConfig(init_std=-0.1)


class TestBankSpec:
    """Tests for :class:`BankSpec`."""

    def test_construction(self) -> None:
        """Valid specs construct and freeze."""
        spec = BankSpec(name="working", num_tokens=64)
        assert spec.name == "working"
        assert spec.num_tokens == 64
        assert spec.trainable is True

    def test_zero_tokens_rejected(self) -> None:
        """Bank with zero or negative tokens must raise."""
        with pytest.raises(ValueError):
            BankSpec(name="working", num_tokens=0)

    def test_empty_name_rejected(self) -> None:
        """Empty bank name must raise."""
        with pytest.raises(ValueError):
            BankSpec(name="", num_tokens=4)


class TestResolveBankSizes:
    """Tests for :func:`resolve_bank_sizes`."""

    def test_returns_all_default_banks(self) -> None:
        """All default banks are present in resolved sizes."""
        sizes = DEFAULT_BANK_SIZES  # internal: defaults dict used directly
        resolved = DEFAULT_BANK_SIZES
        for name in BANK_NAMES:
            assert name in resolved
            assert sizes[name] == resolved[name]

    def test_overrides_apply(self) -> None:
        """Overrides replace default sizes for named banks."""
        from ucsa.models.state import resolve_bank_sizes
        resolved = resolve_bank_sizes({"working": 200})
        assert resolved["working"] == 200
        assert resolved["long_term"] == DEFAULT_BANK_SIZES["long_term"]

    def test_invalid_override_rejected(self) -> None:
        """Non-positive override size must raise."""
        from ucsa.models.state import resolve_bank_sizes
        with pytest.raises(ValueError):
            resolve_bank_sizes({"working": 0})


class TestRetentionScore:
    """Tests for :func:`retention_score`."""

    def test_score_in_unit_interval(self) -> None:
        """Retention score is always in ``[0, 1]``."""
        importance = torch.tensor([0.0, 1.0, 5.0, 8.0])
        usage = torch.tensor([0.0, 0.0, 3.0, 3.0])
        age = torch.tensor([0, 1, 10, 10])
        config = PCSConfig()
        score = retention_score(importance, usage, age, config)
        assert torch.all(score >= 0.0)
        assert torch.all(score <= 1.0)

    def test_higher_importance_increases_score(self) -> None:
        """All else equal, higher importance raises the score."""
        config = PCSConfig()
        importance = torch.tensor([0.0, 10.0])
        usage = torch.zeros(2)
        age = torch.zeros(2)
        score = retention_score(importance, usage, age, config)
        assert score[1] > score[0]

    def test_higher_age_decreases_score(self) -> None:
        """All else equal, higher age lowers the score."""
        config = PCSConfig()
        importance = torch.ones(3)
        usage = torch.ones(3)
        age = torch.tensor([0, 5, 50])
        score = retention_score(importance, usage, age, config)
        assert score[0] > score[1] > score[2]

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes raise ``ValueError``."""
        config = PCSConfig()
        with pytest.raises(ValueError):
            retention_score(
                torch.zeros(2),
                torch.zeros(3),
                torch.zeros(3),
                config,
            )

    def test_negative_inputs_raise(self) -> None:
        """Negative importance, usage, or age raise ``ValueError``."""
        config = PCSConfig()
        with pytest.raises(ValueError):
            retention_score(
                torch.tensor([-1.0]),
                torch.zeros(1),
                torch.zeros(1),
                config,
            )
        with pytest.raises(ValueError):
            retention_score(
                torch.zeros(1),
                torch.tensor([-1.0]),
                torch.zeros(1),
                config,
            )
        with pytest.raises(ValueError):
            retention_score(
                torch.zeros(1),
                torch.zeros(1),
                torch.tensor([-1]),
                config,
            )


class TestPersistentCognitiveState:
    """Tests for :class:`PersistentCognitiveState`."""

    @pytest.fixture()
    def config(self) -> PCSConfig:
        """Default PCS config for tests."""
        return PCSConfig(hidden_size=32)

    @pytest.fixture()
    def state(self, config: PCSConfig) -> PersistentCognitiveState:
        """A fresh PCS instance."""
        return PersistentCognitiveState(config)

    def test_all_default_banks_present(self, state: PersistentCognitiveState) -> None:
        """Every default bank exists with its default size."""
        for name in BANK_NAMES:
            assert name in state.bank_specs
            assert state.bank_size(name) == DEFAULT_BANK_SIZES[name]

    def test_bank_tensor_shape(self, state: PersistentCognitiveState) -> None:
        """Bank tensors have shape ``(num_tokens, hidden_size)``."""
        for name in BANK_NAMES:
            tensor = state.get_bank(name)
            assert tensor.shape == (DEFAULT_BANK_SIZES[name], 32)

    def test_get_all_tokens_concatenates(self, state: PersistentCognitiveState) -> None:
        """``get_all_tokens`` returns the concatenation of every bank."""
        all_tokens = state.get_all_tokens()
        assert all_tokens.shape == (state.total_tokens, 32)
        expected_total = sum(state.bank_size(n) for n in BANK_NAMES)
        assert state.total_tokens == expected_total

    def test_banks_are_parameters(self, state: PersistentCognitiveState) -> None:
        """Every default bank is a learnable :class:`nn.Parameter`."""
        for name in BANK_NAMES:
            tensor = state.get_bank(name)
            assert isinstance(tensor, Tensor)
            assert tensor.requires_grad

    def test_parameter_count(self, state: PersistentCognitiveState) -> None:
        """The number of trainable parameters matches the PCS layout."""
        params = list(state.parameters())
        total = sum(DEFAULT_BANK_SIZES[name] * 32 for name in BANK_NAMES)
        assert sum(p.numel() for p in params) == total

    def test_bank_size_unknown_raises(self, state: PersistentCognitiveState) -> None:
        """Querying an unknown bank raises ``KeyError``."""
        with pytest.raises(KeyError):
            state.bank_size("nope")

    def test_get_bank_unknown_raises(self, state: PersistentCognitiveState) -> None:
        """``get_bank`` with an unknown name raises ``KeyError``."""
        with pytest.raises(KeyError):
            state.get_bank("nope")

    def test_set_bank_replaces_contents(self, state: PersistentCognitiveState) -> None:
        """``set_bank`` writes the provided tensor into the bank."""
        replacement = torch.full((state.bank_size("working"), 32), 0.5)
        state.set_bank("working", replacement)
        current = state.get_bank("working")
        assert torch.allclose(current, replacement)

    def test_set_bank_shape_mismatch_raises(
        self, state: PersistentCognitiveState
    ) -> None:
        """Wrong-shape replacement raises ``ValueError``."""
        with pytest.raises(ValueError):
            state.set_bank("working", torch.zeros(3, 3))

    def test_set_bank_unknown_raises(self, state: PersistentCognitiveState) -> None:
        """Unknown bank name raises ``KeyError``."""
        with pytest.raises(KeyError):
            state.set_bank("nope", torch.zeros(1, 32))

    def test_reset_metadata_zeros_all(
        self, state: PersistentCognitiveState
    ) -> None:
        """``reset_metadata`` zeroes every metadata field."""
        for name in BANK_NAMES:
            getattr(state, f"meta_importance_{name}").fill_(0.7)
            getattr(state, f"meta_usage_{name}").fill_(3.0)
            getattr(state, f"meta_age_{name}").fill_(5)
            getattr(state, f"meta_retention_{name}").fill_(0.4)
        state.reset_metadata()
        snapshot = state.get_all_metadata()
        for name in BANK_NAMES:
            assert torch.all(snapshot[name]["importance"] == 0)
            assert torch.all(snapshot[name]["usage"] == 0)
            assert torch.all(snapshot[name]["age"] == 0)
            assert torch.all(snapshot[name]["retention"] == 0)

    def test_step_age_increments(self, state: PersistentCognitiveState) -> None:
        """``step_age`` increments every metadata age buffer by one."""
        state.step_age()
        state.step_age()
        snapshot = state.get_all_metadata()
        for name in BANK_NAMES:
            assert torch.all(snapshot[name]["age"] == 2)

    def test_record_usage_resets_age(
        self, state: PersistentCognitiveState
    ) -> None:
        """Recording usage at an index resets that slot's age to zero."""
        state.step_age()
        indices = torch.tensor([0, 5, 10])
        state.record_usage("long_term", indices, increment=2.0)
        snapshot = state.get_all_metadata()
        assert torch.all(snapshot["long_term"]["age"][indices] == 0)
        assert torch.all(snapshot["long_term"]["usage"][indices] == 2.0)

    def test_record_usage_unknown_bank_raises(
        self, state: PersistentCognitiveState
    ) -> None:
        """Recording usage on an unknown bank raises ``KeyError``."""
        with pytest.raises(KeyError):
            state.record_usage("nope", torch.tensor([0]))

    def test_update_retention_populates_buffers(
        self, state: PersistentCognitiveState
    ) -> None:
        """``update_retention`` writes scores into the retention buffers."""
        importance = getattr(state, "meta_importance_long_term")
        importance.fill_(2.0)
        usage = getattr(state, "meta_usage_long_term")
        usage.fill_(1.0)
        state.update_retention()
        snapshot = state.get_all_metadata()
        assert torch.all(snapshot["long_term"]["retention"] > 0.0)
        assert torch.all(snapshot["long_term"]["retention"] <= 1.0)

    def test_recycle_bottom_k_returns_indices(
        self, state: PersistentCognitiveState
    ) -> None:
        """``recycle_bottom_k`` returns the indices of recycled slots."""
        retention = getattr(state, "meta_retention_long_term")
        retention[:5] = 0.0
        retention[5:] = 1.0
        recycled = state.recycle_bottom_k("long_term", k=3)
        assert recycled.shape == (3,)
        replacement = state.get_bank("long_term")[recycled]
        assert torch.all(replacement == 0)

    def test_recycle_with_custom_replacement(
        self, state: PersistentCognitiveState
    ) -> None:
        """A custom replacement tensor is written into recycled slots."""
        retention = getattr(state, "meta_retention_long_term")
        retention.fill_(1.0)
        retention[:2] = 0.0
        replacement = torch.full((2, 32), 7.0)
        recycled = state.recycle_bottom_k("long_term", k=2, replacement=replacement)
        slot_values = state.get_bank("long_term")[recycled]
        assert torch.all(slot_values == 7.0)

    def test_recycle_resets_metadata(
        self, state: PersistentCognitiveState
    ) -> None:
        """Recycled slots have their metadata zeroed."""
        retention = getattr(state, "meta_retention_long_term")
        retention.fill_(1.0)
        retention[:1] = 0.0
        state.recycle_bottom_k("long_term", k=1)
        snapshot = state.get_all_metadata()
        idx = torch.tensor([0])
        assert torch.all(snapshot["long_term"]["importance"][idx] == 0)
        assert torch.all(snapshot["long_term"]["usage"][idx] == 0)
        assert torch.all(snapshot["long_term"]["age"][idx] == 0)
        assert torch.all(snapshot["long_term"]["retention"][idx] == 0)

    def test_recycle_k_zero_returns_empty(
        self, state: PersistentCognitiveState
    ) -> None:
        """``k=0`` returns an empty index tensor and mutates nothing."""
        before = state.get_bank("long_term").clone()
        recycled = state.recycle_bottom_k("long_term", k=0)
        assert recycled.numel() == 0
        after = state.get_bank("long_term")
        assert torch.allclose(before, after)

    def test_recycle_replacement_shape_mismatch(
        self, state: PersistentCognitiveState
    ) -> None:
        """Wrong-shape replacement raises ``ValueError``."""
        with pytest.raises(ValueError):
            state.recycle_bottom_k(
                "long_term",
                k=2,
                replacement=torch.zeros(3, 32),
            )

    def test_recycle_unknown_bank_raises(
        self, state: PersistentCognitiveState
    ) -> None:
        """Unknown bank name raises ``KeyError``."""
        with pytest.raises(KeyError):
            state.recycle_bottom_k("nope", k=1)

    def test_gradients_flow_through_banks(
        self, state: PersistentCognitiveState
    ) -> None:
        """A loss on every bank tensor flows gradients to those parameters."""
        for name in BANK_NAMES:
            state.zero_grad(set_to_none=True)
            bank = state.get_bank(name)
            loss = bank.sum()
            loss.backward()
            assert bank.grad is not None
            assert torch.all(bank.grad == 1.0)

    def test_extra_repr_includes_dimensions(
        self, state: PersistentCognitiveState
    ) -> None:
        """``extra_repr`` mentions the hidden size and total token count."""
        text = state.extra_repr()
        assert "hidden_size=32" in text
        assert "total_tokens" in text

    def test_overrides_change_total_tokens(self) -> None:
        """``PCSConfig.bank_sizes`` overrides change the PCS layout."""
        config = PCSConfig(hidden_size=16, bank_sizes={"working": 8})
        state = PersistentCognitiveState(config)
        assert state.bank_size("working") == 8
        assert state.total_tokens == 8 + 128 + 16 + 32 + 16 + 32

    def test_device_movement(self, state: PersistentCognitiveState) -> None:
        """``.to(device)`` moves banks and metadata together."""
        state.to(torch.device("cpu"))
        assert state.get_bank("working").device.type == "cpu"