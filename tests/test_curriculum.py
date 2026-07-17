"""Tests for :mod:`ucsa.training.curriculum`."""

from __future__ import annotations

import pytest

from ucsa.training.curriculum import (
    Curriculum,
    CurriculumSchedule,
    CurriculumStage,
)


class TestCurriculumSchedule:
    """Tests for :class:`CurriculumSchedule`."""

    def test_default_schedule_valid(self) -> None:
        """Default schedule constructs without error."""
        schedule = CurriculumSchedule()
        assert schedule.stage_1_end > 0

    def test_invalid_stage_1_end(self) -> None:
        """Non-positive ``stage_1_end`` is rejected."""
        with pytest.raises(ValueError):
            CurriculumSchedule(stage_1_end=0)

    def test_stage_2_must_be_after_stage_1(self) -> None:
        """``stage_2_end`` must exceed ``stage_1_end``."""
        with pytest.raises(ValueError):
            CurriculumSchedule(stage_1_end=10, stage_2_end=10)

    def test_stage_3_must_be_after_stage_2(self) -> None:
        """``stage_3_end`` must exceed ``stage_2_end``."""
        with pytest.raises(ValueError):
            CurriculumSchedule(stage_2_end=10, stage_3_end=10)


class TestCurriculumStage:
    """Tests for :class:`CurriculumStage`."""

    def test_display_name(self) -> None:
        """``display_name`` lowercases and replaces underscores."""
        assert (
            CurriculumStage.LANGUAGE_JEPA_MEMORY.display_name
            == "language jepa memory"
        )
        assert CurriculumStage.JOINT.display_name == "joint"


class TestCurriculumGetStage:
    """Tests for :meth:`Curriculum.get_stage`."""

    @pytest.fixture()
    def curriculum(self) -> Curriculum:
        """Provide a curriculum with a tiny schedule."""
        return Curriculum(
            CurriculumSchedule(stage_1_end=10, stage_2_end=20, stage_3_end=30)
        )

    def test_stage_1_at_step_zero(
        self, curriculum: Curriculum
    ) -> None:
        """At step 0, the stage is LANGUAGE_ONLY."""
        assert curriculum.get_stage(0) is CurriculumStage.LANGUAGE_ONLY

    def test_stage_1_at_step_9(
        self, curriculum: Curriculum
    ) -> None:
        """At step 9 (just before stage 1 ends), still LANGUAGE_ONLY."""
        assert curriculum.get_stage(9) is CurriculumStage.LANGUAGE_ONLY

    def test_stage_2_at_step_10(
        self, curriculum: Curriculum
    ) -> None:
        """At step 10, the stage is LANGUAGE_JEPA."""
        assert curriculum.get_stage(10) is CurriculumStage.LANGUAGE_JEPA

    def test_stage_3_at_step_20(
        self, curriculum: Curriculum
    ) -> None:
        """At step 20, the stage is LANGUAGE_JEPA_MEMORY."""
        assert curriculum.get_stage(20) is CurriculumStage.LANGUAGE_JEPA_MEMORY

    def test_stage_4_at_step_30(
        self, curriculum: Curriculum
    ) -> None:
        """At step 30, the stage is JOINT."""
        assert curriculum.get_stage(30) is CurriculumStage.JOINT

    def test_stage_4_at_large_step(
        self, curriculum: Curriculum
    ) -> None:
        """At very large step, the stage remains JOINT."""
        assert curriculum.get_stage(1_000_000) is CurriculumStage.JOINT


class TestCurriculumActiveComponents:
    """Tests for :meth:`Curriculum.active_components`."""

    @pytest.fixture()
    def curriculum(self) -> Curriculum:
        """Provide a curriculum."""
        return Curriculum(
            CurriculumSchedule(stage_1_end=10, stage_2_end=20, stage_3_end=30)
        )

    def test_stage_1_components(
        self, curriculum: Curriculum
    ) -> None:
        """Stage 1 has only AR."""
        components = curriculum.active_components(CurriculumStage.LANGUAGE_ONLY)
        assert components == {"ar"}

    def test_stage_2_components(
        self, curriculum: Curriculum
    ) -> None:
        """Stage 2 adds JEPA."""
        components = curriculum.active_components(
            CurriculumStage.LANGUAGE_JEPA
        )
        assert components == {"ar", "jepa"}

    def test_stage_3_components(
        self, curriculum: Curriculum
    ) -> None:
        """Stage 3 adds memory."""
        components = curriculum.active_components(
            CurriculumStage.LANGUAGE_JEPA_MEMORY
        )
        assert components == {"ar", "jepa", "memory"}

    def test_stage_4_components(
        self, curriculum: Curriculum
    ) -> None:
        """Stage 4 adds router (joint)."""
        components = curriculum.active_components(CurriculumStage.JOINT)
        assert components == {"ar", "jepa", "memory", "router"}


class TestCurriculumStep:
    """Tests for :meth:`Curriculum.step`."""

    def test_step_increments_total(self) -> None:
        """``step`` increments the global step counter."""
        curriculum = Curriculum(
            CurriculumSchedule(stage_1_end=10, stage_2_end=20, stage_3_end=30)
        )
        curriculum.step()
        assert curriculum.state.total_step == 1
        curriculum.step()
        assert curriculum.state.total_step == 2

    def test_stage_transition_recorded(self) -> None:
        """Transitions are appended to the history."""
        curriculum = Curriculum(
            CurriculumSchedule(stage_1_end=3, stage_2_end=6, stage_3_end=9)
        )
        for _ in range(9):
            curriculum.step()
        # Three transitions are expected (1->2, 2->3, 3->4).
        assert len(curriculum.state.history) == 3
        assert curriculum.state.history[0][0] == 3
        assert curriculum.state.history[1][0] == 6
        assert curriculum.state.history[2][0] == 9

    def test_stage_step_resets_on_transition(self) -> None:
        """``stage_step`` resets to zero when a transition occurs."""
        curriculum = Curriculum(
            CurriculumSchedule(stage_1_end=3, stage_2_end=6, stage_3_end=9)
        )
        for _ in range(3):
            curriculum.step()
        assert curriculum.state.stage_step == 0
        assert curriculum.state.current_stage is CurriculumStage.LANGUAGE_JEPA

    def test_initial_state(self) -> None:
        """Initial state is LANGUAGE_ONLY at step 0."""
        curriculum = Curriculum()
        assert curriculum.state.total_step == 0
        assert curriculum.state.current_stage is CurriculumStage.LANGUAGE_ONLY
        assert curriculum.state.stage_step == 0
        assert curriculum.state.history == []

    def test_active_components_default_uses_current_stage(self) -> None:
        """``active_components`` with no argument uses the current stage."""
        curriculum = Curriculum(
            CurriculumSchedule(stage_1_end=2, stage_2_end=4, stage_3_end=6)
        )
        assert curriculum.active_components() == {"ar"}
        curriculum.step()
        curriculum.step()
        assert curriculum.active_components() == {"ar", "jepa"}