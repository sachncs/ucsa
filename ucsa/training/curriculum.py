"""Training curriculum.

UCSA's training progresses through four stages:

1. **LANGUAGE_ONLY** -- only the autoregressive loss is active.
2. **LANGUAGE_JEPA** -- adds the JEPA auxiliary loss.
3. **LANGUAGE_JEPA_MEMORY** -- adds the memory-stability loss.
4. **JOINT** -- all four losses are active.

Stage transitions are step-gated: each stage begins at a configured
global step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CurriculumStage(Enum):
    """Enumeration of curriculum stages."""

    LANGUAGE_ONLY = 1
    LANGUAGE_JEPA = 2
    LANGUAGE_JEPA_MEMORY = 3
    JOINT = 4

    @property
    def display_name(self) -> str:
        """Return a human-readable name."""
        return self.name.lower().replace("_", " ")


@dataclass(frozen=True)
class CurriculumSchedule:
    """Step boundaries for each stage.

    Attributes:
        stage_1_end: Step at which stage 1 ends (and stage 2 begins).
        stage_2_end: Step at which stage 2 ends.
        stage_3_end: Step at which stage 3 ends. Anything beyond is
            stage 4.
    """

    stage_1_end: int = 1000
    stage_2_end: int = 5000
    stage_3_end: int = 20000

    def __post_init__(self) -> None:
        if self.stage_1_end <= 0:
            raise ValueError(
                f"stage_1_end must be positive, got {self.stage_1_end}."
            )
        if self.stage_2_end <= self.stage_1_end:
            raise ValueError(
                f"stage_2_end ({self.stage_2_end}) must be > "
                f"stage_1_end ({self.stage_1_end})."
            )
        if self.stage_3_end <= self.stage_2_end:
            raise ValueError(
                f"stage_3_end ({self.stage_3_end}) must be > "
                f"stage_2_end ({self.stage_2_end})."
            )


@dataclass
class CurriculumState:
    """Mutable state tracked by the curriculum.

    Attributes:
        current_stage: The current stage.
        stage_step: Step within the current stage.
        total_step: Global step counter.
        history: List of ``(step, stage)`` transitions.
    """

    current_stage: CurriculumStage = CurriculumStage.LANGUAGE_ONLY
    stage_step: int = 0
    total_step: int = 0
    history: list[tuple[int, CurriculumStage]] = field(default_factory=list)


class Curriculum:
    """Step-gated training curriculum."""

    def __init__(
        self,
        schedule: CurriculumSchedule | None = None,
    ) -> None:
        """Initialise the curriculum.

        Args:
            schedule: Optional :class:`CurriculumSchedule`.
        """
        self.schedule = schedule or CurriculumSchedule()
        self.state = CurriculumState()

    def step(self) -> CurriculumStage:
        """Advance the curriculum by one step.

        Returns:
            The current stage after the advance.
        """
        self.state.total_step += 1
        self.state.stage_step += 1
        new_stage = self.get_stage(self.state.total_step)
        if new_stage is not self.state.current_stage:
            self.state.history.append(
                (self.state.total_step, self.state.current_stage)
            )
            self.state.current_stage = new_stage
            self.state.stage_step = 0
        return new_stage

    def get_stage(self, total_step: int) -> CurriculumStage:
        """Return the stage for a given global step.

        Args:
            total_step: Global training step.

        Returns:
            The stage active at ``total_step``.
        """
        if total_step < self.schedule.stage_1_end:
            return CurriculumStage.LANGUAGE_ONLY
        if total_step < self.schedule.stage_2_end:
            return CurriculumStage.LANGUAGE_JEPA
        if total_step < self.schedule.stage_3_end:
            return CurriculumStage.LANGUAGE_JEPA_MEMORY
        return CurriculumStage.JOINT

    def active_components(self, stage: CurriculumStage | None = None) -> set[str]:
        """Return the set of loss component names active at ``stage``.

        Args:
            stage: Optional stage. Defaults to the current stage.

        Returns:
            Set of component names drawn from ``{"ar", "jepa", "memory",
            "router"}``.
        """
        if stage is None:
            stage = self.state.current_stage
        if stage is CurriculumStage.LANGUAGE_ONLY:
            return {"ar"}
        if stage is CurriculumStage.LANGUAGE_JEPA:
            return {"ar", "jepa"}
        if stage is CurriculumStage.LANGUAGE_JEPA_MEMORY:
            return {"ar", "jepa", "memory"}
        return {"ar", "jepa", "memory", "router"}


__all__ = [
    "Curriculum",
    "CurriculumSchedule",
    "CurriculumStage",
    "CurriculumState",
]