"""
Curriculum learning manager with 6 progressive difficulty stages.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CurriculumStage:
    """Definition of a single curriculum difficulty stage."""
    name: str
    corridor_width: float = 3.0
    obstacle_density: float = 0.0
    enable_turns: bool = False
    enable_moving_obstacles: bool = False
    sensor_noise: float = 0.0
    wind_enabled: bool = False
    wind_sigma: float = 0.3
    reward_threshold: float = 0.0  # avg reward to advance
    success_threshold: float = 0.0  # eval success rate to advance

    def to_env_params(self) -> dict:
        """Convert to environment constructor kwargs."""
        return {
            "corridor_width": self.corridor_width,
            "obstacle_density": self.obstacle_density,
            "enable_turns": self.enable_turns,
            "enable_moving_obstacles": self.enable_moving_obstacles,
            "sensor_noise": self.sensor_noise,
            "wind_enabled": self.wind_enabled,
            "wind_sigma": self.wind_sigma,
        }


# Default 6-stage curriculum
DEFAULT_STAGES = [
    CurriculumStage(
        name="1_wide_straight",
        corridor_width=3.0, obstacle_density=0.0,
        reward_threshold=-40.0,
        success_threshold=0.05,
    ),
    CurriculumStage(
        name="2_narrow_corridor",
        corridor_width=2.6, obstacle_density=0.0,
        reward_threshold=-25.0,
        success_threshold=0.10,
    ),
    CurriculumStage(
        name="3_with_obstacles",
        corridor_width=2.5, obstacle_density=0.7,
        enable_turns=False,
        reward_threshold=-15.0,
        success_threshold=0.20,
    ),
    CurriculumStage(
        name="4_moving_obstacles",
        corridor_width=2.5, obstacle_density=1.0,
        enable_moving_obstacles=True,
        reward_threshold=-8.0,
        success_threshold=0.30,
    ),
    CurriculumStage(
        name="5_wind_intro",
        corridor_width=2.0, obstacle_density=1.0,
        enable_moving_obstacles=True,
        sensor_noise=0.05,
        wind_enabled=True, wind_sigma=0.25,
        reward_threshold=-2.0,
        success_threshold=0.40,
    ),
    CurriculumStage(
        name="6_full_wind_disturbances",
        corridor_width=2.0, obstacle_density=1.0,
        enable_moving_obstacles=True,
        sensor_noise=0.05,
        wind_enabled=True, wind_sigma=0.5,
        reward_threshold=2.0,
        success_threshold=0.55,
    ),
]


class CurriculumManager:
    """
    Manages progressive difficulty stages during training.

    Parameters
    ----------
    enabled : bool
        Whether curriculum learning is active.
    stages : list of CurriculumStage, optional
        Custom stages. Uses DEFAULT_STAGES if None.
    patience : int
        Number of consecutive evaluations above threshold to advance.
    """

    def __init__(self, enabled: bool = False,
                 stages: Optional[List[CurriculumStage]] = None,
                 patience: int = 5):
        self.enabled = enabled
        raw_stages = stages or DEFAULT_STAGES
        self.stages = [
            stage if isinstance(stage, CurriculumStage) else CurriculumStage(**stage)
            for stage in raw_stages
        ]
        self.patience = patience
        self.current_stage = 0
        self._above_threshold_count = 0

    @property
    def current(self) -> CurriculumStage:
        """Current curriculum stage."""
        idx = min(self.current_stage, len(self.stages) - 1)
        return self.stages[idx]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage >= len(self.stages) - 1

    def maybe_advance(self, avg_reward: float, success_rate: float = 0.0) -> bool:
        """
        Check if we should advance to the next stage.

        Parameters
        ----------
        avg_reward : float
            Current average episode reward.

        Returns
        -------
        bool
            True if stage was advanced.
        """
        if not self.enabled or self.is_final_stage:
            return False

        reward_ok = avg_reward >= self.current.reward_threshold
        success_ok = success_rate >= self.current.success_threshold
        if reward_ok and success_ok:
            self._above_threshold_count += 1
        else:
            self._above_threshold_count = 0

        if self._above_threshold_count >= self.patience:
            self.current_stage += 1
            self._above_threshold_count = 0
            return True
        return False

    def get_env_params(self) -> dict:
        """Get environment parameters for the current stage."""
        return self.current.to_env_params()

    def state_dict(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "above_threshold_count": self._above_threshold_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_stage = state.get("current_stage", 0)
        self._above_threshold_count = state.get("above_threshold_count", 0)
