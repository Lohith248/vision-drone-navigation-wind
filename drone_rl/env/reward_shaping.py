"""
Modular reward shaping for drone corridor navigation.

All reward components are configurable via weight parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class RewardWeights:
    """Configurable weights for each reward component."""
    forward_progress: float = 1.0
    distance_scale: float = 100.0
    centering: float = 0.5
    smoothness: float = 0.1
    goal_reached: float = 10.0
    collision: float = -10.0
    timeout_penalty: float = -10.0
    wall_proximity: float = -0.5
    oscillation: float = -0.05
    control_jump: float = -0.05
    time_penalty: float = -0.001


class RewardShaper:
    """
    Compute shaped reward for drone corridor navigation.

    The total reward is a weighted sum of components:
        r_total = Σ w_i * r_i

    Parameters
    ----------
    weights : RewardWeights
        Weight coefficients for each reward component.
    corridor_width : float
        Width of the corridor (for centering reward).
    goal_radius : float
        Distance threshold for goal reached detection.
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        corridor_width: float = 3.0,
        goal_radius: float = 0.5,
        mode: str = "report",
    ):
        self.weights = weights or RewardWeights()
        self.corridor_width = corridor_width
        self.goal_radius = goal_radius
        self.mode = mode

        # State for smoothness tracking
        self._prev_action: Optional[np.ndarray] = None
        self._prev_prev_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal state at the start of an episode."""
        self._prev_action = None
        self._prev_prev_action = None

    def compute(
        self,
        position: np.ndarray,
        prev_position: np.ndarray,
        goal_position: np.ndarray,
        action: np.ndarray,
        collided: bool,
        wall_distances: Optional[np.ndarray] = None,
        initial_goal_dist: float = 1.0,
        step_count: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute all reward components and the total reward.

        Parameters
        ----------
        position : np.ndarray, shape (3,)
            Current drone position (x, y, z).
        prev_position : np.ndarray, shape (3,)
            Previous drone position.
        goal_position : np.ndarray, shape (3,)
            Goal position.
        action : np.ndarray, shape (3,)
            Current action taken (vx, vy, vz).
        collided : bool
            Whether a collision occurred this step.
        wall_distances : np.ndarray, shape (4,), optional
            Distances to [left, right, floor, ceiling] walls.
        initial_goal_dist : float
            Initial distance to goal (for normalisation).

        Returns
        -------
        dict
            Mapping from component name to reward value.
            Includes 'total' key with the weighted sum.
        """
        w = self.weights
        components = {}

        prev_dist = float(np.linalg.norm(prev_position - goal_position))
        curr_dist = float(np.linalg.norm(position - goal_position))
        delta_dist = prev_dist - curr_dist
        if self.mode == "report":
            # Report-spec reward:
            # R_t = r_distance + r_goal + r_time + r_timeout + r_collision
            components["r_distance"] = (
                w.forward_progress
                * w.distance_scale
                * delta_dist
                / max(initial_goal_dist, 1e-6)
            )
            components["r_goal"] = w.goal_reached if curr_dist < self.goal_radius else 0.0
            components["r_time"] = w.time_penalty
            timed_out = (
                step_count is not None
                and max_steps is not None
                and step_count >= max_steps
            )
            components["r_timeout"] = w.timeout_penalty if timed_out else 0.0
            components["r_collision"] = w.collision if collided else 0.0
            components["total"] = sum(components.values())
        else:
            # Optional legacy shaped reward mode
            r_progress = 10.0 * delta_dist / max(initial_goal_dist, 1e-6)
            components["forward_progress"] = w.forward_progress * r_progress

            half_w = self.corridor_width / 2.0
            lateral_offset = abs(position[1])
            centering = 1.0 - min(lateral_offset / half_w, 1.0)
            components["centering"] = w.centering * centering

            if self._prev_action is not None:
                action_diff = action - self._prev_action
                smoothness = -float(np.sum(action_diff ** 2))
            else:
                smoothness = 0.0
            components["smoothness"] = w.smoothness * smoothness

            reached_goal = curr_dist < self.goal_radius
            components["goal_reached"] = w.goal_reached if reached_goal else 0.0
            components["collision"] = w.collision if collided else 0.0

            if wall_distances is not None:
                min_wall_dist = float(np.min(wall_distances))
                wall_penalty = max(0, 1.0 - min_wall_dist / (half_w * 0.5))
                components["wall_proximity"] = w.wall_proximity * wall_penalty
            else:
                components["wall_proximity"] = 0.0

            if self._prev_action is not None and self._prev_prev_action is not None:
                d1 = action - self._prev_action
                d2 = self._prev_action - self._prev_prev_action
                sign_changes = np.sum(np.sign(d1) != np.sign(d2))
                components["oscillation"] = w.oscillation * float(sign_changes)
            else:
                components["oscillation"] = 0.0

            if self._prev_action is not None:
                jump = float(np.max(np.abs(action - self._prev_action)))
                components["control_jump"] = w.control_jump * max(0, jump - 1.0)
            else:
                components["control_jump"] = 0.0

            components["time_penalty"] = w.time_penalty
            components["total"] = sum(components.values())

        # Update action history
        self._prev_prev_action = self._prev_action
        self._prev_action = action.copy()

        return components
