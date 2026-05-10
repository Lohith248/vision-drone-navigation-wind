"""
Discrete action wrapper for DQN with configurable action sets.
"""
import itertools
from typing import Optional
import gymnasium as gym
import numpy as np

SEMANTIC_ACTIONS = {
    "hover":          np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "forward_slow":   np.array([0.3, 0.0, 0.0], dtype=np.float32),
    "forward_fast":   np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "left":           np.array([0.0, 0.5, 0.0], dtype=np.float32),
    "right":          np.array([0.0, -0.5, 0.0], dtype=np.float32),
    "up":             np.array([0.0, 0.0, 0.5], dtype=np.float32),
    "down":           np.array([0.0, 0.0, -0.5], dtype=np.float32),
    "forward_left":   np.array([0.5, 0.3, 0.0], dtype=np.float32),
    "forward_right":  np.array([0.5, -0.3, 0.0], dtype=np.float32),
    "forward_up":     np.array([0.5, 0.0, 0.3], dtype=np.float32),
    "forward_down":   np.array([0.5, 0.0, -0.3], dtype=np.float32),
}
SEMANTIC_ACTION_NAMES = list(SEMANTIC_ACTIONS.keys())
SEMANTIC_ACTION_MAP = np.array(list(SEMANTIC_ACTIONS.values()), dtype=np.float32)

def build_cartesian_action_map(n_bins: int = 3) -> np.ndarray:
    axis_values = np.linspace(-1.0, 1.0, n_bins)
    return np.array(list(itertools.product(axis_values, axis_values, axis_values)), dtype=np.float32)

class DiscreteNavWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action_mode: str = "semantic", n_bins: int = 3):
        super().__init__(env)
        self.action_mode = action_mode
        if action_mode == "semantic":
            self.action_map = SEMANTIC_ACTION_MAP.copy()
            self.action_names = SEMANTIC_ACTION_NAMES.copy()
        elif action_mode == "cartesian":
            self.action_map = build_cartesian_action_map(n_bins)
            self.action_names = [f"({a[0]:+.1f},{a[1]:+.1f},{a[2]:+.1f})" for a in self.action_map]
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")
        self.n_actions = len(self.action_map)
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def step(self, action: int):
        return self.env.step(self.action_map[action])

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def decode_action(self, action: int) -> np.ndarray:
        return self.action_map[action].copy()

    def action_name(self, action: int) -> str:
        return self.action_names[action]
