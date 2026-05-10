"""
Ornstein-Uhlenbeck wind disturbance model.

The OU process generates temporally correlated wind vectors that mimic
realistic indoor air disturbances (HVAC drafts, propeller wash, open windows).

    dW_t = theta * (mu - W_t) * dt + sigma * sqrt(dt) * N(0, I)

Parameters
----------
mu      : mean wind vector (m/s), default zero (calm air)
theta   : mean-reversion rate  — how quickly wind returns to mu
sigma   : volatility           — strength of random gusts
max_speed : hard clip (m/s), set to 5.0 per proposal
"""

from typing import Optional

import numpy as np


class OUWindModel:
    """Ornstein-Uhlenbeck process for 3-D wind disturbance."""

    def __init__(
        self,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.3,
        dt: float = 1.0 / 240.0,
        max_speed: float = 5.0,
        seed: Optional[int] = None,
    ):
        self.mu = np.full(3, mu, dtype=np.float64)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.max_speed = max_speed
        self.rng = np.random.default_rng(seed)
        self._state = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset wind to zero and return initial wind vector."""
        self._state = np.zeros(3, dtype=np.float64)
        return self._state.copy()

    # ------------------------------------------------------------------
    def step(self) -> np.ndarray:
        """Advance one simulation step and return new wind vector (m/s)."""
        noise = self.rng.standard_normal(3)
        self._state += (
            self.theta * (self.mu - self._state) * self.dt
            + self.sigma * np.sqrt(self.dt) * noise
        )
        # Hard-clip magnitude to max_speed (5 m/s as per proposal)
        speed = np.linalg.norm(self._state)
        if speed > self.max_speed:
            self._state = self._state / speed * self.max_speed
        return self._state.copy()

    # ------------------------------------------------------------------
    @property
    def current(self) -> np.ndarray:
        return self._state.copy()
