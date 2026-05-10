"""
DiscreteActionWrapper — Gymnasium wrapper that discretizes DroneNavEnv's action space
======================================================================================
DroneNavEnv exposes a **continuous** action space:

    Box(-3.0, 3.0, shape=(3,), dtype=float32)   →   (vx, vy, vz) velocity targets

DQN (and other value-based RL algorithms) require a **discrete** action space.
This wrapper maps every integer index to a unique (vx, vy, vz) combination by
forming the Cartesian product of ``n_bins`` evenly-spaced values on each axis.

Example (default n_bins=3)
--------------------------
Discrete values per axis : [-3.0,  0.0, +3.0]
Total discrete actions    :  3³ = 27

The integer action index i selects row i from ``self.action_map``, which is the
corresponding continuous velocity vector passed to the wrapped environment.

Usage
-----
>>> from drone_nav_env.env import DroneNavEnv
>>> from drone_nav_env.discrete_wrapper import DiscreteActionWrapper, N_ACTIONS
>>> env = DiscreteActionWrapper(DroneNavEnv())
>>> obs, info = env.reset()
>>> obs, reward, terminated, truncated, info = env.step(0)  # first discrete action
>>> env.decode_action(13)  # → array([0., 0., 0.])  (centre action for n_bins=3)
"""

import itertools

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Module-level constant: total number of discrete actions when n_bins=3
# ---------------------------------------------------------------------------
N_ACTIONS: int = 3**3  # = 27

# Maximum velocity magnitude (must match DroneNavEnv.MAX_VEL)
_MAX_VEL: float = 3.0


class DiscreteActionWrapper(gym.Wrapper):
    """Gymnasium wrapper that converts a continuous (vx, vy, vz) action space
    into a :class:`gym.spaces.Discrete` space.

    The wrapper builds a lookup table ``action_map`` that lists every
    combination of discretized velocity values.  An integer action index is
    converted to its corresponding continuous velocity vector before being
    forwarded to the wrapped environment.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.  Its action space must be a ``Box`` of shape
        ``(3,)`` representing ``(vx, vy, vz)`` velocity targets.
    n_bins : int, optional
        Number of discrete values to sample per axis using
        ``np.linspace(-MAX_VEL, MAX_VEL, n_bins)``.  Defaults to ``3``,
        giving values ``[-3.0, 0.0, +3.0]`` and a total of 27 actions.

    Attributes
    ----------
    action_map : np.ndarray, shape (n_bins**3, 3)
        Lookup table mapping each discrete action index to a continuous
        ``(vx, vy, vz)`` velocity vector.
    action_space : gym.spaces.Discrete
        Discrete action space of size ``n_bins**3``.
    observation_space : gym.spaces.Space
        Unchanged observation space, passed through from the wrapped env.
    """

    def __init__(self, env: gym.Env, n_bins: int = 3) -> None:
        super().__init__(env)

        self._n_bins = n_bins

        # Discretized values for a single axis, e.g. [-3.0, 0.0, 3.0]
        axis_values = np.linspace(-_MAX_VEL, _MAX_VEL, n_bins)

        # Cartesian product across all three axes → (n_bins**3, 3) array
        self.action_map: np.ndarray = np.array(
            list(itertools.product(axis_values, axis_values, axis_values)),
            dtype=np.float32,
        )  # shape: (n_bins**3, 3)

        # Override only the action space; observation space is unchanged
        self.action_space = gym.spaces.Discrete(n_bins**3)
        # self.observation_space is inherited from gym.Wrapper (pass-through)

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def step(self, action: int):
        """Convert a discrete action index to a continuous velocity vector and
        forward it to the wrapped environment.

        Parameters
        ----------
        action : int
            A discrete action index in ``[0, n_bins**3)``.

        Returns
        -------
        observation, reward, terminated, truncated, info
            Unchanged output from the wrapped environment's ``step``.
        """
        continuous_action: np.ndarray = self.action_map[action]
        return self.env.step(continuous_action)

    def reset(self, **kwargs):
        """Reset the wrapped environment, passing through all keyword arguments.

        Returns
        -------
        observation, info
            Unchanged output from the wrapped environment's ``reset``.
        """
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def decode_action(self, action: int) -> np.ndarray:
        """Return the continuous ``(vx, vy, vz)`` vector for a discrete index.

        This is useful for logging or debugging — e.g. printing which velocity
        command was sent by the policy at each step.

        Parameters
        ----------
        action : int
            A discrete action index in ``[0, n_bins**3)``.

        Returns
        -------
        np.ndarray, shape (3,)
            The continuous ``(vx, vy, vz)`` velocity vector corresponding to
            the given discrete index.

        Examples
        --------
        >>> wrapper = DiscreteActionWrapper(env, n_bins=3)
        >>> wrapper.decode_action(0)
        array([-3., -3., -3.], dtype=float32)
        >>> wrapper.decode_action(13)
        array([0., 0., 0.], dtype=float32)
        >>> wrapper.decode_action(26)
        array([3., 3., 3.], dtype=float32)
        """
        return self.action_map[action]
