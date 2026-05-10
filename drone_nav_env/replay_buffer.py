"""
replay_buffer.py
================
Pre-allocated circular experience replay buffer for Deep Q-Learning,
designed to work with the DroneNavEnv observation format:

    obs["image"]  ->  numpy array, shape (64, 64, 3), dtype uint8
    obs["state"]  ->  numpy array, shape (7,),        dtype float32

Transitions of the form (obs, action, reward, next_obs, done) are stored
in fixed-size numpy arrays that are allocated once at construction time.
This avoids repeated memory allocation overhead and keeps all data
contiguous in memory for fast batch sampling.

Typical usage
-------------
    buf = ReplayBuffer(capacity=100_000)

    obs, _ = env.reset()
    action = policy(obs)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    buf.push(obs, action, reward, next_obs, done)

    if buf.is_ready(batch_size=32):
        batch = buf.sample(32)
        # batch["images"], batch["states"], batch["actions"], ...

    buf.save("checkpoints/replay.npz")
    buf = ReplayBuffer.load("checkpoints/replay.npz")
"""

from typing import Dict, Tuple

import numpy as np


class ReplayBuffer:
    """
    Circular (ring) experience replay buffer backed by pre-allocated numpy arrays.

    All storage is reserved at construction time so no dynamic memory
    allocation takes place during training.  When the buffer is full, new
    transitions silently overwrite the oldest ones (FIFO eviction).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    image_shape : tuple of int, optional
        Shape of a single camera frame.  Defaults to (64, 64, 3).
    state_dim : int, optional
        Length of the low-dimensional state vector.  Defaults to 7.
    """

    def __init__(
        self,
        capacity: int,
        image_shape: Tuple[int, ...] = (64, 64, 3),
        state_dim: int = 7,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity}")

        self.capacity = capacity
        self.image_shape = image_shape
        self.state_dim = state_dim

        # ------------------------------------------------------------------ #
        # Pre-allocate all storage arrays.                                     #
        # Using uint8 for images keeps memory footprint ~4× smaller than      #
        # float32 while retaining full pixel precision.                        #
        # ------------------------------------------------------------------ #

        # Current-step observations
        self.images = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)

        # Scalar transition components
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)

        # Next-step observations
        self.next_images = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

        # Internal bookkeeping
        self._ptr = 0  # index of the next write slot (wraps around)
        self._size = 0  # number of valid transitions currently stored

    # ---------------------------------------------------------------------- #
    # Core API                                                                 #
    # ---------------------------------------------------------------------- #

    def push(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        done: bool,
    ) -> None:
        """
        Store one transition in the buffer.

        Writes to the slot pointed to by ``_ptr``, then advances the pointer
        with wrap-around.  ``_size`` is incremented until it reaches
        ``capacity``, after which it stays fixed (old data is being replaced).

        Parameters
        ----------
        obs : dict
            Current observation with keys ``"image"`` and ``"state"``.
        action : int
            Discrete action index chosen by the agent.
        reward : float
            Scalar reward received after taking ``action``.
        next_obs : dict
            Next observation (same structure as ``obs``).
        done : bool
            ``True`` if the episode terminated or was truncated.
        """
        idx = self._ptr  # write position

        # Store current observation
        self.images[idx] = obs["image"]
        self.states[idx] = obs["state"]

        # Store scalar components
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        # Store next observation
        self.next_images[idx] = next_obs["image"]
        self.next_states[idx] = next_obs["state"]

        # Advance circular pointer
        self._ptr = (self._ptr + 1) % self.capacity

        # Track fill level (capped at capacity)
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Randomly sample a batch of transitions without replacement.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        dict
            Dictionary with the following keys, each mapping to a numpy array
            whose first dimension equals ``batch_size``:

            * ``"images"``       – shape ``(B, 64, 64, 3)``, uint8
            * ``"states"``       – shape ``(B, 7)``,          float32
            * ``"actions"``      – shape ``(B,)``,            int64
            * ``"rewards"``      – shape ``(B,)``,            float32
            * ``"next_images"``  – shape ``(B, 64, 64, 3)``,  uint8
            * ``"next_states"``  – shape ``(B, 7)``,          float32
            * ``"dones"``        – shape ``(B,)``,            bool

        Raises
        ------
        ValueError
            If ``batch_size`` exceeds the current number of stored transitions.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only contains "
                f"{self._size} transitions."
            )

        # Draw unique indices from the valid portion of the arrays
        indices = np.random.choice(self._size, size=batch_size, replace=False)

        return {
            "images": self.images[indices],
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_images": self.next_images[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        """Return the number of transitions currently stored in the buffer."""
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        """
        Check whether the buffer holds enough transitions to sample a batch.

        Parameters
        ----------
        batch_size : int
            Desired number of samples.

        Returns
        -------
        bool
            ``True`` if ``_size >= batch_size``, ``False`` otherwise.
        """
        return self._size >= batch_size

    # ---------------------------------------------------------------------- #
    # Persistence                                                              #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """
        Persist the buffer to a compressed ``.npz`` file.

        All seven data arrays plus the two bookkeeping integers (``_ptr`` and
        ``_size``) are saved so that the buffer can be fully restored later.

        Parameters
        ----------
        path : str
            Destination file path.  A ``.npz`` extension will be appended by
            numpy if it is not already present.
        """
        np.savez_compressed(
            path,
            images=self.images,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_images=self.next_images,
            next_states=self.next_states,
            dones=self.dones,
            # Store scalars as 0-d arrays so they round-trip cleanly
            _ptr=np.array(self._ptr, dtype=np.int64),
            _size=np.array(self._size, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        """
        Restore a ``ReplayBuffer`` instance from a ``.npz`` file.

        The capacity and array shapes are inferred from the stored data, so
        the constructor parameters do not need to be supplied manually.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file produced by :meth:`save`.

        Returns
        -------
        ReplayBuffer
            A fully restored buffer with the same state as when it was saved.
        """
        data = np.load(path)

        # Infer constructor arguments from the stored arrays
        capacity = data["images"].shape[0]
        image_shape = data["images"].shape[1:]  # e.g. (64, 64, 3)
        state_dim = data["states"].shape[1]  # e.g. 7

        # Build a fresh (empty) buffer with the correct dimensions
        buf = cls(capacity=capacity, image_shape=image_shape, state_dim=state_dim)

        # Restore all data arrays in-place to keep the pre-allocated memory
        buf.images[:] = data["images"]
        buf.states[:] = data["states"]
        buf.actions[:] = data["actions"]
        buf.rewards[:] = data["rewards"]
        buf.next_images[:] = data["next_images"]
        buf.next_states[:] = data["next_states"]
        buf.dones[:] = data["dones"]

        # Restore bookkeeping integers (stored as 0-d arrays)
        buf._ptr = int(data["_ptr"])
        buf._size = int(data["_size"])

        return buf
