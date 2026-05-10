"""
Learning rate and exploration schedules for RL training.
All schedules are callable with signature: schedule(progress) -> float
where progress is in [0, 1] (fraction of training completed).
"""

import math
from typing import Optional


class LinearSchedule:
    """
    Linearly anneal a value from start to end over training.

    Parameters
    ----------
    start : float
        Initial value.
    end : float
        Final value.
    """

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def __call__(self, progress: float) -> float:
        """
        Parameters
        ----------
        progress : float in [0, 1]
            Fraction of training completed.

        Returns
        -------
        float
            Interpolated value.
        """
        return self.start + (self.end - self.start) * min(progress, 1.0)


class CosineSchedule:
    """
    Cosine annealing from start to end.

    Parameters
    ----------
    start : float
        Initial value (maximum).
    end : float
        Final value (minimum).
    """

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def __call__(self, progress: float) -> float:
        progress = min(progress, 1.0)
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * progress))


class ExponentialSchedule:
    """
    Exponential decay from start to end.

    Parameters
    ----------
    start : float
        Initial value.
    end : float
        Final (asymptotic) value.
    decay_rate : float
        Decay rate constant. Higher = faster decay.
    """

    def __init__(self, start: float, end: float, decay_rate: float = 5.0):
        self.start = start
        self.end = end
        self.decay_rate = decay_rate

    def __call__(self, progress: float) -> float:
        progress = min(progress, 1.0)
        return self.end + (self.start - self.end) * math.exp(-self.decay_rate * progress)


class ConstantSchedule:
    """Returns a constant value regardless of progress."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, progress: float) -> float:
        return self.value
