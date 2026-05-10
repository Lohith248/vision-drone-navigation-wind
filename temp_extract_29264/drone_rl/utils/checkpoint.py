"""
Checkpoint save/load/resume utilities for all algorithms.
"""

import os
import glob
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    metadata: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint to disk.

    Parameters
    ----------
    path : str
        Output file path (.pt).
    model_state : dict
        Model state dict(s). Can contain multiple networks.
    optimizer_state : dict
        Optimizer state dict(s).
    metadata : dict
        Training metadata: timestep, episode, curriculum_stage, best_reward, etc.
    extra : dict, optional
        Extra state (normalizers, replay buffer stats, etc.).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    checkpoint = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "metadata": metadata,
    }
    if extra is not None:
        checkpoint["extra"] = extra
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Load a checkpoint from disk.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    device : torch.device
        Device to map tensors to.

    Returns
    -------
    dict
        Checkpoint dict with keys: model_state, optimizer_state, metadata, extra.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False is required for checkpoints that include optimizer
    # state and metadata objects beyond plain tensor state dicts.
    return torch.load(path, map_location=device, weights_only=False)


def find_latest_checkpoint(directory: str, pattern: str = "checkpoint_*.pt") -> Optional[str]:
    """
    Find the latest checkpoint file in a directory by modification time.

    Parameters
    ----------
    directory : str
        Directory to search.
    pattern : str
        Glob pattern for checkpoint files.

    Returns
    -------
    str or None
        Path to the latest checkpoint, or None if none found.
    """
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files:
        return None
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def save_best_model(
    path: str,
    model_state: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """Save only the model weights (no optimizer) for deployment/evaluation."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({"model_state": model_state, "metadata": metadata}, path)
