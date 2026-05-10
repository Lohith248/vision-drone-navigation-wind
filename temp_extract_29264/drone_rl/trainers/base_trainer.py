"""
Base trainer with shared training infrastructure.
"""
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import yaml

from drone_rl.utils.device import get_device, log_gpu_usage, print_device_info
from drone_rl.utils.seed import set_global_seed
from drone_rl.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from drone_rl.logging.tb_logger import TBLogger
from drone_rl.logging.csv_logger import CSVLogger
from drone_rl.logging.metrics import MetricsTracker
from drone_rl.curriculum.curriculum import CurriculumManager


class BaseTrainer(ABC):
    """
    Abstract base trainer providing shared infrastructure:
    - Config loading, device setup, seeding
    - Logging (TensorBoard + CSV)
    - Checkpoint save/load/resume
    - Evaluation loop
    - GPU monitoring
    - FPS tracking
    - Curriculum learning integration
    """

    def __init__(self, config: dict):
        self.config = config
        self.algo_name = config.get("algorithm", "unknown")

        # Directories
        self.log_dir = Path(config.get("log_dir", f"runs/{self.algo_name}"))
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = get_device(config.get("device", "auto"))
        print_device_info(self.device)

        # Seed
        self.seed = config.get("seed", 42)
        set_global_seed(self.seed)

        # Loggers
        self.tb_logger = TBLogger(str(self.log_dir / "tb"))
        self.csv_logger = CSVLogger(str(self.log_dir / "metrics.csv"))
        self.metrics = MetricsTracker()

        # Curriculum
        curriculum_cfg = config.get("curriculum", {})
        self.curriculum = CurriculumManager(
            enabled=curriculum_cfg.get("enabled", False),
            stages=curriculum_cfg.get("stages", None),
            patience=curriculum_cfg.get("patience", 5),
        )

        # Training state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.best_reward = -float("inf")
        self._train_start_time = 0
        self._last_log_time = 0

        # Save config
        config_path = self.log_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    @abstractmethod
    def _setup(self) -> None:
        """Initialize env, networks, algorithm, buffers."""
        pass

    @abstractmethod
    def _train_step(self) -> Dict[str, float]:
        """Execute one training iteration. Returns metrics dict."""
        pass

    def train(self, total_timesteps: int) -> None:
        """Main training loop."""
        self._setup()
        self._train_start_time = time.time()
        target_timesteps = self.total_timesteps + total_timesteps

        log_interval = self.config.get("log_interval", 10)
        eval_interval = self.config.get("eval_interval", 50000)
        checkpoint_interval = self.config.get("checkpoint_interval", 100000)
        iteration = 0

        print(f"\n{'='*60}")
        print(f"  Training {self.algo_name.upper()} for {total_timesteps:,} timesteps")
        print(f"  Log dir: {self.log_dir}")
        print(f"{'='*60}\n")

        if self.curriculum.enabled:
            self._apply_curriculum_stage()

        while self.total_timesteps < target_timesteps:
            iter_start = time.time()

            # Train step
            step_metrics = self._train_step()
            iteration += 1

            # Timing
            iter_time = time.time() - iter_start
            fps = step_metrics.get("n_steps", 1) / max(iter_time, 1e-6)
            step_metrics["fps"] = fps
            step_metrics["iteration"] = iteration
            step_metrics["total_timesteps"] = self.total_timesteps

            # Curriculum update
            if self.curriculum.enabled:
                step_metrics["curriculum_stage"] = self.curriculum.current_stage

            # GPU stats
            if self.device.type == "cuda":
                gpu_info = log_gpu_usage()
                step_metrics["gpu_mem_mb"] = gpu_info["allocated_mb"]

            # Logging
            if iteration % log_interval == 0:
                self._log_metrics(step_metrics)

            # Evaluation
            if self.total_timesteps % eval_interval < step_metrics.get("n_steps", 1):
                eval_metrics = self._evaluate()
                if eval_metrics:
                    if self.curriculum.enabled:
                        stage_changed = self.curriculum.maybe_advance(
                            avg_reward=eval_metrics.get("mean_reward", 0.0),
                            success_rate=eval_metrics.get("success_rate", 0.0),
                        )
                        eval_metrics["curriculum_stage"] = self.curriculum.current_stage
                        if stage_changed:
                            self._apply_curriculum_stage()
                            print(
                                f"  [Curriculum] Advanced to stage {self.curriculum.current_stage}: "
                                f"{self.curriculum.current.name}"
                            )
                    self._log_eval_metrics(eval_metrics)
                    if eval_metrics["mean_reward"] > self.best_reward:
                        self.best_reward = eval_metrics["mean_reward"]
                        self._save_best()

            # Checkpoint
            if self.total_timesteps % checkpoint_interval < step_metrics.get("n_steps", 1):
                self._save_checkpoint()

        # Final save
        self._save_checkpoint()
        total_time = (time.time() - self._train_start_time) / 60
        print(f"\n{'='*60}")
        print(f"  Training complete! {self.total_timesteps:,} timesteps in {total_time:.1f} min")
        print(f"  Best reward: {self.best_reward:.2f}")
        print(f"{'='*60}")

        self.tb_logger.close()
        self.csv_logger.close()

    def _evaluate(self, n_episodes: int = None) -> Dict[str, float]:
        """Run evaluation episodes. Override in subclass."""
        return {}

    def _apply_curriculum_stage(self) -> None:
        """Apply current curriculum stage params to train/eval envs when supported."""
        params = self.curriculum.get_env_params()
        for env_name in ("env", "eval_env"):
            env_obj = getattr(self, env_name, None)
            if env_obj is not None and hasattr(env_obj, "apply_curriculum"):
                env_obj.apply_curriculum(params)

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log to TensorBoard and CSV."""
        step = self.total_timesteps
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.tb_logger.log_scalar(f"train/{key}", value, step)
        self.csv_logger.log(metrics)

        # Print summary
        elapsed = (time.time() - self._train_start_time) / 60
        reward = metrics.get("mean_episode_reward", 0)
        fps = metrics.get("fps", 0)
        print(f"  [{self.algo_name.upper()}] step={step:>8,} | "
              f"reward={reward:>7.2f} | fps={fps:>6.0f} | "
              f"time={elapsed:>5.1f}m", flush=True)

    def _log_eval_metrics(self, metrics: Dict[str, float]) -> None:
        step = self.total_timesteps
        for key, value in metrics.items():
            self.tb_logger.log_scalar(f"eval/{key}", value, step)

    @abstractmethod
    def _get_model_state(self) -> dict:
        """Return model state dict for checkpointing."""
        pass

    @abstractmethod
    def _get_optimizer_state(self) -> dict:
        """Return optimizer state dict for checkpointing."""
        pass

    def _save_checkpoint(self) -> None:
        path = str(self.checkpoint_dir / f"checkpoint_{self.total_timesteps}.pt")
        metadata = {
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "curriculum_stage": self.curriculum.current_stage,
            "algorithm": self.algo_name,
        }
        save_checkpoint(path, self._get_model_state(),
                        self._get_optimizer_state(), metadata)
        print(f"  [Checkpoint] Saved → {path}")

    def _save_best(self) -> None:
        path = str(self.checkpoint_dir / "best_model.pt")
        metadata = {
            "total_timesteps": self.total_timesteps,
            "best_reward": self.best_reward,
            "algorithm": self.algo_name,
        }
        save_checkpoint(path, self._get_model_state(), {}, metadata)

    def _try_resume(self) -> bool:
        """Attempt to resume from latest checkpoint."""
        if not self.config.get("auto_resume", True):
            return False
        latest = find_latest_checkpoint(str(self.checkpoint_dir))
        if latest is None:
            return False
        print(f"  [Resume] Loading checkpoint: {latest}")
        ckpt = load_checkpoint(latest, self.device)
        self._load_from_checkpoint(ckpt)
        return True

    def _load_from_checkpoint(self, ckpt: dict) -> None:
        """Override in subclass to restore full state."""
        meta = ckpt.get("metadata", {})
        self.total_timesteps = meta.get("total_timesteps", 0)
        self.total_episodes = meta.get("total_episodes", 0)
        self.best_reward = meta.get("best_reward", -float("inf"))
        self.curriculum.current_stage = meta.get("curriculum_stage", 0)


def load_config(config_path: str, overrides: Optional[dict] = None) -> dict:
    """Load YAML config with optional overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base
