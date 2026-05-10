"""
SAC Trainer: off-policy with replay buffer, single-step collection.
"""
from typing import Dict
import numpy as np
import torch

from drone_rl.trainers.base_trainer import BaseTrainer
from drone_rl.env.drone_corridor_env import DroneCorridorEnv
from drone_rl.env.vec_env import DummyVecEnv
from drone_rl.env.reward_shaping import RewardWeights
from drone_rl.networks.sac_networks import SACActor, SACTwinQ
from drone_rl.algorithms.sac import SACAlgorithm
from drone_rl.replay_buffers.replay_buffer import ReplayBuffer
from drone_rl.utils.normalization import ObservationNormalizer


class SACTrainer(BaseTrainer):
    """SAC training loop with replay buffer and automatic entropy tuning."""

    def __init__(self, config: dict):
        config.setdefault("algorithm", "sac")
        super().__init__(config)
        self.env = None
        self.eval_env = None
        self.algorithm = None
        self.replay_buffer = None
        self.obs_normalizer = None
        self._obs_dim = 0
        self._current_obs = None
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_rewards_history = []

    def _setup(self) -> None:
        cfg = self.config
        sac_cfg = cfg.get("sac", {})
        env_cfg = cfg.get("env", {})
        reward_weights = self._build_reward_weights(cfg.get("reward", {}))

        def make_env():
            return DroneCorridorEnv(
                obs_mode=env_cfg.get("obs_mode", "state"),
                reward_mode=env_cfg.get("reward_mode", "report"),
                gui=False,
                wind_enabled=env_cfg.get("wind_enabled", False),
                corridor_width=env_cfg.get("corridor_width", 3.0),
                corridor_length=env_cfg.get("corridor_length", 10.0),
                obstacle_density=env_cfg.get("obstacle_density", 1.0),
                sensor_noise=env_cfg.get("sensor_noise", 0.0),
                max_steps=env_cfg.get("max_steps", 1500),
                reward_weights=reward_weights,
                seed=self.seed,
            )

        self.env = DummyVecEnv([make_env])
        self.eval_env = DummyVecEnv([make_env])

        # Obs dim
        sample_obs, _ = self.eval_env.reset()
        self._obs_dim = self._extract_state(sample_obs).shape[-1]

        # Networks
        hidden_dims = tuple(sac_cfg.get("hidden_dims", [256, 256]))
        actor = SACActor(self._obs_dim, action_dim=3, hidden_dims=hidden_dims)
        critic = SACTwinQ(self._obs_dim, action_dim=3, hidden_dims=hidden_dims)
        total_params = sum(p.numel() for p in actor.parameters()) + \
                       sum(p.numel() for p in critic.parameters())
        print(f"  SAC params: {total_params:,}")

        # Algorithm
        self.algorithm = SACAlgorithm(
            actor=actor, critic=critic,
            obs_dim=self._obs_dim, action_dim=3,
            lr_actor=sac_cfg.get("lr_actor", 3e-4),
            lr_critic=sac_cfg.get("lr_critic", 3e-4),
            lr_alpha=sac_cfg.get("lr_alpha", 3e-4),
            gamma=sac_cfg.get("gamma", 0.99),
            tau=sac_cfg.get("tau", 0.005),
            use_amp=cfg.get("use_amp", True),
            device=self.device,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=sac_cfg.get("buffer_size", 100000),
            obs_dim=self._obs_dim, action_dim=3,
            action_dtype="float32",
        )
        print(f"  Replay buffer: {self.replay_buffer.memory_usage_mb():.1f} MB")

        # Obs normalizer
        if sac_cfg.get("normalize_obs", True):
            self.obs_normalizer = ObservationNormalizer(self._obs_dim)

        self._batch_size = sac_cfg.get("batch_size", 256)
        self._learning_starts = sac_cfg.get("learning_starts", 1000)
        self._train_freq = sac_cfg.get("train_freq", 1)
        self._gradient_steps = sac_cfg.get("gradient_steps", 1)
        self._steps_per_iteration = sac_cfg.get("steps_per_iteration", 1000)

        # Init env
        obs, _ = self.env.reset()
        self._current_obs = self._extract_state(obs)
        self._try_resume()

    def _train_step(self) -> Dict[str, float]:
        """Collect experience and train."""
        ep_rewards = []
        ep_lengths = []
        collisions = 0
        train_metrics_agg = {}

        for step in range(self._steps_per_iteration):
            obs = self._current_obs.squeeze(0) if self._current_obs.ndim > 1 else self._current_obs

            if self.obs_normalizer:
                self.obs_normalizer.update(obs.reshape(1, -1))
                obs_norm = self.obs_normalizer.normalize(obs)
            else:
                obs_norm = obs

            # Select action
            if self.total_timesteps < self._learning_starts:
                action = np.random.uniform(-1, 1, size=(3,)).astype(np.float32)
            else:
                action = self.algorithm.select_action(obs_norm, deterministic=False)

            # Step environment
            next_obs_raw, reward, terminated, truncated, infos = self.env.step(
                action.reshape(1, -1)
            )
            done = (terminated | truncated).any()
            next_obs = self._extract_state(next_obs_raw)
            next_obs_flat = next_obs.squeeze(0) if next_obs.ndim > 1 else next_obs

            # Store transition
            self.replay_buffer.push(obs, action, float(reward[0]),
                                     next_obs_flat, bool(done))

            self._episode_reward += float(reward[0])
            self._episode_steps += 1
            self.total_timesteps += 1

            if done:
                self.total_episodes += 1
                ep_rewards.append(self._episode_reward)
                ep_lengths.append(self._episode_steps)
                info = infos[0] if infos else {}
                if info.get("collided", False):
                    collisions += 1
                self._episode_reward = 0.0
                self._episode_steps = 0
                obs_reset, _ = self.env.reset()
                self._current_obs = self._extract_state(obs_reset)
            else:
                self._current_obs = next_obs

            # Train
            if (self.total_timesteps >= self._learning_starts and
                    self.total_timesteps % self._train_freq == 0 and
                    self.replay_buffer.is_ready(self._batch_size)):
                for _ in range(self._gradient_steps):
                    batch = self.replay_buffer.sample(self._batch_size, self.device)
                    metrics = self.algorithm.update(batch)
                    for k, v in metrics.items():
                        train_metrics_agg.setdefault(k, []).append(v)

        result = {
            "n_steps": self._steps_per_iteration,
            "mean_episode_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "mean_episode_length": float(np.mean(ep_lengths)) if ep_lengths else 0,
            "collision_rate": collisions / max(len(ep_rewards), 1),
            "buffer_size": len(self.replay_buffer),
        }
        for k, v in train_metrics_agg.items():
            result[k] = float(np.mean(v))
        return result

    def _evaluate(self, n_episodes: int = None) -> Dict[str, float]:
        n_episodes = n_episodes or self.config.get("eval_episodes", 10)
        rewards = []
        lengths = []
        successes = 0
        collisions = 0
        out_of_bounds = 0
        timeouts = 0
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            obs_flat = self._extract_state(obs).squeeze(0)
            done = False
            ep_reward = 0.0
            steps = 0
            terminal_info = {}
            terminal_truncated = False
            while not done:
                if self.obs_normalizer:
                    obs_norm = self.obs_normalizer.normalize(obs_flat)
                else:
                    obs_norm = obs_flat
                action = self.algorithm.select_action(obs_norm, deterministic=True)
                next_obs, reward, terminated, truncated, info = self.eval_env.step(
                    action.reshape(1, -1)
                )
                done = (terminated | truncated).any()
                ep_reward += float(reward[0])
                steps += 1
                if done:
                    terminal_info = info[0] if isinstance(info, list) and info else {}
                    terminal_truncated = bool(truncated[0]) if hasattr(truncated, "__len__") else bool(truncated)
                obs_flat = self._extract_state(next_obs).squeeze(0)
            rewards.append(ep_reward)
            lengths.append(steps)
            reached_goal = bool(terminal_info.get("reached_goal", False))
            collided = bool(terminal_info.get("collided", False))
            oob = bool(terminal_info.get("out_of_bounds", False))
            successes += int(reached_goal)
            collisions += int(collided)
            out_of_bounds += int(oob)
            if terminal_truncated and not reached_goal and not collided:
                timeouts += 1
        return {"mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_length": float(np.mean(lengths)),
                "success_rate": float(successes / max(n_episodes, 1)),
                "collision_rate": float(collisions / max(n_episodes, 1)),
                "timeout_rate": float(timeouts / max(n_episodes, 1)),
                "oob_rate": float(out_of_bounds / max(n_episodes, 1))}

    def _extract_state(self, obs):
        if isinstance(obs, dict):
            return obs["state"]
        return obs

    def _get_model_state(self) -> dict:
        state = self.algorithm.state_dict()
        if self.obs_normalizer:
            state["obs_normalizer"] = self.obs_normalizer.state_dict()
        return state

    def _get_optimizer_state(self) -> dict:
        return {}

    def _load_from_checkpoint(self, ckpt: dict) -> None:
        super()._load_from_checkpoint(ckpt)
        model = ckpt.get("model_state", {})
        self.algorithm.load_state_dict(model)

    def _build_reward_weights(self, reward_cfg: dict) -> RewardWeights:
        return RewardWeights(
            forward_progress=reward_cfg.get("forward_progress", 1.0),
            distance_scale=reward_cfg.get("distance_scale", 100.0),
            centering=reward_cfg.get("centering", 0.5),
            smoothness=reward_cfg.get("smoothness", 0.1),
            goal_reached=reward_cfg.get("goal_reached", 10.0),
            collision=reward_cfg.get("collision", -10.0),
            timeout_penalty=reward_cfg.get("timeout_penalty", -10.0),
            wall_proximity=reward_cfg.get("wall_proximity", -0.5),
            oscillation=reward_cfg.get("oscillation", -0.05),
            control_jump=reward_cfg.get("control_jump", -0.05),
            time_penalty=reward_cfg.get("time_penalty", -0.001),
        )
