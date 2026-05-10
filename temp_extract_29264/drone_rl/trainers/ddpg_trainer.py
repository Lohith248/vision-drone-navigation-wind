"""
DDPG Trainer: off-policy deterministic actor-critic with replay buffer.
"""
from typing import Dict
import numpy as np

from drone_rl.trainers.base_trainer import BaseTrainer
from drone_rl.env.drone_corridor_env import DroneCorridorEnv
from drone_rl.env.vec_env import DummyVecEnv
from drone_rl.env.reward_shaping import RewardWeights
from drone_rl.networks.ddpg_networks import DDPGActor, DDPGCritic
from drone_rl.algorithms.ddpg import DDPGAlgorithm
from drone_rl.replay_buffers.replay_buffer import ReplayBuffer
from drone_rl.utils.normalization import ObservationNormalizer
from drone_rl.utils.schedule import LinearSchedule


class DDPGTrainer(BaseTrainer):
    """DDPG training loop with replay buffer and Gaussian action noise."""

    def __init__(self, config: dict):
        config.setdefault("algorithm", "ddpg")
        super().__init__(config)
        self.env = None
        self.eval_env = None
        self.algorithm = None
        self.replay_buffer = None
        self.obs_normalizer = None
        self.noise_schedule = None
        self._obs_dim = 0
        self._current_obs = None
        self._episode_reward = 0.0
        self._episode_steps = 0

    def _setup(self) -> None:
        cfg = self.config
        ddpg_cfg = cfg.get("ddpg", {})
        env_cfg = cfg.get("env", {})
        reward_weights = self._build_reward_weights(cfg.get("reward", {}))

        def make_env():
            return DroneCorridorEnv(
                obs_mode=env_cfg.get("obs_mode", "state"),
                reward_mode=env_cfg.get("reward_mode", "report"),
                gui=False,
                wind_enabled=env_cfg.get("wind_enabled", False),
                wind_sigma=env_cfg.get("wind_sigma", 0.3),
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

        sample_obs, _ = self.eval_env.reset()
        self._obs_dim = self._extract_state(sample_obs).shape[-1]

        hidden_dims = tuple(ddpg_cfg.get("hidden_dims", [256, 256]))
        actor = DDPGActor(self._obs_dim, action_dim=3, hidden_dims=hidden_dims)
        critic = DDPGCritic(self._obs_dim, action_dim=3, hidden_dims=hidden_dims)
        total_params = sum(p.numel() for p in actor.parameters()) + sum(
            p.numel() for p in critic.parameters()
        )
        print(f"  DDPG params: {total_params:,}")

        self.algorithm = DDPGAlgorithm(
            actor=actor,
            critic=critic,
            action_dim=3,
            lr_actor=ddpg_cfg.get("lr_actor", 1e-4),
            lr_critic=ddpg_cfg.get("lr_critic", 1e-3),
            gamma=ddpg_cfg.get("gamma", 0.99),
            tau=ddpg_cfg.get("tau", 0.005),
            max_grad_norm=ddpg_cfg.get("max_grad_norm", 1.0),
            use_amp=cfg.get("use_amp", True),
            device=self.device,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=ddpg_cfg.get("buffer_size", 100000),
            obs_dim=self._obs_dim,
            action_dim=3,
            action_dtype="float32",
        )

        if ddpg_cfg.get("normalize_obs", True):
            self.obs_normalizer = ObservationNormalizer(self._obs_dim)

        self._batch_size = ddpg_cfg.get("batch_size", 256)
        self._learning_starts = ddpg_cfg.get("learning_starts", 1000)
        self._train_freq = ddpg_cfg.get("train_freq", 1)
        self._gradient_steps = ddpg_cfg.get("gradient_steps", 1)
        self._steps_per_iteration = ddpg_cfg.get("steps_per_iteration", 1000)
        self._total_target = cfg.get("total_timesteps", 500000)

        self.noise_schedule = LinearSchedule(
            start=ddpg_cfg.get("noise_start", 0.2),
            end=ddpg_cfg.get("noise_end", 0.05),
        )

        obs, _ = self.env.reset()
        self._current_obs = self._extract_state(obs)
        self._try_resume()

    def _train_step(self) -> Dict[str, float]:
        ep_rewards = []
        ep_lengths = []
        collisions = 0
        train_metrics_agg = {}

        for _ in range(self._steps_per_iteration):
            obs = (
                self._current_obs.squeeze(0)
                if self._current_obs.ndim > 1
                else self._current_obs
            )

            if self.obs_normalizer:
                self.obs_normalizer.update(obs.reshape(1, -1))
                obs_norm = self.obs_normalizer.normalize(obs)
            else:
                obs_norm = obs

            progress = self.total_timesteps / max(self._total_target, 1)
            noise_std = self.noise_schedule(progress)

            if self.total_timesteps < self._learning_starts:
                action = np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
            else:
                action = self.algorithm.select_action(
                    obs_norm, noise_std=noise_std, deterministic=False
                )

            next_obs_raw, reward, terminated, truncated, infos = self.env.step(
                action.reshape(1, -1)
            )
            done = (terminated | truncated).any()
            next_obs = self._extract_state(next_obs_raw)
            next_obs_flat = next_obs.squeeze(0) if next_obs.ndim > 1 else next_obs

            self.replay_buffer.push(
                obs, action, float(reward[0]), next_obs_flat, bool(done)
            )

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

            if (
                self.total_timesteps >= self._learning_starts
                and self.total_timesteps % self._train_freq == 0
                and self.replay_buffer.is_ready(self._batch_size)
            ):
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
            "exploration_noise": noise_std,
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
                action = self.algorithm.select_action(
                    obs_norm, noise_std=0.0, deterministic=True
                )
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
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(successes / max(n_episodes, 1)),
            "collision_rate": float(collisions / max(n_episodes, 1)),
            "timeout_rate": float(timeouts / max(n_episodes, 1)),
            "oob_rate": float(out_of_bounds / max(n_episodes, 1)),
        }

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
        if "obs_normalizer" in model and self.obs_normalizer:
            self.obs_normalizer.load_state_dict(model["obs_normalizer"])

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
