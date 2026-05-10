"""
PPO Trainer: on-policy rollout collection + PPO update loop.
"""
from typing import Dict, Union
import numpy as np
import torch

from drone_rl.trainers.base_trainer import BaseTrainer
from drone_rl.env.drone_corridor_env import DroneCorridorEnv
from drone_rl.env.vec_env import DummyVecEnv, SubprocVecEnv
from drone_rl.env.reward_shaping import RewardWeights
from drone_rl.networks.actor_critic import ActorCritic
from drone_rl.algorithms.ppo import PPOAlgorithm
from drone_rl.replay_buffers.rollout_buffer import RolloutBuffer
from drone_rl.utils.normalization import ObservationNormalizer
from drone_rl.utils.schedule import LinearSchedule


class PPOTrainer(BaseTrainer):
    """PPO training loop with vectorized environments and on-policy rollouts."""

    def __init__(self, config: dict):
        config.setdefault("algorithm", "ppo")
        super().__init__(config)
        self.env = None
        self.eval_env = None
        self.policy = None
        self.algorithm = None
        self.rollout_buffer = None
        self.obs_normalizer = None
        self.lr_schedule = None
        self._dict_obs = False

    def _setup(self) -> None:
        cfg = self.config
        ppo_cfg = cfg.get("ppo", {})

        # Environment
        env_cfg = cfg.get("env", {})
        n_envs = cfg.get("num_envs", 4)
        reward_weights = self._build_reward_weights(cfg.get("reward", {}))

        def make_env(rank: int):
            return lambda: DroneCorridorEnv(
                obs_mode=env_cfg.get("obs_mode", "state"),
                reward_mode=env_cfg.get("reward_mode", "report"),
                gui=False,
                wind_enabled=env_cfg.get("wind_enabled", False),
                wind_sigma=env_cfg.get("wind_sigma", 0.3),
                corridor_width=env_cfg.get("corridor_width", 3.0),
                corridor_length=env_cfg.get("corridor_length", 10.0),
                obstacle_density=env_cfg.get("obstacle_density", 1.0),
                enable_turns=env_cfg.get("enable_turns", False),
                enable_moving_obstacles=env_cfg.get("enable_moving_obstacles", False),
                sensor_noise=env_cfg.get("sensor_noise", 0.0),
                max_steps=env_cfg.get("max_steps", 1500),
                reward_weights=reward_weights,
                seed=self.seed + rank,
                domain_randomization=env_cfg.get("domain_randomization", False),
            )

        use_subproc = cfg.get("use_subproc", False)
        env_fns = [make_env(i) for i in range(n_envs)]
        if use_subproc and n_envs > 1:
            self.env = SubprocVecEnv(env_fns)
        else:
            self.env = DummyVecEnv(env_fns)
        self.eval_env = DummyVecEnv([make_env(10_000)])

        # Determine obs dim
        sample_obs, _ = self.eval_env.reset()
        if isinstance(sample_obs, dict):
            self._dict_obs = True
            obs_dim = sample_obs["state"].shape[-1]
            obs_specs = {
                "state": {
                    "shape": tuple(sample_obs["state"].shape[1:]),
                    "dtype": np.float32,
                },
                "image": {
                    "shape": tuple(sample_obs["image"].shape[1:]),
                    "dtype": np.uint8,
                },
            }
        else:
            self._dict_obs = False
            obs_dim = sample_obs.shape[-1]
            obs_specs = None

        # Policy network
        hidden_dims = tuple(ppo_cfg.get("hidden_dims", [256, 256]))
        activation = ppo_cfg.get("activation", "tanh")
        if self._dict_obs:
            self.policy = ActorCritic(
                action_dim=3,
                hidden_dims=hidden_dims,
                log_std_init=ppo_cfg.get("log_std_init", -0.5),
                activation=activation,
                multimodal=True,
                state_dim=obs_dim,
                state_hidden_dims=tuple(ppo_cfg.get("state_hidden_dims", [128])),
                image_encoder=ppo_cfg.get("image_encoder", "vit"),
                image_feature_dim=ppo_cfg.get("image_feature_dim", 192),
                vit_patch_size=ppo_cfg.get("vit_patch_size", 8),
                vit_embed_dim=ppo_cfg.get("vit_embed_dim", 128),
                vit_depth=ppo_cfg.get("vit_depth", 4),
                vit_heads=ppo_cfg.get("vit_heads", 4),
                vit_dropout=ppo_cfg.get("vit_dropout", 0.0),
            )
        else:
            self.policy = ActorCritic(
                obs_dim=obs_dim, action_dim=3,
                hidden_dims=hidden_dims,
                log_std_init=ppo_cfg.get("log_std_init", -0.5),
                activation=activation,
            )
        print(f"  PPO params: {sum(p.numel() for p in self.policy.parameters()):,}")

        # Algorithm
        lr = ppo_cfg.get("lr", 3e-4)
        self.algorithm = PPOAlgorithm(
            policy=self.policy, lr=lr,
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.01),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            use_amp=cfg.get("use_amp", True),
            normalize_advantages=ppo_cfg.get("normalize_advantages", True),
            device=self.device,
        )

        # Rollout buffer
        n_steps = ppo_cfg.get("n_steps", 2048)
        rollout_kwargs = {
            "n_steps": n_steps,
            "n_envs": n_envs,
            "action_dim": 3,
            "gamma": ppo_cfg.get("gamma", 0.99),
            "gae_lambda": ppo_cfg.get("gae_lambda", 0.95),
        }
        if self._dict_obs:
            rollout_kwargs["obs_specs"] = obs_specs
        else:
            rollout_kwargs["obs_dim"] = obs_dim
        self.rollout_buffer = RolloutBuffer(**rollout_kwargs)

        # Observation normalizer
        if ppo_cfg.get("normalize_obs", True):
            self.obs_normalizer = ObservationNormalizer(obs_dim)

        # LR schedule
        lr_end = ppo_cfg.get("lr_end", 0.0)
        total_ts = cfg.get("total_timesteps", 1_000_000)
        self.lr_schedule = LinearSchedule(lr, lr_end)
        self._total_target = total_ts

        # Minibatch size
        self._batch_size = ppo_cfg.get("batch_size", 512)
        self._n_steps = n_steps
        self._n_envs = n_envs

        # Try resume
        self._try_resume()

    def _train_step(self) -> Dict[str, float]:
        """Collect rollout, compute GAE, run PPO update."""
        # Collect rollout
        obs, _ = self.env.reset()
        proc_obs = self._prepare_obs(obs, update_normalizer=True)

        ep_rewards = []
        ep_lengths = []
        collisions = 0
        forward_prog = 0.0
        corridor_dev = 0.0

        self.rollout_buffer.reset()

        with torch.no_grad():
            for step in range(self._n_steps):
                obs_t = self._to_torch_obs(proc_obs)
                action, log_prob, _, value = self.policy.get_action_and_value(obs_t)
                action_np = action.cpu().numpy()
                log_prob_np = log_prob.cpu().numpy()
                value_np = value.cpu().numpy()

                next_obs, rewards, terminateds, truncateds, infos = self.env.step(action_np)
                dones = terminateds | truncateds
                next_proc_obs = self._prepare_obs(next_obs, update_normalizer=True)

                self.rollout_buffer.add(proc_obs, action_np, rewards,
                                         value_np, log_prob_np, dones)

                # Track episode stats
                for i, info in enumerate(infos):
                    if dones[i]:
                        self.total_episodes += 1
                        ep_r = info.get("episode_reward", rewards[i])
                        ep_rewards.append(ep_r)
                        ep_lengths.append(info.get("step", 1))
                        if info.get("collided", False):
                            collisions += 1
                    forward_prog += info.get("forward_progress", 0)
                    corridor_dev += info.get("corridor_deviation", 0)

                proc_obs = next_proc_obs

            # Bootstrap value for GAE
            obs_t = self._to_torch_obs(proc_obs)
            last_values = self.policy.get_value(obs_t).cpu().numpy()
            last_dones = dones.astype(np.float32)

        # Compute GAE
        self.rollout_buffer.compute_gae(last_values, last_dones)

        # PPO update
        progress = self.total_timesteps / max(self._total_target, 1)
        current_lr = self.lr_schedule(progress)
        train_metrics = self.algorithm.update(
            self.rollout_buffer, batch_size=self._batch_size, lr=current_lr
        )

        # Update timestep count
        n_collected = self._n_steps * self._n_envs
        self.total_timesteps += n_collected

        # Release GPU memory
        torch.cuda.empty_cache() if self.device.type == "cuda" else None

        # Compile metrics
        metrics = {
            "n_steps": n_collected,
            "mean_episode_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "mean_episode_length": float(np.mean(ep_lengths)) if ep_lengths else 0,
            "collision_rate": collisions / max(len(ep_rewards), 1),
            "forward_progress": forward_prog / n_collected,
            "corridor_deviation": corridor_dev / n_collected,
            "learning_rate": current_lr,
        }
        metrics.update(train_metrics)
        return metrics

    def _evaluate(self, n_episodes: int = None) -> Dict[str, float]:
        n_episodes = n_episodes or self.config.get("eval_episodes", 10)
        rewards, lengths = [], []
        successes = 0
        collisions = 0
        out_of_bounds = 0
        timeouts = 0
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            proc_obs = self._prepare_obs(obs, update_normalizer=False)
            done = False
            ep_reward = 0.0
            steps = 0
            terminal_info = {}
            terminal_truncated = False
            while not done:
                obs_t = self._to_torch_obs(proc_obs)
                with torch.no_grad():
                    action = self.policy.get_deterministic_action(obs_t)
                next_obs, reward, terminated, truncated, info = self.eval_env.step(action.cpu().numpy())
                done = (terminated | truncated).any()
                ep_reward += float(reward.sum())
                steps += 1
                if done:
                    terminal_info = info[0] if isinstance(info, list) and info else {}
                    terminal_truncated = bool(truncated[0]) if hasattr(truncated, "__len__") else bool(truncated)
                proc_obs = self._prepare_obs(next_obs, update_normalizer=False)
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

    def _extract_state(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            return obs["state"]
        return obs

    def _prepare_obs(self, obs, update_normalizer: bool) -> Union[np.ndarray, dict]:
        if isinstance(obs, dict):
            state = obs["state"].astype(np.float32, copy=False)
            if self.obs_normalizer:
                if update_normalizer:
                    self.obs_normalizer.update(state)
                state = self.obs_normalizer.normalize(state)
            image = obs["image"]
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return {"state": state.astype(np.float32, copy=False), "image": image}

        state = obs.astype(np.float32, copy=False)
        if self.obs_normalizer:
            if update_normalizer:
                self.obs_normalizer.update(state)
            state = self.obs_normalizer.normalize(state)
        return state.astype(np.float32, copy=False)

    def _to_torch_obs(self, obs):
        if isinstance(obs, dict):
            return {
                "state": torch.from_numpy(obs["state"]).to(self.device),
                "image": torch.from_numpy(obs["image"]).to(self.device),
            }
        return torch.from_numpy(obs).to(self.device)

    def _get_model_state(self) -> dict:
        state = {"policy": self.policy.state_dict()}
        if self.obs_normalizer:
            state["obs_normalizer"] = self.obs_normalizer.state_dict()
        return state

    def _get_optimizer_state(self) -> dict:
        return {"optimizer": self.algorithm.optimizer.state_dict()}

    def _load_from_checkpoint(self, ckpt: dict) -> None:
        super()._load_from_checkpoint(ckpt)
        model = ckpt.get("model_state", {})
        if "policy" in model:
            self.policy.load_state_dict(model["policy"])
        if "obs_normalizer" in model and self.obs_normalizer:
            self.obs_normalizer.load_state_dict(model["obs_normalizer"])
        opt = ckpt.get("optimizer_state", {})
        if "optimizer" in opt:
            self.algorithm.optimizer.load_state_dict(opt["optimizer"])

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
