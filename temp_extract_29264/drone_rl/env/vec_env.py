"""
Vectorized environment wrappers for parallel data collection.
"""
import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym


def _worker(remote, parent_remote, env_fn):
    """Worker process for SubprocVecEnv."""
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                if done:
                    final_info = info.copy()
                    final_info["terminal_observation"] = obs
                    obs, reset_info = env.reset()
                    info = final_info
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                obs, info = env.reset(**data) if data else env.reset()
                remote.send((obs, info))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "seed":
                env.reset(seed=data)
                remote.send(None)
            elif cmd == "apply_curriculum":
                if hasattr(env, "apply_curriculum"):
                    env.apply_curriculum(data or {})
                remote.send(None)
    except Exception as e:
        remote.send(("error", str(e)))
        env.close()


class SubprocVecEnv:
    """
    Multiprocessing vectorized environment.

    Parameters
    ----------
    env_fns : list of callables
        Each callable returns a gym.Env instance.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        ctx = mp.get_context("fork")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            proc = ctx.Process(target=_worker, args=(work_remote, remote, env_fn), daemon=True)
            proc.start()
            self.processes.append(proc)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    @property
    def num_envs(self) -> int:
        return self.n_envs

    def step(self, actions: np.ndarray):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs_list, rewards, terminateds, truncateds, infos = zip(*results)
        obs = self._stack_obs(obs_list)
        return obs, np.array(rewards, dtype=np.float32), \
               np.array(terminateds, dtype=bool), \
               np.array(truncateds, dtype=bool), list(infos)

    def reset(self, **kwargs):
        for remote in self.remotes:
            remote.send(("reset", kwargs if kwargs else None))
        results = [remote.recv() for remote in self.remotes]
        obs_list, infos = zip(*results)
        return self._stack_obs(obs_list), list(infos)

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for proc in self.processes:
            proc.join()
        self.closed = True

    def apply_curriculum(self, params: Dict):
        for remote in self.remotes:
            remote.send(("apply_curriculum", params))
        for remote in self.remotes:
            remote.recv()

    def _stack_obs(self, obs_list):
        if isinstance(obs_list[0], dict):
            return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        return np.stack(obs_list)


class DummyVecEnv:
    """Single-process sequential vectorized environment for debugging."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    @property
    def num_envs(self) -> int:
        return self.n_envs

    def step(self, actions: np.ndarray):
        obs_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                final_info = info.copy()
                final_info["terminal_observation"] = obs
                obs, reset_info = env.reset()
                info = final_info
            obs_list.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        obs = self._stack_obs(obs_list)
        return obs, np.array(rewards, dtype=np.float32), \
               np.array(terminateds, dtype=bool), \
               np.array(truncateds, dtype=bool), infos

    def reset(self, **kwargs):
        obs_list, infos = [], []
        for env in self.envs:
            obs, info = env.reset(**kwargs)
            obs_list.append(obs)
            infos.append(info)
        return self._stack_obs(obs_list), infos

    def close(self):
        for env in self.envs:
            env.close()

    def apply_curriculum(self, params: Dict):
        for env in self.envs:
            if hasattr(env, "apply_curriculum"):
                env.apply_curriculum(params)

    def _stack_obs(self, obs_list):
        if isinstance(obs_list[0], dict):
            return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        return np.stack(obs_list)


def make_vec_env(
    env_fn: Callable[[], gym.Env],
    n_envs: int = 4,
    use_subproc: bool = True,
) -> "SubprocVecEnv | DummyVecEnv":
    """
    Create a vectorized environment.

    Parameters
    ----------
    env_fn : callable
        Function that creates a single env instance.
    n_envs : int
        Number of parallel environments.
    use_subproc : bool
        If True, use multiprocessing; otherwise use single-process.
    """
    env_fns = [env_fn for _ in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)
