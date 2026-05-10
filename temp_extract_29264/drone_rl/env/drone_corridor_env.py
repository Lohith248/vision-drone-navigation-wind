"""
DroneCorridorEnv — Enhanced Gymnasium Environment for Drone Corridor Navigation
================================================================================

Features:
  - Modular observation space (state-only, image, or combined)
  - Configurable corridor parameters for curriculum learning
  - Comprehensive reward shaping
  - Action scaling: policy outputs [-1, 1], env scales to [-MAX_VEL, MAX_VEL]
  - Raycasting for wall distance sensing
  - PyBullet physics with velocity tracking controller
"""

import time
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from drone_nav_env.wind import OUWindModel
from drone_rl.env.corridor_builder import CorridorBuilder
from drone_rl.env.reward_shaping import RewardShaper, RewardWeights

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

IMG_W, IMG_H = 64, 64
CAM_FOV = 90.0
CAM_NEAR = 0.05
CAM_FAR = 20.0

SIM_FREQ = 240
CTRL_FREQ = 30
STEPS_PER_CTRL = SIM_FREQ // CTRL_FREQ

DRONE_RADIUS = 0.10
DRONE_MASS = 0.027
MAX_VEL = 3.0
MAX_STEPS = 1500
GOAL_RADIUS = 0.5

# Observation dimensions for each component
OBS_WALL_DISTS = 4       # left, right, floor, ceiling
OBS_FWD_CLEARANCE = 1    # forward raycast
OBS_VELOCITY = 3         # vx, vy, vz
OBS_YAW_ERROR = 1        # yaw deviation from corridor axis
OBS_GOAL_DIR = 3         # unit vector to goal
OBS_PREV_ACTION = 3      # last action taken


class DroneCorridorEnv(gym.Env):
    """
    Production-quality drone corridor navigation environment.

    Parameters
    ----------
    obs_mode : str
        Observation mode: "state", "image", or "combined".
    gui : bool
        If True, open PyBullet GUI window.
    wind_enabled : bool
        If True, inject OU wind forces.
    wind_sigma : float
        Wind volatility parameter.
    corridor_width : float
        Corridor width in metres.
    corridor_length : float
        Corridor length in metres.
    obstacle_density : float
        Fraction of obstacles to place (0.0 to 1.0).
    enable_turns : bool
        Enable L-shaped corridor turns.
    sensor_noise : float
        Gaussian noise std to add to observations (curriculum stage 5).
    max_steps : int
        Maximum steps per episode.
    reward_weights : RewardWeights, optional
        Custom reward weights.
    seed : int, optional
        RNG seed.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CTRL_FREQ}

    def __init__(
        self,
        obs_mode: str = "state",
        reward_mode: str = "report",
        gui: bool = False,
        wind_enabled: bool = False,
        wind_sigma: float = 0.3,
        corridor_width: float = 3.0,
        corridor_length: float = 10.0,
        obstacle_density: float = 1.0,
        enable_turns: bool = False,
        enable_moving_obstacles: bool = False,
        sensor_noise: float = 0.0,
        max_steps: int = MAX_STEPS,
        reward_weights: Optional[RewardWeights] = None,
        seed: Optional[int] = None,
        domain_randomization: bool = False,
        dr_mass_range: Tuple[float, float] = (0.7, 1.3),
        dr_damping_range: Tuple[float, float] = (0.6, 1.4),
        dr_wind_sigma_range: Tuple[float, float] = (0.0, 0.6),
        dr_sensor_noise_range: Tuple[float, float] = (0.0, 0.1),
    ):
        super().__init__()

        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.gui = gui
        self.wind_enabled = wind_enabled
        self.sensor_noise = sensor_noise
        self.max_steps = max_steps
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # Domain randomization: per-episode random physical parameters
        self.domain_randomization = bool(domain_randomization)
        self.dr_mass_range = tuple(dr_mass_range)
        self.dr_damping_range = tuple(dr_damping_range)
        self.dr_wind_sigma_range = tuple(dr_wind_sigma_range)
        self.dr_sensor_noise_range = tuple(dr_sensor_noise_range)
        # Sampled DR values (re-rolled each reset)
        self._dr_drone_mass = DRONE_MASS
        self._dr_lin_damping = 0.9
        self._dr_ang_damping = 0.99

        # Corridor builder
        self._corridor = CorridorBuilder(
            length=corridor_length,
            width=corridor_width,
            obstacle_density=obstacle_density,
            enable_turns=enable_turns,
            enable_moving_obstacles=enable_moving_obstacles,
        )

        # Reward shaper
        self._reward_shaper = RewardShaper(
            weights=reward_weights,
            corridor_width=corridor_width,
            goal_radius=GOAL_RADIUS,
            mode=reward_mode,
        )

        # Wind model
        self._wind = OUWindModel(sigma=wind_sigma, seed=seed)

        # Compute state observation dimension
        self._state_dim = (OBS_WALL_DISTS + OBS_FWD_CLEARANCE + OBS_VELOCITY +
                           OBS_YAW_ERROR + OBS_GOAL_DIR + OBS_PREV_ACTION)  # = 15

        # ---- Observation space ----
        if obs_mode == "state":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._state_dim,), dtype=np.float32,
            )
        elif obs_mode == "image":
            self.observation_space = spaces.Dict({
                "image": spaces.Box(0, 255, shape=(IMG_H, IMG_W, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(self._state_dim,), dtype=np.float32),
            })
        elif obs_mode == "combined":
            self.observation_space = spaces.Dict({
                "image": spaces.Box(0, 255, shape=(IMG_H, IMG_W, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(self._state_dim,), dtype=np.float32),
            })
        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}")

        # ---- Action space ----
        # Policy outputs [-1, 1]; we scale to [-MAX_VEL, MAX_VEL] internally
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )

        # ---- Internal state ----
        self._physics_client: Optional[int] = None
        self._drone_id: Optional[int] = None
        self._wall_ids: List[int] = []
        self._obstacle_ids: List[int] = []
        self._step_count = 0
        self._prev_position = np.zeros(3, dtype=np.float32)
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._initial_goal_dist = 1.0

        # Connect PyBullet
        self._connect()

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            self._wind = OUWindModel(sigma=self._wind.sigma, seed=seed)

        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)

        # Domain randomization: re-roll physical params at the START of every episode.
        # NOTE: only applied when explicitly enabled. Otherwise behaviour is unchanged.
        if self.domain_randomization:
            self._dr_drone_mass = float(DRONE_MASS * self._rng.uniform(*self.dr_mass_range))
            self._dr_lin_damping = float(0.9 * self._rng.uniform(*self.dr_damping_range))
            self._dr_ang_damping = float(0.99 * self._rng.uniform(*self.dr_damping_range))
            new_sigma = float(self._rng.uniform(*self.dr_wind_sigma_range))
            self._wind.sigma = new_sigma
            self.wind_enabled = bool(new_sigma > 1e-3)
            self.sensor_noise = float(self._rng.uniform(*self.dr_sensor_noise_range))
        self._wind.reset()
        self._reward_shaper.reset()

        # Reset simulation
        p.resetSimulation(physicsClientId=self._physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._physics_client)
        p.setTimeStep(1.0 / SIM_FREQ, physicsClientId=self._physics_client)

        # Load ground plane
        p.loadURDF("plane.urdf", physicsClientId=self._physics_client)

        # Build corridor
        self._wall_ids, self._obstacle_ids = self._corridor.build(
            self._physics_client, self._rng
        )

        # Spawn drone
        self._drone_id = self._spawn_drone()

        # GUI camera
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=7, cameraYaw=50, cameraPitch=-25,
                cameraTargetPosition=[5.0, 0.0, 1.0],
                physicsClientId=self._physics_client,
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                       physicsClientId=self._physics_client)
            self._draw_goal_marker()

        # Settle physics
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._physics_client)

        pos = self._get_position()
        self._prev_position = pos.copy()
        self._initial_goal_dist = float(
            np.linalg.norm(pos - self._corridor.goal_position)
        )

        obs = self._get_obs()
        info = {"wind": self._wind.current.tolist(), "position": pos.tolist()}
        return obs, info

    def step(self, action: np.ndarray):
        # Scale action from [-1, 1] to [-MAX_VEL, MAX_VEL]
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scaled_action = action * MAX_VEL

        # Apply velocity control + wind over sub-steps
        for _ in range(STEPS_PER_CTRL):
            self._apply_velocity_control(scaled_action)
            if self.wind_enabled:
                self._apply_wind_force()
            p.stepSimulation(physicsClientId=self._physics_client)
            if self.gui:
                time.sleep(1.0 / SIM_FREQ)

        self._step_count += 1

        # Collect post-step info
        pos = self._get_position()
        goal_pos = self._corridor.goal_position
        dist = float(np.linalg.norm(pos - goal_pos))
        collided = self._check_collision()
        wall_dists = self._raycast_wall_distances()

        # Compute reward
        reward_components = self._reward_shaper.compute(
            position=pos,
            prev_position=self._prev_position,
            goal_position=goal_pos,
            action=action,
            collided=collided,
            wall_distances=wall_dists,
            initial_goal_dist=self._initial_goal_dist,
            step_count=self._step_count,
            max_steps=self.max_steps,
        )
        reward = reward_components["total"]

        # Termination
        reached_goal = dist < GOAL_RADIUS
        out_of_bounds = self._is_out_of_bounds(pos)
        terminated = reached_goal or collided or out_of_bounds
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "step": self._step_count,
            "distance": dist,
            "reached_goal": reached_goal,
            "collided": collided,
            "out_of_bounds": out_of_bounds,
            "wind": self._wind.current.tolist(),
            "position": pos.tolist(),
            "reward_components": reward_components,
            "wall_distances": wall_dists.tolist() if wall_dists is not None else None,
            "forward_progress": float(pos[0] - self._prev_position[0]),
            "corridor_deviation": float(abs(pos[1])),
        }

        # Update state
        self._prev_position = pos.copy()
        self._prev_action = action.copy()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self._capture_camera_image()

    def close(self):
        if self._physics_client is not None:
            try:
                p.disconnect(physicsClientId=self._physics_client)
            except p.error:
                pass
            self._physics_client = None

    def apply_curriculum(self, params: Dict[str, Any]) -> None:
        """Apply curriculum parameters. New geometry takes effect on next reset."""
        if "corridor_width" in params:
            width = float(params["corridor_width"])
            self._corridor.width = width
            self._reward_shaper.corridor_width = width
        if "obstacle_density" in params:
            self._corridor.obstacle_density = float(np.clip(params["obstacle_density"], 0.0, 1.0))
        if "enable_turns" in params:
            self._corridor.enable_turns = bool(params["enable_turns"])
        if "enable_moving_obstacles" in params:
            self._corridor.enable_moving_obstacles = bool(params["enable_moving_obstacles"])
        if "sensor_noise" in params:
            self.sensor_noise = float(max(params["sensor_noise"], 0.0))
        if "wind_enabled" in params:
            self.wind_enabled = bool(params["wind_enabled"])
        if "wind_sigma" in params:
            self._wind.sigma = float(max(params["wind_sigma"], 0.0))

    # ==================================================================
    # Observation construction
    # ==================================================================

    def _get_obs(self):
        state_vec = self._get_state_vector()

        # Add sensor noise (curriculum stage 5)
        if self.sensor_noise > 0:
            noise = self._rng.normal(0, self.sensor_noise, size=state_vec.shape)
            state_vec = (state_vec + noise).astype(np.float32)

        if self.obs_mode == "state":
            return state_vec
        else:
            return {
                "image": self._capture_camera_image(),
                "state": state_vec,
            }

    def _get_state_vector(self) -> np.ndarray:
        """
        Build the state observation vector:
          [wall_dists(4), fwd_clearance(1), velocity(3), yaw_error(1),
           goal_dir(3), prev_action(3)]  = 15 dims
        """
        pos, quat = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        pos = np.array(pos, dtype=np.float32)
        lin_vel, _ = p.getBaseVelocity(
            self._drone_id, physicsClientId=self._physics_client
        )
        lin_vel = np.array(lin_vel[:3], dtype=np.float32)
        euler = p.getEulerFromQuaternion(quat)
        yaw = float(euler[2])

        # Wall distances (raycasting)
        wall_dists = self._raycast_wall_distances()

        # Forward clearance
        fwd_clearance = self._raycast_forward_clearance()

        # Velocity (normalised by MAX_VEL)
        velocity = lin_vel / MAX_VEL

        # Yaw error (angle from corridor axis, normalised to [-1, 1])
        yaw_error = np.array([yaw / np.pi], dtype=np.float32)

        # Goal direction (unit vector)
        goal_vec = self._corridor.goal_position - pos
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_dir = (goal_vec / max(goal_dist, 1e-6)).astype(np.float32)

        # Previous action
        prev_act = self._prev_action.copy()

        state = np.concatenate([
            wall_dists,          # 4
            fwd_clearance,       # 1
            velocity,            # 3
            yaw_error,           # 1
            goal_dir,            # 3
            prev_act,            # 3
        ]).astype(np.float32)

        return state

    def _raycast_wall_distances(self) -> np.ndarray:
        """Cast rays to left, right, floor, ceiling walls. Returns distances (4,)."""
        pos = self._get_position()
        hw = self._corridor.width / 2.0
        H = self._corridor.height
        ray_len = max(hw, H) + 1.0

        directions = [
            [0, 1, 0],   # left
            [0, -1, 0],  # right
            [0, 0, -1],  # floor
            [0, 0, 1],   # ceiling
        ]

        distances = np.zeros(4, dtype=np.float32)
        for i, d in enumerate(directions):
            ray_end = pos + np.array(d, dtype=np.float32) * ray_len
            result = p.rayTest(pos.tolist(), ray_end.tolist(),
                               physicsClientId=self._physics_client)
            if result and result[0][0] >= 0:
                hit_pos = np.array(result[0][3], dtype=np.float32)
                distances[i] = float(np.linalg.norm(hit_pos - pos))
            else:
                distances[i] = ray_len

        # Normalise by corridor half-width
        distances = distances / ray_len

        return distances

    def _raycast_forward_clearance(self) -> np.ndarray:
        """Cast a forward ray to detect clearance ahead. Returns distance (1,)."""
        pos = self._get_position()
        _, quat = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        forward = rot_mat[:, 0]  # drone's x-axis

        ray_len = 5.0
        ray_end = pos + forward.astype(np.float32) * ray_len
        result = p.rayTest(pos.tolist(), ray_end.tolist(),
                           physicsClientId=self._physics_client)

        if result and result[0][0] >= 0:
            hit_pos = np.array(result[0][3], dtype=np.float32)
            dist = float(np.linalg.norm(hit_pos - pos))
        else:
            dist = ray_len

        return np.array([dist / ray_len], dtype=np.float32)

    # ==================================================================
    # Physics helpers
    # ==================================================================

    def _connect(self):
        if self.gui:
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._physics_client)

    def _spawn_drone(self) -> int:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=DRONE_RADIUS,
                                     physicsClientId=self._physics_client)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=DRONE_RADIUS,
                                  rgbaColor=[0.1, 0.4, 0.9, 1.0],
                                  physicsClientId=self._physics_client)
        # Mass and damping use DR-rolled values; default to fixed when DR is off.
        mass = self._dr_drone_mass if self.domain_randomization else DRONE_MASS
        lin_damp = self._dr_lin_damping if self.domain_randomization else 0.9
        ang_damp = self._dr_ang_damping if self.domain_randomization else 0.99
        drone_id = p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=self._corridor.spawn_position.tolist(),
            physicsClientId=self._physics_client,
        )
        p.changeDynamics(drone_id, -1, linearDamping=lin_damp, angularDamping=ang_damp,
                         physicsClientId=self._physics_client)
        return drone_id

    def _apply_velocity_control(self, vel_target: np.ndarray):
        lin_vel, _ = p.getBaseVelocity(
            self._drone_id, physicsClientId=self._physics_client
        )
        error = vel_target - np.array(lin_vel)
        force = error * DRONE_MASS * SIM_FREQ * 0.5

        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        p.applyExternalForce(self._drone_id, -1, force.tolist(), pos,
                             p.WORLD_FRAME, physicsClientId=self._physics_client)

    def _apply_wind_force(self):
        wind_vec = self._wind.step()
        force = wind_vec * DRONE_MASS
        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        p.applyExternalForce(self._drone_id, -1, force.tolist(), pos,
                             p.WORLD_FRAME, physicsClientId=self._physics_client)

    def _get_position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        return np.array(pos, dtype=np.float32)

    def _check_collision(self) -> bool:
        contacts = p.getContactPoints(bodyA=self._drone_id,
                                      physicsClientId=self._physics_client)
        return len(contacts) > 0

    def _is_out_of_bounds(self, pos: np.ndarray) -> bool:
        x, y, z = pos
        L = self._corridor.length
        W = self._corridor.width
        H = self._corridor.height
        return (x < -0.5 or x > L + 0.5 or
                abs(y) > W / 2.0 + 0.3 or
                z < 0.0 or z > H + 0.3)

    def _capture_camera_image_at(self, width: int, height: int) -> np.ndarray:
        """RGB frame from the onboard camera at arbitrary resolution (observation path uses 64×64)."""
        pos, quat = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        pos = np.array(pos)
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        forward = rot_mat[:, 0]
        cam_pos = pos + forward * 0.05
        target = pos + forward * 2.0

        view_mat = p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=[0, 0, 1],
            physicsClientId=self._physics_client,
        )
        aspect = float(width) / float(max(height, 1))
        proj_mat = p.computeProjectionMatrixFOV(
            fov=CAM_FOV, aspect=aspect, nearVal=CAM_NEAR, farVal=CAM_FAR,
            physicsClientId=self._physics_client,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=width, height=height, viewMatrix=view_mat,
            projectionMatrix=proj_mat, renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._physics_client,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return rgb

    def _capture_camera_image(self) -> np.ndarray:
        return self._capture_camera_image_at(IMG_W, IMG_H)

    def capture_demo_frame(self, width: int = 960, height: int = 540) -> np.ndarray:
        """Higher-res RGB for recordings; does not affect RL observations."""
        w = max(16, int(width))
        h = max(16, int(height))
        return self._capture_camera_image_at(w, h)

    def _draw_goal_marker(self):
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.15, rgbaColor=[0.0, 1.0, 0.2, 0.6],
            physicsClientId=self._physics_client,
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                          basePosition=self._corridor.goal_position.tolist(),
                          physicsClientId=self._physics_client)
