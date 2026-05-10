"""
DroneNavEnv — Vision-Based Continuous Drone Navigation in Wind Environments
===========================================================================
Implements the POMDP described in the project proposal:

    Observation  : o_t = [I_t (64×64×3 RGB), s_low_t (position, velocity, yaw)]
    Action       : a_t = (v_x, v_y, v_z) ∈ ℝ³  continuous velocity targets
    Reward       : R_t = r_distance + r_goal + r_time + r_timeout + r_collision
    Disturbance  : Ornstein-Uhlenbeck wind injected as external forces every step
    Physics      : PyBullet (DIRECT or GUI)
    Interface    : OpenAI Gymnasium (gym.Env)
"""

import time
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from drone_nav_env.corridor import (
    CORRIDOR_HEIGHT,
    CORRIDOR_LENGTH,
    CORRIDOR_WIDTH,
    GOAL_POSITION,
    SPAWN_POSITION,
    build_corridor,
)
from drone_nav_env.wind import OUWindModel

# -----------------------------------------------------------------------
# Environment constants
# -----------------------------------------------------------------------

# Camera
IMG_W, IMG_H = 64, 64  # pixels (as per proposal)
CAM_FOV = 90.0  # degrees
CAM_NEAR = 0.05  # metres
CAM_FAR = 20.0  # metres

# Low-dimensional state: [x, y, z,  vx, vy, vz,  yaw]
LOW_DIM = 7

# Physics
SIM_FREQ = 240  # Hz
CTRL_FREQ = 30  # Hz  (policy step rate)
STEPS_PER_CTRL = SIM_FREQ // CTRL_FREQ

# Drone body (simple sphere approximation for collision)
DRONE_RADIUS = 0.10  # metres

# Episode limits
MAX_STEPS = 1500  # ~50 s at 30 Hz

# Velocity limits for action clipping
MAX_VEL = 3.0  # m/s per axis

# Reward coefficients
R_GOAL = 10.0  # sparse bonus on reaching goal
R_TIMEOUT = -10.0  # penalty for time-out
R_COLLISION = -10.0  # penalty per collision event
R_TIME = -0.001  # small per-step time penalty
GOAL_RADIUS = 0.5  # metres — "close enough" threshold

# Drone mass (kg) — used to convert wind force
DRONE_MASS = 0.027  # ~Crazyflie 2.x mass


class DroneNavEnv(gym.Env):
    """
    Custom Gymnasium environment for vision-based indoor drone navigation
    with Ornstein-Uhlenbeck wind disturbances.

    Parameters
    ----------
    gui : bool
        If True, open the PyBullet GUI window.
    wind_enabled : bool
        If True, inject OU wind forces every simulation step.
    seed : Optional[int]
        RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CTRL_FREQ}

    # ------------------------------------------------------------------
    def __init__(
        self,
        gui: bool = False,
        wind_enabled: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.gui = gui
        self.wind_enabled = wind_enabled
        self._seed = seed

        # ---- Observation space ----
        # Dict space:  "image" + "state"
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(IMG_H, IMG_W, 3),
                    dtype=np.uint8,
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(LOW_DIM,),
                    dtype=np.float32,
                ),
            }
        )

        # ---- Action space ----
        # Continuous (vx, vy, vz) velocity targets clipped to ±MAX_VEL
        self.action_space = spaces.Box(
            low=-MAX_VEL,
            high=MAX_VEL,
            shape=(3,),
            dtype=np.float32,
        )

        # ---- Internal state ----
        self._physics_client: Optional[int] = None
        self._drone_id: Optional[int] = None
        self._obstacle_ids: List[int] = []
        self._wind = OUWindModel(seed=seed)
        self._step_count = 0
        self._prev_dist_to_goal = 0.0
        self._initial_dist_to_goal = float(
            np.linalg.norm(SPAWN_POSITION - GOAL_POSITION)
        )

        # Connect PyBullet
        self._connect()

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment and return initial observation."""
        if seed is not None:
            self._seed = seed
            self._wind = OUWindModel(seed=seed)

        self._step_count = 0
        self._wind.reset()

        # Reset simulation
        p.resetSimulation(physicsClientId=self._physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._physics_client
        )
        p.setTimeStep(1.0 / SIM_FREQ, physicsClientId=self._physics_client)

        # Load ground plane (hidden below the corridor floor)
        p.loadURDF("plane.urdf", physicsClientId=self._physics_client)

        # Build the indoor corridor
        self._obstacle_ids = build_corridor(self._physics_client)

        # Spawn drone (represented as a small sphere)
        self._drone_id = self._spawn_drone()

        # Camera setup (GUI only)
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=7,
                cameraYaw=50,
                cameraPitch=-25,
                cameraTargetPosition=[5.0, 0.0, 1.0],
                physicsClientId=self._physics_client,
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0, physicsClientId=self._physics_client
            )
            # Draw goal marker
            self._draw_goal_marker()

        # Settle physics for a few steps
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._physics_client)

        self._prev_dist_to_goal = float(
            np.linalg.norm(self._get_position() - GOAL_POSITION)
        )

        obs = self._get_obs()
        info = {"wind": self._wind.current.tolist()}
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """
        Apply velocity action, step the simulation, return (obs, reward, terminated, truncated, info).
        """
        action = np.clip(action, -MAX_VEL, MAX_VEL).astype(np.float64)

        # --- Apply velocity control + wind over multiple sub-steps ---
        for _ in range(STEPS_PER_CTRL):
            self._apply_velocity_control(action)
            if self.wind_enabled:
                self._apply_wind_force()
            p.stepSimulation(physicsClientId=self._physics_client)
            if self.gui:
                time.sleep(1.0 / SIM_FREQ)

        self._step_count += 1

        # --- Collect post-step info ---
        pos = self._get_position()
        dist = float(np.linalg.norm(pos - GOAL_POSITION))
        collided = self._check_collision()

        # --- Reward ---
        reward = self._compute_reward(dist, collided)

        # --- Termination conditions ---
        reached_goal = dist < GOAL_RADIUS
        out_of_bounds = self._is_out_of_bounds(pos)
        terminated = reached_goal or collided or out_of_bounds
        truncated = self._step_count >= MAX_STEPS

        obs = self._get_obs()
        info = {
            "step": self._step_count,
            "distance": dist,
            "reached_goal": reached_goal,
            "collided": collided,
            "wind": self._wind.current.tolist(),
        }

        # Update previous distance for next step's shaping
        self._prev_dist_to_goal = dist

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        """Return the front-facing camera image (rgb_array mode)."""
        return self._capture_camera_image()

    # ------------------------------------------------------------------
    def close(self):
        if self._physics_client is not None:
            p.disconnect(physicsClientId=self._physics_client)
            self._physics_client = None

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _connect(self):
        """Connect to the PyBullet physics server."""
        if self.gui:
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._physics_client
        )

    # ------------------------------------------------------------------
    def _spawn_drone(self) -> int:
        """Spawn the drone as a small coloured sphere at SPAWN_POSITION."""
        col = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=DRONE_RADIUS,
            physicsClientId=self._physics_client,
        )
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=DRONE_RADIUS,
            rgbaColor=[0.1, 0.4, 0.9, 1.0],
            physicsClientId=self._physics_client,
        )
        drone_id = p.createMultiBody(
            baseMass=DRONE_MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=SPAWN_POSITION.tolist(),
            physicsClientId=self._physics_client,
        )
        # Disable angular damping so wind effects are visible
        p.changeDynamics(
            drone_id,
            -1,
            linearDamping=0.9,
            angularDamping=0.99,
            physicsClientId=self._physics_client,
        )
        return drone_id

    # ------------------------------------------------------------------
    def _apply_velocity_control(self, vel_target: np.ndarray):
        """
        Simple velocity-tracking controller:
        Computes a proportional force to drive current velocity toward vel_target.
        """
        lin_vel, _ = p.getBaseVelocity(
            self._drone_id, physicsClientId=self._physics_client
        )
        lin_vel = np.array(lin_vel)
        error = vel_target - lin_vel
        force = error * DRONE_MASS * SIM_FREQ * 0.5  # proportional gain

        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        p.applyExternalForce(
            self._drone_id,
            -1,
            force.tolist(),
            pos,
            p.WORLD_FRAME,
            physicsClientId=self._physics_client,
        )

    # ------------------------------------------------------------------
    def _apply_wind_force(self):
        """Apply the current OU wind vector as an external force on the drone."""
        wind_vec = self._wind.step()  # advance OU process
        force = wind_vec * DRONE_MASS  # F = m * a  (wind acceleration)

        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        p.applyExternalForce(
            self._drone_id,
            -1,
            force.tolist(),
            pos,
            p.WORLD_FRAME,
            physicsClientId=self._physics_client,
        )

    # ------------------------------------------------------------------
    def _get_position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        return np.array(pos, dtype=np.float32)

    # ------------------------------------------------------------------
    def _get_low_dim_state(self) -> np.ndarray:
        """
        Returns s_low = [x, y, z,  vx, vy, vz,  yaw]
        as per the proposal's observation definition.
        """
        pos, quat = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        lin_vel, _ = p.getBaseVelocity(
            self._drone_id, physicsClientId=self._physics_client
        )
        euler = p.getEulerFromQuaternion(quat)
        yaw = euler[2]

        return np.array(
            [
                pos[0],
                pos[1],
                pos[2],
                lin_vel[0],
                lin_vel[1],
                lin_vel[2],
                yaw,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def _capture_camera_image(self) -> np.ndarray:
        """
        Render a front-facing 64×64 RGB image from the drone's perspective.
        Returns uint8 array of shape (64, 64, 3).
        """
        pos, quat = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_client
        )
        pos = np.array(pos)

        # Camera is 0.05 m in front of the drone
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        forward = rot_mat[:, 0]  # drone's x-axis = forward
        cam_pos = pos + forward * 0.05
        target = pos + forward * 2.0  # look 2 m ahead

        view_mat = p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=[0, 0, 1],
            physicsClientId=self._physics_client,
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=CAM_FOV,
            aspect=1.0,
            nearVal=CAM_NEAR,
            farVal=CAM_FAR,
            physicsClientId=self._physics_client,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=IMG_W,
            height=IMG_H,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._physics_client,
        )
        # rgba is shape (H, W, 4); drop alpha channel
        rgb = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
        return rgb

    # ------------------------------------------------------------------
    def _get_obs(self) -> dict:
        return {
            "image": self._capture_camera_image(),
            "state": self._get_low_dim_state(),
        }

    # ------------------------------------------------------------------
    def _compute_reward(self, dist: float, collided: bool) -> float:
        """
        R_t = r_distance + r_goal + r_time + r_timeout + r_collision
        Exactly as specified in the proposal.
        """
        # r_distance: reward for getting closer to the goal
        delta_dist = self._prev_dist_to_goal - dist
        r_distance = 10.0 * delta_dist / (self._initial_dist_to_goal + 1e-6)

        # r_goal: sparse bonus on reaching goal
        r_goal = R_GOAL if dist < GOAL_RADIUS else 0.0

        # r_time: small per-step penalty to encourage efficiency
        r_time = R_TIME

        # r_timeout: penalty if episode runs out of time
        r_timeout = R_TIMEOUT if self._step_count >= MAX_STEPS else 0.0

        # r_collision: penalty on any contact with obstacles or walls
        r_collision = R_COLLISION if collided else 0.0

        return float(r_distance + r_goal + r_time + r_timeout + r_collision)

    # ------------------------------------------------------------------
    def _check_collision(self) -> bool:
        """Return True if the drone is in contact with any obstacle or wall."""
        contacts = p.getContactPoints(
            bodyA=self._drone_id,
            physicsClientId=self._physics_client,
        )
        return len(contacts) > 0

    # ------------------------------------------------------------------
    def _is_out_of_bounds(self, pos: np.ndarray) -> bool:
        """Terminate if the drone leaves the corridor bounding box."""
        x, y, z = pos
        return (
            x < -0.5
            or x > CORRIDOR_LENGTH + 0.5
            or abs(y) > CORRIDOR_WIDTH / 2.0 + 0.3
            or z < 0.0
            or z > CORRIDOR_HEIGHT + 0.3
        )

    # ------------------------------------------------------------------
    def _draw_goal_marker(self):
        """Draw a green sphere at the goal location in the GUI."""
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.15,
            rgbaColor=[0.0, 1.0, 0.2, 0.6],
            physicsClientId=self._physics_client,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=GOAL_POSITION.tolist(),
            physicsClientId=self._physics_client,
        )
