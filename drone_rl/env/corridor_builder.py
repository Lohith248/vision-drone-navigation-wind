"""
Configurable corridor geometry builder for PyBullet.

Supports variable corridor width, optional turns, and configurable obstacle
density to enable curriculum learning progression.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

# Default corridor dimensions (metres)
DEFAULT_CORRIDOR_LENGTH = 10.0
DEFAULT_CORRIDOR_WIDTH = 3.0
DEFAULT_CORRIDOR_HEIGHT = 2.5
WALL_THICKNESS = 0.1


def _box(client, half_extents, position, euler=None, color=None, mass=0):
    """Create a box collision+visual shape and return the body ID."""
    if euler is None:
        euler = [0, 0, 0]
    if color is None:
        color = [0.85, 0.82, 0.78, 1.0]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                 physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                              rgbaColor=color, physicsClientId=client)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=position,
                             baseOrientation=p.getQuaternionFromEuler(euler),
                             physicsClientId=client)


def _cylinder(client, radius, height, position, color=None, mass=0):
    """Create a cylinder collision+visual shape and return the body ID."""
    if color is None:
        color = [0.6, 0.6, 0.6, 1.0]
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height,
                                 physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                              rgbaColor=color, physicsClientId=client)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=position,
                             physicsClientId=client)


def _sphere(client, radius, position, color=None, mass=0):
    """Create a sphere collision+visual shape and return the body ID."""
    if color is None:
        color = [1.0, 0.2, 0.2, 1.0]
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius,
                                 physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color,
                              physicsClientId=client)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=position,
                             physicsClientId=client)


class CorridorBuilder:
    """
    Builds parameterised corridor environments for curriculum learning.

    Parameters
    ----------
    length : float
        Corridor length in metres.
    width : float
        Corridor width in metres. Narrower = harder.
    height : float
        Corridor height in metres.
    obstacle_density : float
        Fraction of obstacles to place (0.0 = none, 1.0 = all).
    enable_turns : bool
        If True, add an L-shaped turn at the midpoint.
    enable_moving_obstacles : bool
        If True, some obstacles become dynamic (mass > 0).
    """

    # Standard colours
    C_WALL = [0.88, 0.85, 0.80, 1.0]
    C_CEIL = [0.95, 0.95, 0.95, 1.0]
    C_FLOOR = [0.55, 0.45, 0.35, 1.0]
    C_PILLAR = [0.50, 0.50, 0.55, 1.0]
    C_CRATE = [0.72, 0.53, 0.25, 1.0]
    C_DANGER = [0.90, 0.20, 0.20, 1.0]
    C_GATE = [0.20, 0.75, 0.35, 1.0]
    C_ORANGE = [1.00, 0.55, 0.10, 1.0]

    def __init__(
        self,
        length: float = DEFAULT_CORRIDOR_LENGTH,
        width: float = DEFAULT_CORRIDOR_WIDTH,
        height: float = DEFAULT_CORRIDOR_HEIGHT,
        obstacle_density: float = 1.0,
        enable_turns: bool = False,
        enable_moving_obstacles: bool = False,
    ):
        self.length = length
        self.width = width
        self.height = height
        self.obstacle_density = np.clip(obstacle_density, 0.0, 1.0)
        self.enable_turns = enable_turns
        self.enable_moving_obstacles = enable_moving_obstacles

    @property
    def spawn_position(self) -> np.ndarray:
        """Drone start position."""
        return np.array([0.5, 0.0, 1.0], dtype=np.float32)

    @property
    def goal_position(self) -> np.ndarray:
        """Goal position at far end of corridor."""
        return np.array([self.length - 0.5, 0.0, 1.0], dtype=np.float32)

    def build(self, client: int, rng: Optional[np.random.Generator] = None) -> Tuple[List[int], List[int]]:
        """
        Construct the corridor in the given PyBullet client.

        Parameters
        ----------
        client : int
            PyBullet physics client ID.
        rng : np.random.Generator, optional
            RNG for randomising obstacle placement.

        Returns
        -------
        wall_ids : List[int]
            Body IDs of structural elements (walls, floor, ceiling).
        obstacle_ids : List[int]
            Body IDs of obstacles (for collision detection).
        """
        L = self.length
        W = self.width
        H = self.height
        T = WALL_THICKNESS
        cx = L / 2.0

        wall_ids = []
        obstacle_ids = []

        # ---- Structural elements ----
        wall_ids.append(_box(client, [L/2, W/2, T/2], [cx, 0, -T/2],
                             color=self.C_FLOOR))                         # floor
        wall_ids.append(_box(client, [L/2, W/2, T/2], [cx, 0, H + T/2],
                             color=self.C_CEIL))                          # ceiling
        wall_ids.append(_box(client, [L/2, T/2, H/2], [cx, W/2, H/2],
                             color=self.C_WALL))                          # right wall
        wall_ids.append(_box(client, [L/2, T/2, H/2], [cx, -W/2, H/2],
                             color=self.C_WALL))                          # left wall
        wall_ids.append(_box(client, [T/2, W/2, H/2], [-T/2, 0, H/2],
                             color=self.C_WALL))                          # back wall
        wall_ids.append(_box(client, [T/2, W/2, H/2], [L + T/2, 0, H/2],
                             color=self.C_WALL))                          # front wall

        # ---- Obstacles (density-filtered) ----
        if rng is None:
            rng = np.random.default_rng(0)

        all_obstacles = self._define_obstacles(L, W, H)
        n_to_place = max(0, int(len(all_obstacles) * self.obstacle_density))

        # Randomly select which obstacles to place
        if n_to_place < len(all_obstacles):
            indices = rng.choice(len(all_obstacles), size=n_to_place, replace=False)
            selected = [all_obstacles[i] for i in sorted(indices)]
        else:
            selected = all_obstacles

        for obs_def in selected:
            mass = 0.5 if (self.enable_moving_obstacles and obs_def.get("dynamic", False)) else 0
            shape = obs_def["shape"]
            if shape == "cylinder":
                oid = _cylinder(client, obs_def["radius"], obs_def["height"],
                                obs_def["position"], color=obs_def["color"], mass=mass)
            elif shape == "box":
                oid = _box(client, obs_def["half_extents"], obs_def["position"],
                           color=obs_def["color"], mass=mass)
            elif shape == "sphere":
                oid = _sphere(client, obs_def["radius"], obs_def["position"],
                              color=obs_def["color"], mass=mass)
            else:
                continue
            obstacle_ids.append(oid)

        return wall_ids, obstacle_ids

    def _define_obstacles(self, L: float, W: float, H: float) -> List[Dict]:
        """Define all possible obstacles with their geometry. Returns list of dicts."""
        hw = W / 2.0
        obstacles = []

        # Cylindrical pillars
        if L >= 5.0:
            obstacles.append({"shape": "cylinder", "radius": 0.15, "height": H,
                              "position": [2.5, min(0.5, hw - 0.3), H/2],
                              "color": self.C_PILLAR, "dynamic": False})
        if L >= 7.0:
            obstacles.append({"shape": "cylinder", "radius": 0.15, "height": H,
                              "position": [4.5, max(-0.6, -(hw - 0.3)), H/2],
                              "color": self.C_PILLAR, "dynamic": False})
        if L >= 9.0:
            obstacles.append({"shape": "cylinder", "radius": 0.12, "height": H,
                              "position": [7.0, min(0.3, hw - 0.3), H/2],
                              "color": self.C_PILLAR, "dynamic": False})

        # Floor crates
        obstacles.append({"shape": "box", "half_extents": [0.30, 0.30, 0.30],
                           "position": [1.5, max(-0.8, -(hw - 0.3)), 0.30],
                           "color": self.C_CRATE, "dynamic": True})
        if L >= 6.0:
            obstacles.append({"shape": "box", "half_extents": [0.40, 0.25, 0.25],
                               "position": [3.5, min(0.9, hw - 0.3), 0.25],
                               "color": self.C_CRATE, "dynamic": True})
            obstacles.append({"shape": "box", "half_extents": [0.20, 0.20, 0.50],
                               "position": [6.0, max(-1.0, -(hw - 0.3)), 0.50],
                               "color": self.C_CRATE, "dynamic": True})

        # Hanging danger block
        if L >= 7.0:
            obstacles.append({"shape": "box", "half_extents": [0.15, min(0.8, hw - 0.2), 0.5],
                               "position": [5.0, 0.0, H - 0.5],
                               "color": self.C_DANGER, "dynamic": False})

        # Floating spheres
        if L >= 9.0:
            obstacles.append({"shape": "sphere", "radius": 0.20,
                               "position": [8.0, 0.0, 1.2],
                               "color": self.C_DANGER, "dynamic": True})
            obstacles.append({"shape": "sphere", "radius": 0.15,
                               "position": [3.0, -0.4, 1.5],
                               "color": self.C_ORANGE, "dynamic": True})

        # Navigation gate
        if L >= 9.5:
            gate_half_y = min(0.6, hw - 0.2)
            obstacles.append({"shape": "box", "half_extents": [0.08, 0.08, H/2],
                               "position": [8.5, gate_half_y, H/2],
                               "color": self.C_GATE, "dynamic": False})
            obstacles.append({"shape": "box", "half_extents": [0.08, 0.08, H/2],
                               "position": [8.5, -gate_half_y, H/2],
                               "color": self.C_GATE, "dynamic": False})
            obstacles.append({"shape": "box", "half_extents": [0.08, gate_half_y, 0.08],
                               "position": [8.5, 0.0, 1.8],
                               "color": self.C_GATE, "dynamic": False})

        return obstacles
