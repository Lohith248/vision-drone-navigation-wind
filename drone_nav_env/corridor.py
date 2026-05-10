"""
Indoor corridor geometry builder for PyBullet.

Builds a 10 m × 3 m × 2.5 m corridor with:
  - Floor, ceiling, and 4 walls
  - 3 cylindrical pillars
  - 3 crates (boxes) on the floor
  - 1 hanging danger block from the ceiling
  - 2 floating spheres
  - 1 navigation gate near the exit
"""

from typing import List

import numpy as np
import pybullet as p

# Corridor dimensions (metres)
CORRIDOR_LENGTH = 10.0
CORRIDOR_WIDTH = 3.0
CORRIDOR_HEIGHT = 2.5
WALL_THICKNESS = 0.1

# Goal position: centre of the corridor exit
GOAL_POSITION = np.array([9.5, 0.0, 1.0], dtype=np.float32)

# Drone spawn position
SPAWN_POSITION = np.array([0.5, 0.0, 1.0], dtype=np.float32)


# -----------------------------------------------------------------------
# Primitive helpers
# -----------------------------------------------------------------------


def _box(
    client,
    half_extents,
    position,
    euler=[0, 0, 0],
    color=[0.85, 0.82, 0.78, 1.0],
    mass=0,
):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client
    )
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position,
        baseOrientation=p.getQuaternionFromEuler(euler),
        physicsClientId=client,
    )


def _cylinder(client, radius, height, position, color=[0.6, 0.6, 0.6, 1.0], mass=0):
    col = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client
    )
    vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color,
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position,
        physicsClientId=client,
    )


def _sphere(client, radius, position, color=[1.0, 0.2, 0.2, 1.0], mass=0):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position,
        physicsClientId=client,
    )


# -----------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------


def build_corridor(client: int) -> List[int]:
    """
    Construct the full indoor corridor in the given PyBullet physics client.

    Returns
    -------
    obstacle_ids : List[int]
        PyBullet body IDs of every obstacle (NOT the structural walls/floor/ceiling).
        Used for collision detection.
    """
    L = CORRIDOR_LENGTH
    W = CORRIDOR_WIDTH
    H = CORRIDOR_HEIGHT
    T = WALL_THICKNESS
    cx = L / 2.0

    # ---- Colours ----
    C_WALL = [0.88, 0.85, 0.80, 1.0]
    C_CEIL = [0.95, 0.95, 0.95, 1.0]
    C_FLOOR = [0.55, 0.45, 0.35, 1.0]
    C_PILLAR = [0.50, 0.50, 0.55, 1.0]
    C_CRATE = [0.72, 0.53, 0.25, 1.0]
    C_DANGER = [0.90, 0.20, 0.20, 1.0]
    C_GATE = [0.20, 0.75, 0.35, 1.0]
    C_ORANGE = [1.00, 0.55, 0.10, 1.0]

    # ---- Structure ----
    _box(client, [L / 2, W / 2, T / 2], [cx, 0, -T / 2], color=C_FLOOR)  # floor
    _box(client, [L / 2, W / 2, T / 2], [cx, 0, H + T / 2], color=C_CEIL)  # ceiling
    _box(client, [L / 2, T / 2, H / 2], [cx, W / 2, H / 2], color=C_WALL)  # right wall
    _box(client, [L / 2, T / 2, H / 2], [cx, -W / 2, H / 2], color=C_WALL)  # left wall
    _box(client, [T / 2, W / 2, H / 2], [-T / 2, 0, H / 2], color=C_WALL)  # back wall
    _box(
        client, [T / 2, W / 2, H / 2], [L + T / 2, 0, H / 2], color=C_WALL
    )  # front wall (exit)

    # ---- Obstacles ----
    obstacle_ids = []

    # Cylindrical pillars
    obstacle_ids.append(_cylinder(client, 0.15, H, [2.5, 0.5, H / 2], color=C_PILLAR))
    obstacle_ids.append(_cylinder(client, 0.15, H, [4.5, -0.6, H / 2], color=C_PILLAR))
    obstacle_ids.append(_cylinder(client, 0.12, H, [7.0, 0.3, H / 2], color=C_PILLAR))

    # Floor crates
    obstacle_ids.append(
        _box(client, [0.30, 0.30, 0.30], [1.5, -0.8, 0.30], color=C_CRATE)
    )
    obstacle_ids.append(
        _box(client, [0.40, 0.25, 0.25], [3.5, 0.9, 0.25], color=C_CRATE)
    )
    obstacle_ids.append(
        _box(client, [0.20, 0.20, 0.50], [6.0, -1.0, 0.50], color=C_CRATE)
    )

    # Hanging danger block (low-clearance)
    obstacle_ids.append(
        _box(client, [0.15, 0.8, 0.5], [5.0, 0.0, H - 0.5], color=C_DANGER)
    )

    # Floating spheres
    obstacle_ids.append(_sphere(client, 0.20, [8.0, 0.0, 1.2], color=C_DANGER))
    obstacle_ids.append(_sphere(client, 0.15, [3.0, -0.4, 1.5], color=C_ORANGE))

    # Navigation gate (structural, but counts as collision)
    obstacle_ids.append(
        _box(client, [0.08, 0.08, H / 2], [8.5, 0.6, H / 2], color=C_GATE)
    )
    obstacle_ids.append(
        _box(client, [0.08, 0.08, H / 2], [8.5, -0.6, H / 2], color=C_GATE)
    )
    obstacle_ids.append(_box(client, [0.08, 0.60, 0.08], [8.5, 0.0, 1.8], color=C_GATE))

    return obstacle_ids
