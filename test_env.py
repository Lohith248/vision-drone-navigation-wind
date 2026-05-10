"""
test_env.py
===========
Sanity checks and a short GUI demo for DroneNavEnv.

Usage
-----
# Run sanity checks only (no GUI, fast):
    python test_env.py

# Run GUI demo with random actions:
    python test_env.py --gui

# Run GUI demo with wind disabled:
    python test_env.py --gui --no-wind
"""

import argparse
import sys

import numpy as np

# Make sure the package is importable when running from the project folder
sys.path.insert(0, ".")

from drone_nav_env.env import DroneNavEnv


# -----------------------------------------------------------------------
def test_spaces():
    """Verify observation and action spaces match the proposal spec."""
    print("=" * 55)
    print("TEST 1: Spaces")
    print("=" * 55)

    env = DroneNavEnv(gui=False, wind_enabled=False, seed=0)

    img_space = env.observation_space["image"]
    state_space = env.observation_space["state"]
    act_space = env.action_space

    assert img_space.shape == (64, 64, 3), f"Expected (64,64,3), got {img_space.shape}"
    assert state_space.shape == (7,), f"Expected (7,), got {state_space.shape}"
    assert act_space.shape == (3,), f"Expected (3,), got {act_space.shape}"
    assert act_space.low[0] == -3.0
    assert act_space.high[0] == 3.0

    print(f"  image space  : {img_space.shape}  dtype={img_space.dtype}")
    print(f"  state space  : {state_space.shape}  dtype={state_space.dtype}")
    print(
        f"  action space : {act_space.shape}  range=[{act_space.low[0]}, {act_space.high[0]}]"
    )
    print("  PASSED\n")
    env.close()


# -----------------------------------------------------------------------
def test_reset_and_step():
    """Reset the env and run 10 steps, checking shapes and reward types."""
    print("=" * 55)
    print("TEST 2: Reset + Step")
    print("=" * 55)

    env = DroneNavEnv(gui=False, wind_enabled=True, seed=42)
    obs, info = env.reset()

    assert "image" in obs and "state" in obs
    assert obs["image"].shape == (64, 64, 3)
    assert obs["state"].shape == (7,)
    print(
        f"  reset obs image  : {obs['image'].shape}  "
        f"range=[{obs['image'].min()}, {obs['image'].max()}]"
    )
    print(f"  reset obs state  : {obs['state']}")
    print(f"  initial wind     : {info['wind']}")

    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert isinstance(reward, float)
        assert obs["image"].shape == (64, 64, 3)

    print(f"  10 steps OK  |  total reward = {total_reward:.4f}")
    print(f"  last info    : {info}")
    print("  PASSED\n")
    env.close()


# -----------------------------------------------------------------------
def test_wind():
    """Check that OU wind stays within the 5 m/s cap."""
    print("=" * 55)
    print("TEST 3: Wind Model")
    print("=" * 55)

    from drone_nav_env.wind import OUWindModel

    wind = OUWindModel(sigma=2.0, max_speed=5.0, seed=7)
    wind.reset()

    max_seen = 0.0
    for _ in range(10000):
        w = wind.step()
        speed = np.linalg.norm(w)
        if speed > max_seen:
            max_seen = speed

    print(f"  max wind speed over 10 000 steps: {max_seen:.4f} m/s  (cap=5.0)")
    assert max_seen <= 5.0 + 1e-6, "Wind exceeded max_speed!"
    print("  PASSED\n")


# -----------------------------------------------------------------------
def run_gui_demo(wind_enabled: bool = True, n_steps: int = 500):
    """Open the PyBullet GUI and fly with random actions."""
    import time

    print("=" * 55)
    print("GUI DEMO: Random-action flight")
    print(f"  wind={'ON' if wind_enabled else 'OFF'}  steps={n_steps}")
    print("  Close the PyBullet window or press Ctrl+C to stop.")
    print("=" * 55)

    env = DroneNavEnv(gui=True, wind_enabled=wind_enabled, seed=0)
    obs, info = env.reset()

    total_reward = 0.0
    try:
        for step in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 50 == 0:
                pos = obs["state"][:3]
                print(
                    f"  step {step:4d} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
                    f" | reward={reward:+.4f} | dist={info['distance']:.2f} m"
                    f" | wind_mag={np.linalg.norm(info['wind']):.3f} m/s"
                )

            if terminated or truncated:
                reason = (
                    "GOAL!"
                    if info["reached_goal"]
                    else "COLLISION"
                    if info["collided"]
                    else "TIMEOUT"
                    if truncated
                    else "OOB"
                )
                print(f"\n  Episode ended at step {step} — {reason}")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    print(f"\n  Total reward: {total_reward:.4f}")
    env.close()


# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gui", action="store_true", help="Open PyBullet GUI for live demo"
    )
    parser.add_argument(
        "--no-wind", action="store_true", help="Disable wind disturbances in GUI demo"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps in GUI demo (default 500)",
    )
    args = parser.parse_args()

    if args.gui:
        run_gui_demo(wind_enabled=not args.no_wind, n_steps=args.steps)
    else:
        test_spaces()
        test_reset_and_step()
        test_wind()
        print("=" * 55)
        print("ALL TESTS PASSED!")
        print("  Run  'python test_env.py --gui'  to see the GUI demo.")
        print("=" * 55)
