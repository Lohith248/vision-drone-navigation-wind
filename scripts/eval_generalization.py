#!/usr/bin/env python3
"""
Generalization stress-test for a trained PPO checkpoint.

Runs the same checkpoint against several env perturbations:
  - in_distribution    : training final-stage env (sanity)
  - no_wind            : wind disabled
  - strong_wind        : 2x training wind sigma
  - dense_clutter      : higher obstacle density
  - turns              : enable_turns=True
  - narrow_corridor    : 1.6 m corridor (training was 2.0)
  - noisy_sensors      : sensor_noise=0.15
  - long_corridor      : corridor_length=15 (training was 10)

Writes one CSV with per-scenario success_rate / collision_rate / mean_reward / mean_length.

Usage:
  python scripts/eval_generalization.py \\
    --checkpoint runs/best_vit/ppo_seed0_t1p5m/checkpoints/best_model.pt \\
    --config drone_rl/configs/ppo_best.yaml \\
    --episodes 30 \\
    --out runs/best_vit/generalization.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_rl.utils.checkpoint import load_checkpoint  # noqa: E402
from drone_rl.trainers.ppo_trainer import PPOTrainer  # noqa: E402


SCENARIOS = {
    "in_distribution": {  # mirrors curriculum stage 5 (full wind)
        "env": dict(corridor_width=2.0, obstacle_density=1.0, wind_enabled=True,
                    wind_sigma=0.5, sensor_noise=0.05, corridor_length=10.0,
                    enable_moving_obstacles=True),
    },
    "no_wind": {
        "env": dict(corridor_width=2.0, obstacle_density=1.0, wind_enabled=False,
                    wind_sigma=0.0, sensor_noise=0.05, corridor_length=10.0,
                    enable_moving_obstacles=True),
    },
    "strong_wind": {
        "env": dict(corridor_width=2.0, obstacle_density=1.0, wind_enabled=True,
                    wind_sigma=1.0, sensor_noise=0.05, corridor_length=10.0,
                    enable_moving_obstacles=True),
    },
    "dense_clutter": {
        "env": dict(corridor_width=2.0, obstacle_density=1.5, wind_enabled=True,
                    wind_sigma=0.5, sensor_noise=0.05, corridor_length=10.0,
                    enable_moving_obstacles=True),
    },
    "turns": {
        "env": dict(corridor_width=2.5, obstacle_density=0.7, wind_enabled=True,
                    wind_sigma=0.3, sensor_noise=0.05, corridor_length=10.0,
                    enable_moving_obstacles=True, enable_turns=True),
        # turns are enabled via curriculum.apply_curriculum below if available
        "extras": {"enable_turns": True},
    },
    "narrow_corridor": {
        "env": dict(corridor_width=1.6, obstacle_density=0.7, wind_enabled=True,
                    wind_sigma=0.3, sensor_noise=0.05, corridor_length=10.0,
                    enable_moving_obstacles=True),
    },
    "noisy_sensors": {
        "env": dict(corridor_width=2.0, obstacle_density=1.0, wind_enabled=True,
                    wind_sigma=0.5, sensor_noise=0.15, corridor_length=10.0,
                    enable_moving_obstacles=True),
    },
    "long_corridor": {
        "env": dict(corridor_width=2.0, obstacle_density=1.0, wind_enabled=True,
                    wind_sigma=0.5, sensor_noise=0.05, corridor_length=15.0,
                    enable_moving_obstacles=True),
    },
}


def evaluate_scenario(base_config: dict, checkpoint_path: str,
                      scenario_name: str, scenario: dict,
                      episodes: int, device: str) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("env", {}).update(scenario.get("env", {}))
    cfg["device"] = device
    cfg["auto_resume"] = False
    cfg["eval_episodes"] = episodes
    cfg["log_dir"] = f"runs/_tmp_eval/{scenario_name}"  # write side artefacts here
    cfg["num_envs"] = 1                                   # eval is single-env

    trainer = PPOTrainer(cfg)
    trainer._setup()

    # If the env supports apply_curriculum, push extras (e.g. enable_turns)
    extras = scenario.get("extras", {})
    if extras:
        for env_name in ("env", "eval_env"):
            env_obj = getattr(trainer, env_name, None)
            if env_obj is not None and hasattr(env_obj, "apply_curriculum"):
                env_obj.apply_curriculum(extras)

    ckpt = load_checkpoint(checkpoint_path, trainer.device)
    trainer._load_from_checkpoint(ckpt)

    t0 = time.time()
    metrics = trainer._evaluate(n_episodes=episodes)
    elapsed = time.time() - t0

    # tidy up
    for env_name in ("env", "eval_env"):
        env_obj = getattr(trainer, env_name, None)
        if env_obj is not None and hasattr(env_obj, "close"):
            env_obj.close()
    trainer.tb_logger.close()
    trainer.csv_logger.close()

    row = {
        "scenario": scenario_name,
        "episodes": episodes,
        "elapsed_s": round(elapsed, 1),
        **{k: round(float(v), 4) for k, v in metrics.items()},
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/best_vit/generalization.csv")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="Subset of scenarios to run (default: all)")
    args = ap.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = {k: v for k, v in SCENARIOS.items() if k in args.scenarios}

    rows = []
    for name, scenario in scenarios.items():
        print(f"\n=== {name} ===")
        try:
            row = evaluate_scenario(
                base_config, args.checkpoint, name, scenario,
                episodes=args.episodes, device=args.device,
            )
        except Exception as e:
            print(f"  ! failed: {e}")
            row = {"scenario": name, "episodes": args.episodes, "error": str(e)}
        rows.append(row)
        print(f"  -> {row}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ Wrote {out}")


if __name__ == "__main__":
    main()
