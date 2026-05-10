#!/usr/bin/env python3
"""
Evaluate all seed runs and aggregate metrics by algorithm.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import yaml

# Ensure project root is on path when called as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_rl.utils.checkpoint import load_checkpoint


def _build_trainer(config: dict):
    algo = config.get("algorithm", "ppo")
    if algo == "ppo":
        from drone_rl.trainers.ppo_trainer import PPOTrainer
        return PPOTrainer(config)
    if algo == "ddpg":
        from drone_rl.trainers.ddpg_trainer import DDPGTrainer
        return DDPGTrainer(config)
    if algo == "sac":
        from drone_rl.trainers.sac_trainer import SACTrainer
        return SACTrainer(config)
    if algo == "dqn":
        from drone_rl.trainers.dqn_trainer import DQNTrainer
        return DQNTrainer(config)
    raise ValueError(f"Unsupported algorithm: {algo}")


def _find_checkpoint(run_dir: Path) -> Path | None:
    best = run_dir / "checkpoints" / "best_model.pt"
    if best.is_file():
        return best
    ckpts = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
    return ckpts[-1] if ckpts else None


def _safe_close(trainer) -> None:
    for env_name in ("env", "eval_env"):
        env = getattr(trainer, env_name, None)
        if env is not None and hasattr(env, "close"):
            env.close()
    if hasattr(trainer, "tb_logger"):
        trainer.tb_logger.close()
    if hasattr(trainer, "csv_logger"):
        trainer.csv_logger.close()


def evaluate_run(run_dir: Path, episodes: int, device: str) -> Dict[str, float] | None:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    ckpt = _find_checkpoint(run_dir)
    if ckpt is None:
        return None

    config["device"] = device
    config["auto_resume"] = False
    config["eval_episodes"] = episodes

    trainer = _build_trainer(config)
    try:
        trainer._setup()
        checkpoint = load_checkpoint(str(ckpt), trainer.device)
        trainer._load_from_checkpoint(checkpoint)
        metrics = trainer._evaluate(n_episodes=episodes)
    finally:
        _safe_close(trainer)

    row = {
        "algorithm": config.get("algorithm", "unknown"),
        "run": run_dir.name,
        "seed": int(config.get("seed", -1)),
        "checkpoint": str(ckpt),
    }
    row.update(metrics)
    return row


def aggregate(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)

    summary = []
    for algo, group in sorted(grouped.items()):
        keys = ["success_rate", "collision_rate", "timeout_rate", "mean_reward", "mean_length", "oob_rate"]
        agg_row = {"algorithm": algo, "n_runs": len(group)}
        for key in keys:
            values = [float(g.get(key, 0.0)) for g in group]
            agg_row[f"{key}_mean"] = mean(values)
            agg_row[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary.append(agg_row)
    return summary


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and aggregate all seed runs")
    parser.add_argument("--runs-root", type=str, default="runs/best")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dirs = sorted([d for d in runs_root.glob("*_seed*") if d.is_dir()])
    if not run_dirs:
        print(f"No run directories found under {runs_root}")
        return

    rows = []
    for run_dir in run_dirs:
        row = evaluate_run(run_dir, episodes=args.episodes, device=args.device)
        if row is not None:
            rows.append(row)
            print(
                f"{row['algorithm']:>5} | {row['run']:<20} | "
                f"success={row.get('success_rate', 0.0):.3f} | "
                f"collision={row.get('collision_rate', 0.0):.3f} | "
                f"reward={row.get('mean_reward', 0.0):.2f}"
            )

    if not rows:
        print(f"No evaluable runs found under {runs_root}")
        return

    summary = aggregate(rows)

    detail_csv = runs_root / "eval_results.csv"
    summary_csv = runs_root / "eval_summary.csv"
    write_csv(detail_csv, rows)
    write_csv(summary_csv, summary)

    print("\n=== Aggregated Results ===")
    for row in summary:
        print(
            f"{row['algorithm']:>5} | n={row['n_runs']} | "
            f"success={row['success_rate_mean']:.3f}±{row['success_rate_std']:.3f} | "
            f"collision={row['collision_rate_mean']:.3f}±{row['collision_rate_std']:.3f} | "
            f"reward={row['mean_reward_mean']:.2f}±{row['mean_reward_std']:.2f}"
        )
    print(f"\nSaved: {detail_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
