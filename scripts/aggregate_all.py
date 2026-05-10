#!/usr/bin/env python3
"""
Aggregate every training run into a single comparison CSV/markdown table.

Walks `runs/` looking for directories that contain
  - config.yaml
  - checkpoints/best_model.pt  (or any checkpoint_*.pt)
For each, runs evaluation and records:
  algorithm, run_name, seed, mean_reward, success_rate, collision_rate,
  timeout_rate, oob_rate, mean_length, total_timesteps, params

Usage:
  python scripts/aggregate_all.py --episodes 30 --out report_artifacts/all_runs.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_rl.utils.checkpoint import load_checkpoint  # noqa: E402

# Hardcoded list of "interesting" run roots to scan
INTERESTING_ROOTS = [
    "runs/best_vit",
    "runs/baseline_ddpg",
    "runs/baseline_sac",
    "runs/multi_seed",
    "runs/abl_no_advnorm",
    "runs/abl_low_reward_scale",
    "runs/abl_no_curriculum",
    "runs/abl_state_only",
    "runs/abl_cnn_encoder",
    "runs/abl_domain_random",
]


def find_runs(roots: list[str]) -> list[Path]:
    runs = []
    for r in roots:
        root = Path(r)
        if not root.exists():
            continue
        # Direct config.yaml inside the root?
        if (root / "config.yaml").is_file():
            runs.append(root)
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "config.yaml").is_file():
                runs.append(child)
    return runs


def find_checkpoint(run: Path) -> Path | None:
    # For curriculum experiments, "best_model.pt" can be selected on an easy
    # early curriculum stage. For final reporting we prefer the latest training
    # checkpoint, then evaluate it under the final curriculum stage below.
    def step_num(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    ckpts = sorted((run / "checkpoints").glob("checkpoint_*.pt"), key=step_num)
    if ckpts:
        return ckpts[-1]
    best = run / "checkpoints" / "best_model.pt"
    return best if best.is_file() else None


def apply_final_curriculum_stage(trainer, config: dict) -> None:
    """Evaluate curriculum runs on their final stage for a fair comparison."""
    curriculum = config.get("curriculum", {})
    stages = curriculum.get("stages") or []
    if not (curriculum.get("enabled", False) and stages):
        return
    final_stage = dict(stages[-1])
    final_stage.pop("name", None)
    final_stage.pop("reward_threshold", None)
    final_stage.pop("success_threshold", None)
    for env_name in ("env", "eval_env"):
        env = getattr(trainer, env_name, None)
        if env is not None and hasattr(env, "apply_curriculum"):
            env.apply_curriculum(final_stage)


def build_trainer(config: dict):
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
    raise ValueError(algo)


def safe_close(trainer):
    for env_name in ("env", "eval_env"):
        env = getattr(trainer, env_name, None)
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
    for log_attr in ("tb_logger", "csv_logger"):
        lg = getattr(trainer, log_attr, None)
        if lg is not None:
            try:
                lg.close()
            except Exception:
                pass


def evaluate_run(run: Path, episodes: int, device: str) -> dict | None:
    cfg_path = run / "config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    ckpt = find_checkpoint(run)
    if ckpt is None:
        return None

    config["device"] = device
    config["auto_resume"] = False
    config["eval_episodes"] = episodes
    config["log_dir"] = f"runs/_tmp_eval/{run.parent.name}_{run.name}"

    try:
        trainer = build_trainer(config)
        trainer._setup()
        ck = load_checkpoint(str(ckpt), trainer.device)
        trainer._load_from_checkpoint(ck)
        apply_final_curriculum_stage(trainer, config)
        metrics = trainer._evaluate(n_episodes=episodes)
    except Exception as e:
        traceback.print_exc()
        return {
            "algorithm": config.get("algorithm", "?"),
            "run": f"{run.parent.name}/{run.name}",
            "error": str(e),
        }
    finally:
        try:
            safe_close(trainer)
        except Exception:
            pass

    # Param count
    n_params = 0
    try:
        if hasattr(trainer, "policy") and trainer.policy is not None:
            n_params = sum(p.numel() for p in trainer.policy.parameters())
        elif hasattr(trainer, "algorithm") and hasattr(trainer.algorithm, "actor"):
            n_params = (sum(p.numel() for p in trainer.algorithm.actor.parameters()) +
                        sum(p.numel() for p in trainer.algorithm.critic.parameters()))
    except Exception:
        pass

    meta = ck.get("metadata", {})
    return {
        "algorithm": config.get("algorithm", "?"),
        "run": f"{run.parent.name}/{run.name}",
        "seed": int(config.get("seed", 0)),
        "checkpoint": str(ckpt),
        "total_timesteps": int(meta.get("total_timesteps", 0)),
        "params": int(n_params),
        "mean_reward": round(float(metrics.get("mean_reward", 0.0)), 3),
        "std_reward": round(float(metrics.get("std_reward", 0.0)), 3),
        "mean_length": round(float(metrics.get("mean_length", 0.0)), 1),
        "success_rate": round(float(metrics.get("success_rate", 0.0)), 3),
        "collision_rate": round(float(metrics.get("collision_rate", 0.0)), 3),
        "timeout_rate": round(float(metrics.get("timeout_rate", 0.0)), 3),
        "oob_rate": round(float(metrics.get("oob_rate", 0.0)), 3),
    }


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "No runs found."
    cols = ["run", "algorithm", "seed", "total_timesteps", "params",
            "success_rate", "collision_rate", "timeout_rate",
            "mean_reward", "std_reward", "mean_length"]
    md = ["| " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="report_artifacts/all_runs.csv")
    ap.add_argument("--md-out", default="report_artifacts/all_runs.md")
    ap.add_argument("--roots", nargs="*", default=None)
    args = ap.parse_args()

    runs = find_runs(args.roots or INTERESTING_ROOTS)
    if not runs:
        print("No runs found.")
        return
    print(f"Found {len(runs)} run dirs:")
    for r in runs:
        print(f"  - {r}")

    rows = []
    for run in runs:
        print(f"\n=== {run} ===")
        row = evaluate_run(run, args.episodes, args.device)
        if row:
            rows.append(row)
            print(f"  -> {row}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n✓ Wrote {out}")
        md_out = Path(args.md_out)
        with open(md_out, "w") as f:
            f.write("# All Runs — Evaluation Summary\n\n")
            f.write(f"_n_episodes per run = {args.episodes}_\n\n")
            f.write(to_markdown(rows))
            f.write("\n")
        print(f"✓ Wrote {md_out}")


if __name__ == "__main__":
    main()
