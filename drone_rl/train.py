"""
CLI entry point for the drone RL training framework.

Usage:
    python -m drone_rl.train --algo ppo --config drone_rl/configs/ppo.yaml
    python -m drone_rl.train --algo sac --total-timesteps 500000
    python -m drone_rl.train --algo dqn --config drone_rl/configs/dqn.yaml --resume
    python -m drone_rl.train --algo ddpg --config drone_rl/configs/ddpg.yaml
    python -m drone_rl.train --algo ppo --eval-only --checkpoint runs/ppo/checkpoints/best_model.pt
"""

import argparse
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required to run drone_rl.train. Install dependencies via "
        "`pip install -r requirements.txt`."
    ) from exc


def load_and_merge_config(args) -> dict:
    """Load default config, overlay algorithm config, then CLI overrides."""
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

    # Load default
    default_path = os.path.join(config_dir, "default.yaml")
    with open(default_path) as f:
        config = yaml.safe_load(f)

    # Load algorithm-specific config
    if args.config:
        with open(args.config) as f:
            algo_config = yaml.safe_load(f)
        _deep_update(config, algo_config)
    elif args.algo:
        algo_path = os.path.join(config_dir, f"{args.algo}.yaml")
        if os.path.isfile(algo_path):
            with open(algo_path) as f:
                algo_config = yaml.safe_load(f)
            _deep_update(config, algo_config)

    # CLI overrides (only if explicitly provided)
    if args.algo:
        config["algorithm"] = args.algo
    config.setdefault("algorithm", "ppo")
    if args.total_timesteps:
        config["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if args.num_envs:
        config["num_envs"] = args.num_envs
    if args.log_dir:
        config["log_dir"] = args.log_dir
    if args.no_amp:
        config["use_amp"] = False
    if args.curriculum:
        config.setdefault("curriculum", {})["enabled"] = True
    if args.no_resume:
        config["auto_resume"] = False
    if args.eval_episodes:
        config["eval_episodes"] = args.eval_episodes

    return config


def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Drone RL Training Framework — PPO / SAC / DQN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--algo", type=str, default=None,
                        choices=["ppo", "sac", "dqn", "ddpg"],
                        help="Algorithm to train (overrides config). "
                             "If omitted, taken from --config's `algorithm:` field, "
                             "falling back to 'ppo'.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Log directory")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't auto-resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (requires --checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval/resume")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="Override number of evaluation episodes")

    args = parser.parse_args()

    # Load config
    config = load_and_merge_config(args)

    print(f"\n{'='*60}")
    print(f"  Drone RL Framework v1.0")
    print(f"  Algorithm: {config['algorithm'].upper()}")
    print(f"  Timesteps: {config['total_timesteps']:,}")
    print(f"  Seed: {config['seed']}")
    print(f"  AMP: {config['use_amp']}")
    print(f"{'='*60}\n")

    # Create trainer
    algo = config["algorithm"]
    if algo == "ppo":
        from drone_rl.trainers.ppo_trainer import PPOTrainer
        trainer = PPOTrainer(config)
    elif algo == "sac":
        from drone_rl.trainers.sac_trainer import SACTrainer
        trainer = SACTrainer(config)
    elif algo == "dqn":
        from drone_rl.trainers.dqn_trainer import DQNTrainer
        trainer = DQNTrainer(config)
    elif algo == "ddpg":
        from drone_rl.trainers.ddpg_trainer import DDPGTrainer
        trainer = DDPGTrainer(config)
    else:
        print(f"Unknown algorithm: {algo}")
        sys.exit(1)

    if args.eval_only:
        if not args.checkpoint:
            print("Error: --eval-only requires --checkpoint")
            sys.exit(1)
        config["auto_resume"] = False
        trainer._setup()
        from drone_rl.utils.checkpoint import load_checkpoint
        ckpt = load_checkpoint(args.checkpoint, trainer.device)
        trainer._load_from_checkpoint(ckpt)
        metrics = trainer._evaluate(n_episodes=config.get("eval_episodes", 20))
        print(f"\n  Evaluation Results:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
    else:
        trainer.train(config["total_timesteps"])


if __name__ == "__main__":
    main()
