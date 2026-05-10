#!/usr/bin/env python3
"""
Render the trained PPO+ViT policy flying through the corridor and write an MP4/GIF.

Observations stay 64×64 for the policy; use --demo-width/--demo-height for recording resolution.

Usage:
  python scripts/render_demo.py --episodes 3 --out report_artifacts/demo_wind.mp4 --wind \\
      --demo-width 1280 --demo-height 720 --crf 18
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_rl.utils.checkpoint import load_checkpoint
from drone_rl.trainers.ppo_trainer import PPOTrainer


def _downscale_frames_max_side(frames: list, max_side: int) -> list:
    """Shrink frames so GIF outputs stay reasonably small."""
    from PIL import Image

    h0, w0 = frames[0].shape[:2]
    if max(h0, w0) <= max_side:
        return frames
    scale = max_side / float(max(h0, w0))
    nw, nh = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
    out = []
    for f in frames:
        im = Image.fromarray(f)
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        out.append(np.asarray(im))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/best_vit/ppo_seed0_t1p5m/checkpoints/best_model.pt")
    ap.add_argument("--config", default="drone_rl/configs/ppo_best.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps-per-ep", type=int, default=600)
    ap.add_argument("--wind", action="store_true")
    ap.add_argument("--strong-wind", action="store_true")
    ap.add_argument("--out", default="report_artifacts/demo.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument(
        "--demo-width",
        type=int,
        default=1280,
        help="Recording camera width (policy still uses 64×64 observations).",
    )
    ap.add_argument(
        "--demo-height",
        type=int,
        default=720,
        help="Recording camera height.",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=18,
        help="libx264 CRF for MP4 (lower = higher quality, typical 17–23).",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["device"] = "auto"
    config["auto_resume"] = False
    config["eval_episodes"] = args.episodes
    config["log_dir"] = "runs/_tmp_demo"
    config["num_envs"] = 1
    config.setdefault("env", {})
    config["env"]["wind_enabled"] = bool(args.wind or args.strong_wind)
    if args.strong_wind:
        config["env"]["wind_sigma"] = 1.0
    elif args.wind:
        config["env"]["wind_sigma"] = 0.5
    config["env"]["corridor_width"] = 2.0
    config["env"]["obstacle_density"] = 1.0
    config["env"]["enable_moving_obstacles"] = True
    config["env"]["sensor_noise"] = 0.05
    config["seed"] = args.seed

    trainer = PPOTrainer(config)
    trainer._setup()
    # Match final curriculum-stage semantics for the demo.
    stages = config.get("curriculum", {}).get("stages") or []
    if stages:
        final_stage = dict(stages[-1])
        final_stage.pop("name", None)
        final_stage.pop("reward_threshold", None)
        final_stage.pop("success_threshold", None)
        trainer.eval_env.apply_curriculum(final_stage)
    ckpt = load_checkpoint(args.checkpoint, trainer.device)
    trainer._load_from_checkpoint(ckpt)

    # Get the underlying single env instance
    env = trainer.eval_env.envs[0] if hasattr(trainer.eval_env, "envs") else None
    if env is None:
        print("Could not locate underlying env")
        sys.exit(1)

    frames = []
    ep_summaries = []
    for ep in range(args.episodes):
        obs, _ = trainer.eval_env.reset()
        proc = trainer._prepare_obs(obs, update_normalizer=False)
        ep_reward = 0.0
        for t in range(args.max_steps_per_ep):
            try:
                frame = env.capture_demo_frame(args.demo_width, args.demo_height)
                if frame is not None:
                    frames.append(frame)
            except Exception:
                pass
            obs_t = trainer._to_torch_obs(proc)
            with torch.no_grad():
                action = trainer.policy.get_deterministic_action(obs_t)
            next_obs, rew, term, trunc, info = trainer.eval_env.step(action.cpu().numpy())
            ep_reward += float(rew.sum())
            done = bool((term | trunc).any())
            proc = trainer._prepare_obs(next_obs, update_normalizer=False)
            if done:
                term_info = info[0] if isinstance(info, list) and info else {}
                ep_summaries.append((ep + 1, t + 1, ep_reward,
                                     bool(term_info.get("reached_goal", False)),
                                     bool(term_info.get("collided", False))))
                print(f"  ep {ep+1}: steps={t+1}, reward={ep_reward:.1f}, "
                      f"reached_goal={term_info.get('reached_goal')}, "
                      f"collided={term_info.get('collided')}")
                break

    trainer.eval_env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        print("! No frames captured")
        sys.exit(2)

    if out.suffix.lower() == ".gif":
        import imageio

        gif_frames = frames
        if max(frames[0].shape[:2]) > 640:
            gif_frames = _downscale_frames_max_side(frames, 640)
            print("  (GIF: downscaled to max side 640 px for smaller file size)")
        imageio.mimsave(out, gif_frames, duration=1.0 / args.fps)
    else:
        try:
            import imageio
            # High-quality H.264: CRF + yuv420p for broad player compatibility.
            imageio.mimsave(
                out,
                frames,
                fps=args.fps,
                codec="libx264",
                ffmpeg_params=[
                    "-crf",
                    str(max(0, min(args.crf, 51))),
                    "-preset",
                    "slow",
                ],
            )
        except Exception as e:
            print(f"  ! mp4 write failed ({e}); falling back to gif")
            out = out.with_suffix(".gif")
            import imageio
            imageio.mimsave(out, frames, duration=1.0 / args.fps)
    print(f"\n✓ Wrote {out} ({len(frames)} frames @ {args.fps} fps)")
    print("Episode summaries:")
    for ep, steps, rew, goal, col in ep_summaries:
        flag = "GOAL" if goal else ("COLLIDE" if col else "TIMEOUT")
        print(f"  ep {ep}: steps={steps}, reward={rew:.1f} -> {flag}")


if __name__ == "__main__":
    main()
