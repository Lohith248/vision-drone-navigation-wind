#!/usr/bin/env python3
"""
Build all report figures from the runs/ folder + aggregate CSVs.

Outputs (PNGs) into report_artifacts/report/figures/ (same folder Overleaf uses):
  fig_reward_curves.png       — mean training reward vs steps for every run
  fig_curriculum_progression.png — curriculum stage vs steps (best PPO)
  fig_comparison_bars.png     — success / collision / mean_reward across runs
  fig_generalization.png      — bar chart of generalization scenarios
  fig_generalization_heatmap.png — success-rate heatmap (slide-friendly)
  fig_architecture.png        — network architecture diagram (multimodal ViT+state)

Usage:
  python scripts/figures/make_figures.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
# Single canonical figure dir (bundled with report.tex for Overleaf).
OUT = ROOT / "report_artifacts" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _eval_n_from_all_runs_md() -> int | None:
    md = ROOT / "report_artifacts" / "all_runs.md"
    if not md.exists():
        return None
    import re

    for line in md.read_text().splitlines():
        if "n_episodes per run" in line:
            m = re.search(r"=\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return None

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
})

# Friendly run names for the report
NAME_MAP = {
    "best_vit/ppo_seed0_t1p5m":         "PPO+ViT (seed 0, full)",
    "multi_seed/ppo_seed1":             "PPO+ViT (seed 1)",
    "baseline_ddpg/ddpg_seed0":         "DDPG (state-only)",
    "baseline_sac/sac_seed0":           "SAC (state-only)",
    "abl_no_advnorm/ppo_seed0":         "PPO no adv-norm",
    "abl_low_reward_scale/ppo_seed0":   "PPO low reward-scale",
    "abl_no_curriculum/ppo_seed0":      "PPO no curriculum",
    "abl_state_only/ppo_seed0":         "PPO state-only",
    "abl_cnn_encoder/ppo_seed0":        "PPO+CNN",
    "abl_domain_random/ppo_seed0":      "PPO+ViT + Domain Rand",
    "abl_no_time_penalty/ppo_seed0":    "PPO no time-penalty",
    "abl_no_goal_bonus/ppo_seed0":      "PPO no goal-bonus",
}

GEN_SCENARIO_LABELS = {
    "in_distribution": "In-distribution",
    "no_wind": "No wind",
    "strong_wind": "Strong wind",
    "dense_clutter": "Dense clutter (rho clamped)",
    "turns": "Turns flag (composite)",
    "narrow_corridor": "Narrow corridor",
    "noisy_sensors": "Noisy sensors",
    "long_corridor": "Long corridor (OOD length)",
}

COLORS = {
    "PPO+ViT (seed 0, full)":   "#1f77b4",
    "PPO+ViT (seed 1)":         "#5fa8db",
    "DDPG (state-only)":        "#d62728",
    "SAC (state-only)":         "#2ca02c",
    "PPO no adv-norm":          "#ff7f0e",
    "PPO low reward-scale":     "#9467bd",
    "PPO no curriculum":        "#8c564b",
    "PPO state-only":           "#e377c2",
    "PPO+CNN":                  "#17becf",
    "PPO+ViT + Domain Rand":    "#bcbd22",
    "PPO no time-penalty":      "#3b8686",
    "PPO no goal-bonus":        "#cb4b16",
}


# ==========================================================================
# 1) Reward curves from per-run metrics.csv
# ==========================================================================
def fig_reward_curves():
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0
    for rel, label in NAME_MAP.items():
        path = RUNS / rel / "metrics.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        x_col = "total_timesteps" if "total_timesteps" in df.columns else None
        y_col = "mean_episode_reward" if "mean_episode_reward" in df.columns else None
        if not (x_col and y_col):
            continue
        sub = df[[x_col, y_col]].dropna()
        if len(sub) < 2:
            continue
        # smooth
        y = sub[y_col].rolling(window=5, min_periods=1).mean()
        ax.plot(sub[x_col] / 1e3, y, label=label, color=COLORS.get(label, None), lw=1.6)
        plotted += 1
    ax.set_xlabel("Training steps (×1k)")
    ax.set_ylabel("Mean episode reward (rolling-5)")
    ax.set_title(f"Training reward curves — {plotted} runs")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = OUT / "fig_reward_curves.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


# ==========================================================================
# 2) Curriculum stage progression
# ==========================================================================
def fig_curriculum_progression():
    path = RUNS / "best_vit/ppo_seed0_t1p5m/metrics.csv"
    if not path.exists():
        print(f"  -- skip curriculum (no {path})")
        return
    df = pd.read_csv(path)
    if "curriculum_stage" not in df.columns or "total_timesteps" not in df.columns:
        print("  -- skip curriculum (no columns)")
        return
    sub = df[["total_timesteps", "curriculum_stage", "mean_episode_reward"]].dropna()
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(sub["total_timesteps"] / 1e3, sub["mean_episode_reward"]
             .rolling(window=5, min_periods=1).mean(),
             color="#1f77b4", lw=1.5, label="Reward (rolling-5)")
    ax1.set_xlabel("Training steps (×1k)")
    ax1.set_ylabel("Mean episode reward", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(sub["total_timesteps"] / 1e3, sub["curriculum_stage"],
             color="#d62728", where="post", lw=1.5, label="Curriculum stage")
    ax2.set_ylabel("Curriculum stage", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("PPO+ViT (seed 0): Reward and curriculum stage vs training steps")
    fig.tight_layout()
    out = OUT / "fig_curriculum_progression.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


# ==========================================================================
# 3) Comparison bars from report_artifacts/all_runs.csv
# ==========================================================================
def fig_comparison_bars():
    csv_path = ROOT / "report_artifacts/all_runs.csv"
    if not csv_path.exists():
        print(f"  -- skip comparison bars (run aggregate_all.py first)")
        return
    df = pd.read_csv(csv_path)
    df = df[df["mean_reward"].notna()].copy()
    df["label"] = df["run"].map(lambda r: NAME_MAP.get(r, r))
    df = df.sort_values("mean_reward", ascending=False)

    metrics = [("success_rate", "Success rate"),
               ("collision_rate", "Collision rate"),
               ("mean_reward", "Mean reward")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for ax, (metric, title) in zip(axes, metrics):
        labels = df["label"].tolist()
        vals = df[metric].astype(float).tolist()
        colors = [COLORS.get(l, "#999999") for l in labels]
        ax.barh(labels, vals, color=colors, height=0.72)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(alpha=0.35, axis="x")
        if metric in ("success_rate", "collision_rate"):
            ax.set_xlim(0, 1)
        for i, v in enumerate(vals):
            ax.annotate(
                f"{v:.2f}",
                xy=(v, i),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
    n_ep = _eval_n_from_all_runs_md() or 30
    fig.suptitle(
        f"Algorithm + ablation comparison (latest checkpoint, final curriculum stage, eval N={n_ep})",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    out = OUT / "fig_comparison_bars.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


# ==========================================================================
# 4) Generalization stress-test bars
# ==========================================================================
def fig_generalization():
    csv_path = ROOT / "report_artifacts/generalization.csv"
    if not csv_path.exists():
        print(f"  -- skip generalization (run eval_generalization.py first)")
        return
    df = pd.read_csv(csv_path)
    df = df[df.get("error", pd.Series([None] * len(df))).isna()] if "error" in df.columns else df

    n_ep = int(df["episodes"].iloc[0]) if "episodes" in df.columns and len(df) else "?"
    labels_raw = df["scenario"].tolist()
    labels = [GEN_SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in labels_raw]

    succ = df["success_rate"].astype(float).tolist() if "success_rate" in df.columns else []
    bar_colors_succ = [
        "#27ae60" if v >= 0.95 else "#e67e22" if v >= 0.5 else "#c0392b"
        for v in succ
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    for ax, (metric, title, ylim, palette) in zip(axes, [
        ("success_rate",   "Success rate",   (0, 1), bar_colors_succ),
        ("collision_rate", "Collision rate", (0, 1), None),
        ("mean_reward",    "Mean reward",    None, None),
    ]):
        if metric not in df.columns:
            continue
        vals = df[metric].astype(float).tolist()
        if palette is None and metric == "collision_rate":
            palette = ["#c0392b" if v >= 0.5 else "#27ae60" for v in vals]
        elif palette is None:
            palette = ["#2980b9"] * len(vals)
        ax.barh(labels, vals, color=palette, height=0.68)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(alpha=0.35, axis="x")
        if ylim:
            ax.set_xlim(*ylim)
        for i, v in enumerate(vals):
            ax.annotate(
                f"{v:.2f}",
                xy=(v, i),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
    fig.suptitle(f"Generalization stress-test (frozen PPO+ViT checkpoint, eval N={n_ep} episodes per scenario)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out = OUT / "fig_generalization.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


def fig_generalization_heatmap():
    """Single figure: success rate heatmap (readable at a glance)."""
    csv_path = ROOT / "report_artifacts/generalization.csv"
    if not csv_path.exists():
        print(f"  -- skip generalization heatmap")
        return
    df = pd.read_csv(csv_path)
    df = df[df.get("error", pd.Series([None] * len(df))).isna()] if "error" in df.columns else df
    if "success_rate" not in df.columns:
        return

    labels_raw = df["scenario"].tolist()
    labels = [GEN_SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in labels_raw]
    vals = df["success_rate"].astype(float).values.reshape(-1, 1)
    n_ep = int(df["episodes"].iloc[0]) if "episodes" in df.columns else "?"

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks([0])
    ax.set_xticklabels(["Success rate"], fontsize=11)
    for i in range(len(labels)):
        ax.text(0, i, f"{vals[i, 0]:.2f}", ha="center", va="center",
                color="black" if 0.35 < vals[i, 0] < 0.75 else "white",
                fontsize=10, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate")
    ax.set_title(f"OOD robustness snapshot (N={n_ep} eps / scenario)")
    fig.tight_layout()
    out = OUT / "fig_generalization_heatmap.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


# ==========================================================================
# 5) Architecture diagram (matplotlib boxes)
# ==========================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                   edgecolor="#222", linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, color="#111")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"))

    # Inputs
    box(0.2, 4.0, 2.0, 0.8, "RGB image\n64×64×3 (uint8)", "#cfe2f3")
    box(0.2, 1.2, 2.0, 0.8, "State vector (15)\nwall dists, vel, yaw,\ngoal dir, prev act", "#fce5cd")

    # Encoders
    box(2.7, 3.7, 2.4, 1.4, "ViT Encoder\n(patch=8, depth=4, heads=4)\n→ 192-d feature", "#9fc5e8")
    box(2.7, 0.9, 2.4, 1.4, "MLP State Encoder\n[15 → 128]", "#f9cb9c")

    # Fusion
    box(5.7, 2.2, 2.0, 1.6, "Concat + MLP\n[192+128 → 256 → 256]", "#b6d7a8")

    # Heads
    box(8.2, 3.4, 2.4, 1.0, "Actor head\nμ ∈ ℝ³, log σ\n(continuous a=(vx,vy,vz))",
        "#d9ead3")
    box(8.2, 1.6, 2.4, 1.0, "Critic head\nV(s) ∈ ℝ", "#fff2cc")

    # Arrows
    arrow(2.2, 4.4, 2.7, 4.4)
    arrow(2.2, 1.6, 2.7, 1.6)
    arrow(5.1, 4.4, 5.7, 3.6)
    arrow(5.1, 1.6, 5.7, 2.4)
    arrow(7.7, 3.4, 8.2, 3.9)
    arrow(7.7, 2.6, 8.2, 2.1)

    ax.set_title("Multimodal Actor-Critic for Vision-based Drone Navigation\n"
                 "(PPO with Vision Transformer + state fusion)", fontsize=12)
    out = OUT / "fig_architecture.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


REWARD_ABLATION_RUNS = [
    ("best_vit/ppo_seed0_t1p5m",       "Full reward"),
    ("abl_low_reward_scale/ppo_seed0", "Low reward-scale\n(distance ÷100)"),
    ("abl_no_time_penalty/ppo_seed0",  "No time penalty\n($-0.002\\to 0$)"),
    ("abl_no_goal_bonus/ppo_seed0",    "No goal bonus\n($+30\\to 0$)"),
]


def fig_reward_component_ablation():
    """Three side-by-side bar charts focused on reward-component ablations."""
    csv_path = ROOT / "report_artifacts/all_runs.csv"
    if not csv_path.exists():
        print("  -- skip reward-component ablation (no all_runs.csv)")
        return
    df = pd.read_csv(csv_path).set_index("run", drop=False)

    rows = []
    for run_key, label in REWARD_ABLATION_RUNS:
        if run_key in df.index:
            r = df.loc[run_key]
            rows.append({
                "label": label,
                "success_rate":   float(r.get("success_rate", float("nan"))),
                "collision_rate": float(r.get("collision_rate", float("nan"))),
                "mean_reward":    float(r.get("mean_reward", float("nan"))),
                "mean_length":    float(r.get("mean_length", float("nan"))),
            })
    if len(rows) < 2:
        print("  -- skip reward-component ablation (not enough rows yet)")
        return

    labels = [r["label"] for r in rows]
    palette = ["#1f77b4", "#9467bd", "#3b8686", "#cb4b16"][: len(rows)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    metric_specs = [
        ("success_rate",   "Success rate",   (0, 1.05)),
        ("collision_rate", "Collision rate", (0, 1.05)),
        ("mean_length",    "Mean episode length (steps)", None),
    ]
    for ax, (metric, title, ylim) in zip(axes, metric_specs):
        vals = [r[metric] for r in rows]
        bars = ax.bar(labels, vals, color=palette, edgecolor="#222", linewidth=0.6)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.35, axis="y")
        if ylim:
            ax.set_ylim(*ylim)
        for bar, v in zip(bars, vals):
            ax.annotate(
                f"{v:.2f}" if metric != "mean_length" else f"{v:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )
        ax.tick_params(axis="x", labelsize=9)
    n_ep = _eval_n_from_all_runs_md() or 30
    fig.suptitle(
        f"Reward-component ablations (PPO+ViT, eval N={n_ep} per run)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = OUT / "fig_reward_ablation.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    print("Generating figures into:", OUT)
    fig_architecture()
    fig_reward_curves()
    fig_curriculum_progression()
    fig_comparison_bars()
    fig_generalization()
    fig_generalization_heatmap()
    fig_reward_component_ablation()
    print("\nDone.")


if __name__ == "__main__":
    main()
