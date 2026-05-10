#!/usr/bin/env python3
"""
Splice aggregated results into FINAL_REPORT.md, replacing placeholders.

Inputs:
  - report_artifacts/all_runs.csv   (from aggregate_all.py)
  - report_artifacts/generalization.csv (from eval_generalization.py)

Output:
  - FINAL_REPORT.md is updated in place (a backup is left at .bak)
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "FINAL_REPORT.md"
ALL_CSV = ROOT / "report_artifacts/all_runs.csv"
GEN_CSV = ROOT / "report_artifacts/generalization.csv"

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
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def md_table(rows: list[dict], cols: list[str], header: list[str] | None = None) -> str:
    header = header or cols
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def build_comparison_block(rows: list[dict]) -> str:
    if not rows:
        return "_No runs found in `all_runs.csv`. Run `scripts/aggregate_all.py` first._"
    enriched = []
    for r in rows:
        e = dict(r)
        e["label"] = NAME_MAP.get(r.get("run", ""), r.get("run", ""))
        enriched.append(e)
    enriched.sort(key=lambda r: float(r.get("mean_reward", 0)), reverse=True)
    return md_table(enriched,
                    cols=["label", "algorithm", "seed", "total_timesteps",
                          "params", "success_rate", "collision_rate",
                          "mean_reward", "mean_length"],
                    header=["Run", "Algo", "Seed", "Steps", "Params",
                            "Success", "Collision", "Reward", "Ep len"])


def build_ablation_results(rows: list[dict]) -> dict[str, str]:
    """Map each ablation run name -> short result string for splicing."""
    out = {}
    for r in rows:
        run = r.get("run", "")
        succ = r.get("success_rate", "?")
        coll = r.get("collision_rate", "?")
        rew  = r.get("mean_reward", "?")
        out[run] = f"success={succ} / collision={coll} / reward={rew}"
    return out


def build_seed_table(rows: list[dict]) -> str:
    wanted = ["best_vit/ppo_seed0_t1p5m", "multi_seed/ppo_seed1"]
    sub = [r for r in rows if r.get("run") in wanted]
    if not sub:
        return ""
    out = ["| Run | Steps | Mean reward | Success | Collision |",
           "|---|---|---|---|---|"]
    for r in sub:
        out.append(f"| `{r['run']}` | {r.get('total_timesteps','?')} | "
                   f"{r.get('mean_reward','?')} | {r.get('success_rate','?')} | "
                   f"{r.get('collision_rate','?')} |")
    return "\n".join(out)


def build_gen_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    out = ["| Scenario | Success | Collision | Reward | Ep length |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| `{r.get('scenario','?')}` | "
                   f"{r.get('success_rate','?')} | "
                   f"{r.get('collision_rate','?')} | "
                   f"{r.get('mean_reward','?')} | "
                   f"{r.get('mean_length','?')} |")
    return "\n".join(out)


def main():
    if not REPORT.exists():
        print("FINAL_REPORT.md not found")
        return

    all_rows = read_csv(ALL_CSV)
    gen_rows = read_csv(GEN_CSV)

    text = REPORT.read_text()
    backup = REPORT.with_suffix(".md.bak")
    backup.write_text(text)

    # 1. Big comparison table
    table = build_comparison_block(all_rows)
    text = text.replace("<!-- AUTO-INSERT-COMPARISON-TABLE -->", table)

    # 2. Ablation per-row results
    abl_results = build_ablation_results(all_rows)
    abl_map = {
        "abl_no_advnorm/ppo_seed0":       "**A.",
        "abl_low_reward_scale/ppo_seed0": "**B.",
        "abl_no_curriculum/ppo_seed0":    "**C.",
        "abl_state_only/ppo_seed0":       "**D.",
        "abl_cnn_encoder/ppo_seed0":      "**E.",
        "abl_domain_random/ppo_seed0":    "**F.",
    }
    # Replace the AUTO placeholders in the ablation table row by row.
    # Walk the lines and substitute the first <!-- AUTO --> per ablation row.
    lines = text.split("\n")
    new_lines = []
    for ln in lines:
        replaced = False
        for run_key, anchor in abl_map.items():
            if anchor in ln and "<!-- AUTO -->" in ln:
                res = abl_results.get(run_key, "_(run not finished)_")
                ln = ln.replace("<!-- AUTO -->", res)
                replaced = True
                break
        new_lines.append(ln)
    text = "\n".join(new_lines)

    # 3. Multi-seed table
    seed_table = build_seed_table(all_rows)
    if seed_table:
        # Replace the entire seed table block (lines containing "<!-- AUTO -->" near `multi_seed`)
        # We do it with a marker approach: insert table after `## 7. Multi-seed validation`
        marker = "## 7. Multi-seed validation"
        if marker in text and "<!-- AUTO -->" in text:
            # remove the placeholder table (lines until next blank-line after marker)
            head, sep, tail = text.partition(marker)
            # find end of the placeholder table block
            block_end = tail.find("\n## ")
            if block_end > 0:
                tail_rest = tail[block_end:]
            else:
                tail_rest = ""
            tail_intro = "\n\nWe re-ran the full PPO+ViT recipe with `seed=1` for 800 k steps to validate seed robustness.\n\n"
            text = head + sep + tail_intro + seed_table + "\n\n" + tail_rest

    # 4. Generalization table
    gen_table = build_gen_table(gen_rows)
    if gen_table:
        marker = "## 8. Generalization stress-test (out-of-distribution)"
        if marker in text:
            head, sep, tail = text.partition(marker)
            block_end = tail.find("\n## ")
            if block_end > 0:
                tail_rest = tail[block_end:]
            else:
                tail_rest = ""
            tail_intro = ("\n\nThe proposal motivates \"real-world robustness\". "
                          "We re-evaluate the **best PPO+ViT checkpoint without retraining** "
                          "on the scenarios listed below.\n\n"
                          "![Generalization](report_artifacts/figures/fig_generalization.png)\n\n"
                          "![Generalization success heatmap](report_artifacts/figures/fig_generalization_heatmap.png)\n\n")
            text = head + sep + tail_intro + gen_table + "\n\n" + tail_rest

    REPORT.write_text(text)
    print(f"✓ Updated {REPORT}")
    print(f"  backup: {backup}")


if __name__ == "__main__":
    main()
