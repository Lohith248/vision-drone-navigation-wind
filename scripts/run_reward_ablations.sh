#!/usr/bin/env bash
# Wait for the no_time_penalty training to finish, then:
#  1) train no_goal_bonus
#  2) re-evaluate every interesting run at N=200
#  3) re-evaluate generalization at N=100
#  4) attach Wilson 95% CIs to both outputs
#  5) regenerate figures and splice results into the report
set -e
cd "$(dirname "$0")/.."
PY=.venv310/bin/python
LOG=runs/queue_logs/reward_ablation_pipeline.log
mkdir -p "$(dirname "$LOG")"

echo "=== reward-ablation pipeline ===" >> "$LOG"
date >> "$LOG"

# 1) Wait for no_time_penalty checkpoint to appear and training process to exit.
echo "[1/6] waiting for no_time_penalty training to finish ..." >> "$LOG"
while pgrep -af "ppo_no_time_penalty.yaml" > /dev/null; do
    sleep 30
done
echo "    no_time_penalty done." >> "$LOG"

# 2) Train no_goal_bonus on the freed GPU.
echo "[2/6] training no_goal_bonus ..." >> "$LOG"
$PY -m drone_rl.train --config drone_rl/configs/ablations/ppo_no_goal_bonus.yaml --no-resume \
    >> runs/queue_logs/abl_no_goal_bonus.log 2>&1
echo "    no_goal_bonus done." >> "$LOG"

# 3) Aggregate every run at N=200 (uses existing checkpoints; CPU device to avoid GPU contention with anything else).
echo "[3/6] aggregate_all at N=200 ..." >> "$LOG"
$PY scripts/aggregate_all.py --episodes 200 --device cuda \
    --out report_artifacts/all_runs.csv \
    --md-out report_artifacts/all_runs.md >> "$LOG" 2>&1

# 4) Generalization at N=100 on the best PPO+ViT checkpoint.
BEST_CKPT=$(ls -1 runs/best_vit/*/checkpoints/best_model.pt 2>/dev/null | head -n 1)
BEST_CFG=$(dirname "$(dirname "$BEST_CKPT")")/config.yaml
if [ -n "$BEST_CKPT" ] && [ -f "$BEST_CFG" ]; then
    echo "[4/6] generalization eval (N=100) on $BEST_CKPT" >> "$LOG"
    $PY scripts/eval_generalization.py \
        --checkpoint "$BEST_CKPT" --config "$BEST_CFG" \
        --episodes 100 --device cuda \
        --out report_artifacts/generalization.csv >> "$LOG" 2>&1
else
    echo "[4/6] no best_vit checkpoint found; skipping generalization re-eval" >> "$LOG"
fi

# 5) Wilson 95% CIs.
echo "[5/6] Wilson CIs ..." >> "$LOG"
$PY scripts/wilson_ci.py --in report_artifacts/all_runs.csv --episodes 200 \
    --out report_artifacts/all_runs_ci.csv \
    --md-out report_artifacts/all_runs_ci.md >> "$LOG" 2>&1
$PY scripts/wilson_ci.py --in report_artifacts/generalization.csv --episodes 100 \
    --out report_artifacts/generalization_ci.csv \
    --md-out report_artifacts/generalization_ci.md >> "$LOG" 2>&1

# 6) Regenerate figures and re-splice the report.
echo "[6/6] regenerating figures + filling report ..." >> "$LOG"
$PY scripts/figures/make_figures.py >> "$LOG" 2>&1 || true
$PY scripts/fill_report.py >> "$LOG" 2>&1 || true
$PY scripts/fill_report_tex.py >> "$LOG" 2>&1 || true

echo "=== pipeline complete ===" >> "$LOG"
date >> "$LOG"
