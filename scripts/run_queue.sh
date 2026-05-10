#!/usr/bin/env bash
# Sequential training queue for the drone RL project.
# Runs every required experiment for the report (ablations + baseline + seed 1).
# Idempotent: skips a job whose log dir already has a best_model.pt.
#
# Usage:
#   ./scripts/run_queue.sh           # run everything not already finished
#   ./scripts/run_queue.sh skip-done # same (default)
#   ./scripts/run_queue.sh force     # rerun everything (slow)

set -euo pipefail

ROOT="/home/user35/project_work"
PYTHON="${ROOT}/.venv310/bin/python"
QUEUE_LOG_DIR="${ROOT}/runs/queue_logs"
mkdir -p "${QUEUE_LOG_DIR}"

cd "${ROOT}"

MODE="${1:-skip-done}"

# (config-path, log-dir-relative-to-runs)
JOBS=(
  "drone_rl/configs/ablations/ddpg_curriculum.yaml          | runs/baseline_ddpg/ddpg_seed0"
  "drone_rl/configs/ablations/sac_curriculum.yaml           | runs/baseline_sac/sac_seed0"
  "drone_rl/configs/ablations/ppo_seed1.yaml                | runs/multi_seed/ppo_seed1"
  "drone_rl/configs/ablations/ppo_no_advnorm.yaml           | runs/abl_no_advnorm/ppo_seed0"
  "drone_rl/configs/ablations/ppo_low_reward_scale.yaml     | runs/abl_low_reward_scale/ppo_seed0"
  "drone_rl/configs/ablations/ppo_no_curriculum.yaml        | runs/abl_no_curriculum/ppo_seed0"
  "drone_rl/configs/ablations/ppo_state_only.yaml           | runs/abl_state_only/ppo_seed0"
  "drone_rl/configs/ablations/ppo_cnn_encoder.yaml          | runs/abl_cnn_encoder/ppo_seed0"
  "drone_rl/configs/ablations/ppo_domain_random.yaml        | runs/abl_domain_random/ppo_seed0"
)

echo "================================================================"
echo "  DRONE RL :: TRAINING QUEUE"
echo "  Mode: ${MODE}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPU:"
nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader || echo "  (nvidia-smi unavailable)"
echo "================================================================"

QUEUE_T0=$(date +%s)
PASSED=()
FAILED=()
SKIPPED=()

for entry in "${JOBS[@]}"; do
  cfg=$(echo "${entry}" | cut -d'|' -f1 | xargs)
  log_dir=$(echo "${entry}" | cut -d'|' -f2 | xargs)
  job_name=$(basename "${log_dir}")
  parent=$(basename "$(dirname "${log_dir}")")
  full_name="${parent}/${job_name}"

  echo
  echo "----------------------------------------------------------------"
  echo "  JOB :: ${full_name}"
  echo "  cfg :: ${cfg}"
  echo "  log :: ${log_dir}"
  echo "----------------------------------------------------------------"

  if [[ "${MODE}" != "force" && -f "${log_dir}/checkpoints/best_model.pt" ]]; then
    echo "  [SKIP] best_model.pt already exists"
    SKIPPED+=("${full_name}")
    continue
  fi

  mkdir -p "${log_dir}"
  job_log="${QUEUE_LOG_DIR}/${parent}_${job_name}.log"

  T0=$(date +%s)
  if "${PYTHON}" -m drone_rl.train --config "${cfg}" --no-resume \
       > "${job_log}" 2>&1; then
    T1=$(date +%s); DUR=$(( T1 - T0 ))
    echo "  [OK]   completed in ${DUR}s -> ${job_log}"
    PASSED+=("${full_name} (${DUR}s)")
  else
    T1=$(date +%s); DUR=$(( T1 - T0 ))
    echo "  [FAIL] after ${DUR}s -- see ${job_log}"
    FAILED+=("${full_name} (${DUR}s)")
  fi

  # Free GPU before next job
  nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader || true
  sleep 2
done

QUEUE_T1=$(date +%s)
TOTAL=$(( QUEUE_T1 - QUEUE_T0 ))

echo
echo "================================================================"
echo "  QUEUE COMPLETE in $((TOTAL/60))m $((TOTAL%60))s"
echo "  Passed:  ${#PASSED[@]}"
for j in "${PASSED[@]}";  do echo "    OK $j"; done
echo "  Skipped: ${#SKIPPED[@]}"
for j in "${SKIPPED[@]}"; do echo "    - $j"; done
echo "  Failed:  ${#FAILED[@]}"
for j in "${FAILED[@]}";  do echo "    FAIL $j"; done
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
