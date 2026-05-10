#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-${ROOT_DIR}/.venv310/bin/python}"
TIMESTEPS="${1:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
SEEDS="${SEEDS:-0 1 2}"
NUM_ENVS="${NUM_ENVS:-4}"
RUNS_ROOT="${RUNS_ROOT:-runs/best_vit}"

echo "Using python: ${PY_BIN}"
echo "Timesteps per run: ${TIMESTEPS}"
echo "Seeds: ${SEEDS}"
echo "Runs root: ${RUNS_ROOT}"

for SEED in ${SEEDS}; do
  echo "=== DDPG seed=${SEED} ==="
  "${PY_BIN}" -m drone_rl.train \
    --algo ddpg \
    --config drone_rl/configs/ddpg_best.yaml \
    --total-timesteps "${TIMESTEPS}" \
    --seed "${SEED}" \
    --device auto \
    --no-resume \
    --log-dir "${RUNS_ROOT}/ddpg_seed${SEED}"

  echo "=== PPO seed=${SEED} ==="
  "${PY_BIN}" -m drone_rl.train \
    --algo ppo \
    --config drone_rl/configs/ppo_best.yaml \
    --total-timesteps "${TIMESTEPS}" \
    --seed "${SEED}" \
    --device auto \
    --num-envs "${NUM_ENVS}" \
    --curriculum \
    --no-resume \
    --log-dir "${RUNS_ROOT}/ppo_seed${SEED}"
done

echo "Training complete."
echo "Run evaluation: scripts/eval_suite.sh ${EVAL_EPISODES} ${RUNS_ROOT}"
