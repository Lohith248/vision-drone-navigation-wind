#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-${ROOT_DIR}/.venv310/bin/python}"
EVAL_EPISODES="${1:-100}"
RUNS_ROOT="${2:-runs/best_vit}"

echo "Using python: ${PY_BIN}"
echo "Eval episodes: ${EVAL_EPISODES}"
echo "Runs root: ${RUNS_ROOT}"

"${PY_BIN}" "${ROOT_DIR}/scripts/eval_aggregate.py" \
  --runs-root "${ROOT_DIR}/${RUNS_ROOT}" \
  --episodes "${EVAL_EPISODES}" \
  --device auto
