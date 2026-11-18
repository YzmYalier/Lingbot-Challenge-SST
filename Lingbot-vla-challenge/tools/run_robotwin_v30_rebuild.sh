#!/usr/bin/env bash
# Rebuild into a NEW dataset; no GPU use, no writes to either input dataset.
set -euo pipefail
datafix_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
datafix_workspace="$(cd "$datafix_repo/.." && pwd)"
datafix_python="${DATAFIX_PYTHON:-$datafix_workspace/miniconda3/envs/lingbotvla/bin/python}"
datafix_datasets="$datafix_workspace/RoboTwinData/lerobot_dataset"
exec env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 10 "$datafix_python" "$datafix_repo/tools/rebuild_robotwin_v30_from_v21.py" \
  --source-v21 "${DATAFIX_SOURCE:-$datafix_datasets/RoboTwin_lerobot_v21.zip}" \
  --reference-v30 "${DATAFIX_REFERENCE:-$datafix_datasets/RoboTwin_lerobot_v30}" \
  --out "${DATAFIX_OUTPUT:-$datafix_datasets/RoboTwin_lerobot_v30_framefixed}" \
  --workers "${DATAFIX_WORKERS:-2}" "$@"
