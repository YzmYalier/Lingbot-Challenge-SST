#!/usr/bin/env bash
set -euo pipefail

datafix_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
datafix_workspace="/mnt/data/liangjiaqi/yzm"
datafix_python="${DATAFIX_PYTHON:-$datafix_workspace/miniconda3/envs/lingbotvla/bin/python}"
datafix_datasets="$datafix_workspace/RoboTwinData/lerobot_dataset"

PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 10 "$datafix_python" -u "$datafix_repo/tools/rebuild_robotwin_sam.py" \
  --raw "${DATAFIX_RAW:-$datafix_datasets/RoboTwin_lerobot_v30_framefixed}" \
  --old-sam "${DATAFIX_OLD_SAM:-$datafix_datasets/RoboTwin_lerobot_v30_sam_judged_multi_outline0.1_w2}" \
  --out "${DATAFIX_SAM_OUTPUT:-$datafix_datasets/RoboTwin_lerobot_v30_framefixed_sam_judged_multi_outline0.1_w2}" \
  --work "${DATAFIX_SAM_WORK:-$datafix_workspace/datafix-sam-incremental-work}" \
  --pipeline "$datafix_workspace/auto_mask/pipeline_lerobot.py" \
  --compose-workers "${DATAFIX_COMPOSE_WORKERS:-auto}" \
  --encoder-threads "${DATAFIX_ENCODER_THREADS:-2}" \
  --report-every "${DATAFIX_REPORT_EVERY:-30}" \
  "$@"
