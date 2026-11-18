#!/usr/bin/env bash
# Same command on every node; platform injection matches train_multinode.sh.
set -euo pipefail
datafix_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
datafix_base=/mnt/data/liangjiaqi/yzm
datafix_python="${DATAFIX_PYTHON:-$datafix_base/miniconda3/envs/lingbotvla/bin/python}"
datafix_datasets="$datafix_base/RoboTwinData/lerobot_dataset"
export NODE_RANK="${NODE_RANK:-0}" NNODES="${NNODES:-${HOST_NUM:-1}}"
export MASTER_PORT="${MASTER_PORT:-62500}"
if [[ "$NNODES" -gt 1 && -z "${MASTER_ADDR:-}" ]]; then
  echo 'NNODES > 1 requires MASTER_ADDR, identical on all nodes (same convention as training).' >&2
  exit 2
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# MASTER_ADDR/PORT identify the job; coordination uses the shared filesystem, not a TCP port.
datafix_args=()
if [[ -n "${DATAFIX_REUSE_WORK:-}" ]]; then
  datafix_args+=(--reuse-work "$DATAFIX_REUSE_WORK")
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec nice -n 10 "$datafix_python" -u "$datafix_repo/tools/rebuild_robotwin_sam_multinode.py" \
  --raw "${DATAFIX_RAW:-$datafix_datasets/RoboTwin_lerobot_v30_framefixed}" \
  --old-sam "${DATAFIX_OLD_SAM:-$datafix_datasets/RoboTwin_lerobot_v30_sam_judged_multi_outline0.1_w2}" \
  --out "${DATAFIX_SAM_OUTPUT:-$datafix_datasets/RoboTwin_lerobot_v30_framefixed_sam_judged_multi_outline0.1_w2}" \
  --work "${DATAFIX_SAM_WORK:-$datafix_base/datafix-sam-multinode-work}" \
  --pipeline "$datafix_base/auto_mask/pipeline_lerobot.py" \
  --compose-workers "${DATAFIX_COMPOSE_WORKERS:-auto}" \
  --encoder-threads "${DATAFIX_ENCODER_THREADS:-2}" \
  --report-every "${DATAFIX_REPORT_EVERY:-30}" \
  "${datafix_args[@]}" "$@"
