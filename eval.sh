#!/usr/bin/env bash
# Evaluate train.sh's output, randomized first and clean second.
# Default: global_step_30000, 100 episodes per task, raw + masked videos.
# STEP=24000 or MODEL_PATH=<other checkpoint> overrides the checkpoint.
# --dry-run / --check are forwarded to both evaluations without starting GPUs.
set -euo pipefail

_bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_bundle"
_project="$(python3 -B -c 'from submit_paths import path; print(path("vla_project"))')"

export STEP="${STEP:-30000}"
if [[ ! "$STEP" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEP must be a positive integer: $STEP" >&2
    exit 2
fi
export MODEL_PATH="${MODEL_PATH:-$(python3 -B -c 'import os; from submit_paths import path; print(path("training_run") / "checkpoints" / ("global_step_" + os.environ["STEP"]) / "hf_ckpt")')}"
export TEST_NUM="${TEST_NUM:-100}"
export ENABLE_VIDEO="${ENABLE_VIDEO:-1}"

# Keep the two protocols separate even when a custom run name/directory is set.
TASK_CONFIG=demo_randomized RUN_ID="${RUN_ID:+${RUN_ID}_demo_randomized}" RUN_DIR="${RUN_DIR:+${RUN_DIR%/}/demo_randomized}" bash "$_project/script_sam.sh" "$@"
TASK_CONFIG=demo_clean RUN_ID="${RUN_ID:+${RUN_ID}_demo_clean}" RUN_DIR="${RUN_DIR:+${RUN_DIR%/}/demo_clean}" bash "$_project/script_sam.sh" "$@"
