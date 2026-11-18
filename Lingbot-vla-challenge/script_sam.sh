#!/bin/bash
# Portable RoboTwin evaluation: mask-Qwen -> judged SAM3 -> Lingbot VLA.
# All resource paths and defaults come from ../base.json. Environment variables
# override evaluation settings. Run --check before starting, or --dry-run to
# inspect configuration without model loading.
# Default checkpoint: global_step_30000; 50 tasks x 100 episodes, randomized.
# Checkpoint style: judge=1, outside_scale=0.1, edge_width=2; SAM memory=60.
# External Conda interpreters and model inputs are allowed; code and outputs stay local.
# Examples (from bundle root):
#   bash Lingbot-vla-challenge/script_sam.sh
#   TEST_NUM=1 NUM_TASKS=1 TASK_NAMES=shake_bottle_horizontally bash Lingbot-vla-challenge/script_sam.sh
# Multi-node: set the same JOB_TOKEN on all nodes, plus NODE_RANK and NNODES.
# Results: bash Lingbot-vla-challenge/experiment/robotwin/aggregate_results.sh <run_dir>
set -uo pipefail

# Paths and defaults are resolved once from the bundle's base.json.
_bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
    --dry-run|--check) exec python3 -B "${_bundle}/tools/evaluation_config.py" "$1" ;;
    -h|--help) echo 'Usage: bash script_sam.sh [--dry-run|--check]. Configure base.json; environment variables override evaluation settings.'; exit 0 ;;
    "") ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
_config_exports="$(python3 -B "${_bundle}/tools/evaluation_config.py" --shell)" || exit $?
eval "$_config_exports"
cd "$INFERENCE_WORKDIR" || exit 1

ENABLE_VIDEO="${ENABLE_VIDEO:-0}"
SAM_SAVE_VIDEO="${SAM_SAVE_VIDEO:-}"   # empty = launcher derives it from ENABLE_VIDEO
USE_LENGTH="${USE_LENGTH:-50}"

case "$SAM_FRAMES" in vla|all) ;; *) echo "ERROR: SAM_FRAMES must be vla or all, got '$SAM_FRAMES'" >&2; exit 2 ;; esac
case "$ENABLE_VIDEO" in 0|1|2) ;; *) echo "ERROR: ENABLE_VIDEO must be 0, 1 or 2, got '$ENABLE_VIDEO'" >&2; exit 2 ;; esac
case "$TEST_NUM" in ''|*[!0-9]*|0*) echo "ERROR: TEST_NUM must be a positive integer, got '$TEST_NUM'" >&2; exit 2 ;; esac
case "$USE_LENGTH" in ''|*[!0-9]*|0*) echo "ERROR: USE_LENGTH must be a positive integer, got '$USE_LENGTH'" >&2; exit 2 ;; esac
case "$EDGE_WIDTH" in ''|*[!0-9]*) echo "ERROR: EDGE_WIDTH must be an integer >= 0, got '$EDGE_WIDTH'" >&2; exit 2 ;; esac
EDGE_WIDTH=$((10#$EDGE_WIDTH))
case "$SAM_JUDGE" in 0|1) ;; *) echo "ERROR: SAM_JUDGE must be 0 or 1, got '$SAM_JUDGE'" >&2; exit 2 ;; esac
if ! awk -v s="$OUTSIDE_SCALE" 'BEGIN { exit !(s ~ /^[0-9]*\.?[0-9]+$/ && s + 0 >= 0 && s + 0 <= 1) }'; then
    echo "ERROR: OUTSIDE_SCALE must be a number in [0, 1], got '$OUTSIDE_SCALE'" >&2; exit 2
fi
# The server slices the predicted chunk to [:USE_LENGTH] and asserts it fits;
# check here so a bad value fails at launch, not at the first infer of every slot.
_chunk_size=$(grep -oP '"chunk_size"\s*:\s*\K\d+' "${MODEL_PATH}/config.json" 2>/dev/null | head -n1)
if [ -n "$_chunk_size" ] && [ "$USE_LENGTH" -gt "$_chunk_size" ]; then
    echo "ERROR: USE_LENGTH=${USE_LENGTH} exceeds the checkpoint's chunk_size=${_chunk_size}" >&2; exit 2
fi
# Tag non-default episode counts into RUN_ID so a short run can never share a
# RUN_DIR (and its claims) with a full one. 100 adds no tag, so the names and
# JOB_TOKEN-based RESUME of runs made before this knob existed stay valid.
_ep_tag=""
[ "$TEST_NUM" != "100" ] && _ep_tag="_ep${TEST_NUM}"
# Same for the executed chunk length: 50 (the old fixed value) adds no tag.
_ul_tag=""
[ "$USE_LENGTH" != "50" ] && _ul_tag="_ul${USE_LENGTH}"
# And for the cam_high rendering: judge on -> _judge; outside scale other than
# 0.2 -> _s<scale> (0.10 and .1 both -> _s0.1); outline -> _w<width>. Defaults add nothing.
_style_tag=""
[ "$SAM_JUDGE" = "1" ] && _style_tag="${_style_tag}_judge"
_scale_norm=$(awk -v s="$OUTSIDE_SCALE" 'BEGIN { printf "%g", s }')
[ "$_scale_norm" != "0.2" ] && _style_tag="${_style_tag}_s${_scale_norm}"
[ "$EDGE_WIDTH" -gt 0 ] && _style_tag="${_style_tag}_w${EDGE_WIDTH}"
# sam_frames=all episodes can exceed the launcher's 3600 s stall watchdog (see above).
if [ "$SAM_FRAMES" = "all" ]; then export STALL_TIMEOUT="${STALL_TIMEOUT:-10800}"; fi

# ---- isolated Warp kernel cache ----
_warp_key=$(printf '%s\n%s' "$MODEL_PATH" "$TASK_CONFIG" | md5sum | cut -c1-12)
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${XDG_CACHE_HOME}/warp/${_warp_key}}"
mkdir -p "$WARP_CACHE_PATH"
echo "warp cache: ${WARP_CACHE_PATH}"

# ---- node identity (platform-injected, with single-node fallbacks) ----------
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-${HOST_NUM:-1}}"
NUM_GPUS="${NUM_GPUS:-${HOST_GPU_NUM:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"
[ "${NUM_GPUS:-0}" -gt 0 ] 2>/dev/null || NUM_GPUS=1

# ---- shared run id, identical on every node ---------------------------------
# The _sam<frames> tag keeps masked runs apart from plain script.sh runs of the
# same checkpoint (they would otherwise share a RUN_ID and each other's claims).
_step_num=$(echo "$MODEL_PATH" | grep -oP 'global_step_\K\d+')
if [ -z "${RUN_ID:-}" ] && [ -z "${RUN_DIR:-}" ]; then
    job_token="${JOB_TOKEN:-}"
    if [ -z "$job_token" ]; then
        if [ "$NNODES" -gt 1 ]; then
            echo "ERROR: NNODES=${NNODES} needs an explicit JOB_TOKEN, identical on every node." >&2
            echo "       Take one timestamp at submission time and pass it to all nodes:" >&2
            echo "         JOB_TOKEN=$(date +%Y%m%d_%H%M%S) bash script_sam.sh" >&2
            exit 2
        fi
        job_token="$(date +%Y%m%d_%H%M%S)"
    fi
    RUN_ID="${_step_num}_${TASK_CONFIG}_sam${SAM_FRAMES}${_style_tag}${_ep_tag}${_ul_tag}_${job_token}"
fi

_exp_name=$(echo "$MODEL_PATH" | grep -oP '[^/]+(?=/checkpoints)')
RUN_DIR="${RUN_DIR:-${OUTPUT_BASE}/${_exp_name}_${RUN_ID:-}}"

# ---- refuse to silently resume a finished run (rank 0 only; see script.sh) ---
if [ "$NODE_RANK" = "0" ] && [ -d "${RUN_DIR}/claims" ] && [ -n "$(ls -A "${RUN_DIR}/claims" 2>/dev/null)" ]; then
    _n_claimed=$(ls -1 "${RUN_DIR}/claims" 2>/dev/null | wc -l)
    if [ "${FRESH:-0}" = "1" ]; then
        echo "FRESH=1: discarding the previous run at ${RUN_DIR} (${_n_claimed} claimed tasks)"
        rm -rf "${RUN_DIR}/claims" "${RUN_DIR}/test_num"
    elif [ "${RESUME:-0}" = "1" ]; then
        echo "RESUME=1: keeping ${_n_claimed} existing claims; only unclaimed/freed tasks will run"
    else
        echo "ERROR: ${RUN_DIR} already holds a run (${_n_claimed} tasks claimed)." >&2
        echo "         RUN_ID=<new id> bash script_sam.sh   # separate run (recommended)" >&2
        echo "         FRESH=1        bash script_sam.sh    # wipe and redo this run" >&2
        echo "         RESUME=1       bash script_sam.sh    # continue after a crash" >&2
        exit 2
    fi
fi

mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/launch_n${NODE_RANK}.log"
echo "node ${NODE_RANK}/${NNODES} ($(hostname)) gpus=${NUM_GPUS} run_dir=${RUN_DIR}"
echo "episodes per task: ${TEST_NUM}  video mode: ${ENABLE_VIDEO}"
echo "actions executed per VLA query: ${USE_LENGTH} (model chunk_size: ${_chunk_size:-unknown})"
echo "cam_high mask: sam_frames=${SAM_FRAMES} qwen_gpu=${QWEN_GPU} qwen_gpu_slots=${QWEN_GPU_SLOTS} num_per_gpu=${NUM_PER_GPU}"
echo "cam_high style: judge=${SAM_JUDGE} outside_scale=${OUTSIDE_SCALE} edge_width=${EDGE_WIDTH}"
echo "log -> ${LOG}"

# QWEN3VL_PATH is the VLA's internal backbone (Qwen3-VL-4B), unrelated to mask-Qwen.
RENDER_CHECK_ITERS="${RENDER_CHECK_ITERS:-600}" \
RENDER_CHECK_GPU="${RENDER_CHECK_GPU:-$(( (QWEN_GPU + 1) % NUM_GPUS ))}" \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path     "$MODEL_PATH" \
    --eval_workdir   "$EVAL_WORKDIR" \
    --output_base    "$OUTPUT_BASE" \
    --run_dir        "$RUN_DIR" \
    --start_port "$START_PORT" --sam_start_port "$SAM_START_PORT" --mask_qwen_port "$MASK_QWEN_PORT" \
    --sam_memory_window "$SAM_MEMORY_WINDOW" \
    --use_bf16 "$USE_BF16" --use_fp32 "$USE_FP32" --use_compile "$USE_COMPILE" \
    --task_config    "$TASK_CONFIG" \
    --num_tasks "$NUM_TASKS" --test_num "$TEST_NUM" --num_gpus "$NUM_GPUS" --num_per_gpu "$NUM_PER_GPU" \
    --node_rank "$NODE_RANK" --nnodes "$NNODES" \
    --use_length "$USE_LENGTH" \
    --sam_mask 1 --sam_frames "$SAM_FRAMES" \
    --qwen_gpu "$QWEN_GPU" --qwen_gpu_slots "$QWEN_GPU_SLOTS" \
    --outside_scale "$OUTSIDE_SCALE" --edge_width "$EDGE_WIDTH" --sam_judge "$SAM_JUDGE" \
    --enable_video "$ENABLE_VIDEO" ${SAM_SAVE_VIDEO:+--sam_save_video "$SAM_SAVE_VIDEO"} \
    > "$LOG" 2>&1
rc=$?

echo "node ${NODE_RANK} finished (rc=${rc})"
if [ "$NODE_RANK" = "0" ]; then
    echo "Aggregate the whole run (wait for every node to finish first):"
    echo "  bash experiment/robotwin/aggregate_results.sh ${RUN_DIR}"
fi
exit "$rc"
