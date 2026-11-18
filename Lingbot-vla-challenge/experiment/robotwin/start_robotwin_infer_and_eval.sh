#!/bin/bash
# ============================================================
# Inference + simulation launcher (queue-scheduled).
# num_per_gpu inference servers stay resident per GPU; sim tasks match
# inference slots from a queue. Finishing a task frees its slot and starts the next.
#
# Usage: bash start_robotwin_infer_and_eval.sh [options]
#   --model_path        inference model path (from base.json via script_sam.sh)
#   --inference_script  inference-side module path (default: deploy/lingbot_vla_v2_policy.py)
#   --inference_workdir inference-side working dir (default: current working dir)
#   --eval_workdir      sim-side working directory (from base.json)
#   --output_base       result output path (from base.json)
#   --start_port        starting port (default: 9330)
#   --pid_name          PID file prefix (default: test_pid)
#   --num_tasks         number of sim tasks, taken in order from the task list (default: 50, max: 50)
#   --test_num          eval episodes PER TASK (default: 100 = the benchmark, or $EVAL_TEST_NUM)
#   --num_gpus          total GPUs (default: 1)
#   --num_per_gpu       inference servers per GPU (default: 1)
#   --use_length        actions executed per VLA query; the model still predicts its
#                       full chunk_size, the server returns the first N (default: 50)
#   --robo_name         robot config name (default: robotwin)
#   --video_fps         video recording fps (default: 10)
#   --enable_video      0: no videos (default) | 1: eval video of the VLA query frames + SAM
#                       masked video | 2: like 1, but the eval video renders every step
#   --no_video          same as --enable_video 0
#   --use_bf16          use bfloat16 inference (default: False)
#   --use_fp32          use float32 inference (default: True; release reproduction setting)
#   --use_compile       enable model compile in policy inference (default: True)
#   --task_config       RoboTwin task config (default: demo_clean, or $TASK_CONFIG)
#   --keep_inference    keep inference servers resident after simulation
#   --run_dir           shared result dir (multi-node: every node MUST pass the same
#                       path, so they share one claim dir and one result set). When
#                       omitted the name is derived from model_path + timestamp.
#   --node_rank         this node's index, 0-based (default: 0, or $NODE_RANK)
#   --nnodes            total node count (default: 1, or $NNODES)
#
# Multi-node: nodes coordinate purely through the shared filesystem -- no central
# service, no ssh. Each free inference slot claims the next unclaimed task with an
# atomic `mkdir` under <run_dir>/claims/; the winner owns that task. This balances
# dynamically (a fast node takes more tasks) instead of statically partitioning,
# so no node idles while another still has a queue. Launch this same command on
# every node with the same --run_dir; see script.sh for the wrapper that derives a
# shared run id from the platform's MASTER_ADDR.
#
# Configure device allocation and inference concurrency before launching.
# ============================================================

# ===== Configuration from script_sam.sh / base.json =====
: "${ROLE_RUNNER:?Launch through Lingbot-vla-challenge/script_sam.sh}"
project_root="$INFERENCE_WORKDIR"
inference_workdir="${project_root%/}/"
inference_script="deploy/lingbot_vla_v2_policy.py"
model_path="$MODEL_PATH"
eval_workdir="$EVAL_WORKDIR"
output_base="$OUTPUT_BASE"
start_port=9330
pid_name="test_pid"
num_tasks=50
# Episodes per task (not the task count -- that is num_tasks). 100 is the benchmark.
test_num="${EVAL_TEST_NUM:-100}"
num_gpus=1
num_per_gpu=1
use_length=50
use_bf16=False
use_fp32=True
use_compile=True
robo_name="robotwin"
# Must match the checkpoint's training `data.action_space`. The inference server
# reads it from the checkpoint itself, but the sim-side client needs its own copy
# to send the 16-dim endpose state and call take_action(action_type='ee').
action_space="${ACTION_SPACE:-joint}"
video_fps=10
keep_inference=false
# Multi-node coordination. Defaults make a bare invocation behave exactly as
# before (single node, rank 0 of 1). HOST_NUM / HOST_GPU_NUM are what the
# reference launcher reads from the platform, so honour those too.
run_dir_arg=""
node_rank="${NODE_RANK:-0}"
nnodes="${NNODES:-${HOST_NUM:-1}}"
# Video saving mode (--enable_video):
#   0  nothing: no eval mp4, no SAM masked mp4. The default -- recording is what
#      made the sam_multi run ~5x slower, and per-episode mp4 recording pipes raw
#      frames into ffmpeg over a 64 KB pipe, which deadlocks workers when the
#      output share is busy (the reference launcher's ROBOTWIN_DISABLE_EVAL_VIDEO=1).
#   1  SAM masked mp4 + eval mp4 of the raw cam_high frames the VLA is queried
#      with. No extra rendering: within a chunk every step repeats its query frame.
#   2  SAM masked mp4 + eval mp4 rendered after EVERY step (smooth video, one
#      extra all-camera render per step -- noticeably slower).
enable_video=0
task_config="${TASK_CONFIG:-demo_clean}"
# ---- cam_high SAM3 masking (off by default: the original eval, unchanged) ----
# mask-Qwen = Qwen3.8-27B served by vLLM, used ONLY to turn (frame 0, instruction)
# into SAM3 noun phrases. It is NOT the VLA's internal Qwen3-VL-4B (QWEN3VL_PATH).
sam_mask=0
sam_frames=vla              # vla: frames the VLA is queried with | all: every executed step
qwen_gpu=0                  # GPU that hosts the mask-Qwen vLLM server
qwen_gpu_slots=0            # VLA+SAM3+sim slots that ALSO run on qwen_gpu
mask_qwen_port=8000
sam_start_port=9630         # SAM3 stream server of slot i listens on sam_start_port + i

outside_scale=0.2
edge_width=0                # 0: plain dim (default) | N>0: white N-px ring outside the mask (auto_mask/mask_style.py outline)
sam_judge=0                 # 1: mask-Qwen judges every SAM3 object (/start_judged), as in the *_sam_judged_* sets
sam_save_video=""           # save episode*_cam_high_masked.mp4; empty = follow --enable_video (0 -> False)
# SAM3 tracker memory window in frames (0 = keep all). A bounded window
# limits retained state; choose it for the workload and available memory.
sam_memory_window=60

while [[ $# -gt 0 ]]; do
    case $1 in
        --inference_workdir) inference_workdir="$2"; shift 2 ;;
        --inference_script)  inference_script="$2";  shift 2 ;;
        --model_path)        model_path="$2";        shift 2 ;;
        --eval_workdir)      eval_workdir="$2";      shift 2 ;;
        --output_base)       output_base="$2";       shift 2 ;;
        --start_port)        start_port="$2";        shift 2 ;;
        --pid_name)          pid_name="$2";          shift 2 ;;
        --num_tasks)         num_tasks="$2";         shift 2 ;;
        --test_num)          test_num="$2";          shift 2 ;;
        --num_gpus)          num_gpus="$2";          shift 2 ;;
        --num_per_gpu)       num_per_gpu="$2";       shift 2 ;;
        --use_length)        use_length="$2";        shift 2 ;;
        --use_bf16)          use_bf16="$2";        shift 2 ;;
        --use_fp32)          use_fp32="$2";        shift 2 ;;
        --use_compile)       use_compile="$2";     shift 2 ;;
        --robo_name)         robo_name="$2";         shift 2 ;;
        --action_space)      action_space="$2";      shift 2 ;;
        --task_config)       task_config="$2";       shift 2 ;;
        --video_fps)         video_fps="$2";         shift 2 ;;
        --enable_video)      enable_video="$2";      shift 2 ;;
        --no_video)          enable_video=0;         shift ;;
        --keep_inference)    keep_inference=true;    shift ;;
        --run_dir)           run_dir_arg="$2";       shift 2 ;;
        --node_rank)         node_rank="$2";         shift 2 ;;
        --nnodes)            nnodes="$2";            shift 2 ;;
        --sam_mask)          sam_mask="$2";          shift 2 ;;
        --sam_frames)        sam_frames="$2";        shift 2 ;;
        --qwen_gpu)          qwen_gpu="$2";          shift 2 ;;
        --qwen_gpu_slots)    qwen_gpu_slots="$2";    shift 2 ;;
        --mask_qwen_port)    mask_qwen_port="$2";    shift 2 ;;
        --sam_start_port)    sam_start_port="$2";    shift 2 ;;
        --outside_scale)     outside_scale="$2";     shift 2 ;;
        --edge_width)        edge_width="$2";        shift 2 ;;
        --sam_judge)         sam_judge="$2";         shift 2 ;;
        --sam_save_video)    sam_save_video="$2";    shift 2 ;;
        --sam_memory_window) sam_memory_window="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash $0 [options]"
            echo "  --model_path        inference model path"
            echo "  --inference_script  inference-side module path"
            echo "  --inference_workdir inference-side working dir"
            echo "  --eval_workdir      sim-side (RoboTwin repo) working dir (REQUIRED; or \$EVAL_WORKDIR)"
            echo "  --output_base       result output path"
            echo "  --start_port        starting port (default: 9330)"
            echo "  --pid_name          PID file prefix (default: test_pid)"
            echo "  --num_tasks         number of sim tasks (default: 50, max: 50)"
            echo "  --test_num          eval episodes per task (default: 100, or \$EVAL_TEST_NUM)"
            echo "  --num_gpus          total GPUs (default: 1)"
            echo "  --num_per_gpu       inference servers per GPU (default: 1)"
            echo "  --use_length        actions executed per VLA query, <= chunk_size (default: 50)"
            echo "  --use_bf16          use bfloat16 inference (default: False)"
            echo "  --use_fp32          use float32 inference (default: True; release reproduction setting)"
            echo "  --use_compile       enable model compile (default: True)"
            echo "  --robo_name         robot config name (default: robotwin)"
            echo "  --action_space      joint or ee; must match the checkpoint (default: joint)"
            echo "  --task_config       RoboTwin task config (default: demo_clean)"
            echo "  --keep_inference    keep inference servers resident after simulation"
            echo "  --run_dir           shared result dir (multi-node: same path on every node)"
            echo "  --node_rank         this node's index, 0-based (default: 0, or \$NODE_RANK)"
            echo "  --nnodes            total node count (default: 1, or \$NNODES/\$HOST_NUM)"
            echo "  --video_fps         video recording fps (default: 10)"
            echo "  --enable_video      0 = no videos (default) | 1 = eval video of VLA query frames + SAM video | 2 = eval video of every step + SAM video"
            echo "  --no_video          same as --enable_video 0"
            echo "  --sam_mask          1 = SAM3-mask cam_high (mask-Qwen Qwen3.8-27B -> SAM3); 0 = original eval (default)"
            echo "  --sam_frames        vla (frames the VLA sees, default) | all (every executed step)"
            echo "  --qwen_gpu          GPU hosting the mask-Qwen vLLM server (default: 0)"
            echo "  --qwen_gpu_slots    eval slots that also run on qwen_gpu (default: 0)"
            echo "  --mask_qwen_port    mask-Qwen vLLM port (default: 8000)"
            echo "  --sam_start_port    first SAM3 stream server port (default: 9630)"
            echo "  --outside_scale     brightness kept outside the mask, 0..1 (default: 0.2)"
            echo "  --edge_width        white outline ring width in px just outside the mask; 0 = none (default: 0)"
            echo "  --sam_judge         1 = mask-Qwen judges every SAM3 object (drops distractors); 0 = keep all (default)"
            echo "  --sam_save_video    save masked cam_high videos (default: False for --enable_video 0, else True)"
            echo "  --sam_memory_window SAM3 tracker memory window in frames (default: 60; 0 = keep all)"
            exit 0 ;;
        *)
            echo -e "\033[31mUnknown argument: $1\033[0m"; exit 1 ;;
    esac
done


# ===== Common environment =====
# Cleanup: kill all child processes on exit / Ctrl-C / kill
cleanup() {
    # Idempotent: a signal runs it, then the EXIT trap would run it again.
    [ "${_cleanup_done:-0}" = "1" ] && return
    _cleanup_done=1
    echo ""
    echo -e "\033[33m=== Cleaning up all child processes ===\033[0m"
    # Kill inference servers (and the per-slot SAM3 stream servers)
    for slot in $(seq 0 $((num_slots-1))); do
        for local_pid in ${sam_pids[$slot]:-0}; do
            if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
                kill -TERM -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                    || kill -TERM "$local_pid" 2>/dev/null || true
            fi
        done
        local_pid=${inference_pids[$slot]:-0}
        if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
            kill -TERM -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                || kill -TERM "$local_pid" 2>/dev/null || true
        fi
    done
    # Kill eval workers
    for slot in $(seq 0 $((num_slots-1))); do
        local_pid=${slot_pid[$slot]:-0}
        if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
            kill -TERM -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                || kill -TERM "$local_pid" 2>/dev/null || true
        fi
    done
    sleep 1
    # Force kill survivors
    for slot in $(seq 0 $((num_slots-1))); do
        for local_pid in ${inference_pids[$slot]:-0} ${slot_pid[$slot]:-0} ${sam_pids[$slot]:-0}; do
            if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
                kill -KILL -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                    || kill -KILL "$local_pid" 2>/dev/null || true
            fi
        done
    done
    # Stop the mask-Qwen vLLM server only if this launcher started it
    if [ "${started_mask_qwen:-false}" = "true" ] && [ -n "${mask_qwen_pid:-}" ]; then
        kill -TERM -- -"${mask_qwen_pid}" 2>/dev/null || kill -TERM "${mask_qwen_pid}" 2>/dev/null || true
        echo -e "\033[33m  mask-Qwen vLLM (PID ${mask_qwen_pid}) stopped\033[0m"
    fi
    echo -e "\033[33m=== Cleanup done ===\033[0m"
}
# INT/TERM must also exit: with a bare `trap cleanup INT TERM` bash runs cleanup
# and then RESUMES the scheduling loop, which relaunches the tasks it just killed.
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM



# ===== Working dir (RoboTwin sim side) -- REQUIRED =====
# Must be supplied via --eval_workdir flag or $EVAL_WORKDIR env. No default.
if [ -z "$eval_workdir" ]; then
    eval_workdir="${EVAL_WORKDIR:-}"
fi
if [ -z "$eval_workdir" ]; then
    echo -e "\033[31mError: --eval_workdir (or \$EVAL_WORKDIR) is required (the RoboTwin repo root).\033[0m"
    exit 1
fi
if [ ! -d "$eval_workdir" ]; then
    echo -e "\033[31mError: sim workdir '${eval_workdir}' not found.\033[0m"
    echo "  Pass --eval_workdir <path> or set EVAL_WORKDIR=<path> (the RoboTwin repo root)."
    exit 1
fi
echo -e "\033[36mSim workdir: ${eval_workdir}\033[0m"

# ===== Sim-side interpreter and graphics dependencies (read-only system) =====
_sim_site="$(python3 -B "$ROLE_RUNNER" robotwin -c 'import site;print(site.getsitepackages()[0])')" || exit 1
_egl_vendor="${_sim_site}/sapien/vulkan_library/10_nvidia.json"
[ ! -f "$_egl_vendor" ] || export __EGL_VENDOR_LIBRARY_FILENAMES="$_egl_vendor"
# SAPIEN can render headlessly. An existing DISPLAY may be inherited; this
# submission never installs drivers or creates an X server outside the bundle.

# ===== Render pre-flight =====
# A degraded Vulkan stack does not fail loudly: the eval loop catches the render
# exception, bumps the seed and retries forever, so the run burns hours producing
# nothing. Fail here instead. SAPIEN can also lose a race with a just-started
# Xvfb and pass seconds later, hence the retries.
render_check() {
    # 600 frames x 3 cameras @ 384x320, 3 tries, 120 s timeout: the same budget
    # the reference multi-node launcher uses. Measured ~24 s healthy on this node
    # (ray-traced shader initialization), so the timeout allows a cold start.
    # A weaker check lets a node with a degraded Vulkan state through, and it then
    # fails most tasks.
    local iters="${RENDER_CHECK_ITERS:-600}"
    local tries="${RENDER_CHECK_TRIES:-3}"
    local script="${inference_workdir}experiment/robotwin/render_check.py"
    local t out rc
    [ "$iters" -gt 0 ] || return 0
    if [ ! -f "$script" ]; then
        echo -e "\033[33mWarning: ${script} missing; skipping render pre-flight\033[0m"
        return 0
    fi
    # Pin the check to one GPU. Without this SAPIEN picks a device itself and can
    # land on a card busy with a leftover server from a previous run, which looks
    # like a render failure (100% util, ~200 MB, no throughput) when the stack is fine.
    local check_gpu="${RENDER_CHECK_GPU:-0}"
    for t in $(seq 1 "$tries"); do
        out="$(cd "$TMPDIR" && CUDA_VISIBLE_DEVICES="$check_gpu" \
            timeout "${RENDER_CHECK_TIMEOUT:-120}" python3 -B "$ROLE_RUNNER" robotwin -u "$script" "$iters" 2>&1)"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo -e "\033[32mRender pre-flight passed (attempt ${t}): $(echo "$out" | tail -1)\033[0m"
            return 0
        fi
        local hint=""; [ "$rc" -eq 124 ] && hint=", timed out (render wedged)"
        echo -e "\033[31mRender pre-flight attempt ${t}/${tries} failed (rc=${rc}${hint}): $(echo "$out" | tail -1)\033[0m"
        [ "$t" -lt "$tries" ] && sleep 20
    done
    return 1
}
sim_import_check() {
    (cd "$TMPDIR" && timeout 300 python3 -B "$ROLE_RUNNER" robotwin -c \
        "import torch, sapien, curobo; print('Sim imports OK:', curobo.__file__)")
}
if ! { render_check && sim_import_check; }; then
    echo "Error: simulation preflight failed. Check the host GPU/Vulkan setup; no system files were changed." >&2
    exit 1
fi

# ===== Episodes-per-task validation =====
if ! [[ "$test_num" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "\033[31mError: --test_num must be a positive integer, got '${test_num}'\033[0m"
    exit 1
fi
# The server asserts use_length <= the checkpoint's chunk_size at the first infer;
# script.sh / script_sam.sh check that bound, this only rejects non-integers.
if ! [[ "$use_length" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "\033[31mError: --use_length must be a positive integer, got '${use_length}'\033[0m"
    exit 1
fi

# ===== Video mode =====
case "$enable_video" in
    0) eval_video_log=False; video_every_step=False ;;
    1) eval_video_log=True;  video_every_step=False ;;
    2) eval_video_log=True;  video_every_step=True  ;;
    *) echo -e "\033[31mError: --enable_video must be 0, 1 or 2, got '${enable_video}'\033[0m"; exit 1 ;;
esac
# The SAM masked mp4 follows the mode unless --sam_save_video was given.
[ -z "$sam_save_video" ] && { [ "$enable_video" = "0" ] && sam_save_video=False || sam_save_video=True; }

# ===== Task count validation =====
if [ "$num_tasks" -gt 50 ]; then
    echo -e "\033[31mError: num_tasks ${num_tasks} exceeds max 50; use 1~50\033[0m"
    exit 1
fi

# ===== Action space validation =====
# A typo here would otherwise fall back to the joint path silently: the client
# would send a 14-dim joint state to an EE checkpoint and every episode would
# fail for reasons that look like a policy problem, not a config one.
case "$action_space" in
    joint|ee) ;;
    *)
        echo -e "\033[31mError: --action_space must be 'joint' or 'ee', got '${action_space}'\033[0m"
        exit 1
        ;;
esac

# ===== Sim-side args =====
policy_name=ACT
train_config_name=0
seed=0

# ===== cam_high masking validation =====
case "$sam_mask" in 1|true|True) sam_mask=1 ;; 0|false|False) sam_mask=0 ;;
    *) echo -e "\033[31mError: --sam_mask must be 0 or 1, got '${sam_mask}'\033[0m"; exit 1 ;; esac
case "$sam_frames" in vla|all) ;;
    *) echo -e "\033[31mError: --sam_frames must be 'vla' or 'all', got '${sam_frames}'\033[0m"; exit 1 ;; esac
case "$sam_judge" in 1|true|True) sam_judge=1 ;; 0|false|False) sam_judge=0 ;;
    *) echo -e "\033[31mError: --sam_judge must be 0 or 1, got '${sam_judge}'\033[0m"; exit 1 ;; esac
case "$edge_width" in ''|*[!0-9]*)
    echo -e "\033[31mError: --edge_width must be an integer >= 0, got '${edge_width}'\033[0m"; exit 1 ;; esac
edge_width=$((10#$edge_width))
if ! awk -v s="$outside_scale" 'BEGIN { exit !(s ~ /^[0-9]*\.?[0-9]+$/ && s + 0 >= 0 && s + 0 <= 1) }'; then
    echo -e "\033[31mError: --outside_scale must be a number in [0, 1], got '${outside_scale}'\033[0m"; exit 1
fi
if [ "$sam_mask" != "1" ] && { [ "$sam_judge" = "1" ] || [ "$edge_width" -gt 0 ]; }; then
    echo -e "\033[31mError: --sam_judge / --edge_width need --sam_mask 1\033[0m"; exit 1
fi
if [ "$sam_mask" = "1" ]; then
    if [ "$num_gpus" -lt 2 ] && [ "$qwen_gpu_slots" -lt 1 ]; then
        echo -e "\033[31mError: --sam_mask 1 on ${num_gpus} GPU leaves no GPU for eval slots; pass --qwen_gpu_slots >= 1\033[0m"
        exit 1
    fi
    if [ "$qwen_gpu" -ge "$num_gpus" ] || [ "$qwen_gpu" -lt 0 ]; then
        echo -e "\033[31mError: --qwen_gpu ${qwen_gpu} is outside 0..$((num_gpus-1))\033[0m"; exit 1
    fi
fi

# ===== Slot -> GPU table =====
# sam_mask=0: slot s runs on GPU (s % num_gpus), exactly the original layout.
# sam_mask=1: qwen_gpu hosts the mask-Qwen vLLM server, so the slots go to the
#   other GPUs round-robin (num_per_gpu each), plus qwen_gpu_slots on qwen_gpu.
#   Each slot = 1 VLA server + 1 SAM3 stream server + 1 sim, all on its GPU.
declare -a slot_gpu=()
if [ "$sam_mask" = "1" ]; then
    _slot_gpus=()
    for g in $(seq 0 $((num_gpus-1))); do [ "$g" != "$qwen_gpu" ] && _slot_gpus+=("$g"); done
    for _r in $(seq 1 "$num_per_gpu"); do
        for g in "${_slot_gpus[@]}"; do slot_gpu+=("$g"); done
        # interleave the qwen-GPU slots so they are not the last to be started
        [ "$_r" -le "$qwen_gpu_slots" ] && slot_gpu+=("$qwen_gpu")
    done
    for _r in $(seq $((num_per_gpu + 1)) "$qwen_gpu_slots"); do slot_gpu+=("$qwen_gpu"); done
else
    for s in $(seq 0 $((num_gpus * num_per_gpu - 1))); do slot_gpu+=("$(( s % num_gpus ))"); done
fi

# ===== Compute inference slot count =====
# actual slots = min(num_tasks, len(slot_gpu))  (= num_gpus * num_per_gpu without sam_mask)
# When tasks < max concurrency, spin up only as many servers as tasks to avoid
# the waste of starting then shutting down extras.
num_slots=${#slot_gpu[@]}
if [ "$num_tasks" -lt "$num_slots" ]; then
    echo -e "\033[36mnum_tasks ${num_tasks} < max slots ${num_slots}; starting servers by task count\033[0m"
    num_slots=$num_tasks
fi

# Refuse occupied or overlapping service ports before allocating GPU models.
# This prevents accidentally evaluating against an unrelated VLA/SAM server.
python3 -B "$SUBMIT_ROOT/tools/evaluation_config.py" --check-ports \
    "$start_port" "$sam_start_port" "$mask_qwen_port" "$num_slots" "$sam_mask" || exit 1

# ===== Full task list (50), ordered LONGEST-FIRST =====
# Order matters because slots claim from the head of this list: whatever is left
# at the end determines how long every other slot sits idle. Starting a 4-hour
# task last leaves ~4 h of tail; starting it first lets short tasks fill in around
# it. This is the LPT (longest-processing-time-first) rule -- the classic greedy
# for minimising makespan, guaranteed within 4/3 of optimal.
#
# Ranking measured from three completed 50-task runs (15k clean, 15k randomized,
# 18.5k randomized) using the real per-task seconds in their stats.txt. Tasks are
# sorted by MEAN RANK across the three, not by raw seconds, because total run time
# differs (73 h clean vs 113 h randomized) while the relative order is stable
# (Spearman rho 0.87-0.98 pairwise). Longest put_object_cabinet ~4.2 h avg,
# shortest shake_bottle_horizontally ~1.0 h.
#
# Simulated makespan on those runs (dynamic claiming, vs total/slots lower bound):
#   slots        old order      this order
#     8          1.12x LB        1.05x LB     (-6..11% wall clock)
#    16          1.31x LB        1.09x LB     (-15..20%)
#    32          1.56x LB        1.37x LB     (-12..25%)
# Re-derive after a big model/step change with:
#   bash experiment/robotwin/rank_tasks_by_duration.sh <run_dir>...
#
# NOTE: `--num_tasks N` takes the FIRST N entries, so with this order a partial
# run gets the N slowest tasks, not an arbitrary subset.
task_list_all=(
    "put_object_cabinet" "open_microwave" "stack_bowls_three" "hanging_mug" "blocks_ranking_size"
    "stack_blocks_three" "blocks_ranking_rgb" "place_object_basket" "handover_mic" "put_bottles_dustbin"
    "place_cans_plasticbox" "handover_block" "place_dual_shoes" "scan_object" "place_can_basket"
    "move_can_pot" "move_pillbottle_pad" "lift_pot" "move_stapler_pad" "dump_bin_bigbin"
    "open_laptop" "pick_diverse_bottles" "place_fan" "place_empty_cup" "place_mouse_pad"
    "adjust_bottle" "stack_bowls_two" "place_object_scale" "pick_dual_bottles" "place_bread_basket"
    "place_phone_stand" "place_shoe" "beat_block_hammer" "place_bread_skillet" "click_alarmclock"
    "click_bell" "place_burger_fries" "stack_blocks_two" "place_container_plate" "place_a2b_right"
    "place_object_stand" "place_a2b_left" "move_playingcard_away" "stamp_seal" "rotate_qrcode"
    "shake_bottle" "turn_switch" "grab_roller" "press_stapler" "shake_bottle_horizontally"
)

if [ -n "${TASK_NAMES:-}" ]; then
    IFS=',' read -r -a requested_tasks <<< "$TASK_NAMES"
    for task in "${requested_tasks[@]}"; do
        found=0
        for allowed in "${task_list_all[@]}"; do [ "$task" != "$allowed" ] || found=1; done
        [ "$found" = 1 ] || { echo "Unknown TASK_NAMES entry: $task" >&2; exit 1; }
    done
    task_list_all=("${requested_tasks[@]}")
    num_tasks=${#task_list_all[@]}
fi

# Build the sim task queue
task_queue=()
for i in $(seq 0 $((num_tasks-1))); do
    task_queue+=("${task_list_all[$i]}")
done
echo -e "\033[36mTasks this run (${num_tasks}): ${task_queue[*]}\033[0m"
if [ "$sam_mask" = "1" ]; then
    echo -e "\033[36mInference config: sam_mask ON -- mask-Qwen (Qwen3.8-27B) on GPU ${qwen_gpu}; ${num_slots} slots (VLA+SAM3+sim) on GPUs [${slot_gpu[*]:0:${num_slots}}], ${num_per_gpu}/GPU + ${qwen_gpu_slots} on GPU ${qwen_gpu}; sam_frames=${sam_frames}\033[0m"
else
    echo -e "\033[36mInference config: ${num_gpus} GPU x ${num_per_gpu} servers/GPU = ${num_slots} slots\033[0m"
fi
echo -e "\033[36mInference precision: use_bf16=${use_bf16}, use_fp32=${use_fp32}, use_compile=${use_compile}\033[0m"

# ===== Common variables =====
batch_time=$(date +%Y%m%d_%H%M%S)
# Extract experiment name and step from model_path. The step goes in verbatim
# (18500, not 18k): rounding to thousands collapses 18000/18500/18999 into the
# same name, leaving only the timestamp to tell two runs apart.
_exp_name=$(echo "$model_path" | grep -oP '[^/]+(?=/checkpoints)')
_step_num=$(echo "$model_path" | grep -oP 'global_step_\K\d+')
if [ -n "$run_dir_arg" ]; then
    # Multi-node: the caller supplies one shared path for all nodes. Deriving it
    # per node would give each its own timestamp and split the run into N result
    # sets that never see each other's claims.
    run_dir="$run_dir_arg"
else
    run_dir="${output_base}/${_exp_name}_${_step_num}_${task_config}_${batch_time}"
fi
mkdir -p "${run_dir}/inference_logs" "${run_dir}/eval_logs"
log_dir="${run_dir}"
# Per-node PID files: on a shared mount every node writes into the same run_dir,
# and a single pids.txt would interleave PIDs from different machines (killing by
# PID would then hit unrelated local processes).
inference_pid_file="${run_dir}/inference_pids_n${node_rank}.txt"
eval_pid_file="${run_dir}/eval_pids_n${node_rank}.txt"
> "$inference_pid_file"
> "$eval_pid_file"
# Per-node stats: each node reports what it actually ran. Aggregate across nodes
# by reading eval_logs/ (one log per task, whichever node claimed it).
if [ "$nnodes" -gt 1 ]; then
    stats_file="${run_dir}/stats_n${node_rank}.txt"
else
    stats_file="${run_dir}/stats.txt"
fi
# Claim dir: one subdir per task, created atomically. `mkdir` either creates the
# dir or fails -- there is no window where two nodes both think they won, which is
# what makes this safe without a lock server. Verified on this mount with 64
# concurrent racers over 20 slots: exactly 20 wins, no duplicates.
claim_dir="${run_dir}/claims"
mkdir -p "$claim_dir"
# Episodes per task for this run. aggregate_results.sh reads it to judge which
# tasks are complete. A run dir that already says otherwise (RESUME, or a reused
# --run_dir) would mix two episode counts into one score, so refuse that.
test_num_file="${run_dir}/test_num"
if [ -f "$test_num_file" ] && [ "$(cat "$test_num_file")" != "$test_num" ]; then
    echo -e "\033[31mError: ${run_dir} was run with test_num=$(cat "$test_num_file"), this launch asks for ${test_num}\033[0m"
    exit 1
fi
# tmp + mv: every node writes this, and a peer must never read a half-written file.
echo "$test_num" > "${test_num_file}.tmp.$$" && mv -f "${test_num_file}.tmp.$$" "$test_num_file"
# Same guard for the executed chunk length: a reused --run_dir must not mix
# episodes run with different use_length into one score. Run dirs from before
# this file existed were all use_length=50, so a missing file is not a conflict.
use_length_file="${run_dir}/use_length"
if [ -f "$use_length_file" ] && [ "$(cat "$use_length_file")" != "$use_length" ]; then
    echo -e "\033[31mError: ${run_dir} was run with use_length=$(cat "$use_length_file"), this launch asks for ${use_length}\033[0m"
    exit 1
fi
echo "$use_length" > "${use_length_file}.tmp.$$" && mv -f "${use_length_file}.tmp.$$" "$use_length_file"
echo -e "\033[36mRun directory: ${run_dir}\033[0m"
if [ "$nnodes" -gt 1 ]; then
    echo -e "\033[36mMulti-node: node ${node_rank} of ${nnodes}; claiming tasks from ${claim_dir}\033[0m"
fi

# Result collection arrays
declare -a result_task_names=()
declare -a result_durations=()
declare -a result_logs=()
declare -a result_status=()

# ============================================================
# Phase 1: start inference-side QwenPi servers (resident)
# ============================================================
echo -e "\033[32m========== Starting inference side: ${num_slots} QwenPi servers, ports ${start_port}~$((start_port + num_slots - 1)) ==========\033[0m"

cd "$inference_workdir" || { echo -e "\033[31mError: inference workdir ${inference_workdir} missing\033[0m"; exit 1; }

for slot in $(seq 0 $((num_slots-1))); do
    gpu_id=${slot_gpu[$slot]}
    port=$(( start_port + slot ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    # Tag with node rank: slot/gpu/port repeat on every node, so without it all
    # nodes would write the same inference_logs/ filenames on the shared mount.
    log_file="${run_dir}/inference_logs/qwenpi_n${node_rank}_slot${slot}_gpu${gpu_id}_port${port}.log"

    echo -e "\033[33m[inf slot $slot] GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

    # deploy/qwen3_5vl_dit_policy.py -> deploy.qwen3_5vl_dit_policy
    inference_script_for_module="${inference_script}"
    if [[ "${inference_script_for_module}" = /* ]]; then
        inference_script_for_module="${inference_script_for_module#${inference_workdir%/}/}"
    fi
    inference_module=$(echo "${inference_script_for_module}" | sed 's|/|.|g; s|\.py$||')
    setsid python3 -B "$ROLE_RUNNER" vla -m "$inference_module" \
        --model_path "$model_path" --use_length "$use_length" \
        --use_bf16 "$use_bf16" --use_fp32 "$use_fp32" --use_compile "$use_compile" \
        --port "$port" > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" >> "$inference_pid_file"
done

echo -e "\033[32m${num_slots} inference servers started, PIDs saved to ${inference_pid_file}\033[0m"

# Record each slot's inference server PID for on-demand shutdown
declare -a inference_pids=()
for slot in $(seq 0 $((num_slots-1))); do
    inference_pids[$slot]=0
done
# Re-read PIDs (in launch order, mapping 1:1 to slots)
mapfile -t pid_list < "$inference_pid_file"
for slot in $(seq 0 $((num_slots-1))); do
    if [ $slot -lt ${#pid_list[@]} ]; then
        inference_pids[$slot]=${pid_list[$slot]}
    fi
done
active_inference=$num_slots

# ============================================================
# Phase 1b (sam_mask only): mask-Qwen vLLM + one SAM3 stream server per slot
# ============================================================
# Everything is node-local (127.0.0.1): each node runs its own mask-Qwen and
# SAM3 servers, so multi-node needs nothing beyond the shared run_dir claims.
declare -a sam_pids=()
for slot in $(seq 0 $((num_slots-1))); do sam_pids[$slot]=0; done
started_mask_qwen=false
mask_qwen_pid=""
yzm_root="$SUBMIT_ROOT"
if [ "$sam_mask" = "1" ]; then
    mask_qwen_url="http://127.0.0.1:${mask_qwen_port}"
    # A pre-existing server may load code/weights from another checkout.
    if curl -sf -m 5 "${mask_qwen_url}/v1/models" >/dev/null 2>&1; then
        echo "Error: mask-Qwen port ${mask_qwen_port} is already occupied. Choose a free port in base.json." >&2
        exit 1
    else
        mask_qwen_model="${MASK_QWEN_MODEL:-${yzm_root}/vllm_space/qwen3.8-27B}"
        mask_qwen_log="${run_dir}/inference_logs/mask_qwen_n${node_rank}_gpu${qwen_gpu}_port${mask_qwen_port}.log"
        rpc_dir="$VLLM_RPC_BASE_PATH"; mkdir -p "$rpc_dir"
        echo -e "\033[33m[mask-qwen] starting Qwen3.8-27B on GPU ${qwen_gpu}, port ${mask_qwen_port}, log ${mask_qwen_log}\033[0m"
        # Same flags as auto_mask/start_services.sh (prefix caching: ~5x req/s).
        # Pin the KV cache budget independently of allocations made by the
        # co-located VLA/SAM3 slots. Configure the cache size and memory
        # utilization for the available devices through base.json.
        # GDN prefill uses triton: FlashInfer's JIT gdn_prefill_sm90 kernel needs
        # cuda::ptx::tensormap_replace_* which the system nvcc 12.4 CCCL lacks, so the
        # build fails and EngineCore dies on the first real prefill request.
        CUDA_VISIBLE_DEVICES=${qwen_gpu} OMP_NUM_THREADS=8 VLLM_RPC_BASE_PATH="${rpc_dir}" \
        setsid python3 -B "$ROLE_RUNNER" vllm -m vllm.entrypoints.cli.main serve "${mask_qwen_model}" \
            --served-model-name "$MASK_QWEN_MODEL_NAME" --host 127.0.0.1 --port "${mask_qwen_port}" \
            --max-model-len "${MASK_QWEN_MAX_MODEL_LEN:-8192}" \
            --kv-cache-memory-bytes "$MASK_QWEN_KV_CACHE_MEMORY_BYTES" --gpu-memory-utilization "$MASK_QWEN_GPU_MEMORY_UTILIZATION" \
            --max-num-seqs "${MASK_QWEN_MAX_NUM_SEQS:-16}" \
            --gdn-prefill-backend "${MASK_QWEN_GDN_PREFILL_BACKEND:-triton}" \
            --limit-mm-per-prompt "{\"image\":$(( sam_judge == 1 ? 2 : 1 )),\"video\":0}" --enable-prefix-caching \
            > "$mask_qwen_log" 2>&1 < /dev/null &
        mask_qwen_pid=$!
        started_mask_qwen=true
    fi
    # ---- SAM3 stream servers, one per slot on the slot's GPU ----
    sam_server="$SAM_SERVER"
    [ -f "$sam_server" ] || { echo -e "\033[31mError: ${sam_server} missing\033[0m"; exit 1; }

    # --sam_judge 1: the SAM3 server asks mask-Qwen about every object. Judge
    # settings = pipeline_lerobot.py defaults used for the *_sam_judged_* sets
    # (threshold 0.5, judge image outside x0.2, regrow 2.0 / 0.002, 120 s).
    sam_judge_args=""
    if [ "$sam_judge" = "1" ]; then
        sam_judge_args="--judge-url ${mask_qwen_url} --judge-model ${MASK_QWEN_MODEL_NAME} --judge-threshold 0.5 --judge-outside-scale 0.2 --judge-regrow 2.0 --judge-regrow-min-area 0.002 --judge-timeout 120"
    fi
    for slot in $(seq 0 $((num_slots-1))); do
        gpu_id=${slot_gpu[$slot]}
        sam_port=$(( sam_start_port + slot ))
        sam_log="${run_dir}/inference_logs/sam3_n${node_rank}_slot${slot}_gpu${gpu_id}_port${sam_port}.log"
        CUDA_VISIBLE_DEVICES=${gpu_id} OMP_NUM_THREADS=4  \
        setsid python3 -B "$ROLE_RUNNER" sam3 "$sam_server" --checkpoint "$SAM_CHECKPOINT" \
            --host 127.0.0.1 --port "${sam_port}" \
            --memory-window "${sam_memory_window}" ${sam_judge_args} \
            > "$sam_log" 2>&1 < /dev/null &
        sam_pids[$slot]=$!
        echo "${sam_pids[$slot]}" >> "$inference_pid_file"
        echo -e "\033[33m[sam3 slot $slot] GPU: ${gpu_id}, PORT: ${sam_port}, Log: ${sam_log}\033[0m"
    done
    # ---- wait until every service answers (sims must not start before) ----
    _deadline=$(( $(date +%s) + ${SAM_READY_TIMEOUT:-1800} ))
    until curl -sf -m 5 "${mask_qwen_url}/v1/models" >/dev/null 2>&1; do
        if [ -n "$mask_qwen_pid" ] && ! kill -0 "$mask_qwen_pid" 2>/dev/null; then
            echo -e "\033[31mError: mask-Qwen vLLM exited during startup; see ${mask_qwen_log}\033[0m"; exit 1
        fi
        [ "$(date +%s)" -lt "$_deadline" ] || { echo -e "\033[31mError: mask-Qwen not ready in time\033[0m"; exit 1; }
        sleep 10
    done
    echo -e "\033[32m[mask-qwen] ready at ${mask_qwen_url}\033[0m"
    for slot in $(seq 0 $((num_slots-1))); do
        sam_port=$(( sam_start_port + slot ))
        until curl -sf -m 5 "http://127.0.0.1:${sam_port}/health" | grep -q '"loaded": true'; do
            if ! kill -0 "${sam_pids[$slot]}" 2>/dev/null; then
                echo -e "\033[31mError: SAM3 server of slot ${slot} exited during startup; see inference_logs/sam3_n${node_rank}_slot${slot}_*.log\033[0m"; exit 1
            fi
            [ "$(date +%s)" -lt "$_deadline" ] || { echo -e "\033[31mError: SAM3 slot ${slot} not ready in time\033[0m"; exit 1; }
            sleep 5
        done
    done
    echo -e "\033[32m${num_slots} SAM3 stream servers ready (ports ${sam_start_port}~$((sam_start_port + num_slots - 1)))\033[0m"
fi

# ============================================================
# Phase 2: queue-scheduled sim tasks
# ============================================================
echo -e "\033[32m========== Starting sim side (queue-scheduled, ${num_slots} concurrent slots) ==========\033[0m"

cd "$eval_workdir" || { echo -e "\033[31mError: sim workdir ${eval_workdir} missing\033[0m"; exit 1; }

# ===== Ensure the lingbot eval client is present under RoboTwin/script =====
# The eval client resolves its ../task_config relative to its own location, so it
# must live at <RoboTwin>/script/ for _camera_config.yml to be found.
# Refresh whenever the source is newer: `-f` alone means an edit here never
# reaches a machine that already has a copy, so the sim would silently keep
# running yesterday's client.
eval_client_src="${inference_workdir}experiment/robotwin/eval_policy_client_lingbotvla.py"
eval_client_dst="${eval_workdir}/script/eval_policy_client_lingbotvla.py"
if [ ! -f "$eval_client_src" ]; then
    echo -e "\033[31mError: eval client source not found: ${eval_client_src}\033[0m"
    exit 1
fi
if [ ! -f "$eval_client_dst" ] || [ "$eval_client_src" -nt "$eval_client_dst" ]; then
    echo -e "\033[36mCopying eval client -> ${eval_client_dst}\033[0m"
    cp "$eval_client_src" "$eval_client_dst"
fi

# ===== Ensure the deploy client helpers are present under RoboTwin/script/deploy =====
# The eval client does `from script.deploy.websocket_client_policy import WebsocketClientPolicy`,
# which pulls in a sibling msgpack_numpy. Copy any that are missing or newer from the inference repo.
deploy_pkg_src="${inference_workdir}deploy"
deploy_pkg_dst="${eval_workdir}/script/deploy"
mkdir -p "$deploy_pkg_dst"
for f in __init__.py websocket_client_policy.py msgpack_numpy.py; do
    if [ ! -f "$deploy_pkg_src/$f" ]; then
        echo -e "\033[31mError: deploy helper source not found: ${deploy_pkg_src}/${f}\033[0m"
        exit 1
    fi
    if [ ! -f "$deploy_pkg_dst/$f" ] || [ "$deploy_pkg_src/$f" -nt "$deploy_pkg_dst/$f" ]; then
        echo -e "\033[36mCopying deploy helper -> ${deploy_pkg_dst}/${f}\033[0m"
        cp "$deploy_pkg_src/$f" "$deploy_pkg_dst/$f"
    fi
done
# The cam_high masking wrapper is imported as script.deploy.sam_mask (only when
# --sam_mask 1, but copied always so the sim tree never holds a stale version).
sam_mask_src="${inference_workdir}experiment/robotwin/sam_mask"
sam_mask_dst="${deploy_pkg_dst}/sam_mask"
mkdir -p "$sam_mask_dst"
for f in "$sam_mask_src"/*.py; do
    _dst="$sam_mask_dst/$(basename "$f")"
    if [ ! -f "$_dst" ] || [ "$f" -nt "$_dst" ]; then
        echo -e "\033[36mCopying sam_mask helper -> ${_dst}\033[0m"
        cp "$f" "$_dst"
    fi
done

# Regenerate planner asset paths from templates for this checkout.
_eval_resolved="$(pwd -P)"
_rep_yml="assets/embodiments/aloha-agilex/curobo_left.yml"
if [ ! -f "$_rep_yml" ] || ! grep -qF "${_eval_resolved}/assets" "$_rep_yml"; then
    python3 -B "$ROLE_RUNNER" robotwin script/update_embodiment_config_path.py || exit 1
fi

total_start_time=$(date +%s)

# Slot state: free or busy
# slot_pid[slot]   = sim process PID (0 = free)
# slot_task[slot]  = current task name
# slot_start[slot] = task start time
# slot_log[slot]   = log file path
declare -a slot_pid=()
declare -a slot_task=()
declare -a slot_start=()
declare -a slot_log=()
for slot in $(seq 0 $((num_slots-1))); do
    slot_pid[$slot]=0
    slot_task[$slot]=""
    slot_start[$slot]=0
    slot_log[$slot]=""
done

# ===== Stall watchdog =====
# A task can fail in a way that never exits: the eval loop catches any scene-setup
# exception, bumps the seed and retries forever (see eval_policy_client's
# `except Exception: now_seed += 1; continue`). Seen for real -- a Warp/curobo
# kernel-cache race left one task printing
#   "error occurs ! 'Robot' object has no attribute 'left_planner'"
# 592k times over 11 h with zero episodes completed, holding the run open long
# after the other 49 tasks had finished.
#
# That specific case now fails fast: the eval client recognises it
# (is_fatal_setup_error) and exits 75, so the retry below starts a fresh
# process within seconds. The watchdog stays as the backstop for spins the
# client does not recognise.
#
# Liveness CANNOT be judged from the log's mtime: that pathology writes
# constantly (~1 KB/s). The only real progress signal is completed episodes, so
# count "Success rate" lines -- one per finished episode -- and kill a task whose
# count has not moved for STALL_TIMEOUT.
#
# Default 3600 s: the slowest observed single episode is well under that
# (put_object_cabinet averaged ~185 s/episode), while the reference launcher uses
# the same 3600 s for its dispatcher watchdog. 0 disables the watchdog.
stall_timeout="${STALL_TIMEOUT:-3600}"
declare -a slot_progress=()      # last episode count seen
declare -a slot_progress_at=()   # when that count last changed

episodes_done_in() {  # episodes_done_in <log file>
    local f="$1" n
    [ -f "$f" ] || { echo 0; return; }
    # `grep -c` exits 1 with "0" on no match, so `|| echo 0` would print twice.
    n=$(grep -c "Success rate" "$f" 2>/dev/null)
    echo "${n:-0}"
}

# Queue pointer and retry counters
queue_idx=0
# Claim the next task this node can own, echoing its name (empty = nothing left).
# Single node with nnodes=1 still goes through the claim dir: one code path, and
# it makes a relaunch of the same run_dir skip already-finished tasks for free.
claim_next_task() {
    local t
    while [ $queue_idx -lt ${#task_queue[@]} ]; do
        t="${task_queue[$queue_idx]}"
        queue_idx=$((queue_idx + 1))
        # Winner takes the task; loser moves on. Record who won for debugging.
        if mkdir "${claim_dir}/${t}" 2>/dev/null; then
            echo "node${node_rank} $(hostname) $(date '+%F %T')" > "${claim_dir}/${t}/owner" 2>/dev/null
            printf '%s' "$t"
            return 0
        fi
    done
    printf ''
}
max_retries=3
declare -A task_retries=()
for t in "${task_queue[@]}"; do
    task_retries[$t]=0
done

# Launch a sim task onto a given slot
launch_task() {
    local slot=$1
    local task_name=$2
    local gpu_id=${slot_gpu[$slot]}
    local port=$(( start_port + slot ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    local log_file="${run_dir}/eval_logs/${task_name}.log"
    # cam_high masking overrides (empty without --sam_mask: the original command line)
    local sam_overrides=""
    if [ "$sam_mask" = "1" ]; then
        sam_overrides="${sam_overrides} --sam_mask True --sam_port $(( sam_start_port + slot )) --mask_qwen_url http://127.0.0.1:${mask_qwen_port} --sam_frames ${sam_frames} --outside_scale ${outside_scale} --sam_save_video ${sam_save_video}"
        # only when set, so the default command line is the original one
        if [ "$edge_width" -gt 0 ]; then sam_overrides="${sam_overrides} --edge_width ${edge_width}"; fi
        if [ "$sam_judge" = "1" ]; then sam_overrides="${sam_overrides} --sam_judge True"; fi
    fi

    local attempt=$((task_retries[$task_name]+1))
    echo -e "\033[33m  [launch] ${task_name} -> slot $slot (GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}) [attempt ${attempt}/${max_retries}]\033[0m"

    # On retry, append instead of overwriting the failed log; mark this attempt with a banner
    {
        echo ""
        echo "================================================================"
        echo "  [attempt ${attempt}/${max_retries}] $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  task=${task_name}  slot=${slot}  gpu=${gpu_id}  port=${port}"
        echo "================================================================"
    } >> "$log_file"

    # run_python disables user-site imports and selects the configured sim env.
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    setsid python3 -B "$ROLE_RUNNER" robotwin -u "$eval_client_dst" \
        --config "policy/${policy_name}/deploy_policy.yml" --overrides \
        --task_name "$task_name" --task_config "$task_config" \
        --train_config_name "$train_config_name" --seed "$seed" --policy_name "$policy_name" \
        --port "$port" --robo_name "$robo_name" --action_space "$action_space" \
        --video_fps "$video_fps" --eval_video_log "$eval_video_log" --video_every_step "$video_every_step" \
        --test_num "$test_num" ${sam_overrides} \
        --output_dir "${run_dir}/eval_results" >> "$log_file" 2>&1 &

    local pid=$!
    echo "${pid}" >> "$eval_pid_file"
    slot_pid[$slot]=$pid
    slot_task[$slot]="$task_name"
    slot_start[$slot]=$(date +%s)
    slot_log[$slot]="$log_file"
    # Baseline for the stall watchdog. On a retry the log is appended to, so seed
    # from the current count rather than 0 or the check would fire immediately.
    slot_progress[$slot]=$(episodes_done_in "$log_file")
    slot_progress_at[$slot]=$(date +%s)
}

# Initial fill: assign a task to each free slot
for slot in $(seq 0 $((num_slots-1))); do
    _t="$(claim_next_task)"
    [ -n "$_t" ] && launch_task $slot "$_t"
done

# Shut down the inference server on a given slot
shutdown_inference_slot() {
    local slot=$1
    local inf_pid=${inference_pids[$slot]}
    if [ "$inf_pid" != "0" ] && kill -0 "$inf_pid" 2>/dev/null; then
        kill "$inf_pid" 2>/dev/null
        echo -e "\033[36m  [release] slot $slot inference server (PID: ${inf_pid}) stopped, remaining: $((active_inference - 1))\033[0m"
    fi
    inference_pids[$slot]=0
    local sam_pid=${sam_pids[$slot]:-0}
    if [ "$sam_pid" != "0" ] && kill -0 "$sam_pid" 2>/dev/null; then
        kill "$sam_pid" 2>/dev/null
    fi
    sam_pids[$slot]=0
    active_inference=$((active_inference - 1))
}

# ============================================================
# Phase 3: scheduling loop - detect finished slots, reclaim and dispatch
# ============================================================
completed=0
skipped=0

# Exit when nothing is running: with claim-based dispatch a node cannot know up
# front how many tasks it will own, so "all of MY work is done" is exactly "every
# slot is idle" (a slot only goes idle after failing to claim anything more).
has_running() {
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${slot_pid[$slot]}" != "0" ] && return 0
    done
    return 1
}

# Remaining work for THIS node: running + still-unclaimed tasks it could win.
# Unclaimed is an upper bound (another node may take them first), which is the
# safe direction: it only ever delays shutting an inference server down.
remaining_tasks() {
    local running=0 unclaimed=0 i t
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${slot_pid[$slot]}" != "0" ] && running=$((running + 1))
    done
    for i in $(seq $queue_idx $(( ${#task_queue[@]} - 1 ))); do
        t="${task_queue[$i]}"
        [ -n "$t" ] && [ ! -d "${claim_dir}/${t}" ] && unclaimed=$((unclaimed + 1))
    done
    echo $((running + unclaimed))
}

while has_running; do
    for slot in $(seq 0 $((num_slots-1))); do
        pid=${slot_pid[$slot]}

        # Skip free slots
        [ "$pid" = "0" ] && continue

        # Stall watchdog: kill a live-but-not-progressing task so the normal
        # exit-code path below can retry or skip it. Without this the slot is held
        # forever and the whole run never ends.
        if [ "$stall_timeout" -gt 0 ] 2>/dev/null && kill -0 "$pid" 2>/dev/null; then
            _now=$(date +%s)
            _eps=$(episodes_done_in "${slot_log[$slot]}")
            if [ "$_eps" != "${slot_progress[$slot]:-0}" ]; then
                slot_progress[$slot]=$_eps
                slot_progress_at[$slot]=$_now
            elif [ $(( _now - ${slot_progress_at[$slot]:-$_now} )) -ge "$stall_timeout" ]; then
                echo -e "\033[31m  [stall] ${slot_task[$slot]} slot $slot: no episode completed for $(( (_now - ${slot_progress_at[$slot]}) / 60 )) min (stuck at ${_eps}/${test_num}); killing it\033[0m"
                echo -e "\033[31m         Check ${slot_log[$slot]} -- a repeating 'error occurs !' line means every scene setup is failing.\033[0m"
                # Kill the process group: the worker is started with setsid, so the
                # python process is not a direct child and `kill $pid` alone leaks it.
                kill -KILL -- -"$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                    || kill -KILL "$pid" 2>/dev/null || true
                # Fall through: next loop iteration sees it exited and retries/skips.
            fi
        fi

        # Check whether the process has exited
        if ! kill -0 "$pid" 2>/dev/null; then
            task_end_time=$(date +%s)
            task_duration=$(( task_end_time - slot_start[$slot] ))
            wait "$pid"
            exit_code=$?

            task_name="${slot_task[$slot]}"
            log_file="${slot_log[$slot]}"

            if [ $exit_code -eq 0 ]; then
                completed=$((completed + 1))
                _claimed=$(ls -1 "$claim_dir" 2>/dev/null | wc -l)
                echo -e "\033[32m  [done] ${task_name} slot $slot took ${task_duration}s (this node: ${completed} done, ${skipped} skipped; ${_claimed}/${num_tasks} tasks claimed across all nodes)\033[0m"
                result_status+=("done")
                result_task_names+=("$task_name")
                result_durations+=("$task_duration")
                result_logs+=("$log_file")

                # Free the slot
                slot_pid[$slot]=0
                slot_task[$slot]=""
                slot_log[$slot]=""

                # Claim and dispatch the next task
                _t="$(claim_next_task)"
                [ -n "$_t" ] && launch_task $slot "$_t"
            else
                task_retries[$task_name]=$((task_retries[$task_name] + 1))
                if [ ${task_retries[$task_name]} -lt $max_retries ]; then
                    echo -e "\033[31m  [fail] ${task_name} slot $slot took ${task_duration}s exit=${exit_code}, retrying on this slot (${task_retries[$task_name]}/${max_retries})\033[0m"
                    # Retry immediately on the same slot
                    launch_task $slot "$task_name"
                else
                    skipped=$((skipped + 1))
                    echo -e "\033[31m  [skip] ${task_name} failed ${max_retries} times, no more retries\033[0m"
                    result_status+=("skip:${exit_code}")
                    result_task_names+=("$task_name")
                    result_durations+=("$task_duration")
                    result_logs+=("$log_file")

                    # Free the slot
                    slot_pid[$slot]=0
                    slot_task[$slot]=""
                    slot_log[$slot]=""

                    # Claim and dispatch the next task
                    _t="$(claim_next_task)"
                    [ -n "$_t" ] && launch_task $slot "$_t"
                fi
            fi
        fi
    done

    # Adaptive shutdown: when remaining tasks < active inference servers, stop idle-slot servers
    remaining=$(remaining_tasks)
    for slot in $(seq 0 $((num_slots-1))); do
        [ $remaining -lt $active_inference ] || break
        if [ "${slot_pid[$slot]}" = "0" ] && [ "${inference_pids[$slot]}" != "0" ]; then
            shutdown_inference_slot $slot
        fi
    done

    sleep 5
done

total_end_time=$(date +%s)
total_duration=$(( total_end_time - total_start_time ))

_scope=""
[ "$nnodes" -gt 1 ] && _scope=" on node ${node_rank}"
if [ $skipped -gt 0 ]; then
    echo -e "\033[31m========== Sim finished${_scope}: ${completed} done, ${skipped} skipped, total ${total_duration}s ==========\033[0m"
else
    echo -e "\033[32m========== All sim tasks done${_scope}: ${completed} done, total ${total_duration}s ==========\033[0m"
fi

# ============================================================
# Phase 4: generate the stats file
# ============================================================
echo -e "\033[36mGenerating stats file: ${stats_file}\033[0m"

{
    echo "============================================"
    echo "  Eval Result Stats"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Model: ${_exp_name}_${_step_num}"
    echo "  Model path: ${model_path}"
    echo "  Tasks: ${num_tasks} in this run's task list"
    echo "  Task Config: ${task_config}"
    echo "  Episodes per task: ${test_num}"
    echo "  Video mode: ${enable_video} (0 none | 1 VLA query frames | 2 every step)"
    if [ "$nnodes" -gt 1 ]; then
        echo "  Node: ${node_rank} of ${nnodes} ($(hostname)) -- rows below are THIS node's claimed tasks only"
    fi
    echo "  Inference: ${num_gpus} GPU x ${num_per_gpu}/GPU = ${num_slots} slots"
    echo "  Precision: use_bf16=${use_bf16}, use_fp32=${use_fp32}, use_compile=${use_compile}"
    if [ "$sam_mask" = "1" ]; then
        echo "  cam_high mask: mask-Qwen(Qwen3.8-27B)->SAM3, sam_frames=${sam_frames}, outside_scale=${outside_scale}, edge_width=${edge_width}, judge=${sam_judge}, qwen_gpu=${qwen_gpu} (+${qwen_gpu_slots} slots)"
    fi
    echo "  Result: ${completed} done, ${skipped} skipped"
    echo "============================================"
    echo ""
    printf "%-30s %-10s %-12s %-10s %-10s\n" "Task" "Time(s)" "Done(${test_num})" "Success/Total" "Rate"
    echo "--------------------------------------------------------------------------------"

    total_success=0
    total_episodes=0
    all_complete=true

    for i in "${!result_task_names[@]}"; do
        task_name="${result_task_names[$i]}"
        duration="${result_durations[$i]}"
        log_file="${result_logs[$i]}"
        status="${result_status[$i]}"

        # Extract success rate from the log (strip ANSI color codes)
        success_rate="-"
        episodes_done="-"
        if [ -f "$log_file" ]; then
            last_rate_line=$(sed 's/\x1b\[[0-9;]*m//g' "$log_file" | grep -oP 'Success rate: \K.*' | tail -1)
            if [ -n "$last_rate_line" ]; then
                suc_num=$(echo "$last_rate_line" | grep -oP '^\d+' | head -1)
                total_num=$(echo "$last_rate_line" | grep -oP '/\K\d+' | head -1)
                rate_pct=$(echo "$last_rate_line" | grep -oP '=> \K[\d.]+')
                if [ -n "$suc_num" ] && [ -n "$total_num" ]; then
                    success_rate="${rate_pct}%"
                    episodes_done="${suc_num}/${total_num}"
                    total_success=$((total_success + suc_num))
                    total_episodes=$((total_episodes + total_num))
                fi
            fi
        fi

        # Check whether all test_num episodes ran
        if [ "$episodes_done" != "-" ]; then
            total_ep=$(echo "$episodes_done" | cut -d'/' -f2)
            if [ "$total_ep" = "$test_num" ]; then
                complete_mark="YES"
            else
                complete_mark="NO(${total_ep}/${test_num})"
                all_complete=false
            fi
        else
            complete_mark="NO(0/${test_num})"
            all_complete=false
        fi

        printf "%-30s %-10s %-12s %-10s %-10s\n" "$task_name" "$duration" "$complete_mark" "$episodes_done" "$success_rate"
    done

    echo "--------------------------------------------------------------------------------"
    if [ $total_episodes -gt 0 ]; then
        overall_rate=$(awk "BEGIN {printf \"%.1f\", $total_success/$total_episodes*100}")
    else
        overall_rate="0.0"
    fi
    printf "Summary: total %ds, success %d/%d, overall rate %s%%\n" "$total_duration" "$total_success" "$total_episodes" "$overall_rate"
    if $all_complete; then
        echo "All tasks fully executed ${test_num} episodes"
    else
        echo "Warning: some tasks did not complete ${test_num} episodes; check logs"
    fi
    if [ "$nnodes" -gt 1 ]; then
        echo ""
        echo "NOTE: this is node ${node_rank}'s share only; the rate above is NOT the"
        echo "      benchmark score. Aggregate every node with:"
        echo "      bash ${inference_workdir}experiment/robotwin/aggregate_results.sh ${run_dir}"
    fi
    echo "============================================"
} | tee "$stats_file"

# ============================================================
# Phase 5: handle inference servers (stop the remaining ones)
# ============================================================
if $keep_inference; then
    alive_count=0
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${inference_pids[$slot]}" != "0" ] && alive_count=$((alive_count + 1))
    done
    echo -e "\033[36m========== Sim done, ${alive_count} inference servers kept resident ==========\033[0m"
    echo -e "\033[36mStop inference manually: kill \$(cat ${inference_pid_file})\033[0m"
else
    echo -e "\033[36m========== Stopping remaining inference servers ==========\033[0m"
    for slot in $(seq 0 $((num_slots-1))); do
        inf_pid=${inference_pids[$slot]}
        if [ "$inf_pid" != "0" ] && kill -0 "$inf_pid" 2>/dev/null; then
            kill "$inf_pid" 2>/dev/null
            echo -e "\033[36m  slot $slot inference server (PID: ${inf_pid}) stopped\033[0m"
        fi
    done
    echo -e "\033[32mAll inference servers stopped\033[0m"
fi

echo ""
run_exit_code=0
if [ "$skipped" -gt 0 ] || grep -q '^Warning: some tasks did not complete ' "$stats_file"; then
    run_exit_code=1
    echo -e "\033[31m========== Evaluation incomplete: ${skipped} tasks skipped; see ${stats_file} ==========\033[0m"
else
    echo -e "\033[32m========== All done ==========\033[0m"
fi
echo -e "Run dir: ${run_dir}"
echo -e "Stats file: ${stats_file}"
echo -e "Inference logs: ${run_dir}/inference_logs/"
echo -e "Sim logs: ${run_dir}/eval_logs/"
echo -e "Eval results & videos: ${run_dir}/eval_results/"
if $keep_inference; then
    echo -e "Inference PIDs: ${inference_pid_file} (resident)"
else
    echo -e "Inference PIDs: ${inference_pid_file} (stopped)"
fi
echo -e "Sim PIDs: ${eval_pid_file}"

exit "$run_exit_code"
