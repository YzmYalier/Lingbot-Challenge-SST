#!/usr/bin/env bash
# Start only Qwen on physical GPU0, with independent logs/caches and two-image judge support.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

datafix_workspace="/mnt/data/liangjiaqi/yzm"
datafix_python="${DATAFIX_PYTHON:-$datafix_workspace/miniconda3/envs/lingbotvla/bin/python}"
datafix_vllm="${DATAFIX_VLLM:-$datafix_workspace/miniconda3/envs/vllm_env/bin/vllm}"
datafix_model="${QWEN_MODEL:-$datafix_workspace/vllm_space/qwen3.8-27B}"
datafix_port="${QWEN_PORT:-8000}"
datafix_runtime="${DATAFIX_QWEN_WORK:-$datafix_workspace/datafix-qwen-gpu0/$(hostname)}"
datafix_runtime="$(realpath -m -- "$datafix_runtime")"
datafix_rpc="/tmp/datafix-vllm-$(id -u)-$datafix_port"
datafix_url="http://127.0.0.1:$datafix_port"
datafix_log="$datafix_runtime/qwen_$datafix_port.log"
datafix_timeout="${QWEN_START_TIMEOUT:-600}"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--dry-run' ) ]]; then
  echo "Usage: bash $0 [--dry-run]" >&2
  exit 2
fi
[[ "$datafix_port" =~ ^[0-9]+$ && "$datafix_port" -gt 0 && "$datafix_port" -lt 65536 ]] || { echo 'Invalid QWEN_PORT' >&2; exit 2; }
[[ "$datafix_timeout" =~ ^[0-9]+$ && "$datafix_timeout" -gt 0 ]] || { echo 'Invalid QWEN_START_TIMEOUT' >&2; exit 2; }
[[ -x "$datafix_vllm" && -f "$datafix_model/config.json" ]] || { echo 'Missing vLLM executable or local Qwen model/config.json' >&2; exit 2; }

# Log/cache overrides must not point into existing model, service, training, or dataset trees.
for datafix_protected in "$datafix_workspace/auto_mask" "$datafix_workspace/sam3" \
  "$datafix_workspace/vllm_space" "$datafix_workspace/RoboTwinData" \
  "$datafix_workspace/miniconda3" "$datafix_workspace"/lingbot-vla-v2* "$datafix_model"; do
  datafix_protected="$(realpath -m -- "$datafix_protected")"
  if [[ "$datafix_runtime/" == "$datafix_protected/"* || "$datafix_protected/" == "$datafix_runtime/"* ]]; then
    echo "Refusing log/cache directory overlapping protected code/data: $datafix_runtime" >&2
    exit 2
  fi
done

datafix_command=(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  "TMPDIR=$datafix_runtime/tmp" "VLLM_RPC_BASE_PATH=$datafix_rpc"
  "XDG_CACHE_HOME=$datafix_runtime/cache" "HF_HOME=$datafix_runtime/cache/huggingface"
  "VLLM_CACHE_ROOT=$datafix_runtime/cache/vllm" "TORCH_HOME=$datafix_runtime/cache/torch"
  "TRITON_CACHE_DIR=$datafix_runtime/cache/triton" "TORCHINDUCTOR_CACHE_DIR=$datafix_runtime/cache/torchinductor"
  "CUDA_CACHE_PATH=$datafix_runtime/cache/cuda"
  "FLASHINFER_WORKSPACE_BASE=$datafix_runtime/cache/flashinfer"
  "$datafix_vllm" serve "$datafix_model"
  --served-model-name Qwen/Qwen3.8-27B --host 127.0.0.1 --port "$datafix_port"
  --max-model-len "${QWEN_MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION:-0.65}"
  --max-num-seqs "${QWEN_MAX_NUM_SEQS:-16}"
  --gdn-prefill-backend "${QWEN_GDN_PREFILL_BACKEND:-triton}"
  --limit-mm-per-prompt '{"image":2,"video":0}' --enable-prefix-caching)

if [[ "${1:-}" == '--dry-run' ]]; then
  printf 'GPU0 Qwen command: '
  printf '%q ' "${datafix_command[@]}"
  printf '\nLog: %s\nEndpoint: %s/v1/chat/completions\n' "$datafix_log" "$datafix_url"
  exit 0
fi

mkdir -p "$datafix_runtime/tmp" "$datafix_runtime/cache" "$datafix_rpc"
exec {datafix_lock_fd}>"$datafix_runtime/start_$datafix_port.lock"
flock -n "$datafix_lock_fd" || { echo 'Another Qwen launcher is already running.' >&2; exit 1; }
# Reuse only a healthy service whose PID belongs to this launcher/runtime.
datafix_pid_file="$datafix_runtime/qwen_$datafix_port.pid"
if [[ -f "$datafix_pid_file" ]]; then
  read -r datafix_existing < "$datafix_pid_file"
  if [[ "$datafix_existing" =~ ^[0-9]+$ && -r "/proc/$datafix_existing/cmdline" ]]; then
    datafix_cmdline="$(tr '\0' ' ' < "/proc/$datafix_existing/cmdline")"
    if [[ "$datafix_cmdline" == *"$datafix_vllm serve $datafix_model "* && "$datafix_cmdline" == *"--port $datafix_port "* ]]; then
      if curl -fsS --max-time 2 "$datafix_url/health" >/dev/null 2>&1; then
        printf 'Reusing healthy Qwen on GPU0: PID=%s endpoint=%s/v1/chat/completions\n' "$datafix_existing" "$datafix_url"
        exit 0
      fi
      echo "Owned Qwen PID=$datafix_existing is unhealthy or still loading; see $datafix_log. Refusing to launch a duplicate." >&2
      exit 1
    fi
  fi
fi
# A busy port is never stopped/replaced, including a service launched elsewhere.
"$datafix_python" - "$datafix_port" <<'PY'
import socket,sys
with socket.socket() as sock:
    try:
        sock.bind(('127.0.0.1', int(sys.argv[1])))
    except OSError as exc:
        raise SystemExit(f'Port {sys.argv[1]} is unavailable; existing services were left untouched: {exc}')
PY
(
  exec {datafix_lock_fd}>&-
  cd "$datafix_runtime"
  exec nohup setsid "${datafix_command[@]}"
) >>"$datafix_log" 2>&1 < /dev/null &
datafix_pid=$!
printf '%s\n' "$datafix_pid" > "$datafix_runtime/qwen_$datafix_port.pid"
printf 'Started Qwen on GPU0: PID=%s\nLog: %s\n' "$datafix_pid" "$datafix_log"

datafix_start=$SECONDS
while (( SECONDS - datafix_start < datafix_timeout )); do
  datafix_state=''
  if [[ -r "/proc/$datafix_pid/stat" ]]; then
    read -r datafix_stat < "/proc/$datafix_pid/stat" || true
    datafix_state="${datafix_stat##*) }"
  fi
  if [[ -z "$datafix_state" || "$datafix_state" == Z* ]]; then
    printf 'Qwen exited during startup. See %s\n' "$datafix_log" >&2
    tail -n 30 "$datafix_log" >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "$datafix_url/health" >/dev/null 2>&1; then
    printf 'Qwen ready after %ss: %s/v1/chat/completions\n' "$((SECONDS - datafix_start))" "$datafix_url"
    exit 0
  fi
  printf 'Qwen loading on GPU0: elapsed=%ss; log=%s\n' "$((SECONDS - datafix_start))" "$datafix_log"
  sleep 10
done
printf 'Qwen is not ready after %ss; PID=%s is still running. Check %s\n' "$datafix_timeout" "$datafix_pid" "$datafix_log" >&2
exit 1
