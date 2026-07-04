#!/usr/bin/env bash
set -Eeuo pipefail

# DiffusionGemma 26B-A4B IT NVFP4 vLLM launcher
# Default target: single NVIDIA Blackwell/Hopper GPU, OpenAI-compatible API.

MODEL="${MODEL:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-diffusiongemma}"
CONTAINER_NAME="${CONTAINER_NAME:-diffusiongemma-vllm}"
PORT="${PORT:-8000}"

# Current vLLM Docker Hub has Gemma CUDA 13 builds; fallback below tries :gemma.
IMAGE="${IMAGE:-vllm/vllm-openai:gemma-x86_64-cu130}"
PULL_IMAGE="${PULL_IMAGE:-missing}" # missing|always|never
REUSE_EXISTING_CONTAINER="${REUSE_EXISTING_CONTAINER:-true}"
STOP_TIMEOUT="${STOP_TIMEOUT:-5}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}" # unless-stopped|always|no|on-failure
READY_TIMEOUT="${READY_TIMEOUT:-${DIFFUSIONGEMMA_VLLM_READY_TIMEOUT:-900}}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-${DIFFUSIONGEMMA_VLLM_REQUEST_TIMEOUT:-600}}"
FAILURE_LOG_LINES="${FAILURE_LOG_LINES:-${DIFFUSIONGEMMA_VLLM_FAILURE_LOG_LINES:-80}}"
WARMUP_ON_START="${WARMUP_ON_START:-false}"
SLEEP_LEVEL="${SLEEP_LEVEL:-${DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL:-1}}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/.cache/vllm}"
TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/triton}"
TORCHINDUCTOR_CACHE="${TORCHINDUCTOR_CACHE:-$HOME/.cache/torchinductor}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
CUDA_CACHE="${CUDA_CACHE:-$HOME/.cache/nvidia_compute}"

# For RTX PRO 6000 96/98GB, 65536 is safer.
# Set MAX_MODEL_LEN=262144 if you really need full 256K context and have headroom.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

# Important: keep <=4 for DiffusionGemma.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
ATTN_BACKEND="${ATTN_BACKEND:-TRITON_ATTN}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-}"

ACTION="${1:-start}"
API_BASE="http://127.0.0.1:${PORT}/v1"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
CONTROL_BASE="http://127.0.0.1:${PORT}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

is_truthy() {
  [[ "${1,,}" =~ ^(1|true|yes|on)$ ]]
}

is_healthy() {
  curl -fsS --max-time 5 "${HEALTH_URL}" >/dev/null 2>&1 || \
    curl -fsS --max-time 5 "${API_BASE}/models" >/dev/null 2>&1
}

is_sleeping() {
  local body
  body="$(curl -fsS --max-time 5 "${CONTROL_BASE}/is_sleeping" 2>/dev/null || true)"
  body="${body,,}"
  [[ "${body}" == *true* && "${body}" != *false* ]]
}

post_control() {
  local path="$1"
  curl -fsS --max-time "${REQUEST_TIMEOUT}" -X POST "${CONTROL_BASE}${path}" >/dev/null
}

sleep_server() {
  if ! container_running; then
    echo "${CONTAINER_NAME} is not running; nothing to sleep."
    return 0
  fi
  if is_sleeping; then
    echo "DiffusionGemma vLLM is already sleeping."
    return 0
  fi
  wait_ready
  echo "Putting DiffusionGemma vLLM to sleep (level=${SLEEP_LEVEL}) to release VRAM."
  post_control "/sleep?level=${SLEEP_LEVEL}"
  echo "DiffusionGemma vLLM is sleeping."
}

wake_server() {
  if ! container_running; then
    echo "${CONTAINER_NAME} is not running; start it first."
    return 1
  fi
  wait_ready
  if is_sleeping; then
    echo "Waking DiffusionGemma vLLM."
    post_control "/wake_up"
  fi
  wait_ready
  echo "DiffusionGemma vLLM is awake."
}

show_recent_logs() {
  docker logs --tail "${FAILURE_LOG_LINES}" "${CONTAINER_NAME}" >&2 || true
}

wait_ready() {
  local started_at deadline elapsed
  started_at="${SECONDS}"
  deadline=$((SECONDS + READY_TIMEOUT))
  echo "Waiting up to ${READY_TIMEOUT}s for DiffusionGemma vLLM readiness..."
  while (( SECONDS < deadline )); do
    if is_healthy; then
      elapsed=$((SECONDS - started_at))
      echo "DiffusionGemma vLLM is ready after ${elapsed}s."
      return 0
    fi
    if container_exists && ! container_running; then
      echo "DiffusionGemma vLLM container exited before becoming ready. Recent logs:" >&2
      show_recent_logs
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for DiffusionGemma vLLM at ${API_BASE}. Recent logs:" >&2
  show_recent_logs
  return 1
}

warmup_api() {
  is_truthy "${WARMUP_ON_START}" || return 0
  echo "Running a one-token warmup request..."
  curl -fsS --max-time "${REQUEST_TIMEOUT}" "${API_BASE}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${SERVED_MODEL_NAME}\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"Reply with ok.\"}
      ],
      \"max_tokens\": 1,
      \"temperature\": 0
    }" >/dev/null || echo "WARNING: warmup request failed; continuing." >&2
}

print_urls() {
  echo
  echo "Local API:"
  echo "  ${API_BASE}"

  if [[ -n "${LIGHTNING_CLOUDSPACE_HOST:-}" ]]; then
    echo
    echo "Lightning URL:"
    echo "  https://${PORT}-${LIGHTNING_CLOUDSPACE_HOST}/v1"
  fi

  echo
  echo "OpenAI-compatible model name:"
  echo "  ${SERVED_MODEL_NAME}"
}

check_prereqs() {
  need_cmd docker
  need_cmd curl

  if ! docker info >/dev/null 2>&1; then
    die "Docker is not running or current user cannot access Docker."
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "WARNING: nvidia-smi not found on host. Docker may still work if NVIDIA runtime is configured."
  fi

  mkdir -p \
    "${HF_CACHE}" \
    "${VLLM_CACHE}" \
    "${TRITON_CACHE}" \
    "${TORCHINDUCTOR_CACHE}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${CUDA_CACHE}"
}

pull_image() {
  local mode="${PULL_IMAGE,,}"
  case "${mode}" in
    missing|auto)
      if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        echo "Using cached vLLM image: ${IMAGE}"
        return 0
      fi
      ;;
    never|false|0|no)
      echo "Skipping image pull: PULL_IMAGE=${PULL_IMAGE}"
      docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "Image not found locally: ${IMAGE}"
      return 0
      ;;
    always|true|1|yes)
      ;;
    *)
      die "Invalid PULL_IMAGE=${PULL_IMAGE}. Use missing, always, or never."
      ;;
  esac

  echo "Pulling vLLM image: ${IMAGE}"
  if docker pull "${IMAGE}"; then
    return 0
  fi

  if [[ "${IMAGE}" == "vllm/vllm-openai:gemma-x86_64-cu130" ]]; then
    echo
    echo "Primary image failed. Trying fallback: vllm/vllm-openai:gemma"
    IMAGE="vllm/vllm-openai:gemma"
    if [[ "${mode}" =~ ^(missing|auto)$ ]] && docker image inspect "${IMAGE}" >/dev/null 2>&1; then
      echo "Using cached fallback vLLM image: ${IMAGE}"
      return 0
    fi
    docker pull "${IMAGE}" || die "Could not pull vLLM Gemma image."
  else
    die "Could not pull image: ${IMAGE}"
  fi
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

stop_container() {
  if container_running; then
    echo "Stopping existing container and releasing VRAM: ${CONTAINER_NAME}"
    docker stop --time "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null
  elif container_exists; then
    echo "Container already stopped: ${CONTAINER_NAME}"
  fi
}

apply_restart_policy() {
  container_exists || return 0
  local policy="${RESTART_POLICY}"
  if [[ -z "${policy}" || "${policy,,}" == "none" ]]; then
    policy="no"
  fi
  echo "Applying Docker restart policy '${policy}' to ${CONTAINER_NAME}."
  docker update --restart "${policy}" "${CONTAINER_NAME}" >/dev/null
}

remove_container() {
  if container_exists; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi
}

start_container() {
  local force_recreate="${1:-false}"
  check_prereqs

  if [[ "${force_recreate}" != "true" ]]; then
    if container_running; then
      echo "${CONTAINER_NAME} is already running."
      apply_restart_policy
      wake_server
      warmup_api
      print_urls
      return 0
    fi

    if container_exists && is_truthy "${REUSE_EXISTING_CONTAINER}"; then
      echo "Starting existing container: ${CONTAINER_NAME}"
      echo "Use 'bash $0 restart' or REUSE_EXISTING_CONTAINER=false to recreate it with new settings."
      apply_restart_policy
      docker start "${CONTAINER_NAME}" >/dev/null
      wake_server
      warmup_api
      print_urls
      return 0
    fi
  fi

  pull_image
  remove_container

  env_args=(
    -e "VLLM_USE_V2_MODEL_RUNNER=1"
    -e "XDG_CACHE_HOME=/root/.cache"
    -e "HF_HOME=/root/.cache/huggingface"
    -e "VLLM_CACHE_ROOT=/root/.cache/vllm"
    -e "TRITON_CACHE_DIR=/root/.cache/triton"
    -e "TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor"
    -e "TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions"
    -e "CUDA_CACHE_PATH=/root/.nv/ComputeCache"
    -e "VLLM_SERVER_DEV_MODE=1"
  )

  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_args+=(
      -e "HF_TOKEN=${HF_TOKEN}"
      -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"
    )
  else
    echo "HF_TOKEN not set. If the repo requires Gemma terms/auth, run with: HF_TOKEN=hf_xxx $0"
  fi

  CHAT_TEMPLATE_KWARGS="{\"enable_thinking\":${ENABLE_THINKING}}"

  vllm_args=(
    --model "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --trust-remote-code
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --attention-backend "${ATTN_BACKEND}"
    --generation-config vllm
    --enable-chunked-prefill
    --enable-auto-tool-choice
    --tool-call-parser gemma4
    --reasoning-parser gemma4
    --enable-sleep-mode
    --override-generation-config '{"max_new_tokens": null}'
    --default-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}"
    --host 0.0.0.0
    --port "${PORT}"
  )

  if [[ -n "${LOAD_FORMAT}" ]]; then
    vllm_args+=(--load-format "${LOAD_FORMAT}")
  fi

  if [[ -n "${SAFETENSORS_LOAD_STRATEGY}" ]]; then
    vllm_args+=(--safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}")
  fi

  echo
  echo "Starting ${CONTAINER_NAME}"
  echo "Model:       ${MODEL}"
  echo "Served as:   ${SERVED_MODEL_NAME}"
  echo "Image:       ${IMAGE}"
  echo "Port:        ${PORT}"
  echo "Max context: ${MAX_MODEL_LEN}"
  echo "Restart:     ${RESTART_POLICY}"
  echo

  restart_args=()
  if [[ -n "${RESTART_POLICY}" && "${RESTART_POLICY,,}" != "no" && "${RESTART_POLICY,,}" != "none" ]]; then
    restart_args=(--restart "${RESTART_POLICY}")
  fi

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --ipc=host \
    --network host \
    --shm-size 16g \
    --gpus all \
    "${restart_args[@]}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${VLLM_CACHE}:/root/.cache/vllm" \
    -v "${TRITON_CACHE}:/root/.cache/triton" \
    -v "${TORCHINDUCTOR_CACHE}:/root/.cache/torchinductor" \
    -v "${TORCH_EXTENSIONS_DIR}:/root/.cache/torch_extensions" \
    -v "${CUDA_CACHE}:/root/.nv/ComputeCache" \
    "${env_args[@]}" \
    "${IMAGE}" \
      "${vllm_args[@]}"

  wait_ready
  warmup_api
  print_urls
  echo
  echo "Logs:"
  echo "  bash $0 logs"
  echo
  echo "Test:"
  echo "  bash $0 test"
}

show_logs() {
  docker logs -f "${CONTAINER_NAME}"
}

show_status() {
  docker ps -a --filter "name=${CONTAINER_NAME}"
  if is_healthy; then
    echo
    echo "Health: ready"
    if is_sleeping; then
      echo "Sleep:  sleeping"
    else
      echo "Sleep:  awake"
    fi
  else
    echo
    echo "Health: not ready"
  fi
  print_urls
}

test_api() {
  echo "Testing OpenAI-compatible chat endpoint..."
  wait_ready
  curl -sS --max-time "${REQUEST_TIMEOUT}" "${API_BASE}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${SERVED_MODEL_NAME}\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"Write a short poem about text diffusion.\"}
      ],
      \"max_tokens\": 160,
      \"temperature\": 0.7
    }"
  echo
}

case "${ACTION}" in
  start)
    start_container
    ;;
  restart)
    start_container true
    ;;
  stop)
    stop_container
    ;;
  sleep)
    sleep_server
    ;;
  wake)
    wake_server
    ;;
  remove)
    remove_container
    ;;
  logs)
    show_logs
    ;;
  status)
    show_status
    ;;
  test)
    test_api
    ;;
  *)
    echo "Usage: bash $0 {start|restart|stop|sleep|wake|remove|logs|status|test}"
    echo
    echo "Examples:"
    echo "  HF_TOKEN=hf_xxx bash $0 start"
    echo "  MAX_MODEL_LEN=262144 bash $0 restart"
    echo "  PORT=8001 SERVED_MODEL_NAME=diffusiongemma bash $0 start"
    exit 1
    ;;
esac
