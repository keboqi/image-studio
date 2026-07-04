import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import modal

# Use CUDA 13.2.1
cuda_version = "13.2.1"
flavor = "devel" 
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

MINUTES = 60

DIFFUSIONGEMMA_VLLM_MODEL_ID = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_HF_MODEL",
    "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
)
DIFFUSIONGEMMA_VLLM_SERVED_MODEL_NAME = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_MODEL",
    "diffusiongemma",
)
DIFFUSIONGEMMA_VLLM_PORT = 8001
DIFFUSIONGEMMA_VLLM_MAX_MODEL_LEN = os.environ.get("DIFFUSIONGEMMA_VLLM_MAX_MODEL_LEN", "65536")
DIFFUSIONGEMMA_VLLM_MAX_NUM_SEQS = os.environ.get("DIFFUSIONGEMMA_VLLM_MAX_NUM_SEQS", "2")
DIFFUSIONGEMMA_VLLM_GPU_MEMORY_UTILIZATION = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_GPU_MEMORY_UTILIZATION",
    "0.25",
)
DIFFUSIONGEMMA_VLLM_ATTENTION_BACKEND = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_ATTENTION_BACKEND",
    "TRITON_ATTN",
)
# Keep kernel selection aligned with the Docker launcher: let vLLM auto-select
# the best MoE backend for the current wheel, model, and Blackwell GPU.
DIFFUSIONGEMMA_VLLM_MOE_BACKEND = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_MOE_BACKEND",
    "auto",
)
DIFFUSIONGEMMA_VLLM_LOAD_FORMAT = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_LOAD_FORMAT",
    os.environ.get("LOAD_FORMAT", ""),
)
DIFFUSIONGEMMA_VLLM_SAFETENSORS_LOAD_STRATEGY = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_SAFETENSORS_LOAD_STRATEGY",
    os.environ.get("SAFETENSORS_LOAD_STRATEGY", ""),
)
DIFFUSIONGEMMA_VLLM_ENABLE_THINKING = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_ENABLE_THINKING",
    "true",
)
DIFFUSIONGEMMA_VLLM_ENABLE_REASONING_PARSER = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_ENABLE_REASONING_PARSER",
    "true",
)
DIFFUSIONGEMMA_VLLM_USE_V2_MODEL_RUNNER = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_USE_V2_MODEL_RUNNER",
    "1",
)
# Match the vllm/vllm-openai:gemma-x86_64-cu130 image recommended by the
# DiffusionGemma model card. Latest nightly wheels may not include the native
# DiffusionGemma registration and can fall back to the generic Transformers path.
DIFFUSIONGEMMA_VLLM_WHEEL_INDEX_URL = os.environ.get(
    "DIFFUSIONGEMMA_VLLM_WHEEL_INDEX_URL",
    "https://wheels.vllm.ai/74b5964f02c7e023fadd3004cfac8a61c52eef1f/cu130/vllm/",
)
DIFFUSIONGEMMA_VLLM_WHEEL_URL = os.environ.get("DIFFUSIONGEMMA_VLLM_WHEEL_URL", "")
DIFFUSIONGEMMA_VLLM_SCRIPT = "/root/deploy_diffusiongemma_vllm.sh"
DIFFUSIONGEMMA_VLLM_VENV = "/opt/diffusiongemma-vllm"
DIFFUSIONGEMMA_VLLM_API_BASE = f"http://127.0.0.1:{DIFFUSIONGEMMA_VLLM_PORT}/v1"
DIFFUSIONGEMMA_VLLM_LOG = "/persistent_app/logs/diffusiongemma-vllm.log"
DIFFUSIONGEMMA_VLLM_PID_FILE = "/persistent_app/vllm/diffusiongemma-vllm.pid"
PI_CONFIG_DIR = os.environ.get("PI_CONFIG_DIR", "/root/.pi/agent")
PI_PERSISTENT_DIR = "/persistent_app/pi"
PI_NPM_PACKAGE = os.environ.get("PI_NPM_PACKAGE", "@earendil-works/pi-coding-agent")
PI_NPM_VERSION = os.environ.get("PI_NPM_VERSION", "latest")
PI_MODEL = os.environ.get("PI_MODEL", DIFFUSIONGEMMA_VLLM_SERVED_MODEL_NAME)
BOOGU_IMAGE_DIR = "/root/boogu_image"
BOOGU_IMAGE_MODELS_DIR = "/root/models"
BOOGU_IMAGE_MODEL_REPOS = {
    "Boogu-Image-0.1-Turbo": "Boogu/Boogu-Image-0.1-Turbo",
    "Boogu-Image-0.1-Base": "Boogu/Boogu-Image-0.1-Base",
    "Boogu-Image-0.1-Edit": "Boogu/Boogu-Image-0.1-Edit",
    "Boogu-Image-0.1-Edit-Turbo": "Boogu/Boogu-Image-0.1-Edit-Turbo",
}

# Krea-2 Turbo ComfyUI backend constants
KREA2_COMFY_SCRIPT = "/root/deploy_krea2_comfy.sh"
KREA2_COMFY_PORT = os.environ.get("KREA2_COMFY_PORT", "8188")
KREA2_COMFY_READY_TIMEOUT = os.environ.get("KREA2_COMFY_READY_TIMEOUT", "300")
KREA2_COMFY_START_TIMEOUT = os.environ.get("KREA2_COMFY_START_TIMEOUT", "420")
KREA2_COMFY_REQUEST_TIMEOUT = os.environ.get("KREA2_COMFY_REQUEST_TIMEOUT", "900")
KREA2_COMFY_DIR = "/root/krea2-comfy"
KREA2_COMFY_VENV = os.path.join(KREA2_COMFY_DIR, ".venv")

MODAL_DIFFUSIONGEMMA_VLLM_SCRIPT_SOURCE = r'''
#!/usr/bin/env bash
set -Eeuo pipefail

MODEL="${MODEL:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-diffusiongemma}"
PORT="${PORT:-8001}"
VENV="${DIFFUSIONGEMMA_VLLM_VENV:-/opt/diffusiongemma-vllm}"
HF_CACHE="${HF_CACHE:-/persistent_cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-/persistent_cache/torch}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/persistent_cache/vllm}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/persistent_cache/xdg}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/persistent_cache/triton}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/persistent_cache/torchinductor}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/persistent_cache/torch_extensions}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/persistent_cache/nvidia_compute}"
PID_FILE="${DIFFUSIONGEMMA_VLLM_PID_FILE:-/persistent_app/vllm/diffusiongemma-vllm.pid}"
LOG_FILE="${DIFFUSIONGEMMA_VLLM_LOG_FILE:-/persistent_app/logs/diffusiongemma-vllm.log}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
ATTN_BACKEND="${ATTN_BACKEND:-TRITON_ATTN}"
MOE_BACKEND="${MOE_BACKEND:-${DIFFUSIONGEMMA_VLLM_MOE_BACKEND:-auto}}"
LOAD_FORMAT="${LOAD_FORMAT:-${DIFFUSIONGEMMA_VLLM_LOAD_FORMAT:-}}"
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-${DIFFUSIONGEMMA_VLLM_SAFETENSORS_LOAD_STRATEGY:-}}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
ENABLE_REASONING_PARSER="${ENABLE_REASONING_PARSER:-true}"
USE_V2_MODEL_RUNNER="${USE_V2_MODEL_RUNNER:-1}"
READY_TIMEOUT="${DIFFUSIONGEMMA_VLLM_READY_TIMEOUT:-900}"
REQUEST_TIMEOUT="${DIFFUSIONGEMMA_VLLM_REQUEST_TIMEOUT:-600}"
FAILURE_LOG_LINES="${DIFFUSIONGEMMA_VLLM_FAILURE_LOG_LINES:-40}"
SLEEP_LEVEL="${SLEEP_LEVEL:-${DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL:-1}}"
ACTION="${1:-start}"

API_BASE="http://127.0.0.1:${PORT}/v1"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
CONTROL_BASE="http://127.0.0.1:${PORT}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

pid_value() {
  [[ -f "${PID_FILE}" ]] || return 1
  cat "${PID_FILE}"
}

is_process_running() {
  local pid
  pid="$(pid_value 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
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

is_control_reachable() {
  curl -fsS --max-time 5 "${CONTROL_BASE}/is_sleeping" >/dev/null 2>&1
}

post_control() {
  local path="$1"
  curl -fsS --max-time "${REQUEST_TIMEOUT}" -X POST "${CONTROL_BASE}${path}" >/dev/null
}

print_failure_logs() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "No vLLM log file found at ${LOG_FILE}." >&2
    return 0
  fi

  echo "vLLM final ${FAILURE_LOG_LINES} log lines:" >&2
  tail -n "${FAILURE_LOG_LINES}" "${LOG_FILE}" >&2 || true

  echo "vLLM compact error summary from ${LOG_FILE}:" >&2
  grep -aiE 'traceback|error|exception|runtimeerror|valueerror|cuda|oom|out of memory|failed|no module|modulenotfound|importerror|not found|kv cache|max seq|memory|permission|invalid|unsupported' "${LOG_FILE}" \
    | tail -n "${FAILURE_LOG_LINES}" >&2 || true
}

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if is_healthy; then
      echo "DiffusionGemma vLLM is ready at ${API_BASE}."
      return 0
    fi
    if [[ -f "${PID_FILE}" ]] && ! is_process_running; then
      echo "DiffusionGemma vLLM process exited before becoming ready. Last logs:" >&2
      print_failure_logs
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for DiffusionGemma vLLM at ${API_BASE}. Last logs:" >&2
  print_failure_logs
  return 1
}

setup_env() {
  [[ -x "${VENV}/bin/vllm" ]] || die "vLLM executable not found: ${VENV}/bin/vllm"
  mkdir -p \
    "${HF_CACHE}/hub" \
    "${TORCH_CACHE}" \
    "${VLLM_CACHE_ROOT}" \
    "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "$(dirname "${PID_FILE}")" \
    "$(dirname "${LOG_FILE}")"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_HOME="${HF_CACHE}"
  export HF_HUB_CACHE="${HF_CACHE}/hub"
  export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
  export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
  export TORCH_HOME="${TORCH_CACHE}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}"
  export CUDA_CACHE_PATH="${CUDA_CACHE_PATH}"
  export VLLM_HOST_IP=127.0.0.1
  export VLLM_SERVER_DEV_MODE=1
  if is_enabled "${USE_V2_MODEL_RUNNER}"; then
    export VLLM_USE_V2_MODEL_RUNNER=1
  else
    unset VLLM_USE_V2_MODEL_RUNNER
  fi
  export PYTHONNOUSERSITE=1
  export VIRTUAL_ENV="${VENV}"
  export PATH="${VENV}/bin:${PATH}"
  unset PYTHONPATH
}

start_server_attempt() {
  local chat_template_kwargs
  chat_template_kwargs="{\"enable_thinking\":${ENABLE_THINKING}}"
  local reasoning_args=()
  if is_enabled "${ENABLE_REASONING_PARSER}"; then
    reasoning_args+=(--reasoning-parser gemma4)
  fi
  local moe_args=()
  if [[ -n "${MOE_BACKEND}" && "${MOE_BACKEND,,}" != "auto" ]]; then
    moe_args+=(--moe-backend "${MOE_BACKEND}")
  fi
  local load_args=()
  if [[ -n "${LOAD_FORMAT}" ]]; then
    load_args+=(--load-format "${LOAD_FORMAT}")
  fi
  if [[ -n "${SAFETENSORS_LOAD_STRATEGY}" ]]; then
    load_args+=(--safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}")
  fi
  local vllm_version
  vllm_version="$("${VENV}/bin/python" -m pip show vllm | awk -F': ' '/^Version/{print $2}' || true)"

  echo
  echo "Starting DiffusionGemma vLLM in ${VENV}"
  echo "Model:       ${MODEL}"
  echo "Served as:   ${SERVED_MODEL_NAME}"
  echo "Port:        ${PORT}"
  echo "HF cache:    ${HF_CACHE}"
  echo "vLLM cache:  ${VLLM_CACHE_ROOT}"
  echo "Thinking:    ${ENABLE_THINKING}"
  echo "Reasoning:   ${ENABLE_REASONING_PARSER}"
  echo "MoE backend: ${MOE_BACKEND}"
  echo "V2 runner:   ${USE_V2_MODEL_RUNNER}"
  echo "vLLM version: ${vllm_version}"
  echo "Logs:        ${LOG_FILE}"
  echo "Wheel index: ${DIFFUSIONGEMMA_VLLM_WHEEL_INDEX_URL:-unset}"
  echo

  setsid "${VENV}/bin/vllm" serve "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --attention-backend "${ATTN_BACKEND}" \
    "${moe_args[@]}" \
    "${load_args[@]}" \
    --generation-config vllm \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    "${reasoning_args[@]}" \
    --enable-sleep-mode \
    --override-generation-config '{"max_new_tokens": null}' \
    --default-chat-template-kwargs "${chat_template_kwargs}" \
    > "${LOG_FILE}" 2>&1 &

  echo "$!" > "${PID_FILE}"
  wait_ready
}

start_server() {
  setup_env
  if is_healthy || is_control_reachable; then
    echo "DiffusionGemma vLLM is already reachable at ${CONTROL_BASE}; waking or waiting for readiness."
    wake_server
    return 0
  fi
  if is_process_running; then
    echo "DiffusionGemma vLLM process $(pid_value) is already running; waking or waiting for readiness."
    wake_server
    return $?
  fi

  start_server_attempt
}

sleep_server() {
  if is_sleeping; then
    echo "DiffusionGemma vLLM is already sleeping."
    return 0
  fi
  if ! is_process_running && ! is_healthy && ! is_control_reachable; then
    echo "DiffusionGemma vLLM is not running; nothing to sleep."
    rm -f "${PID_FILE}"
    return 0
  fi
  wait_ready
  echo "Putting DiffusionGemma vLLM to sleep (level=${SLEEP_LEVEL}) to release VRAM."
  post_control "/sleep?level=${SLEEP_LEVEL}"
  echo "DiffusionGemma vLLM is sleeping."
}

wake_server() {
  if ! is_process_running && ! is_healthy && ! is_control_reachable; then
    echo "DiffusionGemma vLLM is not running; start it first."
    rm -f "${PID_FILE}"
    return 1
  fi
  if is_sleeping; then
    echo "Waking DiffusionGemma vLLM."
    post_control "/wake_up"
  fi
  wait_ready
  echo "DiffusionGemma vLLM is awake."
}

stop_server() {
  local pid
  pid="$(pid_value 2>/dev/null || true)"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "DiffusionGemma vLLM is not running."
    rm -f "${PID_FILE}"
    return 0
  fi

  echo "Stopping DiffusionGemma vLLM process ${pid}."
  kill -TERM "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
  local deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      rm -f "${PID_FILE}"
      return 0
    fi
    sleep 1
  done
  kill -KILL "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
  rm -f "${PID_FILE}"
}

show_logs() {
  tail -n 300 -f "${LOG_FILE}"
}

show_status() {
  if is_healthy; then
    echo "healthy: ${API_BASE}"
    if is_sleeping; then
      echo "sleep: sleeping"
    else
      echo "sleep: awake"
    fi
  elif is_sleeping; then
    echo "control reachable: ${CONTROL_BASE}"
    echo "sleep: sleeping"
  elif is_process_running; then
    echo "process running but not healthy: $(pid_value)"
  else
    echo "stopped"
  fi
}

test_api() {
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
    start_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  stop)
    stop_server
    ;;
  sleep)
    sleep_server
    ;;
  wake)
    wake_server
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
    echo "Usage: bash $0 {start|restart|stop|sleep|wake|logs|status|test}" >&2
    exit 1
    ;;
esac
'''.lstrip()


def _write_modal_vllm_script_command() -> str:
    encoded = base64.b64encode(MODAL_DIFFUSIONGEMMA_VLLM_SCRIPT_SOURCE.encode("utf-8")).decode("ascii")
    return (
        'python -c "import base64,pathlib; '
        f"pathlib.Path('{DIFFUSIONGEMMA_VLLM_SCRIPT}').write_text("
        f"base64.b64decode('{encoded}').decode('utf-8'), encoding='utf-8')\""
    )


def _install_diffusiongemma_vllm_command() -> str:
    code = f"""
import os
import re
import subprocess
import urllib.parse
import urllib.request

python_bin = {DIFFUSIONGEMMA_VLLM_VENV!r} + "/bin/python"
wheel_url = {DIFFUSIONGEMMA_VLLM_WHEEL_URL!r}.strip()
wheel_index_url = {DIFFUSIONGEMMA_VLLM_WHEEL_INDEX_URL!r}

if not wheel_url:
    html = urllib.request.urlopen(wheel_index_url, timeout=60).read().decode("utf-8", errors="replace")
    hrefs = re.findall(r'href=["\\']([^"\\']*x86_64\\.whl)["\\']', html)
    if not hrefs:
        raise RuntimeError(f"No x86_64 vLLM wheel found at {{wheel_index_url}}")
    wheel_url = urllib.parse.urljoin(wheel_index_url, hrefs[-1])

print(f"Installing DiffusionGemma vLLM wheel: {{wheel_url}}", flush=True)
subprocess.check_call([
    python_bin,
    "-m",
    "uv",
    "pip",
    "install",
    "--python",
    python_bin,
    "--pre",
    "--torch-backend=cu130",
    "--index-url",
    "https://pypi.org/simple",
    "-U",
    wheel_url,
])
"""
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return f'python -c "import base64; exec(base64.b64decode({encoded!r}).decode())"'


def _patch_diffusiongemma_vllm_instrumentator_command() -> str:
    code = r'''
from pathlib import Path
from sysconfig import get_paths

patch_module = r"""
def _patch() -> None:
    try:
        from prometheus_fastapi_instrumentator import routing as _pfi_routing
        from starlette.routing import Match, Mount
    except Exception:
        return

    if getattr(_pfi_routing, "_vllm_pathless_route_patch", False):
        return

    def _get_route_name(scope, routes, route_name=None):
        for route in routes:
            route_path = getattr(route, "path", None)
            route_matches = getattr(route, "matches", None)
            if route_path is None or route_matches is None:
                continue

            match, child_scope = route_matches(scope)
            if match == Match.FULL:
                route_name = route_path
                child_scope = {**scope, **child_scope}
                if isinstance(route, Mount) and getattr(route, "routes", None):
                    child_route_name = _get_route_name(child_scope, route.routes, route_name)
                    route_name = None if child_route_name is None else route_name + child_route_name
                return route_name

            if match == Match.PARTIAL and route_name is None:
                route_name = route_path

        return None

    _pfi_routing._get_route_name = _get_route_name
    _pfi_routing._vllm_pathless_route_patch = True

_patch()
"""

site_packages = Path(get_paths()["purelib"])
module_path = site_packages / "_vllm_prometheus_route_patch.py"
pth_path = site_packages / "zz_vllm_prometheus_route_patch.pth"
module_path.write_text(patch_module, encoding="utf-8")
pth_path.write_text("import _vllm_prometheus_route_patch\n", encoding="utf-8")
print(f"Installed vLLM Prometheus route patch: {pth_path}")
'''
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return (
        f'{DIFFUSIONGEMMA_VLLM_VENV}/bin/python -c '
        f'"import base64; exec(base64.b64decode({encoded!r}).decode())"'
    )


def _install_modal_pi_command() -> str:
    script = f"""
set -Eeuo pipefail

export NVM_DIR="${{NVM_DIR:-/root/.nvm}}"
export NODE_VERSION="${{NODE_VERSION:-node}}"
export NPM_VERSION="${{NPM_VERSION:-latest}}"
export PI_NPM_PACKAGE="${{PI_NPM_PACKAGE:-{PI_NPM_PACKAGE}}}"
export PI_NPM_VERSION="${{PI_NPM_VERSION:-{PI_NPM_VERSION}}}"

if [[ -s "$NVM_DIR/nvm.sh" ]]; then
  source "$NVM_DIR/nvm.sh"
else
  mkdir -p "$NVM_DIR"
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
  source "$NVM_DIR/nvm.sh"
fi

nvm install "$NODE_VERSION" --latest-npm
nvm use "$NODE_VERSION"
nvm alias default "$(node -v)" >/dev/null

npm install -g "npm@$NPM_VERSION"
npm install -g --force --ignore-scripts "$PI_NPM_PACKAGE@$PI_NPM_VERSION"
hash -r

ln -sf "$(command -v node)" /usr/local/bin/node
ln -sf "$(command -v npm)" /usr/local/bin/npm
ln -sf "$(command -v npx)" /usr/local/bin/npx
ln -sf "$(command -v pi)" /usr/local/bin/pi

node -v
npm -v
pi --version || true
"""
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return (
        'python -c "import base64,subprocess; '
        f"subprocess.check_call(['bash', '-lc', base64.b64decode({encoded!r}).decode('utf-8')])\""
    )


# Define the Modal image strictly following the quick start script sequence
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("ffmpeg", "git", "wget", "curl", "build-essential", "gcc", "g++", "cmake")
    # Keep this layer to true build-time/compiler env only. Runtime knobs are
    # set in ImageStudioWebUI._configure_diffusiongemma_vllm_backend().
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_PATH": "/usr/local/cuda", 
        "TORCH_CUDA_ARCH_LIST": "12.0",  # Optimized specifically for Blackwell (RTX PRO 6000)
        "FORCE_CUDA": "1",
        "CXX": "g++",
        "CC": "gcc",
    })
    .run_commands(_install_modal_pi_command())
    .run_commands("pip install --upgrade pip setuptools wheel")
    .pip_install(
        "torch==2.8.0", 
        "torchaudio==2.8.0",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cu128"
    )
    .pip_install("uv")
    # AI watermark remover setup
    .run_commands(
        "git clone https://github.com/wiltodelta/remove-ai-watermarks.git /root/remove-ai-watermarks",
        "cd /root/remove-ai-watermarks && uv venv .venv --python 3.12",
        "cd /root/remove-ai-watermarks && uv pip install --python .venv/bin/python -e '.[gpu]'",
        "cd /root/remove-ai-watermarks && uv pip install --python .venv/bin/python -U 'transformers>=4.57.1,<5' 'hf_transfer>=0.1.9'"
    )
    .pip_install("Librosa")
    .pip_install("https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.8-cp312-cp312-linux_x86_64.whl")
    .pip_install("gradio")
    .pip_install("diffusers==0.36.0")
    .run_commands("pip install flash-attn --no-build-isolation --no-cache-dir")
    .run_commands("pip install sageattention --no-build-isolation")

    # Boogu-Image generation/edit setup. Checkpoint weights are downloaded into
    # the persistent /root/models link by prepare_models/first use.
    .run_commands(f"git clone https://github.com/boogu-project/Boogu-Image.git {BOOGU_IMAGE_DIR}")
    .run_commands(
        "pip install cache-dit webdataset python-dotenv omegaconf"
    )

    # Ideogram 4 NVFP4 generation setup. Model weights stay out of the image and
    # are downloaded into the mounted Hugging Face cache by prepare_models/first use.
    .run_commands("git clone https://github.com/keboqi/ideogram4.git /root/ideogram4")
    .run_commands(
        "pip install -U 'transformers>=4.57.6' 'huggingface-hub>=1.0' bitsandbytes "
        "'comfy-kitchen[cublas]' json-repair safetensors sentencepiece accelerate "
        "einops requests"
    )
    .run_commands("pip install -U packaging ninja psutil")
    
    # Optional PiD 4x decoder for Z-Image Full, Qwen Image, and Ideogram 4:
    .run_commands("git clone --depth 1 https://github.com/nv-tlabs/PiD.git /root/PiD")
    .pip_install("huggingface_hub", "hf_transfer") # Ensure hf-cli and fast transfers are available
    .run_commands("pip install hydra-core==1.3.2 omegaconf==2.3.0 attrs einops loguru termcolor fvcore iopath pynvml wandb imageio opencv-python-headless pandas safetensors 'huggingface-hub>=1.0' sentencepiece boto3 botocore")
    
    # SeedVR2 Setup
    .run_commands(
        "git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /root/seedvr2_upscaler",
        "pip install -r /root/seedvr2_upscaler/requirements.txt"
    )
    
    # LTX-Web Video Generation Setup
    .run_commands(
        "git clone https://github.com/keboqi/ltx-web.git /root/ltx-web",
        "cd /root/ltx-web && sed -i 's/python api.py//g' run.sh",
        "cd /root/ltx-web && sed -i 's/huggingface-cli/hf/g' run.sh"
    )
    # We install ltx-web requirements in the image, but do NOT run `sh run.sh` here
    # to avoid downloading gigabytes of models into the image layer.
    .run_commands(
        "cd /root/ltx-web && pip install -r requirements.txt",
        "cd /root/ltx-web && pip install sentencepiece",
        "cd /root/ltx-web && pip install -e LTX-2/packages/ltx-core",
        "cd /root/ltx-web && pip install -e LTX-2/packages/ltx-pipelines"
    )
    .pip_install("numpy<2.0")
    .run_commands("cd /root/ideogram4 && git fetch --all && git pull")
    .run_commands(
        f"python -m venv {DIFFUSIONGEMMA_VLLM_VENV}",
        f"{DIFFUSIONGEMMA_VLLM_VENV}/bin/python -m pip install -U pip uv",
        f"{DIFFUSIONGEMMA_VLLM_VENV}/bin/python -m uv pip install "
        f"--python {DIFFUSIONGEMMA_VLLM_VENV}/bin/python "
        "hf_transfer nvidia-modelopt 'numpy<2'",
        _install_diffusiongemma_vllm_command(),
        _patch_diffusiongemma_vllm_instrumentator_command(),
    )
    .run_commands(_write_modal_vllm_script_command())
    .run_commands("cd /root/ltx-web && git fetch --all && git pull")
    # Krea-2 Turbo ComfyUI setup: copy the deploy script into the image.
    # The actual venv install happens at runtime on first use via
    # Krea2ComfyService.ensure_running() because it requires GPU (the
    # doctor check imports torch.cuda).  The persistent cache
    # volume preserves the installed venv across container restarts.
    .add_local_file("deploy_krea2_comfy.sh", remote_path=KREA2_COMFY_SCRIPT, copy=True)
    .run_commands(f"chmod +x {KREA2_COMFY_SCRIPT}")
    # These go last without copy so they are mounted at startup for fast iteration.
    .add_local_dir("image_studio", remote_path="/root/image_studio")
    .add_local_file("image_studio_webui.py", remote_path="/root/image_studio_webui.py")
    .add_local_file("jsondesigner.html", remote_path="/root/jsondesigner.html")
)

app = modal.App("image-studio-webui")

# Create persistent storage volumes
cache_storage = modal.Volume.from_name("image-studio-cache", create_if_missing=True)
app_storage = modal.Volume.from_name("image-studio-app", create_if_missing=True)

PERSISTENT_CACHE_DIR = "/persistent_cache"
PERSISTENT_APP_DIR = "/persistent_app"

DIFFUSIONGEMMA_CACHE_DIRS = (
    "/persistent_cache/huggingface/hub",
    "/persistent_cache/torch",
    "/persistent_cache/vllm",
    "/persistent_cache/xdg",
    "/persistent_cache/triton",
    "/persistent_cache/torchinductor",
    "/persistent_cache/torch_extensions",
    "/persistent_cache/nvidia_compute",
)

PERSISTENT_CACHE_DIRS = (
    "/persistent_cache/huggingface",
    *DIFFUSIONGEMMA_CACHE_DIRS,
    "/persistent_cache/pid_models",
    "/persistent_cache/pid_hf_cache",
    "/persistent_cache/boogu_models",
    "/persistent_cache/ltx_models",
    "/persistent_cache/seedvr2_models",
    "/persistent_cache/krea2_comfy",
    "/persistent_app/outputs",
    "/persistent_app/cache",
    "/persistent_app/logs",
    "/persistent_app/vllm",
    PI_PERSISTENT_DIR,
    os.path.join(PI_PERSISTENT_DIR, "agent"),
)

PERSISTENT_DIR_LINKS = {
    "/root/.cache/huggingface": "/persistent_cache/huggingface",
    "/root/.cache/torch": "/persistent_cache/torch",
    "/root/.cache/transformers": "/persistent_cache/huggingface/hub",
    "/root/.cache/vllm": "/persistent_cache/vllm",
    "/root/.cache/triton": "/persistent_cache/triton",
    "/root/.cache/torchinductor": "/persistent_cache/torchinductor",
    "/root/.cache/torch_extensions": "/persistent_cache/torch_extensions",
    "/root/.nv/ComputeCache": "/persistent_cache/nvidia_compute",
    "/root/PiD/checkpoints": "/persistent_cache/pid_models",
    "/root/PiD/.cache": "/persistent_cache/pid_hf_cache",
    BOOGU_IMAGE_MODELS_DIR: "/persistent_cache/boogu_models",
    "/root/ltx-web/models": "/persistent_cache/ltx_models",
    "/root/seedvr2_upscaler/models": "/persistent_cache/seedvr2_models",
    # Krea2 ComfyUI keeps its venv, cache, and HF downloads in persistent storage
    KREA2_COMFY_DIR: "/persistent_cache/krea2_comfy",
    "/root/outputs": "/persistent_app/outputs",
    "/root/.pi": PI_PERSISTENT_DIR,
}

PERSISTENT_FILE_LINKS = {
    "/root/ideogram4_upsample_cache.json": "/persistent_app/cache/ideogram4_upsample_cache.json",
}


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ensure_diffusiongemma_vllm_cache_dirs():
    for path in DIFFUSIONGEMMA_CACHE_DIRS:
        os.makedirs(path, exist_ok=True)


def _apply_env_defaults(defaults: dict[str, str]):
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _diffusiongemma_vllm_runtime_env_defaults() -> dict[str, str]:
    return {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/persistent_cache/huggingface",
        "HF_HUB_CACHE": "/persistent_cache/huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "/persistent_cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/persistent_cache/huggingface/hub",
        "TORCH_HOME": "/persistent_cache/torch",
        "VLLM_CACHE_ROOT": "/persistent_cache/vllm",
        "XDG_CACHE_HOME": "/persistent_cache/xdg",
        "TRITON_CACHE_DIR": "/persistent_cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/persistent_cache/torchinductor",
        "TORCH_EXTENSIONS_DIR": "/persistent_cache/torch_extensions",
        "CUDA_CACHE_PATH": "/persistent_cache/nvidia_compute",
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_USE_V2_MODEL_RUNNER": DIFFUSIONGEMMA_VLLM_USE_V2_MODEL_RUNNER,
        "VLLM_SERVER_DEV_MODE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def _configure_diffusiongemma_vllm_runtime_env():
    _apply_env_defaults(_diffusiongemma_vllm_runtime_env_defaults())


def _diffusiongemma_vllm_snapshot_command() -> list[str]:
    chat_template_kwargs = json.dumps({
        "enable_thinking": _truthy(DIFFUSIONGEMMA_VLLM_ENABLE_THINKING),
    })
    cmd = [
        f"{DIFFUSIONGEMMA_VLLM_VENV}/bin/vllm",
        "serve",
        DIFFUSIONGEMMA_VLLM_MODEL_ID,
        "--served-model-name",
        DIFFUSIONGEMMA_VLLM_SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(DIFFUSIONGEMMA_VLLM_PORT),
        "--trust-remote-code",
        "--max-model-len",
        DIFFUSIONGEMMA_VLLM_MAX_MODEL_LEN,
        "--max-num-seqs",
        DIFFUSIONGEMMA_VLLM_MAX_NUM_SEQS,
        "--gpu-memory-utilization",
        DIFFUSIONGEMMA_VLLM_GPU_MEMORY_UTILIZATION,
        "--attention-backend",
        DIFFUSIONGEMMA_VLLM_ATTENTION_BACKEND,
        "--generation-config",
        "vllm",
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "gemma4",
        "--enable-sleep-mode",
        "--override-generation-config",
        '{"max_new_tokens": null}',
        "--default-chat-template-kwargs",
        chat_template_kwargs,
    ]
    if DIFFUSIONGEMMA_VLLM_MOE_BACKEND and DIFFUSIONGEMMA_VLLM_MOE_BACKEND.lower() != "auto":
        cmd.extend(["--moe-backend", DIFFUSIONGEMMA_VLLM_MOE_BACKEND])
    if DIFFUSIONGEMMA_VLLM_LOAD_FORMAT:
        cmd.extend(["--load-format", DIFFUSIONGEMMA_VLLM_LOAD_FORMAT])
    if DIFFUSIONGEMMA_VLLM_SAFETENSORS_LOAD_STRATEGY:
        cmd.extend(["--safetensors-load-strategy", DIFFUSIONGEMMA_VLLM_SAFETENSORS_LOAD_STRATEGY])
    if _truthy(DIFFUSIONGEMMA_VLLM_ENABLE_REASONING_PARSER):
        cmd.extend(["--reasoning-parser", "gemma4"])
    return cmd


def _check_vllm_process(process: subprocess.Popen):
    if process.poll() is not None:
        raise RuntimeError(f"DiffusionGemma vLLM exited unexpectedly with code {process.returncode}.")


def _diffusiongemma_vllm_url(path: str) -> str:
    return f"http://127.0.0.1:{DIFFUSIONGEMMA_VLLM_PORT}{path}"


def _diffusiongemma_vllm_ready_once() -> bool:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(_diffusiongemma_vllm_url(path), timeout=5) as response:
                if 200 <= response.status < 300:
                    return True
        except Exception:
            pass
    return False


def _diffusiongemma_vllm_is_sleeping() -> bool:
    try:
        with urllib.request.urlopen(_diffusiongemma_vllm_url("/is_sleeping"), timeout=5) as response:
            if not 200 <= response.status < 300:
                return False
            body = response.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return False

    try:
        data = json.loads(body)
        if isinstance(data, dict):
            return bool(data.get("is_sleeping", data.get("sleeping", False)))
        if isinstance(data, bool):
            return data
    except Exception:
        pass
    body_lc = body.lower()
    return "true" in body_lc and "false" not in body_lc


def _wait_diffusiongemma_vllm_ready(process: subprocess.Popen, timeout: int = 15 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _check_vllm_process(process)
        if _diffusiongemma_vllm_ready_once():
            return
        time.sleep(2)

    _check_vllm_process(process)
    raise TimeoutError(f"DiffusionGemma vLLM was not ready within {timeout} seconds.")


def _warmup_diffusiongemma_vllm():
    req = urllib.request.Request(
        _diffusiongemma_vllm_url("/v1/chat/completions"),
        data=json.dumps({
            "model": DIFFUSIONGEMMA_VLLM_SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply with ok."}],
            "max_tokens": 1,
            "temperature": 0,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        print("DiffusionGemma warmup response status:", response.status)


def _post_diffusiongemma_vllm(path: str, label: str, timeout: int = 300):
    req = urllib.request.Request(
        _diffusiongemma_vllm_url(path),
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"DiffusionGemma vLLM {label} returned HTTP {response.status}.")
        print(f"DiffusionGemma vLLM {label} response status:", response.status)


def _sleep_diffusiongemma_vllm():
    _post_diffusiongemma_vllm("/sleep?level=1", "sleep", timeout=300)


def _wake_diffusiongemma_vllm(process: subprocess.Popen | None = None, timeout: int = 600):
    deadline = time.time() + timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        if process is not None:
            _check_vllm_process(process)
        if _diffusiongemma_vllm_ready_once() and not _diffusiongemma_vllm_is_sleeping():
            print("DiffusionGemma vLLM is already awake.")
            return

        remaining = max(1, int(deadline - time.time()))
        try:
            _post_diffusiongemma_vllm("/wake_up", "wake", timeout=remaining)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)

    if last_error is not None:
        raise TimeoutError(f"Timed out waking DiffusionGemma vLLM: {last_error}") from last_error
    raise TimeoutError("Timed out waking DiffusionGemma vLLM.")


def setup_pi_config():
    """Write Pi's local OpenAI-compatible model config for DiffusionGemma vLLM."""
    config_dir = os.environ.get("PI_CONFIG_DIR", PI_CONFIG_DIR).strip() or PI_CONFIG_DIR
    model_name = (
        os.environ.get("PI_MODEL")
        or os.environ.get("SERVED_MODEL_NAME")
        or os.environ.get("DIFFUSIONGEMMA_VLLM_MODEL")
        or DIFFUSIONGEMMA_VLLM_SERVED_MODEL_NAME
    )
    port = (
        os.environ.get("VLLM_PORT")
        or os.environ.get("PORT")
        or os.environ.get("DIFFUSIONGEMMA_VLLM_PORT")
        or str(DIFFUSIONGEMMA_VLLM_PORT)
    )
    max_model_len = (
        os.environ.get("MAX_MODEL_LEN")
        or os.environ.get("DIFFUSIONGEMMA_VLLM_MAX_MODEL_LEN")
        or DIFFUSIONGEMMA_VLLM_MAX_MODEL_LEN
    )
    try:
        context_window = int(max_model_len)
    except (TypeError, ValueError):
        context_window = 65536

    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "models.json")
    config = {
        "providers": {
            "vllm": {
                "baseUrl": f"http://127.0.0.1:{port}/v1",
                "api": "openai-completions",
                "apiKey": "vllm-local",
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                },
                "models": [
                    {
                        "id": model_name,
                        "name": model_name,
                        "reasoning": True,
                        "input": ["text", "image"],
                        "contextWindow": context_window,
                    }
                ],
            }
        }
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Pi model config written to {config_file} -> {os.path.realpath(config_file)}")


def setup_persistent_cache():
    """Create symlinks mapping container paths to persistent cache."""
    for cache_dir in PERSISTENT_CACHE_DIRS:
        os.makedirs(cache_dir, exist_ok=True)

    for local_path, persistent_path in PERSISTENT_DIR_LINKS.items():
        _link_persistent_path(local_path, persistent_path, is_dir=True)

    for local_path, persistent_path in PERSISTENT_FILE_LINKS.items():
        _link_persistent_path(local_path, persistent_path, is_dir=False)

    setup_pi_config()


def _link_persistent_path(local_path: str, persistent_path: str, is_dir: bool):
    """Replace a container path with a symlink into a Modal volume."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    os.makedirs(os.path.dirname(persistent_path), exist_ok=True)
    if is_dir:
        os.makedirs(persistent_path, exist_ok=True)
    copy_existing_file = (
        not is_dir
        and os.path.isfile(local_path)
        and not os.path.exists(persistent_path)
    )
    if os.path.lexists(local_path):
        if os.path.islink(local_path):
            os.unlink(local_path)
        elif os.path.isdir(local_path):
            shutil.rmtree(local_path)
        else:
            if copy_existing_file:
                shutil.copy2(local_path, persistent_path)
            os.unlink(local_path)
    if not is_dir and not os.path.exists(persistent_path):
        with open(persistent_path, "a", encoding="utf-8"):
            pass
    os.symlink(persistent_path, local_path)


def _hf_download(repo_id: str, *args: str):
    subprocess.run(["hf", "download", repo_id, *args], check=True)


def _hf_download_to_cache(repo_id: str, *include_patterns: str):
    command = ["--cache-dir", "/persistent_cache/huggingface/hub"]
    if include_patterns:
        command.extend(["--include", *include_patterns])
    _hf_download(repo_id, *command)


def _hf_download_local(repo_id: str, local_dir: str, *args: str):
    _hf_download(repo_id, *args, "--local-dir", local_dir)


def _import_webui_module():
    if "/root" not in sys.path:
        sys.path.append("/root")
    import image_studio_webui

    return image_studio_webui


def _download_pid_models():
    print("Downloading PiD models...")
    _hf_download_local(
        "nvidia/PiD",
        "/root/PiD",
        "--include",
        "checkpoints/ae.safetensors",
        "checkpoints/QwenImage_VAE_2d.pth",
        "checkpoints/flux2_ae.safetensors",
        "checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        "checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        "checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth",
        "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/*",
        "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/*",
    )


def _download_ideogram_models():
    print("Downloading Ideogram 4 NVFP4 weights into persistent Hugging Face cache...")
    ideogram_weights_repo = os.environ.get("IDEOGRAM4_NVFP4_WEIGHTS_REPO", "Comfy-Org/Ideogram-4")
    _hf_download_to_cache(
        ideogram_weights_repo,
        "diffusion_models/ideogram4_nvfp4_mixed.safetensors.index.json",
        "diffusion_models/ideogram4_nvfp4_mixed*.safetensors",
        "diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors.index.json",
        "diffusion_models/ideogram4_unconditional_nvfp4_mixed*.safetensors",
        "text_encoders/qwen3vl_8b_nvfp4.safetensors",
        "vae/flux2-vae.safetensors",
    )

    print("Downloading Ideogram 4 tokenizer/config files into persistent Hugging Face cache...")
    ideogram_config_repo = os.environ.get("IDEOGRAM4_NVFP4_CONFIG_REPO", "Qwen/Qwen3-VL-8B-Instruct")
    _hf_download_to_cache(
        ideogram_config_repo,
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    )


def _download_boogu_models():
    print("Downloading Boogu-Image models into persistent /root/models cache...")
    for model_name, repo_id in BOOGU_IMAGE_MODEL_REPOS.items():
        _hf_download_local(repo_id, os.path.join(BOOGU_IMAGE_MODELS_DIR, model_name))


def _download_gemma_models():
    print("Downloading Gemma 4 prompt/chat models into persistent Hugging Face cache...")
    _hf_download_to_cache("google/gemma-4-12B-it")
    gemma_assistant_model = os.environ.get("GEMMA_ASSISTANT_MODEL", "").strip()
    if gemma_assistant_model:
        _hf_download_to_cache(gemma_assistant_model)


def _download_diffusiongemma_vllm_model():
    print("Downloading DiffusionGemma vLLM model into persistent Hugging Face cache...")
    _hf_download_to_cache(DIFFUSIONGEMMA_VLLM_MODEL_ID)


def _download_ltx_models():
    print("Downloading LTX-Web models explicitly...")
    # Bypass run.sh and download models directly to save time and prevent accidental dependency reinstallations.
    _hf_download_local("Lightricks/gemma-3-12b-it-qat-q4_0-unquantized", "/root/ltx-web/models/gemma")
    _hf_download_local("Lightricks/LTX-2.3", "/root/ltx-web/models/checkpoints", "ltx-2.3-22b-distilled-1.1.safetensors")
    _hf_download_local("Lightricks/LTX-2.3", "/root/ltx-web/models/upsamplers", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")


@app.function(
    image=image,
    timeout=4 * 3600,
    volumes={
        PERSISTENT_CACHE_DIR: cache_storage,
        PERSISTENT_APP_DIR: app_storage
    },
    cpu=4.0,
    memory=8192,
    secrets=[modal.Secret.from_name("custom-secret")]
)
def prepare_models():
    """
    Downloads models into the persistent cache volume.
    Run this once before starting the web UI using:
    `modal run modal_deploy.py::prepare_models`
    """
    print("Setting up persistent cache...")
    setup_persistent_cache()

    _download_pid_models()
    _download_boogu_models()
    _download_ideogram_models()
    _download_gemma_models()
    _download_diffusiongemma_vllm_model()
    _download_ltx_models()

    cache_storage.commit()
    app_storage.commit()
    print("Models prepared successfully!")


# ---------------------------------------------------------------------------
# Shared helpers for Modal entry points (used by both WebUI classes)
# ---------------------------------------------------------------------------

def _modal_runtime_env_defaults() -> dict[str, str]:
    return {
        "REMOVE_AI_WATERMARKS_DIR": "/root/remove-ai-watermarks",
        "PID_DIR": "/root/PiD",
        "IDEOGRAM4_DIR": "/root/ideogram4",
        "BOOGU_IMAGE_DIR": BOOGU_IMAGE_DIR,
        "BOOGU_IMAGE_DEVICE": "cuda:0",
        "BOOGU_IMAGE_TURBO_MODEL": os.path.join(BOOGU_IMAGE_MODELS_DIR, "Boogu-Image-0.1-Turbo"),
        "BOOGU_IMAGE_BASE_MODEL": os.path.join(BOOGU_IMAGE_MODELS_DIR, "Boogu-Image-0.1-Base"),
        "BOOGU_IMAGE_EDIT_MODEL": os.path.join(BOOGU_IMAGE_MODELS_DIR, "Boogu-Image-0.1-Edit"),
        "BOOGU_IMAGE_EDIT_TURBO_MODEL": os.path.join(
            BOOGU_IMAGE_MODELS_DIR, "Boogu-Image-0.1-Edit-Turbo"
        ),
        # Krea2 ComfyUI backend env
        "KREA2_COMFY_SCRIPT": KREA2_COMFY_SCRIPT,
        "KREA2_COMFY_BASH": "bash",
        "KREA2_COMFY_PORT": KREA2_COMFY_PORT,
        "KREA2_COMFY_DIR": KREA2_COMFY_DIR,
        "KREA2_COMFY_VENV": KREA2_COMFY_VENV,
        "KREA2_COMFY_READY_TIMEOUT": KREA2_COMFY_READY_TIMEOUT,
        "KREA2_COMFY_START_TIMEOUT": KREA2_COMFY_START_TIMEOUT,
        "KREA2_COMFY_REQUEST_TIMEOUT": KREA2_COMFY_REQUEST_TIMEOUT,
        "KREA2_COMFY_SERVER_BASE": f"http://127.0.0.1:{KREA2_COMFY_PORT}",
        "GEMMA_ASSISTANT_MODEL": "google/gemma-4-12B-it-assistant",
        "IDEOGRAM_API_KEY": os.environ.get("IDEOGRAM_API_KEY", ""),
        "PI_CONFIG_DIR": PI_CONFIG_DIR,
        "PI_MODEL": (
            os.environ.get("SERVED_MODEL_NAME")
            or os.environ.get("DIFFUSIONGEMMA_VLLM_MODEL")
            or PI_MODEL
        ),
        "PI_NPM_PACKAGE": PI_NPM_PACKAGE,
        "PI_NPM_VERSION": PI_NPM_VERSION,
        "VLLM_PORT": str(DIFFUSIONGEMMA_VLLM_PORT),
        "DIFFUSIONGEMMA_VLLM_SCRIPT": DIFFUSIONGEMMA_VLLM_SCRIPT,
        "DIFFUSIONGEMMA_VLLM_BASH": "bash",
        "DIFFUSIONGEMMA_VLLM_VENV": DIFFUSIONGEMMA_VLLM_VENV,
        "DIFFUSIONGEMMA_VLLM_LOG_FILE": DIFFUSIONGEMMA_VLLM_LOG,
        "DIFFUSIONGEMMA_VLLM_PID_FILE": DIFFUSIONGEMMA_VLLM_PID_FILE,
        "DIFFUSIONGEMMA_VLLM_PORT": str(DIFFUSIONGEMMA_VLLM_PORT),
        "DIFFUSIONGEMMA_VLLM_API_BASE": DIFFUSIONGEMMA_VLLM_API_BASE,
        "DIFFUSIONGEMMA_VLLM_MODEL": DIFFUSIONGEMMA_VLLM_SERVED_MODEL_NAME,
        "DIFFUSIONGEMMA_VLLM_HF_MODEL": DIFFUSIONGEMMA_VLLM_MODEL_ID,
        "DIFFUSIONGEMMA_VLLM_START_TIMEOUT": "900",
        "DIFFUSIONGEMMA_VLLM_READY_TIMEOUT": "900",
        "MODEL": DIFFUSIONGEMMA_VLLM_MODEL_ID,
        "SERVED_MODEL_NAME": DIFFUSIONGEMMA_VLLM_SERVED_MODEL_NAME,
        "PORT": str(DIFFUSIONGEMMA_VLLM_PORT),
        "MAX_MODEL_LEN": DIFFUSIONGEMMA_VLLM_MAX_MODEL_LEN,
        "MAX_NUM_SEQS": DIFFUSIONGEMMA_VLLM_MAX_NUM_SEQS,
        "GPU_MEMORY_UTILIZATION": DIFFUSIONGEMMA_VLLM_GPU_MEMORY_UTILIZATION,
        "ATTN_BACKEND": DIFFUSIONGEMMA_VLLM_ATTENTION_BACKEND,
        "MOE_BACKEND": DIFFUSIONGEMMA_VLLM_MOE_BACKEND,
        "ENABLE_THINKING": DIFFUSIONGEMMA_VLLM_ENABLE_THINKING,
        "ENABLE_REASONING_PARSER": DIFFUSIONGEMMA_VLLM_ENABLE_REASONING_PARSER,
        "USE_V2_MODEL_RUNNER": DIFFUSIONGEMMA_VLLM_USE_V2_MODEL_RUNNER,
        "DIFFUSIONGEMMA_VLLM_ENABLE_REASONING_PARSER": DIFFUSIONGEMMA_VLLM_ENABLE_REASONING_PARSER,
        "DIFFUSIONGEMMA_VLLM_USE_V2_MODEL_RUNNER": DIFFUSIONGEMMA_VLLM_USE_V2_MODEL_RUNNER,
        "DIFFUSIONGEMMA_VLLM_MOE_BACKEND": DIFFUSIONGEMMA_VLLM_MOE_BACKEND,
        "DIFFUSIONGEMMA_VLLM_LOAD_FORMAT": DIFFUSIONGEMMA_VLLM_LOAD_FORMAT,
        "DIFFUSIONGEMMA_VLLM_SAFETENSORS_LOAD_STRATEGY": DIFFUSIONGEMMA_VLLM_SAFETENSORS_LOAD_STRATEGY,
        "DIFFUSIONGEMMA_VLLM_UNLOAD_MODE": os.environ.get("DIFFUSIONGEMMA_VLLM_UNLOAD_MODE", "sleep"),
        "DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL": os.environ.get("DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL", "1"),
        "DIFFUSIONGEMMA_VLLM_WHEEL_INDEX_URL": DIFFUSIONGEMMA_VLLM_WHEEL_INDEX_URL,
        "DIFFUSIONGEMMA_VLLM_WHEEL_URL": DIFFUSIONGEMMA_VLLM_WHEEL_URL,
        "LOAD_FORMAT": DIFFUSIONGEMMA_VLLM_LOAD_FORMAT,
        "SAFETENSORS_LOAD_STRATEGY": DIFFUSIONGEMMA_VLLM_SAFETENSORS_LOAD_STRATEGY,
        "HF_CACHE": "/persistent_cache/huggingface",
        "TORCH_CACHE": "/persistent_cache/torch",
        "VLLM_CACHE_ROOT": "/persistent_cache/vllm",
        "XDG_CACHE_HOME": "/persistent_cache/xdg",
        "TRITON_CACHE_DIR": "/persistent_cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/persistent_cache/torchinductor",
        "TORCH_EXTENSIONS_DIR": "/persistent_cache/torch_extensions",
        "CUDA_CACHE_PATH": "/persistent_cache/nvidia_compute",
        "HF_HOME": "/persistent_cache/huggingface",
        "HF_HUB_CACHE": "/persistent_cache/huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "/persistent_cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/persistent_cache/huggingface/hub",
        "TORCH_HOME": "/persistent_cache/torch",
        # uv needs flock for lock files; Modal volumes don't support it.
        # Point uv's cache to the local container filesystem.
        "UV_CACHE_DIR": "/tmp/uv-cache",
    }


def _configure_modal_runtime_env():
    """Set all environment variables the webui module needs to find backends."""
    _apply_env_defaults(_modal_runtime_env_defaults())


def _patch_modal_vllm_service(webui):
    """Wrap the webui's vLLM service so ensure_running/stop commit Modal volumes."""
    if getattr(webui, "_modal_diffusiongemma_vllm_patch", False):
        return

    service = getattr(webui, "_diffusiongemma_vllm_service", None)
    if service is None:
        return

    original_ensure_running = service.ensure_running
    original_stop = service.stop

    def ensure_running_with_persistent_cache():
        try:
            original_ensure_running()
        finally:
            cache_storage.commit()
            app_storage.commit()

    def stop_with_persistent_cache():
        try:
            original_stop()
        finally:
            cache_storage.commit()
            app_storage.commit()

    service.ensure_running = ensure_running_with_persistent_cache
    service.stop = stop_with_persistent_cache
    webui._modal_diffusiongemma_vllm_patch = True


def _patch_modal_krea2_service(webui):
    """Wrap the webui's Krea2 ComfyUI service so ensure_running/stop commit Modal volumes."""
    if getattr(webui, "_modal_krea2_comfy_patch", False):
        return

    service = getattr(webui, "_krea2_comfy_service", None)
    if service is None:
        return

    original_ensure_running = service.ensure_running
    original_stop = service.stop

    def ensure_running_with_persistent_cache():
        try:
            original_ensure_running()
        finally:
            cache_storage.commit()
            app_storage.commit()

    def stop_with_persistent_cache():
        try:
            original_stop()
        finally:
            cache_storage.commit()
            app_storage.commit()

    service.ensure_running = ensure_running_with_persistent_cache
    service.stop = stop_with_persistent_cache
    webui._modal_krea2_comfy_patch = True


def _load_webui_module():
    """Configure env, set up persistent cache, import webui, and patch vLLM service.

    Returns the imported image_studio_webui module.
    """
    _configure_modal_runtime_env()
    setup_persistent_cache()
    app_storage.commit()
    webui = _import_webui_module()
    _patch_modal_vllm_service(webui)
    _patch_modal_krea2_service(webui)
    return webui


def _register_vllm_as_loaded(webui):
    """Tell the webui model manager that vLLM is already running."""
    service = getattr(webui, "_diffusiongemma_vllm_service", None)
    if service is None:
        print("WARNING: webui._diffusiongemma_vllm_service not found; cannot register vLLM as loaded.")
        return
    if service.is_healthy():
        service._register()
        print("DiffusionGemma vLLM registered as loaded in model manager.")
    else:
        print("WARNING: vLLM health check failed; model not registered as loaded.")


def _prefer_local_gemma_upsampler(webui):
    """Default Ideogram prompt upsampling to local Gemma without hiding the API option."""
    local_gemma = getattr(webui, "IDEOGRAM4_UPSAMPLE_GEMMA", None)
    if not local_gemma:
        print("WARNING: webui.IDEOGRAM4_UPSAMPLE_GEMMA not found; cannot set fast upsampler default.")
        return

    webui._ideogram4_default_upsampler = lambda: local_gemma
    print("Fast WebUI Ideogram prompt upsampler default set to local Gemma.")


def _build_modal_fastapi_app(webui):
    """Build the Gradio demo and mount it on a FastAPI app."""
    import gradio as gr
    from fastapi import FastAPI

    demo = webui.build_ui()
    demo.queue()
    fastapi_app = FastAPI()
    attach_proxy = getattr(webui, "attach_vllm_proxy_routes", None)
    attach_designer = getattr(webui, "attach_ideogram_json_designer_route", None)
    proxy_api_key = os.environ.get("IMAGE_STUDIO_VLLM_PROXY_API_KEY", "").strip()
    if callable(attach_proxy):
        # Modal mounts Gradio at "/" below, so parent-level routes must be
        # registered first or the Gradio mount can consume /v1/* requests.
        attach_proxy(type("ModalRouteHost", (), {"app": fastapi_app})(), api_key=proxy_api_key)
    if callable(attach_designer):
        attach_designer(type("ModalRouteHost", (), {"app": fastapi_app})())
    mounted_app = gr.mount_gradio_app(fastapi_app, demo, path="/")
    if callable(attach_proxy):
        attach_proxy(demo, api_key=proxy_api_key)
    if callable(attach_designer):
        attach_designer(demo)
    return mounted_app


# ---------------------------------------------------------------------------
# Entry point 1: Standard WebUI (vLLM started on-demand via script)
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=3600,
    volumes={
        PERSISTENT_CACHE_DIR: cache_storage,
        PERSISTENT_APP_DIR: app_storage
    },
    max_containers=1,  # Gradio requires sticky sessions to avoid state loss
    scaledown_window=5 * 60,
    secrets=[modal.Secret.from_name("custom-secret")]
)
@modal.concurrent(max_inputs=100)  # REQUIRED for Gradio's concurrent websockets/heartbeats
class ImageStudioWebUI:
    @modal.asgi_app(label="web-ui")
    def web_ui(self):
        if not hasattr(self, "fastapi_app"):
            webui = _load_webui_module()
            self.fastapi_app = _build_modal_fastapi_app(webui)
        return self.fastapi_app


# ---------------------------------------------------------------------------
# Entry point 2: Fast WebUI (vLLM pre-loaded via GPU snapshot)
# ---------------------------------------------------------------------------

@app.cls(
    gpu="RTX-PRO-6000",
    image=image,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=5 * 60,
    volumes={
        PERSISTENT_CACHE_DIR: cache_storage,
        PERSISTENT_APP_DIR: app_storage,
    },
    timeout=3600,
    max_containers=1,
    secrets=[modal.Secret.from_name("custom-secret")]
)
@modal.concurrent(max_inputs=100)
class ImageStudioWebUIFast:
    """Image Studio WebUI with DiffusionGemma vLLM pre-loaded via GPU snapshot.

    The snapshot phase starts vLLM, warms it up, and captures the GPU state.
    On restore the web UI is built and the already-running vLLM is registered
    as loaded in the model manager so it appears in the Models tab immediately.
    """

    @modal.enter(snap=True)
    def warmup(self):
        print("Starting DiffusionGemma vLLM snapshot warmup...")
        _ensure_diffusiongemma_vllm_cache_dirs()
        _configure_diffusiongemma_vllm_runtime_env()
        self.server_process = subprocess.Popen(_diffusiongemma_vllm_snapshot_command())

        print(f"Waiting for DiffusionGemma vLLM on port {DIFFUSIONGEMMA_VLLM_PORT}...")
        _wait_diffusiongemma_vllm_ready(self.server_process)

        print("DiffusionGemma vLLM is ready. Sending snapshot warmup request...")
        try:
            _warmup_diffusiongemma_vllm()
        except Exception as exc:
            print("DiffusionGemma warmup request failed; continuing with snapshot:", exc)

        cache_storage.commit()
        time.sleep(5)
        print("Putting DiffusionGemma vLLM to sleep before snapshot to discard the empty KV cache...")
        _sleep_diffusiongemma_vllm()
        print("DiffusionGemma vLLM warmup complete. Capturing sleeping CPU and GPU state.")

    @modal.enter(snap=False)
    def restore(self):
        print("DiffusionGemma vLLM restored from snapshot. Waking before readiness checks...")
        _wake_diffusiongemma_vllm(self.server_process)
        print("Checking DiffusionGemma vLLM health after wake...")
        _wait_diffusiongemma_vllm_ready(self.server_process)
        print("DiffusionGemma vLLM is ready to serve traffic.")

    @modal.asgi_app(label="snapshot-web-ui")
    def serve(self):
        if not hasattr(self, "fastapi_app"):
            # Set the default chat model to DiffusionGemma vLLM before the webui
            # module is imported (it reads this env var at module level).
            os.environ["IMAGE_STUDIO_CHAT_DEFAULT"] = "diffusiongemma_vllm"
            webui = _load_webui_module()
            _register_vllm_as_loaded(webui)
            _prefer_local_gemma_upsampler(webui)
            self.fastapi_app = _build_modal_fastapi_app(webui)
        return self.fastapi_app

    @modal.exit()
    def stop(self):
        if hasattr(self, "server_process") and self.server_process.poll() is None:
            self.server_process.terminate()
