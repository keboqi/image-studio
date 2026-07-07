#!/usr/bin/env bash
set -Eeuo pipefail

# deploy_krea2_comfy.sh
# One-path, long-lived Krea-2-Turbo ComfyUI service.
# Lightning Studio friendly: prefers `uv venv` instead of `python -m venv`.
#
# Primary path:
#   bash deploy_krea2_comfy.sh start
#
# Fresh reset:
#   KREA2_RECREATE_VENV=1 bash deploy_krea2_comfy.sh start
#
# Generate:
#   WIDTH=2048 HEIGHT=2048 bash deploy_krea2_comfy.sh generate "your prompt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND_DIR="${KREA2_COMFY_DIR:-${SCRIPT_DIR}/krea2-comfy}"
COMFY_DIR="${KREA2_COMFY_REPO_DIR:-${BACKEND_DIR}/ComfyUI}"
VENV_DIR="${KREA2_COMFY_VENV:-${BACKEND_DIR}/.venv}"
PYTHON_CMD="${KREA2_COMFY_PYTHON:-python3}"

HOST="${HOST:-${KREA2_COMFY_HOST:-0.0.0.0}}"
PORT="${PORT:-${KREA2_COMFY_PORT:-8188}}"
GPU="${GPU:-${KREA2_COMFY_GPU:-0}}"

# Set INSTALL_TORCH=0 to keep the Studio's existing torch.
# Default auto: install cu130 nightly only if torch is missing or not CUDA 13.x.
INSTALL_TORCH="${INSTALL_TORCH:-auto}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu130}"

# Keep models in the same long-lived ComfyUI process.
# If 2048x2048 OOMs, restart with:
#   COMFY_ARGS="" bash deploy_krea2_comfy.sh restart
COMFY_ARGS="${COMFY_ARGS:---gpu-only --use-pytorch-cross-attention}"

READY_TIMEOUT="${READY_TIMEOUT:-${KREA2_COMFY_READY_TIMEOUT:-300}}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-${KREA2_COMFY_REQUEST_TIMEOUT:-900}}"
LOG_LINES="${LOG_LINES:-${KREA2_COMFY_LOG_LINES:-220}}"
STOP_TIMEOUT="${STOP_TIMEOUT:-${KREA2_COMFY_STOP_TIMEOUT:-20}}"

CACHE_DIR="${KREA2_COMFY_CACHE_DIR:-${BACKEND_DIR}/cache}"
HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_DIR}/xdg}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_DIR}/triton}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_DIR}/torchinductor}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_DIR}/torch_extensions}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${CACHE_DIR}/nvidia_compute}"

PID_FILE="${BACKEND_DIR}/comfyui.pid"
LOG_FILE="${BACKEND_DIR}/comfyui.log"
INSTALL_STAMP="${BACKEND_DIR}/.installed_krea2_comfy"
SERVER_BASE="http://127.0.0.1:${PORT}"

REPO_ID="Comfy-Org/Krea-2"
DIFFUSION_FILE="diffusion_models/krea2_turbo_nvfp4.safetensors"
TEXT_ENCODER_FILE="text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
VAE_FILE="vae/qwen_image_vae.safetensors"

ACTION="${1:-start}"
KREA2_MODEL_MODE="${KREA2_MODEL_MODE:-${KREA2_DOWNLOAD_MODE:-on-demand}}"

if [[ "${OS:-}" == "Windows_NT" ]]; then
  VENV_PYTHON="${VENV_DIR}/Scripts/python.exe"
else
  VENV_PYTHON="${VENV_DIR}/bin/python"
fi

die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

mkdirs() {
  mkdir -p \
    "${BACKEND_DIR}" \
    "${HF_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${CUDA_CACHE_PATH}"
}

export_runtime_env() {
  export HF_HOME XDG_CACHE_HOME
  export TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR CUDA_CACHE_PATH
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  [[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" || true
}

run_pip() {
  if command -v uv >/dev/null 2>&1; then
    uv pip "$@"
  else
    local py_arg=""
    local args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --python) py_arg="$2"; shift 2 ;;
        *) args+=("$1"); shift ;;
      esac
    done
    [[ -n "${py_arg}" ]] || py_arg="${VENV_PYTHON}"
    "${py_arg}" -m pip "${args[@]}"
  fi
}

create_or_recreate_venv() {
  mkdirs

  if [[ "${KREA2_RECREATE_VENV:-0}" == "1" || "${KREA2_RECREATE_VENV:-}" == "true" ]]; then
    echo "Recreating venv: ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
    rm -f "${INSTALL_STAMP}"
  fi

  if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Creating venv: ${VENV_DIR}"
    if command -v uv >/dev/null 2>&1; then
      uv venv "${VENV_DIR}" --python "${PYTHON_CMD}"
    else
      echo "WARNING: uv not found. Falling back to python -m venv, which Lightning Studio may block."
      need_cmd "${PYTHON_CMD}"
      "${PYTHON_CMD}" -m venv "${VENV_DIR}"
    fi
  fi
}

torch_is_cu13() {
  [[ -x "${VENV_PYTHON}" ]] || return 1
  "${VENV_PYTHON}" - <<'PY'
try:
    import torch
    ok = torch.cuda.is_available() and torch.version.cuda and torch.version.cuda.startswith("13")
    raise SystemExit(0 if ok else 1)
except Exception:
    raise SystemExit(1)
PY
}

install_torch_if_needed() {
  if [[ "${INSTALL_TORCH}" == "0" || "${INSTALL_TORCH}" == "false" ]]; then
    echo "INSTALL_TORCH=0, skipping torch install."
    return 0
  fi

  if torch_is_cu13; then
    echo "Torch already has CUDA 13.x support. Skipping torch reinstall."
    return 0
  fi

  echo "Installing CUDA 13 nightly PyTorch for NVFP4 acceleration..."
  run_pip install --python "${VENV_PYTHON}" --pre torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
}

clone_or_update_comfy() {
  need_cmd git

  if [[ ! -d "${COMFY_DIR}/.git" ]]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git "${COMFY_DIR}"
  else
    git -C "${COMFY_DIR}" pull --ff-only || true
  fi
}

install_backend_if_needed() {
  need_cmd curl
  create_or_recreate_venv
  export_runtime_env
  clone_or_update_comfy

  if [[ -f "${INSTALL_STAMP}" && "${KREA2_FORCE_INSTALL:-0}" != "1" ]]; then
    echo "Install stamp found: ${INSTALL_STAMP}"
    maybe_prefetch_models
    return 0
  fi

  echo
  echo "Installing Krea-2 ComfyUI environment"
  echo "ComfyUI: ${COMFY_DIR}"
  echo "Venv:    ${VENV_DIR}"
  echo "Python:  ${VENV_PYTHON}"
  echo

  run_pip install --python "${VENV_PYTHON}" --upgrade pip setuptools wheel
  install_torch_if_needed
  run_pip install --python "${VENV_PYTHON}" -r "${COMFY_DIR}/requirements.txt"
  run_pip install --python "${VENV_PYTHON}" --upgrade "huggingface_hub[hf_xet]" requests websocket-client

  maybe_prefetch_models
  doctor

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${INSTALL_STAMP}"
}

should_prefetch_models() {
  case "${KREA2_MODEL_MODE}" in
    with-models|prefetch|prefetch-models|models|full|1|true|yes)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

maybe_prefetch_models() {
  if should_prefetch_models; then
    download_models
  else
    echo "Skipping Krea-2 model downloads during setup (KREA2_MODEL_MODE=${KREA2_MODEL_MODE})."
    echo "They will be downloaded on first start/generation if missing."
  fi
}

ensure_runtime_models() {
  if [[ "${KREA2_SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
    echo "Skipping Krea-2 runtime model check because KREA2_SKIP_MODEL_DOWNLOAD=1."
    return 0
  fi
  download_models
}

download_models() {
  [[ -x "${VENV_PYTHON}" ]] || die "Missing venv python: ${VENV_PYTHON}"

  mkdir -p \
    "${COMFY_DIR}/models/diffusion_models" \
    "${COMFY_DIR}/models/text_encoders" \
    "${COMFY_DIR}/models/vae"

  "${VENV_PYTHON}" - <<PY
from huggingface_hub import hf_hub_download

repo_id = "${REPO_ID}"
local_dir = "${COMFY_DIR}/models"
files = [
    "${DIFFUSION_FILE}",
    "${TEXT_ENCODER_FILE}",
    "${VAE_FILE}",
]

for filename in files:
    print(f"Downloading/checking {repo_id}/{filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        resume_download=True,
    )
    print(" ->", path)
PY
}

doctor() {
  [[ -x "${VENV_PYTHON}" ]] || die "Missing venv python: ${VENV_PYTHON}"

  (
    cd "${COMFY_DIR}"
    "${VENV_PYTHON}" - <<'PY'
import importlib.metadata as md
import os
import sys

print("python:", sys.executable)
print("cwd:", os.getcwd())

for pkg in ["torch", "torchvision", "torchaudio", "triton", "comfyui-frontend-package", "huggingface_hub"]:
    try:
        print(f"{pkg}:", md.version(pkg))
    except Exception:
        print(f"{pkg}: NOT INSTALLED")

import torch
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

# Light Comfy import check. Do not fully launch the server here.
import folder_paths
print("Comfy folder_paths import: OK")
PY
  )
}

pid_running() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" >/dev/null 2>&1
}

is_ready() {
  curl -fsS --max-time 5 "${SERVER_BASE}/system_stats" >/dev/null 2>&1
}

wait_ready() {
  local start deadline
  start="${SECONDS}"
  deadline=$((SECONDS + READY_TIMEOUT))

  echo "Waiting up to ${READY_TIMEOUT}s for ComfyUI readiness..."
  while (( SECONDS < deadline )); do
    if is_ready; then
      echo "ComfyUI ready after $((SECONDS - start))s."
      return 0
    fi

    if [[ -f "${PID_FILE}" ]] && ! pid_running; then
      echo "ComfyUI exited before readiness. Recent logs:" >&2
      tail -n "${LOG_LINES}" "${LOG_FILE}" >&2 || true
      return 1
    fi

    sleep 2
  done

  echo "Timed out waiting for ${SERVER_BASE}. Recent logs:" >&2
  tail -n "${LOG_LINES}" "${LOG_FILE}" >&2 || true
  return 1
}

start_service() {
  install_backend_if_needed
  export_runtime_env
  ensure_runtime_models

  if is_ready; then
    echo "Krea-2 ComfyUI service already ready at ${SERVER_BASE}."
    print_summary
    return 0
  fi

  if pid_running; then
    echo "Krea-2 ComfyUI process is already running. Waiting for readiness."
    wait_ready
    print_summary
    return 0
  fi

  export CUDA_VISIBLE_DEVICES="${GPU}"

  echo
  echo "Starting long-lived Krea-2 ComfyUI service"
  echo "Mode: persistent ComfyUI server"
  echo "Host: ${HOST}"
  echo "Port: ${PORT}"
  echo "GPU:  ${GPU}"
  echo "Log:  ${LOG_FILE}"
  echo
  echo "Command:"
  # shellcheck disable=SC2086
  printf '  %q' "${VENV_PYTHON}" main.py --listen "${HOST}" --port "${PORT}" ${COMFY_ARGS}
  echo
  echo

  (
    cd "${COMFY_DIR}"
    # shellcheck disable=SC2086
    exec "${VENV_PYTHON}" main.py --listen "${HOST}" --port "${PORT}" ${COMFY_ARGS}
  ) >"${LOG_FILE}" 2>&1 &

  echo "$!" > "${PID_FILE}"

  wait_ready
  print_summary
}

stop_service() {
  if ! pid_running; then
    echo "Krea-2 ComfyUI service is not running."
    rm -f "${PID_FILE}"
    return 0
  fi

  local pid deadline
  pid="$(cat "${PID_FILE}")"
  echo "Stopping Krea-2 ComfyUI service PID ${pid}."
  kill "${pid}" >/dev/null 2>&1 || true

  deadline=$((SECONDS + STOP_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      rm -f "${PID_FILE}"
      echo "Stopped."
      return 0
    fi
    sleep 1
  done

  echo "Force stopping PID ${pid}."
  kill -9 "${pid}" >/dev/null 2>&1 || true
  rm -f "${PID_FILE}"
}

restart_service() {
  stop_service
  start_service
}

status_service() {
  if pid_running; then
    echo "Process: running ($(cat "${PID_FILE}"))"
  else
    echo "Process: not running"
  fi

  if is_ready; then
    echo "Health: ready"
    curl -fsS --max-time 10 "${SERVER_BASE}/system_stats" || true
    echo
  else
    echo "Health: not ready"
  fi

  print_summary
}

show_logs() {
  mkdirs
  touch "${LOG_FILE}"
  tail -f "${LOG_FILE}"
}

print_summary() {
  echo
  echo "URL:   ${SERVER_BASE}"
  echo "UI:    http://${HOST}:${PORT}"
  echo "Log:   ${LOG_FILE}"
  echo "Out:   ${COMFY_DIR}/output"
  echo
  echo "Commands:"
  echo "  bash $0 status"
  echo "  bash $0 logs"
  echo "  WIDTH=2048 HEIGHT=2048 bash $0 generate \"your prompt\""
  echo "  bash $0 stop"
}

generate_image() {
  local prompt="${1:-A cinematic high-fashion photo, natural light, detailed textures, elegant composition}"
  local width="${WIDTH:-1024}"
  local height="${HEIGHT:-1024}"
  local steps="${STEPS:-8}"
  local cfg="${CFG:-1.0}"
  local sampler="${SAMPLER:-euler}"
  local scheduler="${SCHEDULER:-simple}"
  local denoise="${DENOISE:-1.0}"
  local prefix="${PREFIX:-Krea2_turbo}"
  local seed="${SEED:-$(python3 - <<'PY'
import random
print(random.randint(1, 2**63 - 1))
PY
)}"

  if ! is_ready; then
    echo "ComfyUI is not reachable on ${SERVER_BASE}. Starting it first..."
    start_service
  fi

  "${VENV_PYTHON}" - \
    "${PORT}" \
    "${prompt}" \
    "${width}" \
    "${height}" \
    "${steps}" \
    "${cfg}" \
    "${seed}" \
    "${prefix}" \
    "${sampler}" \
    "${scheduler}" \
    "${denoise}" \
    "${COMFY_DIR}" <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

(
    port,
    prompt,
    width,
    height,
    steps,
    cfg,
    seed,
    prefix,
    sampler,
    scheduler,
    denoise,
    comfy_dir,
) = sys.argv[1:]

base = f"http://127.0.0.1:{port}"
width = int(width)
height = int(height)
steps = int(steps)
cfg = float(cfg)
seed = int(seed)
denoise = float(denoise)

# Minimal API prompt matching the official Krea-2 Turbo core graph:
# UNETLoader -> CLIPLoader(type=krea2) -> CLIPTextEncode -> EmptyLatentImage
# -> KSampler(euler/simple, 8 steps) -> VAEDecode -> SaveImage.
workflow = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "krea2_turbo_nvfp4.safetensors",
            "weight_dtype": "default",
        },
    },
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
            "type": "krea2",
            "device": "default",
        },
    },
    "3": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "qwen_image_vae.safetensors",
        },
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["2", 0],
            "text": prompt,
        },
    },
    "5": {
        "class_type": "ConditioningZeroOut",
        "inputs": {
            "conditioning": ["4", 0],
        },
    },
    "6": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": width,
            "height": height,
            "batch_size": 1,
        },
    },
    "7": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": denoise,
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["3", 0],
        },
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["8", 0],
            "filename_prefix": prefix,
        },
    },
}

def request_json(path, payload=None, timeout=60):
    if payload is None:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("ComfyUI API error:", e.code, e.reason)
        print(body)
        raise SystemExit(1)

resp = request_json("/prompt", {"prompt": workflow}, timeout=60)
prompt_id = resp["prompt_id"]

print("prompt_id:", prompt_id)
print("seed:", seed)
print("size:", f"{width}x{height}")
print("steps:", steps)
print("cfg:", cfg)

while True:
    hist = request_json("/history/" + urllib.parse.quote(prompt_id), timeout=60)
    if prompt_id in hist:
        item = hist[prompt_id]
        status = item.get("status", {})
        if status.get("status_str") == "error":
            print(json.dumps(status, indent=2))
            raise SystemExit(1)

        outputs = item.get("outputs", {})
        images = []
        for node_out in outputs.values():
            images.extend(node_out.get("images", []))

        if images:
            print("outputs:")
            for img in images:
                filename = img["filename"]
                subfolder = img.get("subfolder") or ""
                rel = os.path.join("output", subfolder, filename) if subfolder else os.path.join("output", filename)
                print(" ", os.path.join(comfy_dir, rel))
            break

    time.sleep(1)
PY
}

case "${ACTION}" in
  start)
    start_service
    ;;
  restart)
    restart_service
    ;;
  stop)
    stop_service
    ;;
  status)
    status_service
    ;;
  logs)
    show_logs
    ;;
  generate)
    shift || true
    generate_image "${*:-}"
    ;;
  setup|install)
    install_backend_if_needed
    ;;
  download)
    create_or_recreate_venv
    export_runtime_env
    clone_or_update_comfy
    run_pip install --python "${VENV_PYTHON}" --upgrade "huggingface_hub[hf_xet]"
    download_models
    ;;
  doctor|check)
    doctor
    ;;
  *)
    cat <<EOF
Usage: bash $0 {start|restart|stop|status|logs|generate|setup|download|doctor}

Primary path:
  bash $0 start

Fresh reset:
  KREA2_RECREATE_VENV=1 bash $0 start

Prefetch model weights during setup:
  KREA2_MODEL_MODE=with-models bash $0 setup

Generate 2048x2048:
  WIDTH=2048 HEIGHT=2048 bash $0 generate "a cinematic mountain greenhouse at sunrise"

Useful overrides:
  INSTALL_TORCH=0
  HOST=0.0.0.0
  PORT=8188
  GPU=0
  COMFY_ARGS=""
  STEPS=8 CFG=1.0 SEED=123
EOF
    exit 1
    ;;
esac
