#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONSTRAINTS_FILE="$ROOT_DIR/scripts/constraints.txt"
cd "$ROOT_DIR"

pip_install() {
  python -m pip install -c "$CONSTRAINTS_FILE" "$@"
}

clone_if_missing() {
  local repo="$1"
  local destination="$2"
  shift 2

  if [[ -d "$destination/.git" ]]; then
    echo "Using existing checkout: $destination"
    return
  fi
  if [[ -e "$destination" ]]; then
    echo "Cannot clone $repo: $destination exists but is not a Git checkout." >&2
    return 1
  fi
  git clone "$@" "$repo" "$destination"
}

sudo apt update
sudo apt install -y curl ffmpeg
python -m pip install --upgrade pip setuptools wheel
pip_install uv

# Pi chat button setup (Node/npm + Pi + local vLLM model config).
export VLLM_PORT="${VLLM_PORT:-8001}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-diffusiongemma}"
export PI_MODEL="${PI_MODEL:-${SERVED_MODEL_NAME}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export PI_CONFIG_DIR="${PI_CONFIG_DIR:-.pi/agent}"
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ "${IMAGE_STUDIO_SKIP_PI_SETUP:-0}" != "1" ]]; then
  if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    # shellcheck source=/dev/null
    . "$NVM_DIR/nvm.sh"
  else
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
    # shellcheck source=/dev/null
    . "$NVM_DIR/nvm.sh"
  fi
  nvm install "${NODE_VERSION:-node}" --latest-npm
  nvm use "${NODE_VERSION:-node}"
  nvm alias default "$(node -v)"
  npm install -g "npm@${NPM_VERSION:-latest}"
  npm install -g --force --ignore-scripts \
    "@earendil-works/pi-coding-agent@${PI_NPM_VERSION:-latest}"
  mkdir -p "$PI_CONFIG_DIR"
  cat > "$PI_CONFIG_DIR/models.json" <<EOF_JSON
{
  "providers": {
    "vllm": {
      "baseUrl": "http://127.0.0.1:${VLLM_PORT}/v1",
      "api": "openai-completions",
      "apiKey": "vllm-local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "${SERVED_MODEL_NAME}",
          "name": "${SERVED_MODEL_NAME}",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": ${MAX_MODEL_LEN}
        }
      ]
    }
  }
}
EOF_JSON
fi

# AI watermark remover setup. Keep it isolated because it needs diffusers 0.38.
clone_if_missing https://github.com/wiltodelta/remove-ai-watermarks.git remove-ai-watermarks
if [[ ! -x remove-ai-watermarks/.venv/bin/python ]]; then
  uv venv remove-ai-watermarks/.venv --python 3.12
fi
uv pip install --python remove-ai-watermarks/.venv/bin/python -e "remove-ai-watermarks[gpu]"
uv pip install --python remove-ai-watermarks/.venv/bin/python -U \
  "transformers>=4.57.1,<5" "hf_transfer>=0.1.9"

pip_install Librosa gradio "diffusers==0.36.0"
# flash-attn downloads a matching wheel during its build step. Keep pip's
# wheel cache disabled so that wheel is not renamed across mounted filesystems.
pip_install flash-attn --no-build-isolation --no-cache-dir
pip_install sageattention --no-build-isolation
pip_install "https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.8-cp312-cp312-linux_x86_64.whl"
python -c "import nunchaku; print('nunchaku import ok:', getattr(nunchaku, '__version__', 'unknown'))"

# Ideogram 4 NVFP4 generation setup.
clone_if_missing https://github.com/keboqi/ideogram4.git ideogram4
pip_install -U "transformers>=4.57.6" "huggingface-hub>=0.26.0" bitsandbytes \
  "comfy-kitchen[cublas]" json-repair safetensors sentencepiece accelerate einops requests
pip_install -U packaging ninja psutil

# Boogu-Image generation/edit setup.
clone_if_missing https://github.com/boogu-project/Boogu-Image.git boogu_image
pip_install cache-dit webdataset python-dotenv omegaconf
python boogu_image/utils/get_flash_attn.py
hf download Boogu/Boogu-Image-0.1-Turbo --local-dir models/Boogu-Image-0.1-Turbo
hf download Boogu/Boogu-Image-0.1-Base --local-dir models/Boogu-Image-0.1-Base
hf download Boogu/Boogu-Image-0.1-Edit --local-dir models/Boogu-Image-0.1-Edit

# Krea-2 uses its own environment and cannot disturb the main WebUI packages.
export KREA2_COMFY_PORT="${KREA2_COMFY_PORT:-8188}"
bash deploy_krea2_comfy.sh install

# Optional services and gated models require local credentials. Never commit a
# real API key; export it only in the launch environment and run `hf auth login`.
export GEMMA_ASSISTANT_MODEL="${GEMMA_ASSISTANT_MODEL:-google/gemma-4-12B-it-assistant}"
export GEMMA_NVFP4_ASSISTANT_MODEL="${GEMMA_NVFP4_ASSISTANT_MODEL:-$GEMMA_ASSISTANT_MODEL}"
export DIFFUSIONGEMMA_VLLM_PORT="${DIFFUSIONGEMMA_VLLM_PORT:-8001}"
export DIFFUSIONGEMMA_VLLM_MODEL="${DIFFUSIONGEMMA_VLLM_MODEL:-diffusiongemma}"

# Optional PiD 4x decoder for Z-Image Full, Qwen Image, and Ideogram 4.
clone_if_missing https://github.com/nv-tlabs/PiD.git PiD --depth 1
hf download nvidia/PiD --local-dir PiD --include \
  "checkpoints/ae.safetensors" \
  "checkpoints/QwenImage_VAE_2d.pth" \
  "checkpoints/flux2_ae.safetensors" \
  "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/*" \
  "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/*"
pip_install hydra-core==1.3.2 omegaconf==2.3.0 attrs einops loguru termcolor \
  fvcore iopath pynvml wandb imageio opencv-python-headless pandas safetensors \
  "huggingface-hub>=1.0" sentencepiece boto3 botocore

# SeedVR2 setup.
clone_if_missing https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git \
  seedvr2_upscaler --depth 1
pip_install -r seedvr2_upscaler/requirements.txt

# LTX-Web video generation setup. Its launcher installs dependencies and models;
# remove the final API launch because Image Studio owns that subprocess.
clone_if_missing https://github.com/keboqi/ltx-web.git ltx-web
sed -i 's/huggingface-cli/hf/g; s/python api.py//g' ltx-web/run.sh
(
  cd ltx-web
  PIP_CONSTRAINT="$CONSTRAINTS_FILE" sh run.sh
)

# Reconcile binary packages after every third-party requirements file. OpenCV
# 4.12+ requires NumPy 2 on Python 3.12, while this stack and its existing pandas
# wheels require the NumPy 1.x ABI. Use the one headless OpenCV distribution.
python -m pip uninstall -y opencv-python opencv-contrib-python \
  opencv-python-headless opencv-contrib-python-headless || true
pip_install --upgrade --force-reinstall "numpy<2.0" "scipy>=1.13.0" pandas \
  "opencv-python-headless<4.12"
python - <<'PY'
import cv2
import gradio
import numpy
import pandas
import scipy

print(
    "Compatibility check ok:",
    f"numpy={numpy.__version__}",
    f"pandas={pandas.__version__}",
    f"scipy={scipy.__version__}",
    f"opencv={cv2.__version__}",
    f"gradio={gradio.__version__}",
)
PY

python image_studio_webui.py "$@"
