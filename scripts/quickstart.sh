#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y curl ffmpeg
python -m pip install uv Librosa gradio "diffusers==0.36.0"
python -m pip install flash-attn sageattention --no-build-isolation
python -m pip install "https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.8-cp312-cp312-linux_x86_64.whl"

git clone https://github.com/keboqi/ideogram4.git ideogram4
python -m pip install -U "transformers>=4.57.6" "huggingface-hub>=0.26.0" bitsandbytes \
  "comfy-kitchen[cublas]" json-repair safetensors sentencepiece accelerate einops requests

git clone https://github.com/boogu-project/Boogu-Image.git boogu_image
python -m pip install cache-dit webdataset python-dotenv omegaconf
python boogu_image/utils/get_flash_attn.py

bash deploy_krea2_comfy.sh install

git clone --depth 1 https://github.com/nv-tlabs/PiD.git PiD
hf download nvidia/PiD --local-dir PiD --include \
  "checkpoints/ae.safetensors" \
  "checkpoints/QwenImage_VAE_2d.pth" \
  "checkpoints/flux2_ae.safetensors" \
  "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/*" \
  "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/*"

git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git seedvr2_upscaler
python -m pip install -r seedvr2_upscaler/requirements.txt

# Optional services and gated models require local credentials. Never commit a
# real API key; export it only in the launch environment:
# export IDEOGRAM_API_KEY="..."
# hf auth login

python image_studio_webui.py "$@"
