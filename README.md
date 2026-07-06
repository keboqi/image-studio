# Image Studio

Image Studio is a GPU-first Gradio workspace for image generation, image editing, video generation, upscaling, multimodal chat, and model lifecycle management. It brings several local pipelines and companion services into one interface while loading heavyweight models only when they are needed.

## Features

- Generate images with Qwen Image, Z-Image, HiDream-O1, Ideogram 4, Boogu-Image, and Krea 2.
- Edit images with Qwen Image Edit, HiDream-O1, or Boogu-Image.
- Generate keyframe-, audio-, and IC-LoRA-guided video through LTX-Video.
- Upscale images and video with SeedVR2, with optional PiD 4x decoding for supported image generators.
- Remove visible, invisible, and metadata-based AI watermarks.
- Chat with local multimodal Gemma models or the managed DiffusionGemma vLLM service.
- Browse generated media, inspect saved metadata, and unload models from the UI.
- Call the generation tools through Gradio endpoints or use the optional OpenAI-compatible `/v1/*` proxy.

## Requirements

The full setup is intended for an Ubuntu-style Linux host with:

- an NVIDIA GPU and a working CUDA environment;
- Python 3.12;
- Git and `sudo` access;
- enough disk space for several model repositories and checkpoints;
- a Hugging Face account with access to any gated models you enable.

The quick-start script installs system packages, Python dependencies, Node.js, Pi, model repositories, checkpoints, and isolated companion services. It is intentionally a heavyweight machine-setup script; review it before running it on an existing environment.

## Quick start

```bash
git clone https://github.com/keboqi/image-studio.git
cd image-studio

# Install the Hugging Face CLI and authenticate before gated downloads.
python -m pip install --upgrade huggingface-hub
hf auth login

bash scripts/quickstart.sh --no-share
```

The UI listens on `0.0.0.0:7860` by default. Generated media is written to `outputs/`.

To skip installation of the Pi coding agent and its Node.js environment:

```bash
IMAGE_STUDIO_SKIP_PI_SETUP=1 bash scripts/quickstart.sh --no-share
```

After the machine has been provisioned, launch the app directly:

```bash
python image_studio_webui.py --no-share
```

Useful launch options:

```text
--port PORT                 Gradio server port (default: 7860)
--share / --no-share        Enable or disable the Gradio share link
--auth USER:PASSWORD        Protect the UI with basic authentication
--vllm-proxy / --no-vllm-proxy
                            Expose or disable the managed backend at /v1/*
--vllm-proxy-api-key KEY    Require a bearer key for the /v1/* proxy
--selftest                  Run the compatibility test suite and exit
```

> [!IMPORTANT]
> Gradio sharing is enabled by default. If the vLLM proxy is also enabled, set `--vllm-proxy-api-key` or disable sharing so the proxy is not exposed without authentication.

## Configuration

Configuration is read from environment variables at startup. Common overrides include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `IMAGE_STUDIO_NO_BOOTSTRAP` | `0` | Skip optional runtime bootstrap work, useful for development and tests. |
| `IMAGE_STUDIO_VLLM_PROXY` | `1` | Expose the managed DiffusionGemma backend through the UI server. |
| `IMAGE_STUDIO_VLLM_PROXY_API_KEY` | unset | Bearer key required by the OpenAI-compatible proxy. |
| `DIFFUSIONGEMMA_VLLM_PORT` | `8001` | Port used by the managed vLLM service. |
| `DIFFUSIONGEMMA_VLLM_API_BASE` | `http://127.0.0.1:8001/v1` | Existing vLLM endpoint to manage or proxy. |
| `KREA2_COMFY_PORT` | `8188` | Port used by the isolated Krea 2 ComfyUI service. |
| `LTX_WEB_API` | `http://127.0.0.1:8000` | LTX-Web API endpoint. |
| `IDEOGRAM_API_KEY` | unset | Optional Ideogram API credential used for prompt upsampling. |
| `PI_MODEL` | managed model name | Model exposed to the Pi chat integration. |

Additional model paths, repository IDs, timeouts, and service settings are defined in [`image_studio/config.py`](image_studio/config.py).

## API

Every main workflow is available through named Gradio endpoints, including image generation, editing, image/video upscaling, video generation, and watermark removal. Full request examples are in [`image_studio/docs/api.md`](image_studio/docs/api.md).

The managed DiffusionGemma backend can also be exposed as an OpenAI-compatible API:

```bash
python image_studio_webui.py --no-share --vllm-proxy-api-key local-key
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:7860/v1",
    api_key="local-key",
)

response = client.chat.completions.create(
    model="diffusiongemma",
    messages=[{"role": "user", "content": "Describe a cinematic red desert."}],
)
print(response.choices[0].message.content)
```

Use `GET /vllm/health` to inspect the proxy status.

## Development

Install the lightweight development tools and run the compatibility checks:

```bash
python -m pip install -r requirements-dev.txt
python image_studio_webui.py --selftest
ruff check image_studio tests
mypy image_studio
```

The test suite avoids loading GPU models. Optional end-to-end GPU checks are marked with `gpu` and can be selected explicitly with pytest.

## Repository layout

```text
image_studio/
  generators/   Generator adapters and dispatch
  infra/        Lazy imports, bootstrap, and managed model processes
  pipelines/    Model-specific pipeline integrations
  services/     Companion-service clients and lifecycle management
  storage/      Output, metadata, and gallery persistence
  ui/           Gradio components, styling, and event wiring
  web/          API routes, proxying, and embedded designer support
scripts/
  quickstart.sh Full host provisioning and launch workflow
tests/          Compatibility, routing, configuration, and storage tests
```

`image_studio_webui.py` remains the compatibility launcher; application code lives in the `image_studio` package.
