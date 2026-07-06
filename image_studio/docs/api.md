
## API Documentation

This WebUI exposes standard Gradio API endpoints that can be accessed from any 3rd party application using the official `gradio_client` library or via raw HTTP requests.

### Model discovery

`GET /api/models` returns the model catalog used by the image execution framework. Each entry includes
a stable model ID, supported operations, backend ID, and typed parameter schemas. Clients should use
stable IDs for new integrations; the Gradio endpoints below retain their historical display-name and
positional contracts for compatibility.

Image endpoints return a WebP preview path for display and a raw PNG path for download or follow-up processing.

### 1. Generating Images
**Endpoint:** `/run/generate`  (or `api_name="generate"`)

**Python Client Example:**
```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
preview_webp_path, status_text, raw_png_path, vram_markdown = client.predict(
    mode="Qwen Image",               # str: "Qwen Image", "Z-Image", "HiDream-O1", "Ideogram 4", "Boogu-Image", "Krea2"
    prompt="A cute cat",            # str
    neg_prompt="blurry",            # str (used by Qwen Image, Z-Image Best Quality, and Boogu-Image Base)
    width=1024,                     # int
    height=1024,                    # int
    cfg=1.0,                        # float (Qwen Image only)
    steps=8,                        # int (Z-Image Turbo only)
    guidance=0.0,                   # float (Z-Image Turbo only)
    full_steps=50,                  # int (Z-Image Best Quality only, 28-50 recommended)
    full_guidance=4.0,              # float (Z-Image Best Quality only, 3.0-5.0 recommended)
    full_pid_enabled=False,         # bool (Qwen/Z-Image/Ideogram/Krea2: 4x PiD decode)
    full_pid_ckpt="auto",           # str: "auto", "2k", or "2kto4k" (Qwen/Krea2 support auto/2kto4k; Ideogram/Z-Image support all)
    full_pid_steps=4,               # int (PiD distilled checkpoint default)
    full_pid_cfg=1.0,               # float (PiD decoder CFG)
    boogu_version="Turbo",          # str (Boogu-Image only): "Turbo" or "Base"
    boogu_steps=4,                  # int (Boogu-Image only; 4 for Turbo, 50 for Base)
    boogu_base_guidance=4.0,        # float (Boogu-Image Base only)
    krea2_steps=8,                  # int (Krea2 Turbo only; recommended 8)
    krea2_cfg=1.0,                  # float (Krea2 guidance scale; 1 disables CFG, API callers may pass 0)
    ideogram_pipeline="nvfp4 (fast)",          # str (Ideogram only): "nvfp4 (fast)", "fp8-nvfp4-uncond (balanced)", "fp8 (quality)"
    ideogram_sampler="Turbo - 12 steps",      # str (Ideogram only): "Turbo - 12 steps", "Fast Quality - 14 steps", "Default - 20 steps", "Quality - 48 steps"
    ideogram_upsampler="Gemma 4 local",       # str (Ideogram only): "Gemma 4 local", "Ideogram API", "None"
    ideogram_strip_prompt=True,               # bool (Ideogram JSON prompt cleanup)
    ideogram_reuse_cache=True,                # bool (reuse prompt upsample cache)
    ideogram_gemma_tokens=2048,               # int (local Gemma upsampler)
    ideogram_gemma_thinking=False,            # bool (local Gemma upsampler)
    ideogram_cfg_one_final_steps=0,           # int (Ideogram only; final steps use CFG=1 and skip unconditional branch)
    ideogram_lora_mode="Off",                 # str (Ideogram only): "Off", "Runtime adapter", "Fused in memory"
    ideogram_lora_weight="Realism_Engine_Ideogram_V2.safetensors",  # str (Ideogram Realism Engine LoRA file)
    ideogram_lora_cond_strength=0.9,          # float (conditional transformer LoRA strength)
    ideogram_lora_uncond_strength=0.4,        # float (unconditional transformer LoRA strength)
    ideogram_api_key="",                      # str (remote Ideogram API upsampler)
    seed=-1,                        # float (-1 for random)
    zimage_version="Turbo",         # str (Z-Image only): "Turbo" or "Best Quality"
    hidream_version="Dev",          # str (HiDream-O1 only): "Dev" or "Best Quality"
    api_name="/generate"
)
```

### 2. Editing Images
**Endpoint:** `/run/edit` (or `api_name="edit"`)

**Python Client Example:**
```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
preview_webp_path, status_text, raw_png_path, vram_markdown = client.predict(
    model_name="Qwen Image Edit",             # str: "Qwen Image Edit", "HiDream-O1", or "Boogu-Image"
    img1=handle_file("path/to/img1.png"),  # File path or URL
    img2=None,                             # Optional file 2
    img3=None,                             # Optional file 3
    prompt="Make it winter",               # str
    neg_prompt="",                         # str (Qwen and Boogu-Image)
    cfg=1.0,                               # float (Qwen only)
    qwen_seed=-1,                          # float
    boogu_version="Turbo",                 # str (Boogu-Image only): "Turbo" or "Base"
    boogu_steps=4,                         # int (Boogu-Image only; 4 for Turbo, 50 for Base)
    boogu_text_guidance=4.0,               # float (Boogu-Image Base only)
    boogu_image_guidance=1.0,              # float (Boogu-Image Base only)
    boogu_seed=-1,                         # float (Boogu-Image only)
    width=1024,                            # int (HiDream or Boogu-Image)
    height=1024,                           # int (HiDream or Boogu-Image)
    keep_original_aspect=True,             # bool (HiDream/Boogu-Image single source)
    hidream_seed=-1,                       # float
    hidream_version="Dev",                 # str (HiDream-O1 only): "Dev" or "Best Quality"
    api_name="/edit"
)
```

### 3. Upscaling Images (SeedVR2)
**Endpoint:** `/run/upscale` (or `api_name="upscale"`)

**Python Client Example:**
```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
preview_webp_path, status_text, raw_png_path, vram_markdown = client.predict(
    image=handle_file("source.png"), # File path or URL
    resolution=2160,                 # int
    max_resolution=3584,             # int
    dit_model="seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    color_correction="lab",          # str: "lab", "wavelet", "hsv", etc.
    vae_tiling=False,                # bool
    vae_tile_size=1024,              # int
    blocks_to_swap=0,                # int
    seed=42,                         # float
    api_name="/upscale"
)
```

### 4. AI Watermark Remover
**Endpoint:** `/run/ai_remover` (or `api_name="ai_remover"`)

**Python Client Example:**
```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
preview_webp_path, status_text, raw_png_path, vram_markdown = client.predict(
    image=handle_file("watermarked.png"),  # File path or URL
    mode="all",                            # str: "all", "visible", "invisible", or "metadata"
    humanize=0.0,                          # float: 0.0 - 6.0 (analog humanizer strength)
    api_name="/ai_remover"
)
```

### 5. Video Generation (LTX-Video)
**Endpoint:** `/run/generate_video` (or `api_name="generate_video"`)

Requires the `ltx-web` backend to be running (started automatically via quick start).

**Python Client Example:**
```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
video_path, status_text, vram_markdown = client.predict(
    image1=None,                      # Optional first keyframe image
    image2=None,                      # Optional second keyframe image
    image3=None,                      # Optional third keyframe image
    audio=None,                       # Optional audio file; enables audio-guided video
    ic_lora_name="Off",               # str: "Off" or one of the IC-LoRA adapter names
    ic_lora_reference_image=None,      # Optional reference sheet/still image
    ic_lora_reference_video=None,      # Optional reference/control video
    ic_lora_reference_text="",        # Optional Ingredients sheet description
    ic_lora_strength=1.0,             # float (0.0-2.0)
    ic_lora_attention_strength=1.0,   # float (0.0-1.0)
    prompt="A cat walking",           # str
    neg_prompt="",                    # str
    width=1024,                       # int
    height=1024,                      # int
    frames=121,                       # int (9-1201; audio paths require 16n+9)
    fps=24,                           # float
    skip_memory_cleanup=True,         # bool
    api_name="/generate_video"
)
```
Pass `handle_file("start.png")`/`handle_file("end.png")` into image slots to guide generation with keyframes, or pass `handle_file("speech.wav")` as `audio` for audio-to-video / portrait lipsync.
For IC-LoRA, set `ic_lora_name` and provide either `ic_lora_reference_image` or `ic_lora_reference_video`; audio is not used by the generic IC-LoRA path.

### 6. Video Upscaling (SeedVR2)
**Endpoint:** `/run/upscale_video` (or `api_name="upscale_video"`)

**Python Client Example:**
```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
video_path, status_text = client.predict(
    video=handle_file("source.mp4"),  # File path or URL
    resolution=2160,                  # int
    max_resolution=3584,              # int
    dit_model="seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    color_correction="lab",           # str: "lab", "wavelet", "hsv", etc.
    vae_tiling=False,                 # bool
    vae_tile_size=1024,               # int
    blocks_to_swap=0,                 # int
    batch_size=5,                     # int: 1, 5, 9, 13, ...
    chunk_size=25,                    # int: 0 = whole video
    temporal_overlap=0,               # int
    seed=42,                          # float
    api_name="/upscale_video"
)
```

### 7. Chat (Gemma 4 12B)
**Endpoint:** `/run/chat` (or via the Chat tab UI)

The Chat tab supports local multimodal Gemma models plus a managed DiffusionGemma vLLM backend. The local Gemma choices support text, image, and audio inputs; DiffusionGemma vLLM supports text and image inputs, but not audio. It is launched through `deploy_diffusiongemma_vllm.sh` on first use. Prompt enhancement and Ideogram's Gemma upsampler reuse whichever Chat Model is selected. You can also call the underlying `chat_respond` function via the Gradio client.

### 8. vLLM / OpenAI-Compatible Proxy
By default, this WebUI exposes the managed DiffusionGemma backend on the same host as the UI. Use `--no-vllm-proxy` or set `IMAGE_STUDIO_VLLM_PROXY=0` to disable it.

```bash
python image_studio_webui.py --port 7860 --vllm-proxy-api-key local-key
```

Then use it like a regular vLLM/OpenAI-compatible endpoint:

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

The proxy forwards `/v1/*` requests to the managed backend at `DIFFUSIONGEMMA_VLLM_API_BASE` and starts it on demand. It also preserves streaming responses for requests with `"stream": true`. A lightweight status endpoint is available at `/vllm/health`.

---

**Raw HTTP JSON API:**
Gradio exposes standard REST endpoints. Click the **"Use via API"** link at the very bottom of this WebUI page (in the footer) to see comprehensive interactive JSON payloads and CURL examples.
