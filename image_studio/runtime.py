"""Image Studio runtime composition.

See ``scripts/quickstart.sh`` for installation and launch instructions.
"""

import base64
import contextlib
import hashlib
import io
import importlib
import json
import math
import os
import re
import sys
import subprocess
import shutil
import tempfile
import time
import logging
import gc
import threading
import atexit
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from image_studio.config import AppConfig
from image_studio.core.backends import BackendRegistry, CallableBackend
from image_studio.core.executor import ModelExecutor
from image_studio.core.models import Operation
from image_studio.errors import AppError, BackendUnavailableError, ModelLoadError, UserInputError
from image_studio.infra.bootstrap import GitBootstrap, RepoSpec
from image_studio.infra.model_manager import ManagedModelSpec, ModelManager
from image_studio.infra.model_storage import ModelStorageCatalog, ModelStorageTarget
from image_studio.generators.base import EditRequest, GenerationRequest, UIRequest
from image_studio.generators.registry import RequestHandlerRegistry
from image_studio.parsing import extract_json_object as pure_extract_json_object
from image_studio.parsing import fix_unescaped_json_newlines, parse_enhance_json
from image_studio.pipelines.ideogram.prompting import clean_malformed_json_caption, parse_caption
from image_studio.pipelines.pid import patchify_flux2_raw_latents
from image_studio.storage.output_store import OutputStore
from image_studio.storage.output_store import OUTPUT_PREVIEW_QUALITY, OUTPUT_PREVIEW_SUFFIX
from image_studio.storage.metadata import JsonFileCache, PromptMetadataStore
from image_studio.validation import (
    snap_ltx_audio_video_frames,
    validate_boogu_dims as pure_validate_boogu_dims,
    validate_dims as pure_validate_dims,
    validate_ideogram_dims,
)
from image_studio.web.routes import PUBLIC_API_ENDPOINTS, promote_routes_before_fallback
from image_studio.runtime_binding import export_module, rebind_modules
from image_studio.ui.theme import build_theme, load_css
from image_studio.logging_setup import configure_logging
from image_studio.context import AppContext
from image_studio.integrations.image_models import ImageModelFunctions, build_image_model_registry

_RUNTIME_MODULES = []


def check_dependencies():
    missing = []
    for pkg in ["diffusers", "gradio"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(f"Missing: {missing}. Run: pip install -r requirements.txt")


check_dependencies()

import torch
import numpy as np
import gradio as gr
import cv2
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline, QwenImageEditPlusPipeline
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from image_studio.services.ltx_video import LtxVideoService

_NUNCHAKU_IMPORT_ERROR: Exception | None = None
try:
    from nunchaku import NunchakuZImageTransformer2DModel
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
    from nunchaku.utils import get_gpu_memory, get_precision, is_turing
except Exception as exc:
    _NUNCHAKU_IMPORT_ERROR = exc
    NunchakuZImageTransformer2DModel = None
    NunchakuQwenImageTransformer2DModel = None

    def get_gpu_memory() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))

    def get_precision() -> str:
        override = APP_CONFIG.nunchaku_precision
        if override:
            return override
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability(0)
            return "fp4" if major >= 10 else "int4"
        return "int4"

    def is_turing() -> bool:
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(0)
        return major == 7 and minor == 5


def require_nunchaku() -> None:
    if _NUNCHAKU_IMPORT_ERROR is None:
        return
    raise gr.Error(
        "This generator requires nunchaku, but it is not importable in the current "
        "Python environment. Install it with the quick-start nunchaku command, then "
        f"restart the app. Import error: {_NUNCHAKU_IMPORT_ERROR}"
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
log = configure_logging("image_studio.webui")

APP_CONFIG = AppConfig.from_env(
    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    argv=sys.argv,
)
BASE_DIR = str(APP_CONFIG.paths.base_dir)
OUTPUT_DIR = str(APP_CONFIG.paths.output_dir)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SELFTEST_ARG = "--selftest"
_NO_BOOTSTRAP = APP_CONFIG.no_bootstrap


def _bootstrap_allowed(name: str) -> bool:
    """Return whether lazy setup may clone/install external projects."""
    if not _NO_BOOTSTRAP:
        return True
    log.info("%s bootstrap skipped because no-bootstrap/selftest mode is active.", name)
    return False

# Auto-detect precision: 'fp4' (NVFP4) for Blackwell/50-series, 'int4' for older GPUs
_PRECISION = get_precision()
log.info("Detected GPU quantization format: %s", _PRECISION)

GEN_MODEL = f"nunchaku-tech/nunchaku-qwen-image/svdq-{_PRECISION}_r128-qwen-image-lightningv1.0-4steps.safetensors"
EDIT_MODEL = f"nunchaku-ai/nunchaku-qwen-image-edit-2509/lightning-251115/svdq-{_PRECISION}_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors"
HIDREAM_O1_REPO = "https://github.com/HiDream-ai/HiDream-O1-Image.git"
HIDREAM_O1_DIR = os.path.join(BASE_DIR, "hidream_o1_image")
HIDREAM_O1_FULL_MODEL = APP_CONFIG.hidream_full_model
HIDREAM_O1_DEV_MODEL = APP_CONFIG.hidream_dev_model
HIDREAM_O1_MODE = "HiDream-O1"
BOOGU_IMAGE_REPO = "https://github.com/boogu-project/Boogu-Image.git"
BOOGU_IMAGE_DIR = str(APP_CONFIG.boogu.directory)
BOOGU_IMAGE_MODE = "Boogu-Image"
BOOGU_IMAGE_TURBO_NAME = "Boogu-Image-0.1-Turbo"
BOOGU_IMAGE_BASE_NAME = "Boogu-Image-0.1-Base"
BOOGU_IMAGE_EDIT_NAME = "Boogu-Image-0.1-Edit"
BOOGU_IMAGE_EDIT_TURBO_NAME = "Boogu-Image-0.1-Edit-Turbo"
KREA2_MODE = "Krea2"
KREA2_DEFAULT_STEPS = 8
KREA2_DEFAULT_CFG = 1.0
BOOGU_IMAGE_MAX_SEQUENCE_LENGTH = APP_CONFIG.boogu.max_sequence_length
REMOVE_AI_WATERMARKS_REPO = "https://github.com/wiltodelta/remove-ai-watermarks.git"
REMOVE_AI_WATERMARKS_DIR = str(APP_CONFIG.paths.remove_ai_watermarks_dir)
REMOVE_AI_WATERMARKS_TRANSFORMERS_SPEC = "transformers>=4.57.1,<5"
REMOVE_AI_WATERMARKS_HF_TRANSFER_SPEC = "hf_transfer>=0.1.9"
PID_REPO = "https://github.com/nv-tlabs/PiD.git"
PID_DIR = str(APP_CONFIG.pid.directory)
PID_HF_REPO = "nvidia/PiD"

MODEL_GEN = "gen"
MODEL_EDIT = "edit"
MODEL_ZIMAGE_TURBO = "zimage_turbo"
MODEL_ZIMAGE_FULL = "zimage_full"
MODEL_ZIMAGE_PID_2K = "zimage_pid_2k"
MODEL_ZIMAGE_PID_2KTO4K = "zimage_pid_2kto4k"
MODEL_QWEN_PID_2KTO4K = "qwen_pid_2kto4k"
MODEL_IDEOGRAM4_PID_2K = "ideogram4_pid_2k"
MODEL_IDEOGRAM4_PID_2KTO4K = "ideogram4_pid_2kto4k"
MODEL_HIDREAM_O1_FULL = "hidream_o1_full"
MODEL_HIDREAM_O1_DEV = "hidream_o1_dev"
MODEL_BOOGU_IMAGE_TURBO = "boogu_image_turbo"
MODEL_BOOGU_IMAGE_BASE = "boogu_image_base"
MODEL_BOOGU_IMAGE_EDIT = "boogu_image_edit"
MODEL_BOOGU_IMAGE_EDIT_TURBO = "boogu_image_edit_turbo"
MODEL_KREA2_TURBO_NVFP4 = "krea2_turbo_nvfp4"
MODEL_IDEOGRAM4_NVFP4 = "ideogram4_nvfp4"
MODEL_IDEOGRAM4_FP8_NVFP4_UNCOND = "ideogram4_fp8_nvfp4_uncond"
MODEL_IDEOGRAM4_FP8 = "ideogram4_fp8"
MODEL_SEEDVR2 = "seedvr2"
MODEL_GEMMA = "gemma4"
MODEL_GEMMA_CHAT = "gemma4_nvfp4"
MODEL_DIFFUSIONGEMMA_VLLM = "diffusiongemma_vllm"
MODEL_LTX_VIDEO = "ltx_video"
IDEOGRAM4_MODE = "Ideogram 4"
IDEOGRAM4_REPO = "https://github.com/keboqi/ideogram4.git"
IDEOGRAM4_DIR = str(APP_CONFIG.ideogram.directory)
IDEOGRAM4_SRC_DIR = os.path.join(IDEOGRAM4_DIR, "src")
IDEOGRAM4_PIPELINE_NVFP4 = "nvfp4"
IDEOGRAM4_PIPELINE_FP8_NVFP4_UNCOND = "fp8-nvfp4-uncond"
IDEOGRAM4_PIPELINE_FP8 = "fp8"
IDEOGRAM4_DEFAULT_PIPELINE = IDEOGRAM4_PIPELINE_NVFP4
IDEOGRAM4_NVFP4_WEIGHTS_REPO = APP_CONFIG.ideogram.nvfp4_weights_repo
IDEOGRAM4_NVFP4_CONFIG_REPO = APP_CONFIG.ideogram.nvfp4_config_repo
IDEOGRAM4_FP8_WEIGHTS_REPO = APP_CONFIG.ideogram.fp8_weights_repo
IDEOGRAM4_FP8_NVFP4_UNCOND_WEIGHTS_REPO = APP_CONFIG.ideogram.fp8_nvfp4_uncond_weights_repo
IDEOGRAM4_NVFP4_CONDITIONAL_FILENAME = "diffusion_models/ideogram4_nvfp4_mixed.safetensors.index.json"
IDEOGRAM4_NVFP4_UNCONDITIONAL_FILENAME = "diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors.index.json"
IDEOGRAM4_NVFP4_TEXT_ENCODER_FILENAME = "text_encoders/qwen3vl_8b_nvfp4.safetensors"
IDEOGRAM4_NVFP4_AUTOENCODER_FILENAME = "vae/flux2-vae.safetensors"
IDEOGRAM4_PIPELINE_LABELS = {
    IDEOGRAM4_PIPELINE_NVFP4: "nvfp4 (fast)",
    IDEOGRAM4_PIPELINE_FP8_NVFP4_UNCOND: "fp8-nvfp4-uncond (balanced)",
    IDEOGRAM4_PIPELINE_FP8: "fp8 (quality)",
}
IDEOGRAM4_PIPELINE_CHOICES = list(IDEOGRAM4_PIPELINE_LABELS.values())
IDEOGRAM4_PIPELINE_LABEL_TO_KEY = {
    label: key for key, label in IDEOGRAM4_PIPELINE_LABELS.items()
}
IDEOGRAM4_PIPELINE_MODEL_KEYS = {
    IDEOGRAM4_PIPELINE_NVFP4: MODEL_IDEOGRAM4_NVFP4,
    IDEOGRAM4_PIPELINE_FP8_NVFP4_UNCOND: MODEL_IDEOGRAM4_FP8_NVFP4_UNCOND,
    IDEOGRAM4_PIPELINE_FP8: MODEL_IDEOGRAM4_FP8,
}
IDEOGRAM4_UPSAMPLE_GEMMA = "Gemma 4 local"
IDEOGRAM4_UPSAMPLE_REMOTE = "Ideogram API"
IDEOGRAM4_UPSAMPLE_NONE = "None"
IDEOGRAM4_UPSAMPLERS = [
    IDEOGRAM4_UPSAMPLE_GEMMA,
    IDEOGRAM4_UPSAMPLE_REMOTE,
    IDEOGRAM4_UPSAMPLE_NONE,
]
IDEOGRAM4_SAMPLER_PRESETS = {
    "Turbo - 12 steps": "V4_TURBO_12",
    "Fast Quality - 14 steps": "V4_FAST_QUALITY_14",
    "Default - 20 steps": "V4_DEFAULT_20",
    "Quality - 48 steps": "V4_QUALITY_48",
}
IDEOGRAM4_SAMPLER_CHOICES = list(IDEOGRAM4_SAMPLER_PRESETS.keys())
IDEOGRAM4_LORA_OFF = "Off"
IDEOGRAM4_LORA_RUNTIME = "Runtime adapter"
IDEOGRAM4_LORA_FUSED = "Fused in memory"
IDEOGRAM4_LORA_CHOICES = [
    IDEOGRAM4_LORA_OFF,
    IDEOGRAM4_LORA_RUNTIME,
    IDEOGRAM4_LORA_FUSED,
]
IDEOGRAM4_REALISM_LORA_REPO = APP_CONFIG.ideogram.realism_lora_repo
IDEOGRAM4_REALISM_LORA_WEIGHTS = [
    "Realism_Engine_Ideogram_V2.safetensors",
    "Realism_Engine_Ideogram_V3.safetensors",
    "Realism_Engine_Ideogram_V4.safetensors",
    "Realism_Engine_Ideogram4_V1.safetensors",
    "Realism_Engine_Ideogram4_beta.safetensors",
]
IDEOGRAM4_REALISM_LORA_DEFAULT = APP_CONFIG.ideogram.realism_lora_weight
if IDEOGRAM4_REALISM_LORA_DEFAULT not in IDEOGRAM4_REALISM_LORA_WEIGHTS:
    IDEOGRAM4_REALISM_LORA_WEIGHTS.insert(0, IDEOGRAM4_REALISM_LORA_DEFAULT)
IDEOGRAM4_UPSAMPLE_CACHE_SCHEMA_VERSION = 2
IDEOGRAM4_UPSAMPLE_CACHE_PATH = os.path.join(BASE_DIR, "ideogram4_upsample_cache.json")
IDEOGRAM4_JSON_DESIGNER_FILE = os.path.join(BASE_DIR, "jsondesigner.html")
IDEOGRAM4_JSON_DESIGNER_PATH = "/ideogram4/json-designer"
IDEOGRAM4_JSON_DESIGNER_ROUTE_NAME = "ideogram4_json_designer"
IDEOGRAM4_PROMPT_METADATA_SCHEMA_VERSION = 1
IDEOGRAM4_PROMPT_METADATA_SUFFIX = ".ideogram4_prompt.json"


@dataclass(frozen=True)
class HiDreamSpec:
    model_id: str
    label: str
    short_label: str
    steps: int
    guidance_scale: float
    shift: float
    scheduler_name: str
    use_default_timesteps: bool
    noise_scale_start: float
    noise_scale_end: float
    noise_clip_std: float


@dataclass(frozen=True)
class PIDCheckpointSpec:
    registry_key: str
    experiment: str
    relative_checkpoint_path: str
    label: str


_git_bootstrap = GitBootstrap(_bootstrap_allowed)


PID_SCALE = 4
PID_MAX_LOW_SIDE = 1024
PID_CKPT_AUTO = "auto"
PID_CKPT_2K = "2k"
PID_CKPT_2KTO4K = "2kto4k"
PID_BACKBONE_ZIMAGE = "zimage"
PID_BACKBONE_QWEN = "qwenimage"
PID_BACKBONE_IDEOGRAM4 = "ideogram4_flux2"
PID_ZIMAGE_CKPT_CHOICES = [PID_CKPT_AUTO, PID_CKPT_2K, PID_CKPT_2KTO4K]
PID_QWEN_CKPT_CHOICES = [PID_CKPT_AUTO, PID_CKPT_2KTO4K]
PID_IDEOGRAM4_CKPT_CHOICES = [PID_CKPT_AUTO, PID_CKPT_2K, PID_CKPT_2KTO4K]
PID_FLUX_AE_PATH = "checkpoints/ae.safetensors"
PID_FLUX2_AE_PATH = "checkpoints/flux2_ae.safetensors"
PID_QWEN_VAE_PATH = "checkpoints/QwenImage_VAE_2d.pth"
PID_VAE_ASSETS = {
    PID_BACKBONE_ZIMAGE: PID_FLUX_AE_PATH,
    PID_BACKBONE_QWEN: PID_QWEN_VAE_PATH,
    PID_BACKBONE_IDEOGRAM4: PID_FLUX2_AE_PATH,
}
PID_CHECKPOINTS = {
    PID_BACKBONE_ZIMAGE: {
        PID_CKPT_2K: PIDCheckpointSpec(
            registry_key=MODEL_ZIMAGE_PID_2K,
            experiment="PiD_res2k_sr4x_official_flux_distill_4step",
            relative_checkpoint_path=(
                "checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/"
                "model_ema_bf16.pth"
            ),
            label="PiD 2k",
        ),
        PID_CKPT_2KTO4K: PIDCheckpointSpec(
            registry_key=MODEL_ZIMAGE_PID_2KTO4K,
            experiment="PiD_res2kto4k_sr4x_official_flux_distill_4step",
            relative_checkpoint_path=(
                "checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/"
                "model_ema_bf16.pth"
            ),
            label="PiD 2kto4k",
        ),
    },
    PID_BACKBONE_QWEN: {
        PID_CKPT_2KTO4K: PIDCheckpointSpec(
            registry_key=MODEL_QWEN_PID_2KTO4K,
            experiment="PiD_res2kto4k_sr4x_official_qwenimage_distill_4step",
            relative_checkpoint_path=(
                "checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/"
                "model_ema_bf16.pth"
            ),
            label="Qwen PiD 2kto4k",
        ),
    },
    PID_BACKBONE_IDEOGRAM4: {
        PID_CKPT_2K: PIDCheckpointSpec(
            registry_key=MODEL_IDEOGRAM4_PID_2K,
            experiment="PiD_res2k_sr4x_official_flux2_distill_4step",
            relative_checkpoint_path=(
                "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/"
                "model_ema_bf16.pth"
            ),
            label="Ideogram Flux2 PiD 2k",
        ),
        PID_CKPT_2KTO4K: PIDCheckpointSpec(
            registry_key=MODEL_IDEOGRAM4_PID_2KTO4K,
            experiment="PiD_res2kto4k_sr4x_official_flux2_distill_4step",
            relative_checkpoint_path=(
                "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/"
                "model_ema_bf16.pth"
            ),
            label="Ideogram Flux2 PiD 2kto4k",
        ),
    },
}


HIDREAM_O1_SPECS = {
    MODEL_HIDREAM_O1_FULL: HiDreamSpec(
        model_id=HIDREAM_O1_FULL_MODEL,
        label="HiDream-O1-Image",
        short_label="HiDream-O1 Full",
        steps=50,
        guidance_scale=5.0,
        shift=3.0,
        scheduler_name="default",
        use_default_timesteps=False,
        noise_scale_start=8.0,
        noise_scale_end=8.0,
        noise_clip_std=0.0,
    ),
    MODEL_HIDREAM_O1_DEV: HiDreamSpec(
        model_id=HIDREAM_O1_DEV_MODEL,
        label="HiDream-O1-Image-Dev",
        short_label="HiDream-O1 Dev",
        steps=28,
        guidance_scale=0.0,
        shift=1.0,
        scheduler_name="flash",
        use_default_timesteps=True,
        noise_scale_start=7.5,
        noise_scale_end=7.5,
        noise_clip_std=2.5,
    ),
}

BOOGU_IMAGE_VERSION_TURBO = "Turbo"
BOOGU_IMAGE_VERSION_BASE = "Base"
BOOGU_IMAGE_GENERATION_VERSIONS = [BOOGU_IMAGE_VERSION_TURBO, BOOGU_IMAGE_VERSION_BASE]
BOOGU_IMAGE_EDIT_VERSIONS = [BOOGU_IMAGE_VERSION_TURBO, BOOGU_IMAGE_VERSION_BASE]
BOOGU_IMAGE_VERSION_KEYS = {
    BOOGU_IMAGE_VERSION_TURBO: MODEL_BOOGU_IMAGE_TURBO,
    BOOGU_IMAGE_VERSION_BASE: MODEL_BOOGU_IMAGE_BASE,
}
BOOGU_IMAGE_EDIT_VERSION_KEYS = {
    BOOGU_IMAGE_VERSION_TURBO: MODEL_BOOGU_IMAGE_EDIT_TURBO,
    BOOGU_IMAGE_VERSION_BASE: MODEL_BOOGU_IMAGE_EDIT,
}
BOOGU_IMAGE_MODEL_NAMES = {
    MODEL_BOOGU_IMAGE_TURBO: BOOGU_IMAGE_TURBO_NAME,
    MODEL_BOOGU_IMAGE_BASE: BOOGU_IMAGE_BASE_NAME,
    MODEL_BOOGU_IMAGE_EDIT: BOOGU_IMAGE_EDIT_NAME,
    MODEL_BOOGU_IMAGE_EDIT_TURBO: BOOGU_IMAGE_EDIT_TURBO_NAME,
}
BOOGU_IMAGE_TURBO_DEFAULT_STEPS = 4
BOOGU_IMAGE_BASE_DEFAULT_STEPS = 50
BOOGU_IMAGE_MAX_SIDE = 2048
BOOGU_IMAGE_MAX_PIXELS = BOOGU_IMAGE_MAX_SIDE * BOOGU_IMAGE_MAX_SIDE

LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# ---------------------------------------------------------------------------
# SeedVR2 bootstrap: auto-clone + import
# ---------------------------------------------------------------------------
SEEDVR2_REPO = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
SEEDVR2_DIR = str(APP_CONFIG.seedvr2.directory)
SEEDVR2_REPO_SPEC = RepoSpec(
    name="SeedVR2",
    repo_url=SEEDVR2_REPO,
    target_dir=SEEDVR2_DIR,
    sentinel_files=("inference_cli.py",),
    requirements=("requirements.txt", "numpy<2.0"),
)


_runtime_pipelines_seedvr2 = importlib.import_module('image_studio.pipelines.seedvr2')
export_module(_runtime_pipelines_seedvr2, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_seedvr2)




# Lazy-loaded SeedVR2 modules (imported after bootstrap)




# ---------------------------------------------------------------------------
# Boogu-Image bootstrap: lazy clone/import of the official pipelines
# ---------------------------------------------------------------------------
BOOGU_IMAGE_REPO_SPEC = RepoSpec(
    name="Boogu-Image",
    repo_url=BOOGU_IMAGE_REPO,
    target_dir=BOOGU_IMAGE_DIR,
    sentinel_files=(
        "boogu/pipelines/boogu/pipeline_boogu.py",
        "boogu/pipelines/boogu/pipeline_boogu_turbo.py",
    ),
)


_runtime_pipelines_boogu = importlib.import_module('image_studio.pipelines.boogu')
export_module(_runtime_pipelines_boogu, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_boogu)








# ---------------------------------------------------------------------------
# HiDream-O1 bootstrap: lazy clone + import of the official image pipeline
# ---------------------------------------------------------------------------
HIDREAM_O1_REPO_SPEC = RepoSpec(
    name="HiDream-O1",
    repo_url=HIDREAM_O1_REPO,
    target_dir=HIDREAM_O1_DIR,
    sentinel_files=("models/pipeline.py", "models/qwen3_vl_transformers.py"),
)


_runtime_pipelines_hidream = importlib.import_module('image_studio.pipelines.hidream')
export_module(_runtime_pipelines_hidream, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_hidream)
















# ---------------------------------------------------------------------------
# PiD bootstrap: lazy clone/import/checkpoint download for 4x latent decoding
# ---------------------------------------------------------------------------
PID_REPO_SPEC = RepoSpec(
    name="PiD",
    repo_url=PID_REPO,
    target_dir=PID_DIR,
    sentinel_files=("pid/_src/utils/model_loader.py", "pid/_src/configs/pid/config.py"),
)


_runtime_pipelines_pid = importlib.import_module('image_studio.pipelines.pid')
export_module(_runtime_pipelines_pid, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_pid)


























































# ---------------------------------------------------------------------------
# LTX-Web Video Backend Manager
# ---------------------------------------------------------------------------
LTX_WEB_DIR = str(APP_CONFIG.ltx.directory)
LTX_WEB_API = APP_CONFIG.ltx.api_base
_ltx_video_service = LtxVideoService(LTX_WEB_DIR, LTX_WEB_API)

def _start_ltx_video_backend():
    spec = MODEL_SPECS[MODEL_LTX_VIDEO]
    model_mgr.ensure_vram(spec.vram_mb, exclude=MODEL_LTX_VIDEO)
    started = _ltx_video_service.start()
    if started and not model_mgr.is_loaded(MODEL_LTX_VIDEO):
        model_mgr.register(
            MODEL_LTX_VIDEO,
            _ltx_video_service,
            spec.vram_mb,
            unload_fn=_ltx_video_service.stop,
        )
    return started

def _stop_ltx_video_backend():
    _ltx_video_service.stop()

atexit.register(_stop_ltx_video_backend)

def check_ltx_video_health():
    if _NO_BOOTSTRAP:
        return False, None
    return _ltx_video_service.health()

def unload_ltx_video_pipeline():
    return _ltx_video_service.unload_pipeline()


# ---------------------------------------------------------------------------
# Model Manager - tracks loaded pipelines, VRAM budgets, and LRU eviction
# ---------------------------------------------------------------------------


# Singleton
model_mgr = ModelManager()
_model_load_lock = threading.RLock()
_inprocess_gpu_lock = threading.RLock()

atexit.register(model_mgr.unload_all)

LIGHTNING_STEPS = 4
CHAT_MAX_TOKENS = 2048
CHAT_MIN_TOKENS = 64
CHAT_MAX_TOKEN_LIMIT = 32768



IDEOGRAM4_EXCLUSIVE_GROUP = "ideogram4"
BOOGU_IMAGE_EXCLUSIVE_GROUP = "boogu_image"
KREA2_EXCLUSIVE_GROUP = "krea2"


# Estimated VRAM values are rough upper-bounds so the manager errs on the side
# of caution when deciding to evict.
MODEL_SPECS: dict[str, ManagedModelSpec] = {
    MODEL_GEN: ManagedModelSpec(MODEL_GEN, "Qwen Image (Gen)", 8_500),
    MODEL_EDIT: ManagedModelSpec(MODEL_EDIT, "Qwen Image Edit", 9_000),
    MODEL_ZIMAGE_TURBO: ManagedModelSpec(MODEL_ZIMAGE_TURBO, "Z-Image Turbo (FP4)", 7_000),
    MODEL_ZIMAGE_FULL: ManagedModelSpec(MODEL_ZIMAGE_FULL, "Z-Image Full (BF16)", 16_000),
    MODEL_ZIMAGE_PID_2K: ManagedModelSpec(MODEL_ZIMAGE_PID_2K, "PiD 4x Decoder (2k)", 12_000),
    MODEL_ZIMAGE_PID_2KTO4K: ManagedModelSpec(MODEL_ZIMAGE_PID_2KTO4K, "PiD 4x Decoder (2kto4k)", 18_000),
    MODEL_QWEN_PID_2KTO4K: ManagedModelSpec(MODEL_QWEN_PID_2KTO4K, "Qwen PiD 4x Decoder (2kto4k)", 18_000),
    MODEL_HIDREAM_O1_FULL: ManagedModelSpec(MODEL_HIDREAM_O1_FULL, "HiDream-O1-Image", 24_000),
    MODEL_HIDREAM_O1_DEV: ManagedModelSpec(MODEL_HIDREAM_O1_DEV, "HiDream-O1-Image-Dev", 24_000),
    MODEL_BOOGU_IMAGE_TURBO: ManagedModelSpec(
        MODEL_BOOGU_IMAGE_TURBO, "Boogu-Image-0.1-Turbo", 40_000, BOOGU_IMAGE_EXCLUSIVE_GROUP
    ),
    MODEL_BOOGU_IMAGE_BASE: ManagedModelSpec(
        MODEL_BOOGU_IMAGE_BASE, "Boogu-Image-0.1-Base", 40_000, BOOGU_IMAGE_EXCLUSIVE_GROUP
    ),
    MODEL_BOOGU_IMAGE_EDIT: ManagedModelSpec(
        MODEL_BOOGU_IMAGE_EDIT, "Boogu-Image-0.1-Edit", 40_000, BOOGU_IMAGE_EXCLUSIVE_GROUP
    ),
    MODEL_BOOGU_IMAGE_EDIT_TURBO: ManagedModelSpec(
        MODEL_BOOGU_IMAGE_EDIT_TURBO,
        "Boogu-Image-0.1-Edit-Turbo",
        40_000,
        BOOGU_IMAGE_EXCLUSIVE_GROUP,
    ),
    MODEL_KREA2_TURBO_NVFP4: ManagedModelSpec(
        MODEL_KREA2_TURBO_NVFP4, "Krea2 Turbo (ComfyUI)", 38_000, KREA2_EXCLUSIVE_GROUP
    ),
    MODEL_IDEOGRAM4_NVFP4: ManagedModelSpec(
        MODEL_IDEOGRAM4_NVFP4, "Ideogram 4 NVFP4", 22_000, IDEOGRAM4_EXCLUSIVE_GROUP
    ),
    MODEL_IDEOGRAM4_FP8_NVFP4_UNCOND: ManagedModelSpec(
        MODEL_IDEOGRAM4_FP8_NVFP4_UNCOND,
        "Ideogram 4 FP8/NVFP4 Uncond",
        24_000,
        IDEOGRAM4_EXCLUSIVE_GROUP,
    ),
    MODEL_IDEOGRAM4_FP8: ManagedModelSpec(
        MODEL_IDEOGRAM4_FP8, "Ideogram 4 FP8", 24_000, IDEOGRAM4_EXCLUSIVE_GROUP
    ),
    MODEL_IDEOGRAM4_PID_2K: ManagedModelSpec(
        MODEL_IDEOGRAM4_PID_2K, "Ideogram PiD 4x Decoder (2k)", 12_000
    ),
    MODEL_IDEOGRAM4_PID_2KTO4K: ManagedModelSpec(
        MODEL_IDEOGRAM4_PID_2KTO4K, "Ideogram PiD 4x Decoder (2kto4k)", 18_000
    ),
    MODEL_SEEDVR2: ManagedModelSpec(MODEL_SEEDVR2, "SeedVR2 Upscaler", 12_000),
    MODEL_GEMMA: ManagedModelSpec(MODEL_GEMMA, "Gemma 4 12B (google BF16)", 26_000),
    MODEL_GEMMA_CHAT: ManagedModelSpec(MODEL_GEMMA_CHAT, "Gemma 4 12B (Huihui NVFP4)", 12_000),
    MODEL_DIFFUSIONGEMMA_VLLM: ManagedModelSpec(
        MODEL_DIFFUSIONGEMMA_VLLM, "DiffusionGemma vLLM", 26_000
    ),
    MODEL_LTX_VIDEO: ManagedModelSpec(MODEL_LTX_VIDEO, "LTX-Web Video", 18_000),
}
BOOGU_IMAGE_MODEL_LOCATIONS = {
    MODEL_BOOGU_IMAGE_TURBO: APP_CONFIG.boogu.turbo_model,
    MODEL_BOOGU_IMAGE_BASE: APP_CONFIG.boogu.base_model,
    MODEL_BOOGU_IMAGE_EDIT: APP_CONFIG.boogu.edit_model,
    MODEL_BOOGU_IMAGE_EDIT_TURBO: APP_CONFIG.boogu.edit_turbo_model,
}

def _hf_repo_prefix(model_ref: str) -> str:
    parts = [part for part in str(model_ref or "").split("/") if part]
    return "/".join(parts[:2]) if len(parts) >= 2 else str(model_ref or "").strip()

def _looks_like_hf_repo_id(value: str) -> bool:
    value = (value or "").strip()
    if not value or "://" in value or "\\" in value:
        return False
    if value.startswith(("/", "./", "../", "~", "models/")):
        return False
    if Path(value).expanduser().exists():
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts)

def _configured_model_storage_location(value: str) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    value = (value or "").strip()
    if not value:
        return (), ()
    if _looks_like_hf_repo_id(value):
        return (), (value,)
    return (Path(value),), ()

def _boogu_storage_targets() -> list[ModelStorageTarget]:
    targets = []
    for key, model_name in BOOGU_IMAGE_MODEL_NAMES.items():
        paths, repos = _configured_model_storage_location(BOOGU_IMAGE_MODEL_LOCATIONS[key])
        if not paths and not repos:
            paths = (Path(BASE_DIR) / "models" / model_name,)
            repos = (f"Boogu/{model_name}",)
        targets.append(
            ModelStorageTarget(
                key=key,
                display_name=MODEL_SPECS[key].display_name,
                paths=paths,
                hf_repos=repos,
                active_model_keys=(key,),
            )
        )
    return targets

def _pid_storage_targets() -> list[ModelStorageTarget]:
    targets = []
    for backbone, checkpoint_specs in PID_CHECKPOINTS.items():
        vae_asset = PID_VAE_ASSETS.get(backbone)
        for spec in checkpoint_specs.values():
            paths = [Path(PID_DIR).joinpath(*spec.relative_checkpoint_path.split("/"))]
            if vae_asset:
                paths.append(Path(PID_DIR).joinpath(*vae_asset.split("/")))
            targets.append(
                ModelStorageTarget(
                    key=spec.registry_key,
                    display_name=MODEL_SPECS[spec.registry_key].display_name,
                    paths=tuple(paths),
                    active_model_keys=(spec.registry_key,),
                )
            )
    return targets

def _build_model_storage_catalog() -> ModelStorageCatalog:
    krea_models = Path(KREA2_COMFY_DIR) / "ComfyUI" / "models"
    targets: list[ModelStorageTarget] = [
        ModelStorageTarget(
            key=MODEL_GEN,
            display_name=MODEL_SPECS[MODEL_GEN].display_name,
            hf_repos=("Qwen/Qwen-Image", _hf_repo_prefix(GEN_MODEL)),
            active_model_keys=(MODEL_GEN,),
        ),
        ModelStorageTarget(
            key=MODEL_EDIT,
            display_name=MODEL_SPECS[MODEL_EDIT].display_name,
            hf_repos=("Qwen/Qwen-Image-Edit-2509", _hf_repo_prefix(EDIT_MODEL)),
            active_model_keys=(MODEL_EDIT,),
        ),
        ModelStorageTarget(
            key=MODEL_ZIMAGE_TURBO,
            display_name=MODEL_SPECS[MODEL_ZIMAGE_TURBO].display_name,
            hf_repos=(
                "Tongyi-MAI/Z-Image-Turbo",
                "nunchaku-tech/nunchaku-z-image-turbo",
            ),
            active_model_keys=(MODEL_ZIMAGE_TURBO,),
        ),
        ModelStorageTarget(
            key=MODEL_ZIMAGE_FULL,
            display_name=MODEL_SPECS[MODEL_ZIMAGE_FULL].display_name,
            hf_repos=("Tongyi-MAI/Z-Image",),
            active_model_keys=(MODEL_ZIMAGE_FULL,),
        ),
        ModelStorageTarget(
            key=MODEL_HIDREAM_O1_FULL,
            display_name=MODEL_SPECS[MODEL_HIDREAM_O1_FULL].display_name,
            hf_repos=(HIDREAM_O1_SPECS[MODEL_HIDREAM_O1_FULL].model_id,),
            active_model_keys=(MODEL_HIDREAM_O1_FULL,),
        ),
        ModelStorageTarget(
            key=MODEL_HIDREAM_O1_DEV,
            display_name=MODEL_SPECS[MODEL_HIDREAM_O1_DEV].display_name,
            hf_repos=(HIDREAM_O1_SPECS[MODEL_HIDREAM_O1_DEV].model_id,),
            active_model_keys=(MODEL_HIDREAM_O1_DEV,),
        ),
        ModelStorageTarget(
            key=MODEL_KREA2_TURBO_NVFP4,
            display_name=MODEL_SPECS[MODEL_KREA2_TURBO_NVFP4].display_name,
            paths=(
                krea_models / "diffusion_models" / "krea2_turbo_nvfp4.safetensors",
                krea_models / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors",
                krea_models / "vae" / "qwen_image_vae.safetensors",
            ),
            active_model_keys=(MODEL_KREA2_TURBO_NVFP4,),
        ),
        ModelStorageTarget(
            key=MODEL_IDEOGRAM4_NVFP4,
            display_name=MODEL_SPECS[MODEL_IDEOGRAM4_NVFP4].display_name,
            hf_repos=(IDEOGRAM4_NVFP4_WEIGHTS_REPO, IDEOGRAM4_NVFP4_CONFIG_REPO),
            active_model_keys=(MODEL_IDEOGRAM4_NVFP4,),
        ),
        ModelStorageTarget(
            key=MODEL_IDEOGRAM4_FP8_NVFP4_UNCOND,
            display_name=MODEL_SPECS[MODEL_IDEOGRAM4_FP8_NVFP4_UNCOND].display_name,
            hf_repos=(
                IDEOGRAM4_FP8_NVFP4_UNCOND_WEIGHTS_REPO,
                IDEOGRAM4_NVFP4_WEIGHTS_REPO,
            ),
            active_model_keys=(MODEL_IDEOGRAM4_FP8_NVFP4_UNCOND,),
        ),
        ModelStorageTarget(
            key=MODEL_IDEOGRAM4_FP8,
            display_name=MODEL_SPECS[MODEL_IDEOGRAM4_FP8].display_name,
            hf_repos=(IDEOGRAM4_FP8_WEIGHTS_REPO,),
            active_model_keys=(MODEL_IDEOGRAM4_FP8,),
        ),
        ModelStorageTarget(
            key="ideogram4_realism_lora",
            display_name="Ideogram Realism Engine LoRA",
            hf_repos=(IDEOGRAM4_REALISM_LORA_REPO,),
            active_model_keys=(
                MODEL_IDEOGRAM4_NVFP4,
                MODEL_IDEOGRAM4_FP8_NVFP4_UNCOND,
                MODEL_IDEOGRAM4_FP8,
            ),
        ),
        ModelStorageTarget(
            key=MODEL_SEEDVR2,
            display_name=MODEL_SPECS[MODEL_SEEDVR2].display_name,
            paths=(Path(SEEDVR2_DIR) / "models",),
            active_model_keys=(MODEL_SEEDVR2,),
        ),
        ModelStorageTarget(
            key=MODEL_GEMMA,
            display_name=MODEL_SPECS[MODEL_GEMMA].display_name,
            hf_repos=tuple(repo for repo in (GEMMA_MODEL_ID, GEMMA_ASSISTANT_MODEL_ID) if repo),
            active_model_keys=(MODEL_GEMMA,),
        ),
        ModelStorageTarget(
            key=MODEL_GEMMA_CHAT,
            display_name=MODEL_SPECS[MODEL_GEMMA_CHAT].display_name,
            hf_repos=tuple(repo for repo in (GEMMA_HUIHUI_MODEL_ID, GEMMA_NVFP4_ASSISTANT_MODEL_ID) if repo),
            active_model_keys=(MODEL_GEMMA_CHAT,),
        ),
        ModelStorageTarget(
            key=MODEL_DIFFUSIONGEMMA_VLLM,
            display_name=MODEL_SPECS[MODEL_DIFFUSIONGEMMA_VLLM].display_name,
            hf_repos=(DIFFUSIONGEMMA_VLLM_HF_MODEL or "nvidia/diffusiongemma-26B-A4B-it-NVFP4",),
            active_model_keys=(MODEL_DIFFUSIONGEMMA_VLLM,),
        ),
        ModelStorageTarget(
            key=MODEL_LTX_VIDEO,
            display_name=MODEL_SPECS[MODEL_LTX_VIDEO].display_name,
            paths=(
                Path(LTX_WEB_DIR) / "models",
                Path(LTX_WEB_DIR) / "checkpoints",
            ),
            active_model_keys=(MODEL_LTX_VIDEO,),
        ),
    ]
    targets.extend(_boogu_storage_targets())
    targets.extend(_pid_storage_targets())
    return ModelStorageCatalog(targets)

def _model_display_name(key: str) -> str:
    spec = MODEL_SPECS.get(key)
    return spec.display_name if spec else key

# ---------------------------------------------------------------------------
# Pipelines (lazy-loaded via ModelManager)
# ---------------------------------------------------------------------------


def _unload_pipe_to_cpu(pipe):
    """Move every submodule of a diffusers pipeline to CPU."""
    try:
        pipe.to("cpu")
    except Exception:
        for attr in dir(pipe):
            mod = getattr(pipe, attr, None)
            if isinstance(mod, torch.nn.Module):
                try:
                    mod.to("cpu")
                except Exception:
                    pass


def _unload_exclusive_peers(key: str) -> None:
    spec = MODEL_SPECS.get(key)
    if spec is None or not spec.exclusive_group:
        return
    for other_key, other_spec in MODEL_SPECS.items():
        if other_key == key or other_spec.exclusive_group != spec.exclusive_group:
            continue
        if model_mgr.is_loaded(other_key):
            model_mgr.unload(other_key)


def _load_managed_model(
    key: str,
    factory: Callable[[], Any],
    unload_fn_factory: Callable[[Any], Callable[[], None]] | None = None,
    vram_key: str | None = None,
) -> Any:
    """Return an already-loaded model or create/register it with ModelManager."""
    with _model_load_lock:
        spec = MODEL_SPECS[key]
        if vram_key is not None:
            estimate = MODEL_SPECS[vram_key]
            spec = ManagedModelSpec(
                key=spec.key,
                display_name=spec.display_name,
                vram_mb=estimate.vram_mb,
                exclusive_group=spec.exclusive_group,
            )
        make_unload = unload_fn_factory or (lambda loaded: lambda: _unload_pipe_to_cpu(loaded))
        return model_mgr.get_or_load(
            spec,
            factory,
            make_unload,
            specs=MODEL_SPECS,
        )


_runtime_pipelines_qwen = importlib.import_module('image_studio.pipelines.qwen')
export_module(_runtime_pipelines_qwen, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_qwen)












def validate_boogu_dims(w: int | None, h: int | None) -> tuple[int, int]:
    return pure_validate_boogu_dims(w, h)


















_runtime_pipelines_ideogram_pipeline = importlib.import_module('image_studio.pipelines.ideogram.pipeline')
export_module(_runtime_pipelines_ideogram_pipeline, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_ideogram_pipeline)






















_runtime_pipelines_ideogram_lora = importlib.import_module('image_studio.pipelines.ideogram.lora')
export_module(_runtime_pipelines_ideogram_lora, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_ideogram_lora)














































_runtime_pipelines_zimage = importlib.import_module('image_studio.pipelines.zimage')
export_module(_runtime_pipelines_zimage, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_zimage)










# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def timed_result(action: Callable[[], Any]) -> tuple[Any, float]:
    """Run an action and return its result plus elapsed wall-clock seconds."""
    start = time.time()
    return action(), time.time() - start


def ok_status(elapsed: float, *parts: Any) -> str:
    """Build the common Gradio status line used by image jobs."""
    fields = [f"OK **{elapsed:.1f}s**", *(str(part) for part in parts)]
    return " | ".join(fields)


def finalize_image_result(
    prefix: str,
    image: Image.Image,
    status: str,
    seed: int,
    always_seed: bool = False,
) -> tuple[str, str, str]:
    """Persist an output image and append seed information to its status."""
    preview_path, raw_path = save_output_image_pair(prefix, image)
    return preview_path, append_seed_status(status, seed, always=always_seed), raw_path
def validate_dims(w: int, h: int) -> tuple[int, int]:
    return pure_validate_dims(w, h)


def require_prompt(prompt: str, message: str = "Please enter a prompt.") -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error(message)
    return prompt


def normalize_seed(seed: int | float | str) -> int:
    return int(seed)


def make_cuda_generator(seed: int):
    if seed < 0:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device).manual_seed(seed)


output_store = OutputStore(OUTPUT_DIR)
_ideogram4_prompt_metadata_store = PromptMetadataStore(
    output_store.contained_path,
    suffix=IDEOGRAM4_PROMPT_METADATA_SUFFIX,
    schema_version=IDEOGRAM4_PROMPT_METADATA_SCHEMA_VERSION,
)


_runtime_storage_output_store = importlib.import_module('image_studio.storage.output_store')
export_module(_runtime_storage_output_store, globals())
_RUNTIME_MODULES.append(_runtime_storage_output_store)






































def append_seed_status(status: str, seed: int, always: bool = False) -> str:
    if always or seed >= 0:
        return f"{status} | seed {seed}"
    return status


def cleanup_old_files(directory: str, max_age_seconds: float):
    if not os.path.isdir(directory):
        return
    now = time.time()
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or now - os.path.getmtime(path) <= max_age_seconds:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def chat_temp_dir(cleanup: bool = True) -> str:
    directory = os.path.join(OUTPUT_DIR, ".chat_cache")
    os.makedirs(directory, exist_ok=True)
    if cleanup:
        cleanup_old_files(directory, 3600)
    return directory


def save_chat_attachment(image: Any, prefix: str) -> str:
    img = coerce_rgb_image(image)
    path = os.path.join(chat_temp_dir(), f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.png")
    img.save(path)
    return path


_runtime_generators_qwen = importlib.import_module('image_studio.generators.qwen')
export_module(_runtime_generators_qwen, globals())
_RUNTIME_MODULES.append(_runtime_generators_qwen)




_runtime_generators_boogu = importlib.import_module('image_studio.generators.boogu')
export_module(_runtime_generators_boogu, globals())
_RUNTIME_MODULES.append(_runtime_generators_boogu)


_runtime_generators_krea2 = importlib.import_module('image_studio.generators.krea2')
export_module(_runtime_generators_krea2, globals())
_RUNTIME_MODULES.append(_runtime_generators_krea2)










_runtime_generators_hidream = importlib.import_module('image_studio.generators.hidream')
export_module(_runtime_generators_hidream, globals())
_RUNTIME_MODULES.append(_runtime_generators_hidream)










def validate_ideogram4_dims(w: int, h: int) -> tuple[int, int]:
    return validate_ideogram_dims(w, h)


_runtime_generators_ideogram = importlib.import_module('image_studio.generators.ideogram')
export_module(_runtime_generators_ideogram, globals())
_RUNTIME_MODULES.append(_runtime_generators_ideogram)








_runtime_generators_zimage = importlib.import_module('image_studio.generators.zimage')
export_module(_runtime_generators_zimage, globals())
_RUNTIME_MODULES.append(_runtime_generators_zimage)




# ---------------------------------------------------------------------------
# SeedVR2 Upscale (self-contained, runs the pipeline in-process)
# ---------------------------------------------------------------------------

# Known model list (fallback when SeedVR2 isn't loaded yet for the dropdown)
SEEDVR2_DIT_MODELS = [
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_3b-Q4_K_M.gguf",
    "seedvr2_ema_3b-Q8_0.gguf",
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b-Q4_K_M.gguf",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
]
SEEDVR2_DEFAULT_DIT = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"










SEEDVR2_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v")






























LTX_DISTILLED_STEPS = 8
LTX_VIDEO_MAX_FRAMES = 1201
LTX_AUDIO_VIDEO_RESOLUTION_MULTIPLE = 128
LTX_IC_LORA_OFF = "Off"
LTX_IC_LORA_OPTIONS = {
    LTX_IC_LORA_OFF: None,
    "Ingredients / Reference Sheet": "ltx-2.3-22b-ic-lora-ingredients-0.9",
    "Union Control (Depth/Canny/Pose)": "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    "Motion Track Control": "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5",
    "Day to Night": "ltx-2.3-22b-ic-lora-day-to-night-0.9",
    "Colorization": "ltx-2.3-22b-ic-lora-colorization-0.9",
    "Decompression": "ltx-2.3-22b-ic-lora-decompression-0.9",
    "Deblur": "ltx-2.3-22b-ic-lora-deblur-0.9",
    "In / Outpainting": "ltx-2.3-22b-ic-lora-in-outpainting-0.9",
    "Water Simulation": "ltx-2.3-22b-ic-lora-water-simulation-0.9",
}
LTX_IC_LORA_CHOICES = list(LTX_IC_LORA_OPTIONS.keys())


_runtime_services_ltx_video = importlib.import_module('image_studio.services.ltx_video')
export_module(_runtime_services_ltx_video, globals())
_RUNTIME_MODULES.append(_runtime_services_ltx_video)























# ---------------------------------------------------------------------------
# Gemma 4 12B - prompt enhancement & multimodal chat
# ---------------------------------------------------------------------------
GEMMA_MODEL_ID = "google/gemma-4-12B-it"
GEMMA_ASSISTANT_MODEL_ID = APP_CONFIG.chat.assistant_model
GEMMA_NVFP4_ASSISTANT_MODEL_ID = APP_CONFIG.chat.nvfp4_assistant_model
GEMMA_NUM_ASSISTANT_TOKENS = APP_CONFIG.chat.assistant_tokens
GEMMA_MODEL_URL = "https://huggingface.co/google/gemma-4-12B-it"
GEMMA_HUIHUI_MODEL_ID = "sakamakismile/Huihui-gemma-4-12B-it-abliterated-NVFP4A16"
GEMMA_HUIHUI_MODEL_URL = "https://huggingface.co/sakamakismile/Huihui-gemma-4-12B-it-abliterated-NVFP4A16"
DIFFUSIONGEMMA_VLLM_SCRIPT = str(APP_CONFIG.vllm.script)
DIFFUSIONGEMMA_VLLM_BASH = APP_CONFIG.vllm.shell
DIFFUSIONGEMMA_VLLM_PORT = str(APP_CONFIG.vllm.port)
DIFFUSIONGEMMA_VLLM_API_BASE = APP_CONFIG.vllm.api_base
DIFFUSIONGEMMA_VLLM_MODEL = APP_CONFIG.vllm.model
DIFFUSIONGEMMA_VLLM_HF_MODEL = APP_CONFIG.vllm.hf_model
DIFFUSIONGEMMA_VLLM_READY_TIMEOUT = APP_CONFIG.vllm.ready_timeout
DIFFUSIONGEMMA_VLLM_START_TIMEOUT = APP_CONFIG.vllm.start_timeout
DIFFUSIONGEMMA_VLLM_REQUEST_TIMEOUT = APP_CONFIG.vllm.request_timeout
DIFFUSIONGEMMA_VLLM_RESTART_POLICY = APP_CONFIG.vllm.restart_policy
DIFFUSIONGEMMA_VLLM_WARMUP_ON_START = APP_CONFIG.vllm.warmup_on_start
DIFFUSIONGEMMA_VLLM_UNLOAD_MODE = APP_CONFIG.vllm.unload_mode
DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL = APP_CONFIG.vllm.sleep_level
KREA2_COMFY_SCRIPT = str(APP_CONFIG.krea2.script)
KREA2_COMFY_BASH = APP_CONFIG.krea2.shell
KREA2_COMFY_PORT = str(APP_CONFIG.krea2.port)
KREA2_COMFY_SERVER_BASE = APP_CONFIG.krea2.server_base
KREA2_COMFY_DIR = str(APP_CONFIG.krea2.directory)
KREA2_COMFY_READY_TIMEOUT = APP_CONFIG.krea2.ready_timeout
KREA2_COMFY_START_TIMEOUT = APP_CONFIG.krea2.start_timeout
KREA2_COMFY_REQUEST_TIMEOUT = APP_CONFIG.krea2.request_timeout
IMAGE_STUDIO_VLLM_PROXY_DEFAULT = APP_CONFIG.vllm.proxy_enabled
IMAGE_STUDIO_VLLM_PROXY_API_KEY = APP_CONFIG.vllm.proxy_api_key

CHAT_GEMMA_GOOGLE = "google"
CHAT_GEMMA_HUIHUI = "huihui"
CHAT_DIFFUSIONGEMMA_VLLM = "diffusiongemma_vllm"
CHAT_GEMMA_DEFAULT = APP_CONFIG.chat.default_model
CHAT_GEMMA_CHOICES = {
    CHAT_GEMMA_GOOGLE: "Google Gemma 4 12B-it (official)",
    CHAT_GEMMA_HUIHUI: "Huihui Gemma 4 NVFP4 (abliterated)",
    CHAT_DIFFUSIONGEMMA_VLLM: "DiffusionGemma vLLM (managed)",
}
CHAT_GEMMA_LABEL_TO_KEY = {label: key for key, label in CHAT_GEMMA_CHOICES.items()}


_runtime_pipelines_gemma = importlib.import_module('image_studio.pipelines.gemma')
export_module(_runtime_pipelines_gemma, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_gemma)
_chat_selector = ChatModelSelector(CHAT_GEMMA_DEFAULT)


GEMMA_MODEL_SPECS = {
    CHAT_GEMMA_GOOGLE: GemmaModelSpec(
        key=CHAT_GEMMA_GOOGLE,
        model_id=GEMMA_MODEL_ID,
        label="Gemma 4 12B-it",
        mgr_key=MODEL_GEMMA,
        vram_mb=MODEL_SPECS[MODEL_GEMMA].vram_mb,
        assistant_model_id=GEMMA_ASSISTANT_MODEL_ID,
    ),
    CHAT_GEMMA_HUIHUI: GemmaModelSpec(
        key=CHAT_GEMMA_HUIHUI,
        model_id=GEMMA_HUIHUI_MODEL_ID,
        label="Huihui Gemma 4 NVFP4",
        mgr_key=MODEL_GEMMA_CHAT,
        vram_mb=MODEL_SPECS[MODEL_GEMMA_CHAT].vram_mb,
        assistant_model_id=GEMMA_NVFP4_ASSISTANT_MODEL_ID,
        trust_remote_code=True,
    ),
}




_runtime_services_managed_runtime = importlib.import_module('image_studio.services.managed_runtime')
export_module(_runtime_services_managed_runtime, globals())
_RUNTIME_MODULES.append(_runtime_services_managed_runtime)




_runtime_services_vllm = importlib.import_module('image_studio.services.vllm')
export_module(_runtime_services_vllm, globals())
_RUNTIME_MODULES.append(_runtime_services_vllm)


_diffusiongemma_vllm_service = DiffusionGemmaVllmService()


_runtime_services_krea2_comfy = importlib.import_module('image_studio.services.krea2_comfy')
export_module(_runtime_services_krea2_comfy, globals())
_RUNTIME_MODULES.append(_runtime_services_krea2_comfy)


_krea2_comfy_service = Krea2ComfyService()
MODEL_STORAGE = _build_model_storage_catalog()

APP_CONTEXT = AppContext(
    config=APP_CONFIG,
    model_manager=model_mgr,
    output_store=output_store,
    ltx_video=_ltx_video_service,
    diffusiongemma=_diffusiongemma_vllm_service,
    krea2=_krea2_comfy_service,
    chat_selector=_chat_selector,
    model_load_lock=_model_load_lock,
    gpu_lock=_inprocess_gpu_lock,
)


_VLLM_PROXY_ROUTE_NAME = "image_studio_vllm_proxy"
_VLLM_PROXY_HEALTH_ROUTE_NAME = "image_studio_vllm_proxy_health"
_VLLM_PROXY_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


_runtime_web_proxy = importlib.import_module('image_studio.web.proxy')
export_module(_runtime_web_proxy, globals())
_RUNTIME_MODULES.append(_runtime_web_proxy)




















_runtime_web_designer = importlib.import_module('image_studio.web.designer')
export_module(_runtime_web_designer, globals())
_RUNTIME_MODULES.append(_runtime_web_designer)






















# ---- Ideogram 4 prompt upsampling ----
_ideogram4_upsample_cache = JsonFileCache(
    IDEOGRAM4_UPSAMPLE_CACHE_PATH,
    schema_version=IDEOGRAM4_UPSAMPLE_CACHE_SCHEMA_VERSION,
)


_runtime_pipelines_ideogram_prompting = importlib.import_module('image_studio.pipelines.ideogram.prompting')
export_module(_runtime_pipelines_ideogram_prompting, globals())
_RUNTIME_MODULES.append(_runtime_pipelines_ideogram_prompting)


















































# ---- Prompt Enhancement ----
_ENHANCE_SYSTEM = """
You are an expert AI image prompt engineering engine and visual creative director.
Rewrite the user's rough image request into a clear, detailed, self-contained English prompt that can be used directly by an image generation or image editing model.

Use the SCALIST framework when it helps the image:
- Subject: exact identity, appearance, colors, materials, textures, clothing, expression, pose.
- Composition: shot size, camera angle, foreground/midground/background, subject placement, focal point, negative space.
- Action: what is happening, direction of motion, gestures, interactions, causal relationships.
- Location: place, era, weather, time of day, environmental details.
- Image style: photorealistic, cinematic, anime, oil painting, watercolor, 3D render, product photo, UI mockup, etc. Choose lighting and color mood that match the request.
- Specs: lens, depth of field, focus, render/photography terms, material finish, sharpness.
- Text rendering: if the user requests visible text, preserve the exact requested text inside double quotes and specify font style, color, size, material, and precise placement.

Resolve implicit knowledge before rewriting. For poems, quotes, formulas, historical figures, landmarks, famous paintings, UI layouts, cultural symbols, scientific concepts, or real-world objects, explicitly describe the visible features the image model must draw. Do not leave vague references like "Mona Lisa", "freedom", or "Einstein equation" without concrete visual details.

Convert abstract ideas into visible symbols, scenery, lighting, color, and composition. Anchor spatial relationships with precise terms such as centered in the foreground, top-left corner, behind the subject, aligned along the bottom edge, background softly out of focus.

Output requirements:
- Output ONLY valid JSON, no Markdown fences and no extra text.
- JSON schema: {"prompt": "English single-paragraph prompt", "reasoning": "brief explanation of visual choices", "resolved_knowledge": "implicit knowledge resolved, or 'none'"}.
- The prompt should be one natural paragraph, usually 80-220 words. Simple requests can be shorter.
- Put the main subject and visual intent first, then composition, action, location, style, technical specs, and text rendering details.
- Do not include phrases that require the image model to infer missing facts.
""".strip()


_ENHANCE_VIDEO_SYSTEM = """
You are an expert AI video prompt engineering engine and visual creative director, specializing in prompts for LTX-2.3 video generation.
Rewrite the user's rough video request into a clear, detailed, self-contained English prompt.

Follow the LTX-2.3 optimal prompt structure:
1. Subject: What is happening, exact identity, appearance, actions.
2. Environment: Where the scene takes place, setting details.
3. Camera motion: How the camera moves (e.g., drone shot, slow motion, tracking shot, cinematic zoom, orbit shot, wide angle).
4. Style: Visual aesthetic (e.g., cinematic, ultra realistic, anime, epic movie style, futuristic, commercial).
5. Lighting: Mood and realism (e.g., dramatic lighting, golden hour, neon lights, soft lighting).

Output requirements:
- Output ONLY valid JSON, no Markdown fences and no extra text.
- JSON schema: {"prompt": "English single-paragraph prompt", "reasoning": "brief explanation of visual choices", "resolved_knowledge": "implicit knowledge resolved, or 'none'"}.
- The prompt should be highly descriptive and natural. Do not use bullet points in the prompt itself.
- Do not include phrases that require the video model to infer missing facts.
""".strip()

_runtime_generators_chat = importlib.import_module('image_studio.generators.chat')
export_module(_runtime_generators_chat, globals())
_RUNTIME_MODULES.append(_runtime_generators_chat)














# ---- Multimodal Chat ----
_CHAT_SYSTEM = (
    "You are a helpful, knowledgeable assistant. "
    "Respond concisely and accurately. When given images or audio, "
    "analyze them carefully before answering."
)
















# ---------------------------------------------------------------------------
# Model Manager UI helpers
# ---------------------------------------------------------------------------

_runtime_ui_models = importlib.import_module('image_studio.ui.models')
export_module(_runtime_ui_models, globals())
_RUNTIME_MODULES.append(_runtime_ui_models)










# ---------------------------------------------------------------------------
# CSS & Theme
# ---------------------------------------------------------------------------

CSS = load_css()

THEME = build_theme(gr)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_runtime_storage_gallery = importlib.import_module('image_studio.storage.gallery')
export_module(_runtime_storage_gallery, globals())
_RUNTIME_MODULES.append(_runtime_storage_gallery)
_runtime_ui_gallery_actions = importlib.import_module('image_studio.ui.gallery_actions')
export_module(_runtime_ui_gallery_actions, globals())
_RUNTIME_MODULES.append(_runtime_ui_gallery_actions)



MAX_GALLERY_IMAGES = 50
MAX_OUTPUT_FILES = 500
GALLERY_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
GALLERY_VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov")





API_DOCS = (Path(__file__).with_name("docs") / "api.md").read_text(encoding="utf-8")

_runtime_ui_components_generate = importlib.import_module('image_studio.ui.components.generate')
export_module(_runtime_ui_components_generate, globals())
_RUNTIME_MODULES.append(_runtime_ui_components_generate)


_runtime_ui_components_edit = importlib.import_module('image_studio.ui.components.edit')
export_module(_runtime_ui_components_edit, globals())
_RUNTIME_MODULES.append(_runtime_ui_components_edit)


TAB_GENERATE = 0
TAB_EDIT = 1
TAB_UPSCALE = 2
TAB_GALLERY = 3
TAB_CHAT = 4
TAB_MODELS = 5
TAB_AI_REMOVER = 6
TAB_VIDEO = 7


GEN_SIZE_BASE_PRESETS = {
    "1:1": (1024, 1024),
    "4:5": (768, 960),
    "5:4": (960, 768),
    "3:4": (768, 1024),
    "4:3": (1024, 768),
    "2:3": (768, 1152),
    "3:2": (1152, 768),
    "9:16": (576, 1024),
    "16:9": (1024, 576),
    "21:9": (1344, 576),
}
GEN_SIZE_SCALE_FACTORS = {
    "Small": 1.0,
    "Medium": 1.5,
    "Large": 2.0,
}
GEN_SIZE_PRESETS = {
    size: {
        aspect: (int(width * scale), int(height * scale))
        for aspect, (width, height) in GEN_SIZE_BASE_PRESETS.items()
    }
    for size, scale in GEN_SIZE_SCALE_FACTORS.items()
}
GEN_SIZE_ASPECT_CHOICES = list(GEN_SIZE_BASE_PRESETS.keys())










HIDREAM_MODE_KEYS = {
    "Best Quality": MODEL_HIDREAM_O1_FULL,
    "Dev": MODEL_HIDREAM_O1_DEV,
}








_runtime_generators_dispatch = importlib.import_module('image_studio.generators.dispatch')
export_module(_runtime_generators_dispatch, globals())
_RUNTIME_MODULES.append(_runtime_generators_dispatch)














IMAGE_BACKEND_REGISTRY = BackendRegistry()
IMAGE_BACKEND_REGISTRY.register(
    CallableBackend(
        id="local-gpu",
        label="In-process GPU runtime",
        max_concurrency=1,
    )
)
IMAGE_BACKEND_REGISTRY.register(
    CallableBackend(
        id="krea2-comfy",
        label="Krea2 ComfyUI",
        start_fn=_krea2_comfy_service.ensure_running,
        stop_fn=_krea2_comfy_service.stop,
        health_fn=_krea2_comfy_service.is_healthy,
        max_concurrency=1,
    )
)
IMAGE_MODEL_REGISTRY = build_image_model_registry(
    ImageModelFunctions(
        qwen_generate=run_generate,
        qwen_edit=run_edit,
        zimage_generate=run_zimage,
        zimage_full_generate=run_zimage_full,
        hidream_generate=run_hidream_generate,
        hidream_edit=run_hidream_edit,
        boogu_generate=run_boogu_generate,
        boogu_edit=run_boogu_edit,
        krea2_generate=run_krea2_generate,
        ideogram_generate=run_ideogram4_generate,
        hidream_model_keys=HIDREAM_MODE_KEYS,
    )
)
IMAGE_MODEL_EXECUTOR = ModelExecutor(IMAGE_MODEL_REGISTRY, IMAGE_BACKEND_REGISTRY)


# Deprecated compatibility registries. New dispatch goes through IMAGE_MODEL_EXECUTOR.
GENERATION_HANDLERS: dict[str, Callable[[GenerationRequest, Any], Any]] = {
    "Qwen Image": _run_qwen_generation,
    "Z-Image": _run_zimage_generation,
    HIDREAM_O1_MODE: _run_hidream_generation,
    IDEOGRAM4_MODE: _run_ideogram_generation,
    BOOGU_IMAGE_MODE: _run_boogu_generation,
    KREA2_MODE: _run_krea2_generation,
}
GENERATION_REGISTRY = RequestHandlerRegistry[GenerationRequest](_run_qwen_generation)
for _mode, _handler in GENERATION_HANDLERS.items():
    GENERATION_REGISTRY.register(_mode, _handler)
GENERATOR_MODES = [
    adapter.spec.display_name
    for adapter in IMAGE_MODEL_REGISTRY.for_operation(Operation.IMAGE_GENERATE)
]










EDIT_HANDLERS: dict[str, Callable[[EditRequest, Any], Any]] = {
    "Qwen Image Edit": _run_qwen_edit_request,
    HIDREAM_O1_MODE: _run_hidream_edit_request,
    BOOGU_IMAGE_MODE: _run_boogu_edit_request,
}
EDIT_REGISTRY = RequestHandlerRegistry[EditRequest](_run_qwen_edit_request)
for _mode, _handler in EDIT_HANDLERS.items():
    EDIT_REGISTRY.register(_mode, _handler)
EDITOR_MODES = [
    adapter.spec.display_name
    for adapter in IMAGE_MODEL_REGISTRY.for_operation(Operation.IMAGE_EDIT)
]










































_runtime_services_ai_remover = importlib.import_module('image_studio.services.ai_remover')
export_module(_runtime_services_ai_remover, globals())
_RUNTIME_MODULES.append(_runtime_services_ai_remover)




































_runtime_ui_layout = importlib.import_module('image_studio.ui.layout')
export_module(_runtime_ui_layout, globals())
_RUNTIME_MODULES.append(_runtime_ui_layout)






_runtime_ui_components_upscale = importlib.import_module('image_studio.ui.components.upscale')
export_module(_runtime_ui_components_upscale, globals())
_RUNTIME_MODULES.append(_runtime_ui_components_upscale)


_runtime_ui_components_chat = importlib.import_module('image_studio.ui.components.chat')
export_module(_runtime_ui_components_chat, globals())
_RUNTIME_MODULES.append(_runtime_ui_components_chat)


_runtime_ui_components_gallery = importlib.import_module('image_studio.ui.components.gallery')
export_module(_runtime_ui_components_gallery, globals())
_RUNTIME_MODULES.append(_runtime_ui_components_gallery)


_runtime_ui_components_models = importlib.import_module('image_studio.ui.components.models')
export_module(_runtime_ui_components_models, globals())
_RUNTIME_MODULES.append(_runtime_ui_components_models)


_runtime_ui_components_ai_remover = importlib.import_module('image_studio.ui.components.ai_remover')
export_module(_runtime_ui_components_ai_remover, globals())
_RUNTIME_MODULES.append(_runtime_ui_components_ai_remover)


VIDEO_QUICK_RATIO_CHOICES = [
    "3:2 Default",
    "2:3 Portrait",
    "16:9 HQ",
    "9:16 Vertical",
    "1:1 Square",
]
VIDEO_QUICK_RATIO_PRESETS = {
    "large": {
        "3:2": (768, 1152),
        "2:3": (1152, 768),
        "16:9": (640, 1152),
        "9:16": (1152, 640),
        "1:1": (1024, 1024),
    },
    "small": {
        "3:2": (512, 768),
        "2:3": (768, 512),
        "16:9": (384, 640),
        "9:16": (640, 384),
        "1:1": (512, 512),
    },
}


_runtime_ui_components_video = importlib.import_module('image_studio.ui.components.video')
export_module(_runtime_ui_components_video, globals())
_RUNTIME_MODULES.append(_runtime_ui_components_video)












_runtime_ui_wiring = importlib.import_module('image_studio.ui.wiring')
export_module(_runtime_ui_wiring, globals())
_RUNTIME_MODULES.append(_runtime_ui_wiring)
































IDEOGRAM_JSON_DESIGNER_OPEN_JS = (Path(__file__).with_name("ui") / "designer.js").read_text(encoding="utf-8")




































_runtime_web_routes = importlib.import_module('image_studio.web.routes')
export_module(_runtime_web_routes, globals())
_RUNTIME_MODULES.append(_runtime_web_routes)


rebind_modules(_RUNTIME_MODULES, globals())


def run_selftest() -> int:
    """Run the pytest compatibility suite without launching Gradio."""
    environment = os.environ.copy()
    environment["IMAGE_STUDIO_NO_BOOTSTRAP"] = "1"
    tests_dir = Path(__file__).resolve().parents[1] / "tests"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tests_dir)],
        cwd=BASE_DIR,
        env=environment,
        check=False,
    )
    return int(result.returncode)




def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=True, help="Enable Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--auth", help="user:password for basic auth")
    parser.add_argument(
        "--vllm-proxy",
        action=argparse.BooleanOptionalAction,
        default=IMAGE_STUDIO_VLLM_PROXY_DEFAULT,
        help="Expose the managed DiffusionGemma vLLM backend at /v1/* on the WebUI server (default: enabled)",
    )
    parser.add_argument(
        "--vllm-proxy-api-key",
        default=IMAGE_STUDIO_VLLM_PROXY_API_KEY,
        help="Optional bearer/API key required by the /v1/* vLLM proxy",
    )
    parser.add_argument("--selftest", action="store_true", help="Run the pytest compatibility suite without launching")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()

    auth = tuple(args.auth.split(":", 1)) if args.auth else None

    app = build_ui(APP_CONTEXT)
    app.queue(max_size=4, default_concurrency_limit=1)
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": args.port,
        "share": args.share,
        "auth": auth,
        "css": CSS,
        "theme": THEME,
    }
    fastapi_app = attach_app_routes(
        app,
        vllm_proxy=args.vllm_proxy,
        api_key=args.vllm_proxy_api_key,
        model_catalog_provider=IMAGE_MODEL_EXECUTOR.catalog,
    )
    if args.vllm_proxy and fastapi_app is not None:
        launch_kwargs["_app"] = fastapi_app
    if args.vllm_proxy and args.share and not args.vllm_proxy_api_key:
        log.warning(
            "vLLM proxy is enabled with Gradio share and no proxy API key. "
            "Use --vllm-proxy-api-key or IMAGE_STUDIO_VLLM_PROXY_API_KEY for public links."
        )
    app.launch(**launch_kwargs, prevent_thread_lock=True)
    attach_app_routes(
        app,
        vllm_proxy=args.vllm_proxy,
        api_key=args.vllm_proxy_api_key,
        model_catalog_provider=IMAGE_MODEL_EXECUTOR.catalog,
    )
    block_thread = getattr(app, "block_thread", None)
    if callable(block_thread):
        block_thread()
    else:
        while True:
            time.sleep(3600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
