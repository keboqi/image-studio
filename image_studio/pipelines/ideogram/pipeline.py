"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import UserInputError
from image_studio.infra.lazy_modules import LazyModuleGroup

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def enable_ideogram4_cuda_fast_paths():
    """Enable PyTorch CUDA attention fast paths when available."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    for backend_flag in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
        "enable_cudnn_sdp",
    ):
        flag_setter = getattr(torch.backends.cuda, backend_flag, None)
        if flag_setter is None:
            continue
        try:
            flag_setter(True)
        except Exception as exc:
            log.warning("Could not enable torch.backends.cuda.%s: %s", backend_flag, exc)

def make_ideogram4_sdpa_context():
    """Prefer fused SDPA backends while preserving math fallback."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = [
            getattr(SDPBackend, name)
            for name in (
                "CUDNN_ATTENTION",
                "FLASH_ATTENTION",
                "EFFICIENT_ATTENTION",
                "MATH",
            )
            if hasattr(SDPBackend, name)
        ]
        if not backends:
            return contextlib.nullcontext()
        try:
            return sdpa_kernel(backends, set_priority=True)
        except TypeError:
            try:
                return sdpa_kernel(backends, set_priority_order=True)
            except TypeError:
                return sdpa_kernel(backends)
    except Exception:
        return contextlib.nullcontext()

def _import_ideogram4():
    """Lazy-import the local Ideogram 4 checkout."""
    pipeline_file = os.path.join(IDEOGRAM4_SRC_DIR, "ideogram4", "pipeline_ideogram4.py")
    if not os.path.isfile(pipeline_file):
        raise UserInputError(
            "Ideogram 4 source is not installed. Run the quick start setup to clone "
            f"{IDEOGRAM4_REPO} into {IDEOGRAM4_DIR}."
        )
    if IDEOGRAM4_SRC_DIR not in sys.path:
        sys.path.insert(0, IDEOGRAM4_SRC_DIR)

    try:
        from ideogram4 import PRESETS, Ideogram4Pipeline, Ideogram4PipelineConfig
        from ideogram4.magic_prompt import (
            MAGIC_PROMPTS,
            build_messages,
            reorder_caption_keys,
            strip_aspect_ratio,
            strip_aspect_ratio_and_bboxes,
        )
    except Exception as exc:
        raise UserInputError(
            "Ideogram 4 imports failed. Install the Ideogram quick-start pip dependencies "
            "on the server, then restart this WebUI."
        ) from exc

    return {
        "PRESETS": PRESETS,
        "Ideogram4Pipeline": Ideogram4Pipeline,
        "Ideogram4PipelineConfig": Ideogram4PipelineConfig,
        "MAGIC_PROMPTS": MAGIC_PROMPTS,
        "build_messages": build_messages,
        "reorder_caption_keys": reorder_caption_keys,
        "strip_aspect_ratio": strip_aspect_ratio,
        "strip_aspect_ratio_and_bboxes": strip_aspect_ratio_and_bboxes,
    }


_ideogram_modules = LazyModuleGroup("Ideogram 4", lambda: True, _import_ideogram4)


def _get_ideogram4():
    return _ideogram_modules.get()

def build_ideogram_manual_upsampler_messages(prompt, width, height):
    """Return the exact prompt-guide messages used by the local Gemma upsampler."""
    try:
        width = int(width)
        height = int(height)
        d = math.gcd(width, height) or 1
        aspect_ratio = f"{width // d}:{height // d}"
        
        mods = _get_ideogram4()
        build_messages = mods["build_messages"]
        messages = build_messages("v1.txt", prompt, aspect_ratio)
        return json.dumps(messages, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error building JSON prompt: {str(e)}"

def _ideogram4_config_subfolder(default_subfolder: str) -> str:
    repo = str(IDEOGRAM4_NVFP4_CONFIG_REPO).strip().lower()
    if repo == "qwen/qwen3-vl-8b-instruct":
        return ""
    return default_subfolder

def _normalize_ideogram4_pipeline(pipeline: str | None) -> str:
    value = str(pipeline or "").strip()
    if value in IDEOGRAM4_PIPELINE_LABEL_TO_KEY:
        return IDEOGRAM4_PIPELINE_LABEL_TO_KEY[value]
    if value in IDEOGRAM4_PIPELINE_LABELS:
        return value
    lowered = value.lower()
    for label, key in IDEOGRAM4_PIPELINE_LABEL_TO_KEY.items():
        if lowered == label.lower():
            return key
    return IDEOGRAM4_DEFAULT_PIPELINE

def _ideogram4_pipeline_label(pipeline: str | None) -> str:
    return IDEOGRAM4_PIPELINE_LABELS[_normalize_ideogram4_pipeline(pipeline)]

def _ideogram4_pipeline_config(pipeline: str | None = None):
    pipeline = _normalize_ideogram4_pipeline(pipeline)
    mods = _get_ideogram4()
    config_cls = mods["Ideogram4PipelineConfig"]
    if pipeline == IDEOGRAM4_PIPELINE_NVFP4:
        return config_cls(
            weights_repo=IDEOGRAM4_NVFP4_WEIGHTS_REPO,
            conditional_index_filename=IDEOGRAM4_NVFP4_CONDITIONAL_FILENAME,
            unconditional_index_filename=IDEOGRAM4_NVFP4_UNCONDITIONAL_FILENAME,
            autoencoder_filename=IDEOGRAM4_NVFP4_AUTOENCODER_FILENAME,
            tokenizer_repo=IDEOGRAM4_NVFP4_CONFIG_REPO,
            text_encoder_config_repo=IDEOGRAM4_NVFP4_CONFIG_REPO,
            tokenizer_subfolder=_ideogram4_config_subfolder("tokenizer"),
            text_encoder_subfolder=_ideogram4_config_subfolder("text_encoder"),
            text_encoder_weights_repo=IDEOGRAM4_NVFP4_WEIGHTS_REPO,
            text_encoder_weights_filename=IDEOGRAM4_NVFP4_TEXT_ENCODER_FILENAME,
        )
    if pipeline == IDEOGRAM4_PIPELINE_FP8_NVFP4_UNCOND:
        return config_cls(
            weights_repo=IDEOGRAM4_FP8_NVFP4_UNCOND_WEIGHTS_REPO,
            unconditional_weights_repo=IDEOGRAM4_NVFP4_WEIGHTS_REPO,
            unconditional_index_filename=IDEOGRAM4_NVFP4_UNCONDITIONAL_FILENAME,
        )
    return config_cls(weights_repo=IDEOGRAM4_FP8_WEIGHTS_REPO)

def _unload_ideogram4_pipe(pipe):
    for attr in ("conditional_transformer", "unconditional_transformer", "text_encoder", "autoencoder"):
        module = getattr(pipe, attr, None)
        if module is not None and hasattr(module, "to"):
            try:
                module.to("cpu")
            except Exception:
                pass

def get_ideogram4_pipe(pipeline: str | None = None):
    pipeline = _normalize_ideogram4_pipeline(pipeline)
    model_key = IDEOGRAM4_PIPELINE_MODEL_KEYS[pipeline]
    existing = model_mgr.get(model_key)
    if existing is not None:
        return existing

    def factory():
        if not torch.cuda.is_available():
            raise UserInputError("Ideogram 4 requires CUDA on the server.")
        mods = _get_ideogram4()
        enable_ideogram4_cuda_fast_paths()
        log.info("Loading Ideogram 4 %s pipeline...", _ideogram4_pipeline_label(pipeline))
        pipe = mods["Ideogram4Pipeline"].from_pretrained(
            config=_ideogram4_pipeline_config(pipeline),
            device="cuda",
            dtype=torch.bfloat16,
        )
        log.info("Ideogram 4 %s pipeline ready.", _ideogram4_pipeline_label(pipeline))
        return pipe

    return _load_managed_model(
        model_key,
        factory,
        unload_fn_factory=lambda pipe: lambda: _unload_ideogram4_pipe(pipe),
    )

__all__ = (
    'enable_ideogram4_cuda_fast_paths',
    'make_ideogram4_sdpa_context',
    '_get_ideogram4',
    'build_ideogram_manual_upsampler_messages',
    '_ideogram4_config_subfolder',
    '_normalize_ideogram4_pipeline',
    '_ideogram4_pipeline_label',
    '_ideogram4_pipeline_config',
    '_unload_ideogram4_pipe',
    'get_ideogram4_pipe',
)
_seal_runtime_module(globals())
