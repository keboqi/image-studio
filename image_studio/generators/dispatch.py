"""Typed image workflow dispatch with legacy Gradio endpoint adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from image_studio.core.executor import ModelExecutor
from image_studio.core.models import Operation
from image_studio.progress import NO_PROGRESS

from .base import EditRequest, GenerationRequest

GENERATION_MODEL_IDS = {
    "Qwen Image": "qwen-image",
    "Z-Image": "z-image",
    "HiDream-O1": "hidream-o1",
    "SenseNova U1.5": "sensenova-u1.5",
    "Ideogram 4": "ideogram-4",
    "Boogu-Image": "boogu-image",
    "Krea2": "krea2",
}

EDIT_MODEL_IDS = {
    "Qwen Image Edit": "qwen-image-edit",
    "HiDream-O1": "hidream-o1",
    "SenseNova U1.5": "sensenova-u1.5",
    "Boogu-Image": "boogu-image",
}

SEEDVR2_DIT_MODELS = [
    "seedvr2_distill_6L_1.4B_sharp_fp16.safetensors",
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
SEEDVR2_FAST_DIT = "seedvr2_distill_6L_1.4B_sharp_fp16.safetensors"
SEEDVR2_DEFAULT_DIT = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"

_executor: ModelExecutor | None = None
_seedvr2_available: Callable[[], bool] | None = None
_seedvr2_loader: Callable[[], dict[str, Any]] | None = None


def configure_dispatch(
    executor: ModelExecutor,
    *,
    seedvr2_available: Callable[[], bool],
    seedvr2_loader: Callable[[], dict[str, Any]],
) -> None:
    """Inject application-owned dispatch collaborators at composition time."""
    global _executor, _seedvr2_available, _seedvr2_loader
    _executor = executor
    _seedvr2_available = seedvr2_available
    _seedvr2_loader = seedvr2_loader


def _get_seedvr2_model_options() -> tuple[list[str], str, bool]:
    if _seedvr2_available is None or _seedvr2_loader is None:
        return SEEDVR2_DIT_MODELS, SEEDVR2_DEFAULT_DIT, False
    if not _seedvr2_available():
        return SEEDVR2_DIT_MODELS, SEEDVR2_DEFAULT_DIT, False
    try:
        seedvr2 = _seedvr2_loader()
        models = seedvr2["get_available_dit_models"]()
        default = (
            SEEDVR2_DEFAULT_DIT
            if SEEDVR2_DEFAULT_DIT in models
            else seedvr2["DEFAULT_DIT"]
        )
        return models, default, True
    except Exception:
        return SEEDVR2_DIT_MODELS, SEEDVR2_DEFAULT_DIT, False


def generation_parameters(request: GenerationRequest) -> dict[str, Any]:
    common = {
        "prompt": request.prompt,
        "width": request.width,
        "height": request.height,
        "seed": request.seed,
        "pid_enabled": request.full_pid_enabled,
        "pid_ckpt": request.full_pid_ckpt,
        "pid_steps": request.full_pid_steps,
        "pid_cfg": request.full_pid_cfg,
    }
    by_model = {
        "qwen-image": {
            **common,
            "neg_prompt": request.neg_prompt,
            "cfg": request.cfg,
        },
        "z-image": {
            **common,
            "neg_prompt": request.neg_prompt,
            "version": request.zimage_version,
            "steps": request.steps,
            "guidance": request.guidance,
            "full_steps": request.full_steps,
            "full_guidance": request.full_guidance,
        },
        "hidream-o1": {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "version": request.hidream_version,
            "seed": request.seed,
        },
        "sensenova-u1.5": {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "quality": request.hidream_version,
            "seed": request.seed,
        },
        "boogu-image": {
            "prompt": request.prompt,
            "neg_prompt": request.neg_prompt,
            "width": request.width,
            "height": request.height,
            "version": request.boogu_version,
            "steps": request.boogu_steps,
            "base_guidance": request.boogu_base_guidance,
            "seed": request.seed,
        },
        "krea2": {
            **common,
            "steps": request.krea2_steps,
            "cfg": request.krea2_cfg,
        },
        "ideogram-4": {
            **common,
            "pipeline": request.ideogram_pipeline,
            "sampler": request.ideogram_sampler,
            "upsampler": request.ideogram_upsampler,
            "strip_prompt": request.ideogram_strip_prompt,
            "reuse_cache": request.ideogram_reuse_cache,
            "gemma_tokens": request.ideogram_gemma_tokens,
            "gemma_thinking": request.ideogram_gemma_thinking,
            "cfg_one_final_steps": request.ideogram_cfg_one_final_steps,
            "lora_mode": request.ideogram_lora_mode,
            "lora_weight": request.ideogram_lora_weight,
            "lora_cond_strength": request.ideogram_lora_cond_strength,
            "lora_uncond_strength": request.ideogram_lora_uncond_strength,
            "api_key": request.ideogram_api_key,
        },
    }
    model_id = GENERATION_MODEL_IDS.get(request.mode, request.mode)
    return by_model.get(model_id, asdict(request))


def edit_parameters(request: EditRequest) -> dict[str, Any]:
    common = {
        "img1": request.img1,
        "img2": request.img2,
        "img3": request.img3,
        "prompt": request.prompt,
    }
    by_model = {
        "qwen-image-edit": {
            **common,
            "neg_prompt": request.neg_prompt,
            "cfg": request.cfg,
            "seed": request.qwen_seed,
        },
        "hidream-o1": {
            **common,
            "width": request.width,
            "height": request.height,
            "keep_original_aspect": request.keep_original_aspect,
            "version": request.hidream_version,
            "seed": request.hidream_seed,
        },
        "sensenova-u1.5": {
            **common,
            "width": request.width,
            "height": request.height,
            "keep_original_aspect": request.keep_original_aspect,
            "quality": request.hidream_version,
            "seed": request.hidream_seed,
        },
        "boogu-image": {
            **common,
            "neg_prompt": request.neg_prompt,
            "version": request.boogu_version,
            "width": request.width,
            "height": request.height,
            "keep_original_aspect": request.keep_original_aspect,
            "steps": request.boogu_steps,
            "text_guidance": request.boogu_text_guidance,
            "image_guidance": request.boogu_image_guidance,
            "seed": request.boogu_seed,
        },
    }
    model_id = EDIT_MODEL_IDS.get(request.model_name, request.model_name)
    return by_model.get(model_id, asdict(request))


def run_generation_request(
    executor: ModelExecutor,
    request: GenerationRequest,
    progress: Any = NO_PROGRESS,
) -> Any:
    model_id = GENERATION_MODEL_IDS.get(request.mode, request.mode)
    return executor.execute(
        model_id,
        Operation.IMAGE_GENERATE,
        generation_parameters(request),
        progress,
        strict=model_id in GENERATION_MODEL_IDS.values(),
    )


def run_edit_request(
    executor: ModelExecutor,
    request: EditRequest,
    progress: Any = NO_PROGRESS,
) -> Any:
    model_id = EDIT_MODEL_IDS.get(request.model_name, request.model_name)
    return executor.execute(
        model_id,
        Operation.IMAGE_EDIT,
        edit_parameters(request),
        progress,
        strict=model_id in EDIT_MODEL_IDS.values(),
    )


def _run_generation_request(
    request: GenerationRequest,
    progress: Any = NO_PROGRESS,
) -> Any:
    if _executor is None:
        raise RuntimeError("Image model executor is not configured.")
    return run_generation_request(_executor, request, progress)


def _run_edit_request(request: EditRequest, progress: Any = NO_PROGRESS) -> Any:
    if _executor is None:
        raise RuntimeError("Image model executor is not configured.")
    return run_edit_request(_executor, request, progress)


def _dispatch_generate(
    mode,
    prompt,
    neg_prompt,
    width,
    height,
    cfg,
    steps,
    guidance,
    full_steps,
    full_guidance,
    full_pid_enabled,
    full_pid_ckpt,
    full_pid_steps,
    full_pid_cfg,
    boogu_version,
    boogu_steps,
    boogu_base_guidance,
    krea2_steps,
    krea2_cfg,
    ideogram_pipeline,
    ideogram_sampler,
    ideogram_upsampler,
    ideogram_strip_prompt,
    ideogram_reuse_cache,
    ideogram_gemma_tokens,
    ideogram_gemma_thinking,
    ideogram_cfg_one_final_steps,
    ideogram_lora_mode,
    ideogram_lora_weight,
    ideogram_lora_cond_strength,
    ideogram_lora_uncond_strength,
    ideogram_api_key,
    seed,
    zimage_version,
    hidream_version,
    progress=NO_PROGRESS,
):
    """Dispatch the stable flat Gradio generation endpoint."""
    request = GenerationRequest.from_mapping(locals())
    return _run_generation_request(request, progress)


def _dispatch_edit(
    model_name,
    img1,
    img2,
    img3,
    prompt,
    neg_prompt,
    cfg,
    qwen_seed,
    boogu_version,
    boogu_steps,
    boogu_text_guidance,
    boogu_image_guidance,
    boogu_seed,
    width,
    height,
    keep_original_aspect,
    hidream_seed,
    hidream_version,
    progress=NO_PROGRESS,
):
    """Dispatch the stable flat Gradio edit endpoint."""
    request = EditRequest.from_mapping(locals())
    return _run_edit_request(request, progress)


__all__ = (
    "EDIT_MODEL_IDS",
    "GENERATION_MODEL_IDS",
    "SEEDVR2_DEFAULT_DIT",
    "SEEDVR2_DIT_MODELS",
    "SEEDVR2_FAST_DIT",
    "_dispatch_edit",
    "_dispatch_generate",
    "_get_seedvr2_model_options",
    "_run_edit_request",
    "_run_generation_request",
    "configure_dispatch",
    "edit_parameters",
    "generation_parameters",
    "run_edit_request",
    "run_generation_request",
)
