"""Extracted runtime implementation."""

from __future__ import annotations
from dataclasses import asdict

from image_studio.core.models import Operation
from image_studio.progress import NO_PROGRESS

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _get_seedvr2_model_options() -> tuple[list[str], str, bool]:
    if not is_seedvr2_available():
        return SEEDVR2_DIT_MODELS, SEEDVR2_DEFAULT_DIT, False
    try:
        s = _get_seedvr2()
        models = s["get_available_dit_models"]()
        default = SEEDVR2_DEFAULT_DIT if SEEDVR2_DEFAULT_DIT in models else s["DEFAULT_DIT"]
        return models, default, True
    except Exception:
        return SEEDVR2_DIT_MODELS, SEEDVR2_DEFAULT_DIT, False

def _run_zimage_generation(req: GenerationRequest, progress=NO_PROGRESS):
    if req.zimage_version == "Turbo":
        return run_zimage(
            req.prompt, req.width, req.height, req.steps, req.guidance,
            req.full_pid_enabled, req.full_pid_ckpt, req.full_pid_steps, req.full_pid_cfg,
            req.seed, progress=progress,
        )
    return run_zimage_full(
        req.prompt, req.neg_prompt, req.width, req.height,
        req.full_steps, req.full_guidance,
        req.full_pid_enabled, req.full_pid_ckpt, req.full_pid_steps, req.full_pid_cfg,
        req.seed, progress=progress,
    )

def _run_hidream_generation(req: GenerationRequest, progress=NO_PROGRESS):
    model_key = HIDREAM_MODE_KEYS.get(req.hidream_version, MODEL_HIDREAM_O1_DEV)
    return run_hidream_generate(
        req.prompt, req.width, req.height, req.seed,
        model_key=model_key, progress=progress,
    )

def _run_ideogram_generation(req: GenerationRequest, progress=NO_PROGRESS):
    return run_ideogram4_generate(
        req.prompt, req.width, req.height,
        req.ideogram_pipeline, req.ideogram_sampler, req.ideogram_upsampler,
        req.ideogram_strip_prompt, req.ideogram_reuse_cache,
        req.ideogram_gemma_tokens, req.ideogram_gemma_thinking,
        req.ideogram_cfg_one_final_steps,
        req.ideogram_lora_mode, req.ideogram_lora_weight,
        req.ideogram_lora_cond_strength, req.ideogram_lora_uncond_strength,
        req.full_pid_enabled, req.full_pid_ckpt, req.full_pid_steps, req.full_pid_cfg,
        req.ideogram_api_key, req.seed, progress=progress,
    )

def _run_boogu_generation(req: GenerationRequest, progress=NO_PROGRESS):
    return run_boogu_generate(
        req.prompt, req.neg_prompt, req.width, req.height,
        req.boogu_version, req.boogu_steps, req.boogu_base_guidance,
        req.seed, progress=progress,
    )

def _run_krea2_generation(req: GenerationRequest, progress=NO_PROGRESS):
    return run_krea2_generate(
        req.prompt, req.width, req.height,
        req.krea2_steps, req.krea2_cfg,
        req.full_pid_enabled, req.full_pid_ckpt, req.full_pid_steps, req.full_pid_cfg,
        req.seed, progress=progress,
    )

def _run_qwen_generation(req: GenerationRequest, progress=NO_PROGRESS):
    return run_generate(
        req.prompt, req.neg_prompt, req.width, req.height, req.cfg,
        req.full_pid_enabled, req.full_pid_ckpt, req.full_pid_steps, req.full_pid_cfg,
        req.seed, progress=progress,
    )

def _run_generation_request(req: GenerationRequest, progress=NO_PROGRESS):
    values = asdict(req)
    common = {
        "prompt": req.prompt,
        "width": req.width,
        "height": req.height,
        "seed": req.seed,
        "pid_enabled": req.full_pid_enabled,
        "pid_ckpt": req.full_pid_ckpt,
        "pid_steps": req.full_pid_steps,
        "pid_cfg": req.full_pid_cfg,
    }
    model_parameters = {
        "Qwen Image": {
            **common,
            "neg_prompt": req.neg_prompt,
            "cfg": req.cfg,
        },
        "Z-Image": {
            **common,
            "neg_prompt": req.neg_prompt,
            "version": req.zimage_version,
            "steps": req.steps,
            "guidance": req.guidance,
            "full_steps": req.full_steps,
            "full_guidance": req.full_guidance,
        },
        HIDREAM_O1_MODE: {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "version": req.hidream_version,
            "seed": req.seed,
        },
        BOOGU_IMAGE_MODE: {
            "prompt": req.prompt,
            "neg_prompt": req.neg_prompt,
            "width": req.width,
            "height": req.height,
            "version": req.boogu_version,
            "steps": req.boogu_steps,
            "base_guidance": req.boogu_base_guidance,
            "seed": req.seed,
        },
        KREA2_MODE: {
            **common,
            "steps": req.krea2_steps,
            "cfg": req.krea2_cfg,
        },
        IDEOGRAM4_MODE: {
            **common,
            "pipeline": req.ideogram_pipeline,
            "sampler": req.ideogram_sampler,
            "upsampler": req.ideogram_upsampler,
            "strip_prompt": req.ideogram_strip_prompt,
            "reuse_cache": req.ideogram_reuse_cache,
            "gemma_tokens": req.ideogram_gemma_tokens,
            "gemma_thinking": req.ideogram_gemma_thinking,
            "cfg_one_final_steps": req.ideogram_cfg_one_final_steps,
            "lora_mode": req.ideogram_lora_mode,
            "lora_weight": req.ideogram_lora_weight,
            "lora_cond_strength": req.ideogram_lora_cond_strength,
            "lora_uncond_strength": req.ideogram_lora_uncond_strength,
            "api_key": req.ideogram_api_key,
        },
    }
    parameters = model_parameters.get(req.mode, values)
    return IMAGE_MODEL_EXECUTOR.execute(
        req.mode,
        Operation.IMAGE_GENERATE,
        parameters,
        progress,
        strict=req.mode in model_parameters,
    )

def _run_hidream_edit_request(req: EditRequest, progress=NO_PROGRESS):
    model_key = HIDREAM_MODE_KEYS.get(req.hidream_version, MODEL_HIDREAM_O1_DEV)
    return run_hidream_edit(
        req.img1, req.img2, req.img3, req.prompt, req.width, req.height,
        req.keep_original_aspect, req.hidream_seed,
        model_key=model_key, progress=progress,
    )

def _run_boogu_edit_request(req: EditRequest, progress=NO_PROGRESS):
    return run_boogu_edit(
        req.img1, req.img2, req.img3, req.prompt, req.neg_prompt,
        req.boogu_version,
        req.width, req.height, req.keep_original_aspect,
        req.boogu_steps, req.boogu_text_guidance, req.boogu_image_guidance, req.boogu_seed,
        progress=progress,
    )

def _run_qwen_edit_request(req: EditRequest, progress=NO_PROGRESS):
    return run_edit(
        req.img1, req.img2, req.img3, req.prompt, req.neg_prompt,
        req.cfg, req.qwen_seed, progress=progress,
    )

def _run_edit_request(req: EditRequest, progress=NO_PROGRESS):
    common = {
        "img1": req.img1,
        "img2": req.img2,
        "img3": req.img3,
        "prompt": req.prompt,
    }
    model_parameters = {
        "Qwen Image Edit": {
            **common,
            "neg_prompt": req.neg_prompt,
            "cfg": req.cfg,
            "seed": req.qwen_seed,
        },
        HIDREAM_O1_MODE: {
            **common,
            "width": req.width,
            "height": req.height,
            "keep_original_aspect": req.keep_original_aspect,
            "version": req.hidream_version,
            "seed": req.hidream_seed,
        },
        BOOGU_IMAGE_MODE: {
            **common,
            "neg_prompt": req.neg_prompt,
            "version": req.boogu_version,
            "width": req.width,
            "height": req.height,
            "keep_original_aspect": req.keep_original_aspect,
            "steps": req.boogu_steps,
            "text_guidance": req.boogu_text_guidance,
            "image_guidance": req.boogu_image_guidance,
            "seed": req.boogu_seed,
        },
    }
    parameters = model_parameters.get(req.model_name, asdict(req))
    return IMAGE_MODEL_EXECUTOR.execute(
        req.model_name,
        Operation.IMAGE_EDIT,
        parameters,
        progress,
        strict=req.model_name in model_parameters,
    )

def _dispatch_generate(
    mode, prompt, neg_prompt, width, height,
    cfg, steps, guidance, full_steps, full_guidance,
    full_pid_enabled, full_pid_ckpt, full_pid_steps, full_pid_cfg,
    boogu_version, boogu_steps, boogu_base_guidance,
    krea2_steps, krea2_cfg,
    ideogram_pipeline, ideogram_sampler, ideogram_upsampler, ideogram_strip_prompt,
    ideogram_reuse_cache, ideogram_gemma_tokens, ideogram_gemma_thinking,
    ideogram_cfg_one_final_steps,
    ideogram_lora_mode, ideogram_lora_weight,
    ideogram_lora_cond_strength, ideogram_lora_uncond_strength,
    ideogram_api_key,
    seed, zimage_version, hidream_version, progress=NO_PROGRESS,
):
    """Route generation requests to the model-specific endpoint."""
    req = GenerationRequest.from_mapping(locals())
    return _run_generation_request(req, progress=progress)

def _dispatch_edit(
    model_name, img1, img2, img3, prompt, neg_prompt, cfg, qwen_seed,
    boogu_version, boogu_steps, boogu_text_guidance, boogu_image_guidance, boogu_seed,
    width, height, keep_original_aspect, hidream_seed, hidream_version,
    progress=NO_PROGRESS,
):
    req = EditRequest.from_mapping(locals())
    return _run_edit_request(req, progress=progress)

__all__ = (
    '_get_seedvr2_model_options',
    '_run_zimage_generation',
    '_run_hidream_generation',
    '_run_ideogram_generation',
    '_run_boogu_generation',
    '_run_krea2_generation',
    '_run_qwen_generation',
    '_run_generation_request',
    '_run_hidream_edit_request',
    '_run_boogu_edit_request',
    '_run_qwen_edit_request',
    '_run_edit_request',
    '_dispatch_generate',
    '_dispatch_edit',
)
_seal_runtime_module(globals())
