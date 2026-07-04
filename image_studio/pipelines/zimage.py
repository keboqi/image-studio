"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

_zimage_cfg = None

def get_zimage_pipe():
    global _zimage_cfg
    precision = _PRECISION
    dtype = torch.float16 if is_turing() else torch.bfloat16
    rank = 128
    cfg = (rank, precision, str(dtype))

    existing = model_mgr.get(MODEL_ZIMAGE_TURBO)
    if existing is not None and _zimage_cfg == cfg:
        return existing

    # Config changed - unload the old one first if present
    if model_mgr.is_loaded(MODEL_ZIMAGE_TURBO):
        model_mgr.unload(MODEL_ZIMAGE_TURBO)

    def factory():
        require_nunchaku()
        log.info(
            "Loading Z-Image Turbo pipeline (precision=%s, rank=%s, dtype=%s)...",
            precision, rank, dtype,
        )
        transformer = NunchakuZImageTransformer2DModel.from_pretrained(
            f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors",
            torch_dtype=dtype,
        )
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            transformer=transformer,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        log.info("Z-Image Turbo pipeline ready.")
        return pipe

    pipe = _load_managed_model(MODEL_ZIMAGE_TURBO, factory)
    _zimage_cfg = cfg
    return pipe

def get_zimage_full_pipe():
    """Load the full (non-distilled) Z-Image pipeline for best quality.

    This uses the original Tongyi-MAI/Z-Image weights at bfloat16 precision
    (no nunchaku quantization). It supports full CFG, negative prompts, and
    produces the highest quality images at the cost of slower inference."""
    
    if model_mgr.is_loaded(MODEL_ZIMAGE_TURBO):
        model_mgr.unload(MODEL_ZIMAGE_TURBO)
        
    def factory():
        log.info("Loading Z-Image (full, best quality) pipeline at bfloat16...")
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        if get_gpu_memory() > 18:
            pipe.to("cuda")
        else:
            pipe.enable_sequential_cpu_offload()
        log.info("Z-Image (full) pipeline ready.")
        return pipe

    return _load_managed_model(MODEL_ZIMAGE_FULL, factory)

__all__ = (
    'get_zimage_pipe',
    'get_zimage_full_pipe',
)
_seal_runtime_module(globals())
