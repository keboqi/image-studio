"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
from image_studio.runtime_access import runtime_namespace as _runtime

_zimage_cfg = None

def get_zimage_pipe():
    global _zimage_cfg
    precision = _runtime()._PRECISION
    dtype = _runtime().torch.float16 if _runtime().is_turing() else _runtime().torch.bfloat16
    rank = 128
    cfg = (rank, precision, str(dtype))

    existing = _runtime().model_mgr.get(_runtime().MODEL_ZIMAGE_TURBO)
    if existing is not None and _zimage_cfg == cfg:
        return existing

    # Config changed - unload the old one first if present
    if _runtime().model_mgr.is_loaded(_runtime().MODEL_ZIMAGE_TURBO):
        _runtime().model_mgr.unload(_runtime().MODEL_ZIMAGE_TURBO)

    def factory():
        _runtime().require_nunchaku()
        _runtime().log.info(
            "Loading Z-Image Turbo pipeline (precision=%s, rank=%s, dtype=%s)...",
            precision, rank, dtype,
        )
        transformer = _runtime().NunchakuZImageTransformer2DModel.from_pretrained(
            f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors",
            torch_dtype=dtype,
        )
        pipe = _runtime().ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            transformer=transformer,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        _runtime().log.info("Z-Image Turbo pipeline ready.")
        return pipe

    pipe = _runtime()._load_managed_model(_runtime().MODEL_ZIMAGE_TURBO, factory)
    _zimage_cfg = cfg
    return pipe

def get_zimage_full_pipe():
    """Load the full (non-distilled) Z-Image pipeline for best quality.

    This uses the original Tongyi-MAI/Z-Image weights at bfloat16 precision
    (no nunchaku quantization). It supports full CFG, negative prompts, and
    produces the highest quality images at the cost of slower inference."""
    
    if _runtime().model_mgr.is_loaded(_runtime().MODEL_ZIMAGE_TURBO):
        _runtime().model_mgr.unload(_runtime().MODEL_ZIMAGE_TURBO)
        
    def factory():
        _runtime().log.info("Loading Z-Image (full, best quality) pipeline at bfloat16...")
        pipe = _runtime().ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image",
            torch_dtype=_runtime().torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        if _runtime().get_gpu_memory() > 18:
            pipe.to("cuda")
        else:
            pipe.enable_sequential_cpu_offload()
        _runtime().log.info("Z-Image (full) pipeline ready.")
        return pipe

    return _runtime()._load_managed_model(_runtime().MODEL_ZIMAGE_FULL, factory)

__all__ = (
    'get_zimage_pipe',
    'get_zimage_full_pipe',
)
