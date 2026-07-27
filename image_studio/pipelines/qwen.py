"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
from image_studio.runtime_access import runtime_namespace as _runtime


def get_gen_pipe():
    def factory():
        _runtime().require_nunchaku()
        _runtime().log.info("Loading Qwen Image gen pipeline (precision=%s)...", _runtime()._PRECISION)
        transformer = _runtime().NunchakuQwenImageTransformer2DModel.from_pretrained(_runtime().GEN_MODEL)
        scheduler = _runtime().FlowMatchEulerDiscreteScheduler.from_config(_runtime().LIGHTNING_SCHEDULER)
        pipe = _runtime().QwenImagePipeline.from_pretrained(
            "Qwen/Qwen-Image", transformer=transformer, scheduler=scheduler, torch_dtype=_runtime().torch.bfloat16,
        )
        if _runtime().get_gpu_memory() > 18:
            pipe.to("cuda")
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipe._exclude_from_cpu_offload.append("transformer")
            pipe.enable_sequential_cpu_offload()
        _runtime().log.info("Gen pipeline ready.")
        return pipe

    return _runtime()._load_managed_model(_runtime().MODEL_GEN, factory)

def get_edit_pipe():
    def factory():
        _runtime().require_nunchaku()
        _runtime().log.info("Loading Qwen Image Edit pipeline (precision=%s)...", _runtime()._PRECISION)
        transformer = _runtime().NunchakuQwenImageTransformer2DModel.from_pretrained(_runtime().EDIT_MODEL)
        pipe = _runtime().QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", transformer=transformer, torch_dtype=_runtime().torch.bfloat16,
        )
        pipe.to("cuda")
        _runtime().log.info("Edit pipeline ready.")
        return pipe

    return _runtime()._load_managed_model(_runtime().MODEL_EDIT, factory)

__all__ = (
    'get_gen_pipe',
    'get_edit_pipe',
)
