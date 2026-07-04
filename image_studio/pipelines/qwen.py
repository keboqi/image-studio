"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def get_gen_pipe():
    def factory():
        require_nunchaku()
        log.info("Loading Qwen Image gen pipeline (precision=%s)...", _PRECISION)
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(GEN_MODEL)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
        pipe = QwenImagePipeline.from_pretrained(
            "Qwen/Qwen-Image", transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16,
        )
        if get_gpu_memory() > 18:
            pipe.to("cuda")
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipe._exclude_from_cpu_offload.append("transformer")
            pipe.enable_sequential_cpu_offload()
        log.info("Gen pipeline ready.")
        return pipe

    return _load_managed_model(MODEL_GEN, factory)

def get_edit_pipe():
    def factory():
        require_nunchaku()
        log.info("Loading Qwen Image Edit pipeline (precision=%s)...", _PRECISION)
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(EDIT_MODEL)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", transformer=transformer, torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        log.info("Edit pipeline ready.")
        return pipe

    return _load_managed_model(MODEL_EDIT, factory)

__all__ = (
    'get_gen_pipe',
    'get_edit_pipe',
)
_seal_runtime_module(globals())
