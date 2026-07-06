"""Built-in image model adapters with model-specific parameter contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.models import (
    DataclassModelAdapter,
    ModelRegistry,
    ModelSpec,
    Operation,
    OperationBinding,
)


def _dimension(default: int = 1024):
    return field(default=default, metadata={"minimum": 256, "maximum": 4096})


def _seed():
    return field(default=-1, metadata={"description": "Use -1 to select a random seed."})


@dataclass(frozen=True)
class QwenGenerateParameters:
    prompt: str = ""
    neg_prompt: str = ""
    width: int = _dimension()
    height: int = _dimension()
    cfg: float = 1.0
    pid_enabled: bool = False
    pid_ckpt: str = "auto"
    pid_steps: int = 4
    pid_cfg: float = 1.0
    seed: int = _seed()


@dataclass(frozen=True)
class QwenEditParameters:
    img1: Any = None
    img2: Any = None
    img3: Any = None
    prompt: str = ""
    neg_prompt: str = ""
    cfg: float = 1.0
    seed: int = _seed()


@dataclass(frozen=True)
class ZImageGenerateParameters:
    prompt: str = ""
    neg_prompt: str = ""
    width: int = _dimension()
    height: int = _dimension()
    version: str = field(default="Turbo", metadata={"choices": ("Turbo", "Best Quality")})
    steps: int = 8
    guidance: float = 0.0
    full_steps: int = 50
    full_guidance: float = 4.0
    pid_enabled: bool = False
    pid_ckpt: str = "auto"
    pid_steps: int = 4
    pid_cfg: float = 1.0
    seed: int = _seed()


@dataclass(frozen=True)
class HiDreamGenerateParameters:
    prompt: str = ""
    width: int = _dimension()
    height: int = _dimension()
    version: str = field(default="Dev", metadata={"choices": ("Dev", "Best Quality")})
    seed: int = _seed()


@dataclass(frozen=True)
class HiDreamEditParameters:
    img1: Any = None
    img2: Any = None
    img3: Any = None
    prompt: str = ""
    width: int = _dimension()
    height: int = _dimension()
    keep_original_aspect: bool = True
    version: str = field(default="Dev", metadata={"choices": ("Dev", "Best Quality")})
    seed: int = _seed()


@dataclass(frozen=True)
class BooguGenerateParameters:
    prompt: str = ""
    neg_prompt: str = ""
    width: int = _dimension()
    height: int = _dimension()
    version: str = field(default="Turbo", metadata={"choices": ("Turbo", "Base")})
    steps: int = 4
    base_guidance: float = 4.0
    seed: int = _seed()


@dataclass(frozen=True)
class BooguEditParameters:
    img1: Any = None
    img2: Any = None
    img3: Any = None
    prompt: str = ""
    neg_prompt: str = ""
    version: str = field(default="Turbo", metadata={"choices": ("Turbo", "Base")})
    width: int = _dimension()
    height: int = _dimension()
    keep_original_aspect: bool = True
    steps: int = 4
    text_guidance: float = 4.0
    image_guidance: float = 1.0
    seed: int = _seed()


@dataclass(frozen=True)
class Krea2GenerateParameters:
    prompt: str = ""
    width: int = _dimension()
    height: int = _dimension()
    steps: int = 8
    cfg: float = 1.0
    pid_enabled: bool = False
    pid_ckpt: str = "auto"
    pid_steps: int = 4
    pid_cfg: float = 1.0
    seed: int = _seed()


@dataclass(frozen=True)
class IdeogramGenerateParameters:
    prompt: str = ""
    width: int = _dimension()
    height: int = _dimension()
    pipeline: str = "nvfp4 (fast)"
    sampler: str = "Turbo - 12 steps"
    upsampler: str = "Gemma 4 local"
    strip_prompt: bool = True
    reuse_cache: bool = True
    gemma_tokens: int = 2048
    gemma_thinking: bool = False
    cfg_one_final_steps: int = 0
    lora_mode: str = "Off"
    lora_weight: str = "Realism_Engine_Ideogram_V2.safetensors"
    lora_cond_strength: float = 0.9
    lora_uncond_strength: float = 0.4
    pid_enabled: bool = False
    pid_ckpt: str = "auto"
    pid_steps: int = 4
    pid_cfg: float = 1.0
    api_key: str = field(default="", metadata={"secret": True})
    seed: int = _seed()


@dataclass(frozen=True)
class ImageModelFunctions:
    qwen_generate: Callable[..., Any]
    qwen_edit: Callable[..., Any]
    zimage_generate: Callable[..., Any]
    zimage_full_generate: Callable[..., Any]
    hidream_generate: Callable[..., Any]
    hidream_edit: Callable[..., Any]
    boogu_generate: Callable[..., Any]
    boogu_edit: Callable[..., Any]
    krea2_generate: Callable[..., Any]
    ideogram_generate: Callable[..., Any]
    hidream_model_keys: Mapping[str, str]


def build_image_model_registry(functions: ImageModelFunctions) -> ModelRegistry:
    """Create the built-in registry without importing heavyweight model libraries."""
    registry = ModelRegistry()

    registry.register(
        DataclassModelAdapter(
            ModelSpec(
                id="qwen-image",
                display_name="Qwen Image",
                backend_id="local-gpu",
                operations=(Operation.IMAGE_GENERATE,),
                description="Qwen image generation through the in-process Diffusers pipeline.",
                order=10,
            ),
            {
                Operation.IMAGE_GENERATE: OperationBinding(
                    QwenGenerateParameters,
                    lambda p, progress: functions.qwen_generate(
                        p.prompt,
                        p.neg_prompt,
                        p.width,
                        p.height,
                        p.cfg,
                        p.pid_enabled,
                        p.pid_ckpt,
                        p.pid_steps,
                        p.pid_cfg,
                        p.seed,
                        progress=progress,
                    ),
                )
            },
        )
    )

    registry.register(
        DataclassModelAdapter(
            ModelSpec(
                id="qwen-image-edit",
                display_name="Qwen Image Edit",
                backend_id="local-gpu",
                operations=(Operation.IMAGE_EDIT,),
                order=10,
            ),
            {
                Operation.IMAGE_EDIT: OperationBinding(
                    QwenEditParameters,
                    lambda p, progress: functions.qwen_edit(
                        p.img1,
                        p.img2,
                        p.img3,
                        p.prompt,
                        p.neg_prompt,
                        p.cfg,
                        p.seed,
                        progress=progress,
                    ),
                )
            },
        )
    )

    def run_zimage(p: ZImageGenerateParameters, progress: Any):
        if p.version == "Turbo":
            return functions.zimage_generate(
                p.prompt,
                p.width,
                p.height,
                p.steps,
                p.guidance,
                p.pid_enabled,
                p.pid_ckpt,
                p.pid_steps,
                p.pid_cfg,
                p.seed,
                progress=progress,
            )
        return functions.zimage_full_generate(
            p.prompt,
            p.neg_prompt,
            p.width,
            p.height,
            p.full_steps,
            p.full_guidance,
            p.pid_enabled,
            p.pid_ckpt,
            p.pid_steps,
            p.pid_cfg,
            p.seed,
            progress=progress,
        )

    registry.register(
        DataclassModelAdapter(
            ModelSpec(
                id="z-image",
                display_name="Z-Image",
                backend_id="local-gpu",
                operations=(Operation.IMAGE_GENERATE,),
                order=20,
            ),
            {Operation.IMAGE_GENERATE: OperationBinding(ZImageGenerateParameters, run_zimage)},
        )
    )

    def hidream_key(version: str) -> str:
        return functions.hidream_model_keys.get(version, functions.hidream_model_keys["Dev"])

    registry.register(
        DataclassModelAdapter(
            ModelSpec(
                id="hidream-o1",
                display_name="HiDream-O1",
                backend_id="local-gpu",
                operations=(Operation.IMAGE_GENERATE, Operation.IMAGE_EDIT),
                order=30,
            ),
            {
                Operation.IMAGE_GENERATE: OperationBinding(
                    HiDreamGenerateParameters,
                    lambda p, progress: functions.hidream_generate(
                        p.prompt,
                        p.width,
                        p.height,
                        p.seed,
                        model_key=hidream_key(p.version),
                        progress=progress,
                    ),
                ),
                Operation.IMAGE_EDIT: OperationBinding(
                    HiDreamEditParameters,
                    lambda p, progress: functions.hidream_edit(
                        p.img1,
                        p.img2,
                        p.img3,
                        p.prompt,
                        p.width,
                        p.height,
                        p.keep_original_aspect,
                        p.seed,
                        model_key=hidream_key(p.version),
                        progress=progress,
                    ),
                ),
            },
        )
    )

    registry.register(
        DataclassModelAdapter(
            ModelSpec(
                id="boogu-image",
                display_name="Boogu-Image",
                backend_id="local-gpu",
                operations=(Operation.IMAGE_GENERATE, Operation.IMAGE_EDIT),
                order=50,
            ),
            {
                Operation.IMAGE_GENERATE: OperationBinding(
                    BooguGenerateParameters,
                    lambda p, progress: functions.boogu_generate(
                        p.prompt,
                        p.neg_prompt,
                        p.width,
                        p.height,
                        p.version,
                        p.steps,
                        p.base_guidance,
                        p.seed,
                        progress=progress,
                    ),
                ),
                Operation.IMAGE_EDIT: OperationBinding(
                    BooguEditParameters,
                    lambda p, progress: functions.boogu_edit(
                        p.img1,
                        p.img2,
                        p.img3,
                        p.prompt,
                        p.neg_prompt,
                        p.version,
                        p.width,
                        p.height,
                        p.keep_original_aspect,
                        p.steps,
                        p.text_guidance,
                        p.image_guidance,
                        p.seed,
                        progress=progress,
                    ),
                ),
            },
        )
    )

    registry.register(
        DataclassModelAdapter(
            ModelSpec(
                id="krea2",
                display_name="Krea2",
                backend_id="krea2-comfy",
                operations=(Operation.IMAGE_GENERATE,),
                aliases=("Krea2 Turbo",),
                order=60,
            ),
            {
                Operation.IMAGE_GENERATE: OperationBinding(
                    Krea2GenerateParameters,
                    lambda p, progress: functions.krea2_generate(
                        p.prompt,
                        p.width,
                        p.height,
                        p.steps,
                        p.cfg,
                        p.pid_enabled,
                        p.pid_ckpt,
                        p.pid_steps,
                        p.pid_cfg,
                        p.seed,
                        progress=progress,
                    ),
                )
            },
        )
    )

    registry.register(
        DataclassModelAdapter(
            ModelSpec(
                id="ideogram-4",
                display_name="Ideogram 4",
                backend_id="local-gpu",
                operations=(Operation.IMAGE_GENERATE,),
                order=40,
            ),
            {
                Operation.IMAGE_GENERATE: OperationBinding(
                    IdeogramGenerateParameters,
                    lambda p, progress: functions.ideogram_generate(
                        p.prompt,
                        p.width,
                        p.height,
                        p.pipeline,
                        p.sampler,
                        p.upsampler,
                        p.strip_prompt,
                        p.reuse_cache,
                        p.gemma_tokens,
                        p.gemma_thinking,
                        p.cfg_one_final_steps,
                        p.lora_mode,
                        p.lora_weight,
                        p.lora_cond_strength,
                        p.lora_uncond_strength,
                        p.pid_enabled,
                        p.pid_ckpt,
                        p.pid_steps,
                        p.pid_cfg,
                        p.api_key,
                        p.seed,
                        progress=progress,
                    ),
                )
            },
        )
    )
    return registry


__all__ = (
    "BooguEditParameters",
    "BooguGenerateParameters",
    "HiDreamEditParameters",
    "HiDreamGenerateParameters",
    "IdeogramGenerateParameters",
    "ImageModelFunctions",
    "Krea2GenerateParameters",
    "QwenEditParameters",
    "QwenGenerateParameters",
    "ZImageGenerateParameters",
    "build_image_model_registry",
)
