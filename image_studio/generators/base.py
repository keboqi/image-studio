"""Stable interfaces for image generators."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Protocol


class UIRequest:
    """Build typed requests from flat Gradio component mappings."""

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(item.name for item in fields(cls))

    @classmethod
    def from_mapping(cls, values: dict[str, Any]):
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def component_inputs(
        cls,
        components: dict[str, Any],
        aliases: dict[str, str] | None = None,
    ) -> list[Any]:
        aliases = aliases or {}
        return [components[aliases.get(name, name)] for name in cls.field_names()]


@dataclass(frozen=True)
class GenerationRequest(UIRequest):
    mode: str
    prompt: str
    neg_prompt: str
    width: int
    height: int
    cfg: float
    steps: int
    guidance: float
    full_steps: int
    full_guidance: float
    full_pid_enabled: bool
    full_pid_ckpt: str
    full_pid_steps: int
    full_pid_cfg: float
    boogu_version: str
    boogu_steps: int
    boogu_base_guidance: float
    krea2_steps: int
    krea2_cfg: float
    ideogram_pipeline: str
    ideogram_sampler: str
    ideogram_upsampler: str
    ideogram_strip_prompt: bool
    ideogram_reuse_cache: bool
    ideogram_gemma_tokens: int
    ideogram_gemma_thinking: bool
    ideogram_cfg_one_final_steps: int
    ideogram_lora_mode: str
    ideogram_lora_weight: str
    ideogram_lora_cond_strength: float
    ideogram_lora_uncond_strength: float
    ideogram_api_key: str
    seed: int
    zimage_version: str
    hidream_version: str


@dataclass(frozen=True)
class EditRequest(UIRequest):
    model_name: str
    img1: Any
    img2: Any
    img3: Any
    prompt: str
    neg_prompt: str
    cfg: float
    qwen_seed: int
    boogu_version: str
    boogu_steps: int
    boogu_text_guidance: float
    boogu_image_guidance: float
    boogu_seed: int
    width: int
    height: int
    keep_original_aspect: bool
    hidream_seed: int
    hidream_version: str


@dataclass(frozen=True)
class ImageResult:
    image: Any
    status_parts: list[str]
    elapsed: float
    seed: int
    prefix: str
    metadata: dict[str, Any] | None = None


class Generator(Protocol):
    mode: str

    def generate(self, request: GenerationRequest, progress: Any) -> ImageResult: ...
