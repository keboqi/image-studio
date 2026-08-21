"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
from dataclasses import dataclass
from typing import Any

from image_studio.runtime_access import runtime_namespace as _runtime
from image_studio.ui.components.base import ComponentSet

GEN_VISIBILITY_ORDER = (
    "negative", "qwen", "zimage", "zimage_turbo", "zimage_full", "pid",
    "hidream", "variant", "ideogram", "boogu", "krea2", "boogu_base", "boogu_steps", "pid_checkpoint",
)


def _gen_mode_visibility_updates(mode: str, zimage_version: str, hidream_version: str, boogu_version: str):
    """Return Gradio visibility updates for generator-specific controls."""
    mode = mode or "Qwen Image"
    zimage_version = zimage_version or "Turbo"
    hidream_version = hidream_version or "Dev"
    boogu_version = _runtime()._normalize_boogu_generation_version(boogu_version)
    
    qwen = mode == "Qwen Image"
    zimage = mode == "Z-Image"
    turbo = zimage and zimage_version == "Turbo"
    zfull = zimage and zimage_version == "Best Quality"
    hidream = mode == _runtime().HIDREAM_O1_MODE
    sensenova = mode == _runtime().SENSENOVA_MODE
    ideogram = mode == _runtime().IDEOGRAM4_MODE
    boogu = mode == _runtime().BOOGU_IMAGE_MODE
    krea2 = mode == _runtime().KREA2_MODE
    variant_choices = (
        _runtime().SENSENOVA_QUALITY_CHOICES
        if sensenova else ["Dev", "Best Quality"]
    )
    variant_default = (
        _runtime().SENSENOVA_QUALITY_FAST if sensenova else "Dev"
    )
    variant_value = (
        hidream_version if hidream_version in variant_choices else variant_default
    )
    boogu_base = boogu and boogu_version == _runtime().BOOGU_IMAGE_VERSION_BASE
    zimage_family = turbo or zfull
    pid_capable = qwen or zimage_family or ideogram or krea2

    if qwen or krea2:
        pid_ckpt_choices = _runtime().PID_QWEN_CKPT_CHOICES
    elif ideogram:
        pid_ckpt_choices = _runtime().PID_IDEOGRAM4_CKPT_CHOICES
    else:
        pid_ckpt_choices = _runtime().PID_ZIMAGE_CKPT_CHOICES

    if boogu_base:
        boogu_steps_update = _runtime().gr.update(
            visible=True,
            minimum=25,
            maximum=50,
            value=_runtime().BOOGU_IMAGE_BASE_DEFAULT_STEPS,
            label="Steps (Base)",
        )
    else:
        boogu_steps_update = _runtime().gr.update(
            visible=boogu,
            minimum=3,
            maximum=8,
            value=_runtime().BOOGU_IMAGE_TURBO_DEFAULT_STEPS,
            label="Steps (Turbo)",
        )

    updates = {
        "negative": _runtime().gr.update(visible=qwen or zfull or boogu_base),
        "qwen": _runtime().gr.update(visible=qwen),
        "zimage": _runtime().gr.update(visible=zimage),
        "zimage_turbo": _runtime().gr.update(visible=turbo),
        "zimage_full": _runtime().gr.update(visible=zfull),
        "pid": _runtime().gr.update(visible=pid_capable),
        "hidream": _runtime().gr.update(visible=hidream or sensenova),
        "variant": _runtime().gr.update(
            choices=variant_choices,
            value=variant_value,
            label=("Inference preset" if sensenova else "HiDream-O1 Version"),
        ),
        "ideogram": _runtime().gr.update(visible=ideogram),
        "boogu": _runtime().gr.update(visible=boogu),
        "krea2": _runtime().gr.update(visible=krea2),
        "boogu_base": _runtime().gr.update(visible=boogu_base),
        "boogu_steps": boogu_steps_update,
        "pid_checkpoint": _runtime().gr.update(choices=pid_ckpt_choices, value=_runtime().PID_CKPT_AUTO),
    }
    return tuple(updates[name] for name in GEN_VISIBILITY_ORDER)

def _apply_gen_size_preset(size: str, aspect: str):
    dims = _runtime().GEN_SIZE_PRESETS.get(size, {}).get(aspect or "")
    if dims is None:
        return _runtime().gr.update(), _runtime().gr.update(), _runtime().gr.update(), _runtime().gr.update(), _runtime().gr.update()
    width, height = dims
    return (
        _runtime().gr.update(value=aspect if size == "Small" else None),
        _runtime().gr.update(value=aspect if size == "Medium" else None),
        _runtime().gr.update(value=aspect if size == "Large" else None),
        _runtime().gr.update(value=width),
        _runtime().gr.update(value=height),
    )

def _apply_gen_small_size_preset(aspect: str):
    return _apply_gen_size_preset("Small", aspect)

def _apply_gen_medium_size_preset(aspect: str):
    return _apply_gen_size_preset("Medium", aspect)

def _apply_gen_large_size_preset(aspect: str):
    return _apply_gen_size_preset("Large", aspect)

@dataclass
class GenerateTab(ComponentSet):
    mode: Any
    prompt: Any
    enhance_btn: Any
    size_preset_small: Any
    size_preset_medium: Any
    size_preset_large: Any
    negative: Any
    width: Any
    height: Any
    cfg: Any
    steps: Any
    guidance: Any
    full_steps: Any
    full_guidance: Any
    full_pid_enabled: Any
    full_pid_ckpt: Any
    full_pid_steps: Any
    full_pid_cfg: Any
    boogu_version: Any
    boogu_steps: Any
    boogu_base_guidance: Any
    krea2_steps: Any
    krea2_cfg: Any
    ideogram_pipeline: Any
    ideogram_sampler: Any
    ideogram_upsampler: Any
    ideogram_strip_prompt: Any
    ideogram_reuse_cache: Any
    ideogram_gemma_tokens: Any
    ideogram_gemma_thinking: Any
    ideogram_cfg_one_final_steps: Any
    ideogram_lora_mode: Any
    ideogram_lora_weight: Any
    ideogram_lora_cond_strength: Any
    ideogram_lora_uncond_strength: Any
    ideogram_api_key: Any
    ideogram_open_designer_btn: Any
    ideogram_designer_payload: Any
    ideogram_build_prompt_btn: Any
    ideogram_raw_prompt: Any
    seed: Any
    zimage_version: Any
    hidream_version: Any
    button: Any
    output: Any
    raw: Any
    status: Any
    to_edit: Any
    to_upscale: Any
    to_ai_remover: Any
    to_video: Any


def _build_generate_tab() -> GenerateTab:
    with _runtime().gr.Tab("Generate", id=_runtime().TAB_GENERATE):
        with _runtime().gr.Row(equal_height=False):
            with _runtime().gr.Column(scale=5):
                with _runtime().gr.Row():
                    gen_mode = _runtime().gr.Radio(
                        _runtime().GENERATOR_MODES,
                        value="Z-Image",
                        label="Generator",
                        elem_id="gen-mode",
                    )
                gen_prompt = _runtime().gr.Textbox(
                    label="Prompt", lines=4,
                    placeholder="Describe the image you want to create",
                    elem_id="gen-prompt",
                )
                gen_enhance_btn = _runtime().gr.Button(
                    "Enhance Prompt (Gemma 4)", size="sm",
                    elem_classes=["enhance-btn"],
                )
                gen_btn = _runtime().gr.Button("Generate", variant="primary", elem_id="gen-btn")
                with _runtime().gr.Accordion("Generation Parameters (Advanced)", open=False):
                    with _runtime().gr.Group(visible=False, elem_id="gen-neg-group") as gen_neg_group:
                        gen_neg = _runtime().gr.Textbox(label="Negative Prompt", lines=1, value="")
                    with _runtime().gr.Row():
                        gen_size_small = _runtime().gr.Dropdown(
                            _runtime().GEN_SIZE_ASPECT_CHOICES,
                            value="1:1",
                            label="Small",
                        )
                        gen_size_medium = _runtime().gr.Dropdown(
                            _runtime().GEN_SIZE_ASPECT_CHOICES,
                            value=None,
                            label="Medium",
                        )
                        gen_size_large = _runtime().gr.Dropdown(
                            _runtime().GEN_SIZE_ASPECT_CHOICES,
                            value=None,
                            label="Large",
                        )
                    with _runtime().gr.Row():
                        gen_w = _runtime().gr.Slider(256, 4096, 1024, step=32, label="Width")
                        gen_h = _runtime().gr.Slider(256, 4096, 1024, step=32, label="Height")
                    with _runtime().gr.Row(visible=False, elem_id="gen-lightning-group") as gen_lightning_group:
                        gen_cfg = _runtime().gr.Slider(0.5, 5.0, 1.0, step=0.1, label="CFG Scale")
                    with _runtime().gr.Group(visible=True, elem_id="gen-zimage-group") as gen_zimage_group:
                        _runtime().gr.Markdown("**Z-Image Settings**")
                        gen_zimage_version = _runtime().gr.Dropdown(["Turbo", "Best Quality"], value="Turbo", label="Z-Image Version")
                        with _runtime().gr.Row(visible=True, elem_id="gen-zimage-turbo-group") as gen_zimage_turbo_group:
                            gen_steps = _runtime().gr.Slider(4, 16, 8, step=1, label="Steps (DiT forwards)")
                            gen_guidance = _runtime().gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Guidance (Turbo=0)")
                        with _runtime().gr.Row(visible=False, elem_id="gen-zimage-full-group") as gen_zimage_full_group:
                            gen_full_steps = _runtime().gr.Slider(20, 60, 30, step=1, label="Steps")
                            gen_full_guidance = _runtime().gr.Slider(1.0, 8.0, 4.0, step=0.25, label="Guidance Scale (CFG)")
                    with _runtime().gr.Group(visible=False, elem_id="gen-hidream-group") as gen_hidream_group:
                        gen_hidream_version = _runtime().gr.Dropdown(["Dev", "Best Quality"], value="Dev", label="HiDream-O1 Version")
                    with _runtime().gr.Group(visible=False, elem_id="gen-boogu-group") as gen_boogu_group:
                        _runtime().gr.Markdown("**Boogu-Image Settings**")
                        gen_boogu_version = _runtime().gr.Dropdown(
                            _runtime().BOOGU_IMAGE_GENERATION_VERSIONS,
                            value=_runtime().BOOGU_IMAGE_VERSION_TURBO,
                            label="Boogu-Image Version",
                        )
                        gen_boogu_steps = _runtime().gr.Slider(
                            3,
                            8,
                            _runtime().BOOGU_IMAGE_TURBO_DEFAULT_STEPS,
                            step=1,
                            label="Steps (Turbo)",
                        )
                        with _runtime().gr.Row(visible=False, elem_id="gen-boogu-base-group") as gen_boogu_base_group:
                            gen_boogu_base_guidance = _runtime().gr.Slider(2.0, 5.0, 4.0, step=0.25, label="Text Guidance")
                    with _runtime().gr.Group(visible=False, elem_id="gen-krea2-group") as gen_krea2_group:
                        _runtime().gr.Markdown("**Krea2 Turbo (ComfyUI)**")
                        with _runtime().gr.Row():
                            gen_krea2_steps = _runtime().gr.Slider(
                                1,
                                16,
                                _runtime().KREA2_DEFAULT_STEPS,
                                step=1,
                                label="Steps",
                            )
                            gen_krea2_cfg = _runtime().gr.Slider(
                                0.0,
                                5.0,
                                _runtime().KREA2_DEFAULT_CFG,
                                step=0.1,
                                label="CFG (1 = disabled)",
                            )
                    with _runtime().gr.Group(visible=True, elem_id="gen-zimage-pid-group") as gen_zimage_pid_group:
                        _runtime().gr.Markdown("**PiD 4x Decode**")
                        with _runtime().gr.Row():
                            gen_full_pid_enabled = _runtime().gr.Checkbox(value=False, label="PiD 4x Decode")
                            gen_full_pid_ckpt = _runtime().gr.Dropdown(
                                _runtime().PID_ZIMAGE_CKPT_CHOICES,
                                value=_runtime().PID_CKPT_AUTO,
                                label="PiD Checkpoint",
                            )
                        with _runtime().gr.Row():
                            gen_full_pid_steps = _runtime().gr.Slider(1, 8, 4, step=1, label="PiD Steps")
                            gen_full_pid_cfg = _runtime().gr.Slider(0.0, 4.0, 1.0, step=0.25, label="PiD CFG")
                    with _runtime().gr.Group(visible=False, elem_id="gen-ideogram-group") as gen_ideogram_group:
                        _runtime().gr.Markdown("**Ideogram 4**")
                        with _runtime().gr.Row():
                            gen_ideogram_pipeline = _runtime().gr.Dropdown(
                                _runtime().IDEOGRAM4_PIPELINE_CHOICES,
                                value=_runtime().IDEOGRAM4_PIPELINE_LABELS[_runtime().IDEOGRAM4_DEFAULT_PIPELINE],
                                label="Pipeline",
                            )
                            gen_ideogram_sampler = _runtime().gr.Dropdown(
                                _runtime().IDEOGRAM4_SAMPLER_CHOICES,
                                value="Turbo - 12 steps",
                                label="Sampler Preset",
                            )
                            gen_ideogram_upsampler = _runtime().gr.Radio(
                                _runtime().IDEOGRAM4_UPSAMPLERS,
                                value=_runtime()._ideogram4_default_upsampler(),
                                label="Prompt Upsampler",
                                elem_id="gen-ideogram-upsampler",
                            )
                        with _runtime().gr.Row():
                            gen_ideogram_strip_prompt = _runtime().gr.Checkbox(
                                value=True,
                                label="Strip aspect ratio/bboxes",
                                elem_id="gen-ideogram-strip-prompt",
                            )
                            gen_ideogram_reuse_cache = _runtime().gr.Checkbox(
                                value=True,
                                label="Reuse upsample cache",
                            )
                            gen_ideogram_gemma_thinking = _runtime().gr.Checkbox(
                                value=False,
                                label="Gemma Thinking",
                            )
                        gen_ideogram_gemma_tokens = _runtime().gr.Slider(
                            512, 4096, 2048, step=128,
                            label="Gemma Max New Tokens",
                        )
                        gen_ideogram_cfg_one_final_steps = _runtime().gr.Slider(
                            0, 8, 0, step=1,
                            label="CFG=1 Final Steps",
                        )
                        with _runtime().gr.Row():
                            gen_ideogram_lora_mode = _runtime().gr.Dropdown(
                                _runtime().IDEOGRAM4_LORA_CHOICES,
                                value=_runtime().IDEOGRAM4_LORA_OFF,
                                label="Realism Engine LoRA",
                            )
                            gen_ideogram_lora_weight = _runtime().gr.Dropdown(
                                _runtime().IDEOGRAM4_REALISM_LORA_WEIGHTS,
                                value=_runtime().IDEOGRAM4_REALISM_LORA_DEFAULT,
                                label="LoRA Weight",
                            )
                        with _runtime().gr.Row():
                            gen_ideogram_lora_cond_strength = _runtime().gr.Slider(
                                0.0, 1.5, 0.9, step=0.05,
                                label="LoRA Conditional Strength",
                            )
                            gen_ideogram_lora_uncond_strength = _runtime().gr.Slider(
                                0.0, 1.5, 0.4, step=0.05,
                                label="LoRA Unconditional Strength",
                            )
                        gen_ideogram_api_key = _runtime().gr.Textbox(
                            label="Ideogram API Key",
                            type="password",
                            value="",
                        )
                        gen_ideogram_designer_payload = _runtime().gr.Textbox(
                            visible=False,
                            elem_id="gen-ideogram-designer-payload",
                        )
                        gen_ideogram_open_designer_btn = _runtime().gr.Button(
                            "Open JSON Designer",
                            size="sm",
                        )
                        with _runtime().gr.Accordion("Raw JSON Build Prompt", open=False):
                            gen_ideogram_build_prompt_btn = _runtime().gr.Button("Build Prompt JSON", size="sm")
                            gen_ideogram_raw_prompt = _runtime().gr.Textbox(label="Raw JSON", lines=10, interactive=False)
                    gen_seed = _runtime().gr.Number(-1, label="Seed (-1 = random)", precision=0)
            with _runtime().gr.Column(scale=5):
                gen_out = _runtime().gr.Image(label="Preview", type="filepath", height=520, interactive=False, format="webp")
                with _runtime().gr.Row():
                    gen_to_edit = _runtime().gr.Button("Send to Edit", size="sm", elem_classes=["send-btn"])
                    gen_to_upscale = _runtime().gr.Button("Send to Upscale", size="sm", elem_classes=["send-btn"])
                    gen_to_ai_remover = _runtime().gr.Button("Send to AI Remover", size="sm", elem_classes=["send-btn"])
                    gen_to_video = _runtime().gr.Button("Send to Video", size="sm", elem_classes=["send-btn"])
                gen_st = _runtime().gr.Markdown("", elem_id="gen-status")
                gen_raw = _runtime().gr.File(label="Raw PNG Download", interactive=False)
        visibility_inputs = [gen_mode, gen_zimage_version, gen_hidream_version, gen_boogu_version]
        visibility_outputs = [
            gen_neg_group,
            gen_lightning_group,
            gen_zimage_group,
            gen_zimage_turbo_group,
            gen_zimage_full_group,
            gen_zimage_pid_group,
            gen_hidream_group,
            gen_hidream_version,
            gen_ideogram_group,
            gen_boogu_group,
            gen_krea2_group,
            gen_boogu_base_group,
            gen_boogu_steps,
            gen_full_pid_ckpt,
        ]
        gen_mode.change(fn=_gen_mode_visibility_updates, inputs=visibility_inputs, outputs=visibility_outputs)
        gen_zimage_version.change(fn=_gen_mode_visibility_updates, inputs=visibility_inputs, outputs=visibility_outputs)
        gen_hidream_version.change(fn=_gen_mode_visibility_updates, inputs=visibility_inputs, outputs=visibility_outputs)
        gen_boogu_version.change(fn=_gen_mode_visibility_updates, inputs=visibility_inputs, outputs=visibility_outputs)
        gen_preset_outputs = [gen_size_small, gen_size_medium, gen_size_large, gen_w, gen_h]
        gen_size_small.change(
            fn=_apply_gen_small_size_preset,
            inputs=[gen_size_small],
            outputs=gen_preset_outputs,
        )
        gen_size_medium.change(
            fn=_apply_gen_medium_size_preset,
            inputs=[gen_size_medium],
            outputs=gen_preset_outputs,
        )
        gen_size_large.change(
            fn=_apply_gen_large_size_preset,
            inputs=[gen_size_large],
            outputs=gen_preset_outputs,
        )

    return GenerateTab(**{
        "mode": gen_mode,
        "prompt": gen_prompt,
        "enhance_btn": gen_enhance_btn,
        "size_preset_small": gen_size_small,
        "size_preset_medium": gen_size_medium,
        "size_preset_large": gen_size_large,
        "negative": gen_neg,
        "width": gen_w,
        "height": gen_h,
        "cfg": gen_cfg,
        "steps": gen_steps,
        "guidance": gen_guidance,
        "full_steps": gen_full_steps,
        "full_guidance": gen_full_guidance,
        "full_pid_enabled": gen_full_pid_enabled,
        "full_pid_ckpt": gen_full_pid_ckpt,
        "full_pid_steps": gen_full_pid_steps,
        "full_pid_cfg": gen_full_pid_cfg,
        "boogu_version": gen_boogu_version,
        "boogu_steps": gen_boogu_steps,
        "boogu_base_guidance": gen_boogu_base_guidance,
        "krea2_steps": gen_krea2_steps,
        "krea2_cfg": gen_krea2_cfg,
        "ideogram_pipeline": gen_ideogram_pipeline,
        "ideogram_sampler": gen_ideogram_sampler,
        "ideogram_upsampler": gen_ideogram_upsampler,
        "ideogram_strip_prompt": gen_ideogram_strip_prompt,
        "ideogram_reuse_cache": gen_ideogram_reuse_cache,
        "ideogram_gemma_tokens": gen_ideogram_gemma_tokens,
        "ideogram_gemma_thinking": gen_ideogram_gemma_thinking,
        "ideogram_cfg_one_final_steps": gen_ideogram_cfg_one_final_steps,
        "ideogram_lora_mode": gen_ideogram_lora_mode,
        "ideogram_lora_weight": gen_ideogram_lora_weight,
        "ideogram_lora_cond_strength": gen_ideogram_lora_cond_strength,
        "ideogram_lora_uncond_strength": gen_ideogram_lora_uncond_strength,
        "ideogram_api_key": gen_ideogram_api_key,
        "ideogram_open_designer_btn": gen_ideogram_open_designer_btn,
        "ideogram_designer_payload": gen_ideogram_designer_payload,
        "ideogram_build_prompt_btn": gen_ideogram_build_prompt_btn,
        "ideogram_raw_prompt": gen_ideogram_raw_prompt,
        "seed": gen_seed,
        "zimage_version": gen_zimage_version,
        "hidream_version": gen_hidream_version,
        "button": gen_btn,
        "output": gen_out,
        "raw": gen_raw,
        "status": gen_st,
        "to_edit": gen_to_edit,
        "to_upscale": gen_to_upscale,
        "to_ai_remover": gen_to_ai_remover,
        "to_video": gen_to_video,
    })

__all__ = (
    '_gen_mode_visibility_updates',
    '_apply_gen_size_preset',
    '_apply_gen_small_size_preset',
    '_apply_gen_medium_size_preset',
    '_apply_gen_large_size_preset',
    '_build_generate_tab',
)
