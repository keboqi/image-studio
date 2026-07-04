"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.ui.components.base import ComponentSet

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

GEN_VISIBILITY_ORDER = (
    "negative", "qwen", "zimage", "zimage_turbo", "zimage_full", "pid",
    "hidream", "ideogram", "boogu", "krea2", "boogu_base", "boogu_steps", "pid_checkpoint",
)


def _gen_mode_visibility_updates(mode: str, zimage_version: str, hidream_version: str, boogu_version: str):
    """Return Gradio visibility updates for generator-specific controls."""
    mode = mode or "Qwen Image"
    zimage_version = zimage_version or "Turbo"
    hidream_version = hidream_version or "Dev"
    boogu_version = _normalize_boogu_generation_version(boogu_version)
    
    qwen = mode == "Qwen Image"
    zimage = mode == "Z-Image"
    turbo = zimage and zimage_version == "Turbo"
    zfull = zimage and zimage_version == "Best Quality"
    hidream = mode == HIDREAM_O1_MODE
    ideogram = mode == IDEOGRAM4_MODE
    boogu = mode == BOOGU_IMAGE_MODE
    krea2 = mode == KREA2_MODE
    boogu_base = boogu and boogu_version == BOOGU_IMAGE_VERSION_BASE
    zimage_family = turbo or zfull
    pid_capable = qwen or zimage_family or ideogram or krea2

    if qwen or krea2:
        pid_ckpt_choices = PID_QWEN_CKPT_CHOICES
    elif ideogram:
        pid_ckpt_choices = PID_IDEOGRAM4_CKPT_CHOICES
    else:
        pid_ckpt_choices = PID_ZIMAGE_CKPT_CHOICES

    if boogu_base:
        boogu_steps_update = gr.update(
            visible=True,
            minimum=25,
            maximum=50,
            value=BOOGU_IMAGE_BASE_DEFAULT_STEPS,
            label="Steps (Base)",
        )
    else:
        boogu_steps_update = gr.update(
            visible=boogu,
            minimum=3,
            maximum=8,
            value=BOOGU_IMAGE_TURBO_DEFAULT_STEPS,
            label="Steps (Turbo)",
        )

    updates = {
        "negative": gr.update(visible=qwen or zfull or boogu_base),
        "qwen": gr.update(visible=qwen),
        "zimage": gr.update(visible=zimage),
        "zimage_turbo": gr.update(visible=turbo),
        "zimage_full": gr.update(visible=zfull),
        "pid": gr.update(visible=pid_capable),
        "hidream": gr.update(visible=hidream),
        "ideogram": gr.update(visible=ideogram),
        "boogu": gr.update(visible=boogu),
        "krea2": gr.update(visible=krea2),
        "boogu_base": gr.update(visible=boogu_base),
        "boogu_steps": boogu_steps_update,
        "pid_checkpoint": gr.update(choices=pid_ckpt_choices, value=PID_CKPT_AUTO),
    }
    return tuple(updates[name] for name in GEN_VISIBILITY_ORDER)

def _apply_gen_size_preset(size: str, aspect: str):
    dims = GEN_SIZE_PRESETS.get(size, {}).get(aspect or "")
    if dims is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    width, height = dims
    return (
        gr.update(value=aspect if size == "Small" else None),
        gr.update(value=aspect if size == "Medium" else None),
        gr.update(value=aspect if size == "Large" else None),
        gr.update(value=width),
        gr.update(value=height),
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


def _build_generate_tab() -> dict[str, Any]:
    with gr.Tab("Generate", id=TAB_GENERATE):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Row():
                    gen_mode = gr.Radio(
                        GENERATOR_MODES,
                        value="Z-Image",
                        label="Generator",
                        elem_id="gen-mode",
                    )
                gen_prompt = gr.Textbox(
                    label="Prompt", lines=4,
                    placeholder="Describe the image you want to create",
                    elem_id="gen-prompt",
                )
                gen_enhance_btn = gr.Button(
                    "Enhance Prompt (Gemma 4)", size="sm",
                    elem_classes=["enhance-btn"],
                )
                gen_btn = gr.Button("Generate", variant="primary", elem_id="gen-btn")
                with gr.Accordion("Generation Parameters (Advanced)", open=False):
                    with gr.Group(visible=False, elem_id="gen-neg-group") as gen_neg_group:
                        gen_neg = gr.Textbox(label="Negative Prompt", lines=1, value="")
                    with gr.Row():
                        gen_size_small = gr.Dropdown(
                            GEN_SIZE_ASPECT_CHOICES,
                            value="1:1",
                            label="Small",
                        )
                        gen_size_medium = gr.Dropdown(
                            GEN_SIZE_ASPECT_CHOICES,
                            value=None,
                            label="Medium",
                        )
                        gen_size_large = gr.Dropdown(
                            GEN_SIZE_ASPECT_CHOICES,
                            value=None,
                            label="Large",
                        )
                    with gr.Row():
                        gen_w = gr.Slider(256, 4096, 1024, step=32, label="Width")
                        gen_h = gr.Slider(256, 4096, 1024, step=32, label="Height")
                    with gr.Row(visible=False, elem_id="gen-lightning-group") as gen_lightning_group:
                        gen_cfg = gr.Slider(0.5, 5.0, 1.0, step=0.1, label="CFG Scale")
                    with gr.Group(visible=True, elem_id="gen-zimage-group") as gen_zimage_group:
                        gr.Markdown("**Z-Image Settings**")
                        gen_zimage_version = gr.Dropdown(["Turbo", "Best Quality"], value="Turbo", label="Z-Image Version")
                        with gr.Row(visible=True, elem_id="gen-zimage-turbo-group") as gen_zimage_turbo_group:
                            gen_steps = gr.Slider(4, 16, 8, step=1, label="Steps (DiT forwards)")
                            gen_guidance = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Guidance (Turbo=0)")
                        with gr.Row(visible=False, elem_id="gen-zimage-full-group") as gen_zimage_full_group:
                            gen_full_steps = gr.Slider(20, 60, 30, step=1, label="Steps")
                            gen_full_guidance = gr.Slider(1.0, 8.0, 4.0, step=0.25, label="Guidance Scale (CFG)")
                    with gr.Group(visible=False, elem_id="gen-hidream-group") as gen_hidream_group:
                        gr.Markdown("**HiDream-O1 Settings**")
                        gen_hidream_version = gr.Dropdown(["Dev", "Best Quality"], value="Dev", label="HiDream-O1 Version")
                    with gr.Group(visible=False, elem_id="gen-boogu-group") as gen_boogu_group:
                        gr.Markdown("**Boogu-Image Settings**")
                        gen_boogu_version = gr.Dropdown(
                            BOOGU_IMAGE_GENERATION_VERSIONS,
                            value=BOOGU_IMAGE_VERSION_TURBO,
                            label="Boogu-Image Version",
                        )
                        gen_boogu_steps = gr.Slider(
                            3,
                            8,
                            BOOGU_IMAGE_TURBO_DEFAULT_STEPS,
                            step=1,
                            label="Steps (Turbo)",
                        )
                        with gr.Row(visible=False, elem_id="gen-boogu-base-group") as gen_boogu_base_group:
                            gen_boogu_base_guidance = gr.Slider(2.0, 5.0, 4.0, step=0.25, label="Text Guidance")
                    with gr.Group(visible=False, elem_id="gen-krea2-group") as gen_krea2_group:
                        gr.Markdown("**Krea2 Turbo (ComfyUI)**")
                        with gr.Row():
                            gen_krea2_steps = gr.Slider(
                                1,
                                16,
                                KREA2_DEFAULT_STEPS,
                                step=1,
                                label="Steps",
                            )
                            gen_krea2_cfg = gr.Slider(
                                0.0,
                                5.0,
                                KREA2_DEFAULT_CFG,
                                step=0.1,
                                label="CFG (1 = disabled)",
                            )
                    with gr.Group(visible=True, elem_id="gen-zimage-pid-group") as gen_zimage_pid_group:
                        gr.Markdown("**PiD 4x Decode**")
                        with gr.Row():
                            gen_full_pid_enabled = gr.Checkbox(value=False, label="PiD 4x Decode")
                            gen_full_pid_ckpt = gr.Dropdown(
                                PID_ZIMAGE_CKPT_CHOICES,
                                value=PID_CKPT_AUTO,
                                label="PiD Checkpoint",
                            )
                        with gr.Row():
                            gen_full_pid_steps = gr.Slider(1, 8, 4, step=1, label="PiD Steps")
                            gen_full_pid_cfg = gr.Slider(0.0, 4.0, 1.0, step=0.25, label="PiD CFG")
                    with gr.Group(visible=False, elem_id="gen-ideogram-group") as gen_ideogram_group:
                        gr.Markdown("**Ideogram 4**")
                        with gr.Row():
                            gen_ideogram_pipeline = gr.Dropdown(
                                IDEOGRAM4_PIPELINE_CHOICES,
                                value=IDEOGRAM4_PIPELINE_LABELS[IDEOGRAM4_DEFAULT_PIPELINE],
                                label="Pipeline",
                            )
                            gen_ideogram_sampler = gr.Dropdown(
                                IDEOGRAM4_SAMPLER_CHOICES,
                                value="Turbo - 12 steps",
                                label="Sampler Preset",
                            )
                            gen_ideogram_upsampler = gr.Radio(
                                IDEOGRAM4_UPSAMPLERS,
                                value=_ideogram4_default_upsampler(),
                                label="Prompt Upsampler",
                                elem_id="gen-ideogram-upsampler",
                            )
                        with gr.Row():
                            gen_ideogram_strip_prompt = gr.Checkbox(
                                value=True,
                                label="Strip aspect ratio/bboxes",
                                elem_id="gen-ideogram-strip-prompt",
                            )
                            gen_ideogram_reuse_cache = gr.Checkbox(
                                value=True,
                                label="Reuse upsample cache",
                            )
                            gen_ideogram_gemma_thinking = gr.Checkbox(
                                value=False,
                                label="Gemma Thinking",
                            )
                        gen_ideogram_gemma_tokens = gr.Slider(
                            512, 4096, 2048, step=128,
                            label="Gemma Max New Tokens",
                        )
                        gen_ideogram_cfg_one_final_steps = gr.Slider(
                            0, 8, 0, step=1,
                            label="CFG=1 Final Steps",
                        )
                        with gr.Row():
                            gen_ideogram_lora_mode = gr.Dropdown(
                                IDEOGRAM4_LORA_CHOICES,
                                value=IDEOGRAM4_LORA_OFF,
                                label="Realism Engine LoRA",
                            )
                            gen_ideogram_lora_weight = gr.Dropdown(
                                IDEOGRAM4_REALISM_LORA_WEIGHTS,
                                value=IDEOGRAM4_REALISM_LORA_DEFAULT,
                                label="LoRA Weight",
                            )
                        with gr.Row():
                            gen_ideogram_lora_cond_strength = gr.Slider(
                                0.0, 1.5, 0.9, step=0.05,
                                label="LoRA Conditional Strength",
                            )
                            gen_ideogram_lora_uncond_strength = gr.Slider(
                                0.0, 1.5, 0.4, step=0.05,
                                label="LoRA Unconditional Strength",
                            )
                        gen_ideogram_api_key = gr.Textbox(
                            label="Ideogram API Key",
                            type="password",
                            value="",
                        )
                        gen_ideogram_designer_payload = gr.Textbox(
                            visible=False,
                            elem_id="gen-ideogram-designer-payload",
                        )
                        gen_ideogram_open_designer_btn = gr.Button(
                            "Open JSON Designer",
                            size="sm",
                        )
                        with gr.Accordion("Raw JSON Build Prompt", open=False):
                            gen_ideogram_build_prompt_btn = gr.Button("Build Prompt JSON", size="sm")
                            gen_ideogram_raw_prompt = gr.Textbox(label="Raw JSON", lines=10, interactive=False)
                    gen_seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
            with gr.Column(scale=5):
                gen_out = gr.Image(label="Preview", type="filepath", height=520, interactive=False, format="webp")
                with gr.Row():
                    gen_to_edit = gr.Button("Send to Edit", size="sm", elem_classes=["send-btn"])
                    gen_to_upscale = gr.Button("Send to Upscale", size="sm", elem_classes=["send-btn"])
                    gen_to_ai_remover = gr.Button("Send to AI Remover", size="sm", elem_classes=["send-btn"])
                    gen_to_video = gr.Button("Send to Video", size="sm", elem_classes=["send-btn"])
                gen_st = gr.Markdown("", elem_id="gen-status")
                gen_raw = gr.File(label="Raw PNG Download", interactive=False)
        visibility_inputs = [gen_mode, gen_zimage_version, gen_hidream_version, gen_boogu_version]
        visibility_outputs = [
            gen_neg_group,
            gen_lightning_group,
            gen_zimage_group,
            gen_zimage_turbo_group,
            gen_zimage_full_group,
            gen_zimage_pid_group,
            gen_hidream_group,
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
_seal_runtime_module(globals())
