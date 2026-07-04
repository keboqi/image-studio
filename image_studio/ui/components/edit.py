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

EDIT_VISIBILITY_ORDER = (
    "negative", "qwen", "boogu", "hidream", "size", "aspect", "hidream_version",
    "boogu_guidance", "boogu_steps",
)


def _edit_mode_visibility_updates(model_name: str, boogu_version: str):
    """Return Gradio visibility updates for editor-specific controls."""
    model_name = model_name or "Qwen Image Edit"
    boogu_version = _normalize_boogu_edit_version(boogu_version)
    hidream = model_name == HIDREAM_O1_MODE
    boogu = model_name == BOOGU_IMAGE_MODE
    boogu_base = boogu and boogu_version == BOOGU_IMAGE_VERSION_BASE
    qwen = not hidream and not boogu
    sized = hidream or boogu
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
        "negative": gr.update(visible=qwen or boogu_base),
        "qwen": gr.update(visible=qwen),
        "boogu": gr.update(visible=boogu),
        "hidream": gr.update(visible=hidream),
        "size": gr.update(visible=sized),
        "aspect": gr.update(visible=sized),
        "hidream_version": gr.update(visible=hidream),
        "boogu_guidance": gr.update(visible=boogu_base),
        "boogu_steps": boogu_steps_update,
    }
    return tuple(updates[name] for name in EDIT_VISIBILITY_ORDER)

@dataclass
class EditTab(ComponentSet):
    model: Any
    img1: Any
    img2: Any
    img3: Any
    prompt: Any
    enhance_btn: Any
    negative: Any
    cfg: Any
    seed: Any
    boogu_version: Any
    boogu_steps: Any
    boogu_text_guidance: Any
    boogu_image_guidance: Any
    boogu_seed: Any
    width: Any
    height: Any
    keep_aspect: Any
    hidream_seed: Any
    hidream_version: Any
    button: Any
    output: Any
    raw: Any
    status: Any
    to_upscale: Any
    to_ai_remover: Any
    to_video: Any


def _build_edit_tab() -> dict[str, Any]:
    with gr.Tab("Edit", id=TAB_EDIT):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                e_model = gr.Radio(
                    EDITOR_MODES,
                    value="Qwen Image Edit",
                    label="Editor",
                    elem_id="edit-model",
                )
                with gr.Row():
                    e_img1 = gr.Image(label="Image 1", type="pil", height=200)
                    e_img2 = gr.Image(label="Image 2 (opt)", type="pil", height=200)
                    e_img3 = gr.Image(label="Image 3 (opt)", type="pil", height=200)
                e_prompt = gr.Textbox(
                    label="Edit Prompt",
                    lines=3,
                    placeholder="Describe how to combine or edit the images",
                )
                e_enhance_btn = gr.Button(
                    "Enhance Prompt (Gemma 4)", size="sm",
                    elem_classes=["enhance-btn"],
                )
                with gr.Accordion("Editing Parameters (Advanced)", open=False):
                    with gr.Group(elem_id="edit-qwen-neg-group") as e_qwen_neg_group:
                        e_neg = gr.Textbox(label="Negative Prompt", lines=1, value="")
                    with gr.Row(elem_id="edit-qwen-param-group") as e_qwen_param_group:
                        e_cfg = gr.Slider(0.5, 5.0, 1.0, step=0.1, label="CFG Scale")
                        e_seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
                    with gr.Group(visible=False, elem_id="edit-boogu-param-group") as e_boogu_param_group:
                        gr.Markdown("**Boogu-Image Settings**")
                        e_boogu_version = gr.Dropdown(
                            BOOGU_IMAGE_EDIT_VERSIONS,
                            value=BOOGU_IMAGE_VERSION_TURBO,
                            label="Boogu-Image Edit Version",
                        )
                        with gr.Row():
                            e_boogu_steps = gr.Slider(
                                3,
                                8,
                                BOOGU_IMAGE_TURBO_DEFAULT_STEPS,
                                step=1,
                                label="Steps (Turbo)",
                            )
                            e_boogu_seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
                        with gr.Row(visible=False, elem_id="edit-boogu-guidance-group") as e_boogu_guidance_group:
                            e_boogu_text_guidance = gr.Slider(2.0, 5.0, 4.0, step=0.25, label="Text Guidance")
                            e_boogu_image_guidance = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Image Guidance")
                    with gr.Row(visible=False, elem_id="edit-hidream-version-group") as e_hidream_version_group:
                        e_hd_version = gr.Dropdown(["Dev", "Best Quality"], value="Dev", label="HiDream-O1 Version")
                    with gr.Row(visible=False, elem_id="edit-hidream-size-group") as e_hidream_size_group:
                        e_w = gr.Slider(256, 4096, 1024, step=64, label="Width")
                        e_h = gr.Slider(256, 4096, 1024, step=64, label="Height")
                    with gr.Row(visible=False, elem_id="edit-hidream-aspect-group") as e_hidream_aspect_group:
                        e_keep_aspect = gr.Checkbox(True, label="Preserve single-source aspect")
                    with gr.Row(visible=False, elem_id="edit-hidream-seed-group") as e_hidream_seed_group:
                        e_hd_seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
                e_btn = gr.Button("Generate", variant="primary", elem_id="edit-btn")
            with gr.Column(scale=5):
                e_out = gr.Image(label="Preview", type="filepath", height=520, interactive=False, format="webp")
                with gr.Row():
                    edit_to_upscale = gr.Button("Send to Upscale", size="sm", elem_classes=["send-btn"])
                    edit_to_ai_remover = gr.Button("Send to AI Remover", size="sm", elem_classes=["send-btn"])
                    edit_to_video = gr.Button("Send to Video", size="sm", elem_classes=["send-btn"])
                e_st = gr.Markdown("", elem_id="edit-status")
                e_raw = gr.File(label="Raw PNG Download", interactive=False)
        visibility_inputs = [e_model, e_boogu_version]
        visibility_outputs = [
            e_qwen_neg_group,
            e_qwen_param_group,
            e_boogu_param_group,
            e_hidream_version_group,
            e_hidream_size_group,
            e_hidream_aspect_group,
            e_hidream_seed_group,
            e_boogu_guidance_group,
            e_boogu_steps,
        ]
        e_model.change(
            fn=_edit_mode_visibility_updates,
            inputs=visibility_inputs,
            outputs=visibility_outputs,
        )
        e_boogu_version.change(
            fn=_edit_mode_visibility_updates,
            inputs=visibility_inputs,
            outputs=visibility_outputs,
        )

    return EditTab(**{
        "model": e_model,
        "img1": e_img1,
        "img2": e_img2,
        "img3": e_img3,
        "prompt": e_prompt,
        "enhance_btn": e_enhance_btn,
        "negative": e_neg,
        "cfg": e_cfg,
        "seed": e_seed,
        "boogu_version": e_boogu_version,
        "boogu_steps": e_boogu_steps,
        "boogu_text_guidance": e_boogu_text_guidance,
        "boogu_image_guidance": e_boogu_image_guidance,
        "boogu_seed": e_boogu_seed,
        "width": e_w,
        "height": e_h,
        "keep_aspect": e_keep_aspect,
        "hidream_seed": e_hd_seed,
        "hidream_version": e_hd_version,
        "button": e_btn,
        "output": e_out,
        "raw": e_raw,
        "status": e_st,
        "to_upscale": edit_to_upscale,
        "to_ai_remover": edit_to_ai_remover,
        "to_video": edit_to_video,
    })

__all__ = (
    '_edit_mode_visibility_updates',
    '_build_edit_tab',
)
_seal_runtime_module(globals())
