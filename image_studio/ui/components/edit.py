"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
from dataclasses import dataclass
from typing import Any

from image_studio.runtime_access import runtime_namespace as _runtime
from image_studio.ui.components.base import ComponentSet

EDIT_VISIBILITY_ORDER = (
    "negative", "qwen", "boogu", "hidream", "size", "aspect", "hidream_version",
    "boogu_guidance", "boogu_steps",
)


def _edit_mode_visibility_updates(model_name: str, boogu_version: str):
    """Return Gradio visibility updates for editor-specific controls."""
    model_name = model_name or "Qwen Image Edit"
    boogu_version = _runtime()._normalize_boogu_edit_version(boogu_version)
    hidream = model_name == _runtime().HIDREAM_O1_MODE
    boogu = model_name == _runtime().BOOGU_IMAGE_MODE
    sensenova = model_name == _runtime().SENSENOVA_MODE
    boogu_base = boogu and boogu_version == _runtime().BOOGU_IMAGE_VERSION_BASE
    qwen = not hidream and not boogu and not sensenova
    sized = hidream or boogu or sensenova
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
        "negative": _runtime().gr.update(visible=qwen or boogu_base),
        "qwen": _runtime().gr.update(visible=qwen),
        "boogu": _runtime().gr.update(visible=boogu),
        "hidream": _runtime().gr.update(visible=hidream),
        "size": _runtime().gr.update(visible=sized),
        "aspect": _runtime().gr.update(visible=sized),
        "hidream_version": _runtime().gr.update(visible=hidream or sensenova),
        "boogu_guidance": _runtime().gr.update(visible=boogu_base),
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


def _build_edit_tab() -> EditTab:
    with _runtime().gr.Tab("Edit", id=_runtime().TAB_EDIT):
        with _runtime().gr.Row(equal_height=False):
            with _runtime().gr.Column(scale=5):
                e_model = _runtime().gr.Radio(
                    _runtime().EDITOR_MODES,
                    value="Qwen Image Edit",
                    label="Editor",
                    elem_id="edit-model",
                )
                with _runtime().gr.Row():
                    e_img1 = _runtime().gr.Image(label="Image 1", type="pil", height=200)
                    e_img2 = _runtime().gr.Image(label="Image 2 (opt)", type="pil", height=200)
                    e_img3 = _runtime().gr.Image(label="Image 3 (opt)", type="pil", height=200)
                e_prompt = _runtime().gr.Textbox(
                    label="Edit Prompt",
                    lines=3,
                    placeholder="Describe how to combine or edit the images",
                )
                e_enhance_btn = _runtime().gr.Button(
                    "Enhance Prompt (Gemma 4)", size="sm",
                    elem_classes=["enhance-btn"],
                )
                with _runtime().gr.Accordion("Editing Parameters (Advanced)", open=False):
                    with _runtime().gr.Group(elem_id="edit-qwen-neg-group") as e_qwen_neg_group:
                        e_neg = _runtime().gr.Textbox(label="Negative Prompt", lines=1, value="")
                    with _runtime().gr.Row(elem_id="edit-qwen-param-group") as e_qwen_param_group:
                        e_cfg = _runtime().gr.Slider(0.5, 5.0, 1.0, step=0.1, label="CFG Scale")
                        e_seed = _runtime().gr.Number(-1, label="Seed (-1 = random)", precision=0)
                    with _runtime().gr.Group(visible=False, elem_id="edit-boogu-param-group") as e_boogu_param_group:
                        _runtime().gr.Markdown("**Boogu-Image Settings**")
                        e_boogu_version = _runtime().gr.Dropdown(
                            _runtime().BOOGU_IMAGE_EDIT_VERSIONS,
                            value=_runtime().BOOGU_IMAGE_VERSION_TURBO,
                            label="Boogu-Image Edit Version",
                        )
                        with _runtime().gr.Row():
                            e_boogu_steps = _runtime().gr.Slider(
                                3,
                                8,
                                _runtime().BOOGU_IMAGE_TURBO_DEFAULT_STEPS,
                                step=1,
                                label="Steps (Turbo)",
                            )
                            e_boogu_seed = _runtime().gr.Number(-1, label="Seed (-1 = random)", precision=0)
                        with _runtime().gr.Row(visible=False, elem_id="edit-boogu-guidance-group") as e_boogu_guidance_group:
                            e_boogu_text_guidance = _runtime().gr.Slider(2.0, 5.0, 4.0, step=0.25, label="Text Guidance")
                            e_boogu_image_guidance = _runtime().gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Image Guidance")
                    with _runtime().gr.Row(visible=False, elem_id="edit-hidream-version-group") as e_hidream_version_group:
                        e_hd_version = _runtime().gr.Dropdown(["Dev", "Best Quality"], value="Dev", label="HiDream-O1 Version")
                    with _runtime().gr.Row(visible=False, elem_id="edit-hidream-size-group") as e_hidream_size_group:
                        e_w = _runtime().gr.Slider(256, 4096, 1024, step=64, label="Width")
                        e_h = _runtime().gr.Slider(256, 4096, 1024, step=64, label="Height")
                    with _runtime().gr.Row(visible=False, elem_id="edit-hidream-aspect-group") as e_hidream_aspect_group:
                        e_keep_aspect = _runtime().gr.Checkbox(True, label="Preserve single-source aspect")
                    with _runtime().gr.Row(visible=False, elem_id="edit-hidream-seed-group") as e_hidream_seed_group:
                        e_hd_seed = _runtime().gr.Number(-1, label="Seed (-1 = random)", precision=0)
                e_btn = _runtime().gr.Button("Generate", variant="primary", elem_id="edit-btn")
            with _runtime().gr.Column(scale=5):
                e_out = _runtime().gr.Image(label="Preview", type="filepath", height=520, interactive=False, format="webp")
                with _runtime().gr.Row():
                    edit_to_upscale = _runtime().gr.Button("Send to Upscale", size="sm", elem_classes=["send-btn"])
                    edit_to_ai_remover = _runtime().gr.Button("Send to AI Remover", size="sm", elem_classes=["send-btn"])
                    edit_to_video = _runtime().gr.Button("Send to Video", size="sm", elem_classes=["send-btn"])
                e_st = _runtime().gr.Markdown("", elem_id="edit-status")
                e_raw = _runtime().gr.File(label="Raw PNG Download", interactive=False)
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
