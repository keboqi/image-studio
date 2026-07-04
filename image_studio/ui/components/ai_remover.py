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

@dataclass
class AiRemoverTab(ComponentSet):
    image: Any
    mode: Any
    humanize: Any
    button: Any
    output: Any
    raw: Any
    to_edit: Any
    to_upscale: Any
    status: Any


def _build_ai_remover_tab() -> dict[str, Any]:
    with gr.Tab("AI Remover", id=TAB_AI_REMOVER):
        gr.Markdown(
            "Remove visible/invisible watermarks and metadata from an image using **`remove-ai-watermarks`**."
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                air_img = gr.Image(label="Source Image", type="pil", height=320)
                with gr.Group():
                    air_mode = gr.Dropdown(
                        choices=["all", "visible", "invisible", "metadata"],
                        value="all",
                        label="Removal Mode (Subcommand)",
                    )
                    air_humanize = gr.Slider(
                        minimum=0.0,
                        maximum=6.0,
                        value=0.0,
                        step=0.5,
                        label="Analog Humanizer",
                    )
                air_btn = gr.Button("Remove Watermarks", variant="primary", elem_id="ai-remover-btn")
            with gr.Column(scale=5):
                air_out = gr.Image(label="Cleaned Preview", type="filepath", height=520, interactive=False, format="webp")
                with gr.Row():
                    air_to_edit = gr.Button("Send to Edit", size="sm", elem_classes=["send-btn"])
                    air_to_upscale = gr.Button("Send to Upscale", size="sm", elem_classes=["send-btn"])
                air_st = gr.Markdown("", elem_id="ai-remover-status")
                air_raw = gr.File(label="Raw PNG Download", interactive=False)
    return AiRemoverTab(**{
        "image": air_img,
        "mode": air_mode,
        "humanize": air_humanize,
        "button": air_btn,
        "output": air_out,
        "raw": air_raw,
        "to_edit": air_to_edit,
        "to_upscale": air_to_upscale,
        "status": air_st,
    })

__all__ = (
    '_build_ai_remover_tab',
)
_seal_runtime_module(globals())
