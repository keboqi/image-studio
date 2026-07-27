"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
from dataclasses import dataclass
from typing import Any

from image_studio.runtime_access import runtime_namespace as _runtime
from image_studio.ui.components.base import ComponentSet


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


def _build_ai_remover_tab() -> AiRemoverTab:
    with _runtime().gr.Tab("AI Remover", id=_runtime().TAB_AI_REMOVER):
        _runtime().gr.Markdown(
            "Remove visible/invisible watermarks and metadata from an image using **`remove-ai-watermarks`**."
        )
        with _runtime().gr.Row(equal_height=False):
            with _runtime().gr.Column(scale=5):
                air_img = _runtime().gr.Image(label="Source Image", type="pil", height=320)
                with _runtime().gr.Group():
                    air_mode = _runtime().gr.Dropdown(
                        choices=["all", "visible", "invisible", "metadata"],
                        value="all",
                        label="Removal Mode (Subcommand)",
                    )
                    air_humanize = _runtime().gr.Slider(
                        minimum=0.0,
                        maximum=6.0,
                        value=0.0,
                        step=0.5,
                        label="Analog Humanizer",
                    )
                air_btn = _runtime().gr.Button("Remove Watermarks", variant="primary", elem_id="ai-remover-btn")
            with _runtime().gr.Column(scale=5):
                air_out = _runtime().gr.Image(label="Cleaned Preview", type="filepath", height=520, interactive=False, format="webp")
                with _runtime().gr.Row():
                    air_to_edit = _runtime().gr.Button("Send to Edit", size="sm", elem_classes=["send-btn"])
                    air_to_upscale = _runtime().gr.Button("Send to Upscale", size="sm", elem_classes=["send-btn"])
                air_st = _runtime().gr.Markdown("", elem_id="ai-remover-status")
                air_raw = _runtime().gr.File(label="Raw PNG Download", interactive=False)
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
