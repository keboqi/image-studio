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
class ModelsTab(ComponentSet):
    status: Any
    refresh: Any
    unload_all: Any
    picker: Any
    unload: Any


def _build_models_tab() -> dict[str, Any]:
    with gr.Tab("Models", id=TAB_MODELS):
        gr.Markdown(
            "Manage GPU-resident models.  The manager automatically "
            "evicts the **least-recently-used** model when VRAM is "
            "insufficient for a new load, but you can also unload "
            "models manually here."
        )
        models_status = gr.Markdown(_build_models_md(), elem_id="models-status")
        with gr.Row():
            models_refresh_btn = gr.Button("Refresh", size="sm")
            models_unload_all_btn = gr.Button("Unload All Models", size="sm", variant="stop")
        with gr.Row():
            models_picker = gr.Dropdown(
                choices=_get_loaded_model_choices(),
                value=None,
                label="Select model to unload",
                interactive=True,
            )
            models_unload_btn = gr.Button("Unload Selected", size="sm", variant="stop")

    return ModelsTab(**{
        "status": models_status,
        "refresh": models_refresh_btn,
        "unload_all": models_unload_all_btn,
        "picker": models_picker,
        "unload": models_unload_btn,
    })

__all__ = (
    '_build_models_tab',
)
_seal_runtime_module(globals())
