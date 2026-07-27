"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
from dataclasses import dataclass
from typing import Any

from image_studio.runtime_access import runtime_namespace as _runtime
from image_studio.ui.components.base import ComponentSet


@dataclass
class ModelsTab(ComponentSet):
    status: Any
    refresh: Any
    unload_all: Any
    picker: Any
    unload: Any
    storage_picker: Any
    remove_files: Any
    remove_all_files: Any


def _build_models_tab() -> ModelsTab:
    with _runtime().gr.Tab("Models", id=_runtime().TAB_MODELS):
        _runtime().gr.Markdown(
            "Manage GPU-resident models and downloaded model files. The manager automatically "
            "evicts the **least-recently-used** model when VRAM is "
            "insufficient for a new load, but you can also unload "
            "models manually here and remove cached weights from disk."
        )
        models_status = _runtime().gr.Markdown(_runtime()._build_models_md(), elem_id="models-status")
        with _runtime().gr.Row():
            models_refresh_btn = _runtime().gr.Button("Refresh", size="sm")
            models_unload_all_btn = _runtime().gr.Button("Unload All Models", size="sm", variant="stop")
        with _runtime().gr.Row():
            models_picker = _runtime().gr.Dropdown(
                choices=_runtime()._get_loaded_model_choices(),
                value=None,
                label="Select model to unload",
                interactive=True,
            )
            models_unload_btn = _runtime().gr.Button("Unload Selected", size="sm", variant="stop")
        with _runtime().gr.Row():
            models_storage_picker = _runtime().gr.Dropdown(
                choices=_runtime()._get_downloaded_model_choices(),
                value=None,
                label="Select downloaded files to remove",
                interactive=True,
            )
            models_remove_files_btn = _runtime().gr.Button("Remove Selected Files", size="sm", variant="stop")
            models_remove_all_files_btn = _runtime().gr.Button("Remove All Downloaded Files", size="sm", variant="stop")

    return ModelsTab(**{
        "status": models_status,
        "refresh": models_refresh_btn,
        "unload_all": models_unload_all_btn,
        "picker": models_picker,
        "unload": models_unload_btn,
        "storage_picker": models_storage_picker,
        "remove_files": models_remove_files_btn,
        "remove_all_files": models_remove_all_files_btn,
    })

__all__ = (
    '_build_models_tab',
)
