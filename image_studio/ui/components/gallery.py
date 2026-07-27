"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
from dataclasses import dataclass
from typing import Any

from image_studio.runtime_access import runtime_namespace as _runtime
from image_studio.ui.components.base import ComponentSet


@dataclass
class GalleryTab(ComponentSet):
    tab: Any
    refresh: Any
    download: Any
    to_edit: Any
    to_upscale: Any
    to_ai_remover: Any
    to_video: Any
    delete: Any
    remove_all: Any
    gallery: Any
    selected: Any


def _build_gallery_tab() -> GalleryTab:
    with _runtime().gr.Tab("Gallery", id=_runtime().TAB_GALLERY) as gallery_tab:
        _runtime().gr.Markdown("View generated, edited, and upscaled images below. Click an image to view it full-size or download.")
        with _runtime().gr.Row():
            refresh_btn = _runtime().gr.Button("Refresh Gallery", size="sm")
            download_btn = _runtime().gr.DownloadButton("Download Selected", size="sm", interactive=False)
            gal_to_edit = _runtime().gr.Button("Send Selected to Edit", size="sm", elem_classes=["send-btn"])
            gal_to_upscale = _runtime().gr.Button("Send Selected to Upscale", size="sm", elem_classes=["send-btn"])
            gal_to_ai_remover = _runtime().gr.Button("Send Selected to AI Remover", size="sm", elem_classes=["send-btn"])
            gal_to_video = _runtime().gr.Button("Send to Video", size="sm", elem_classes=["send-btn"])
            delete_btn = _runtime().gr.Button("Delete Selected", size="sm", variant="stop")
            remove_all_img_btn = _runtime().gr.Button("Remove All", size="sm", variant="stop")
        gallery = _runtime().gr.Gallery(
            value=_runtime().get_gallery_images(),
            label="Image Gallery",
            show_label=False,
            elem_id="gallery",
            columns=[4],
            rows=[3],
            object_fit="contain",
            height=600,
            allow_preview=True,
        )
        selected_gallery_item = _runtime().gr.State(None)

    return GalleryTab(**{
        "tab": gallery_tab,
        "refresh": refresh_btn,
        "download": download_btn,
        "to_edit": gal_to_edit,
        "to_upscale": gal_to_upscale,
        "to_ai_remover": gal_to_ai_remover,
        "to_video": gal_to_video,
        "delete": delete_btn,
        "remove_all": remove_all_img_btn,
        "gallery": gallery,
        "selected": selected_gallery_item,
    })

__all__ = (
    '_build_gallery_tab',
)
