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


def _build_gallery_tab() -> dict[str, Any]:
    with gr.Tab("Gallery", id=TAB_GALLERY) as gallery_tab:
        gr.Markdown("View generated, edited, and upscaled images below. Click an image to view it full-size or download.")
        with gr.Row():
            refresh_btn = gr.Button("Refresh Gallery", size="sm")
            download_btn = gr.DownloadButton("Download Selected", size="sm", interactive=False)
            gal_to_edit = gr.Button("Send Selected to Edit", size="sm", elem_classes=["send-btn"])
            gal_to_upscale = gr.Button("Send Selected to Upscale", size="sm", elem_classes=["send-btn"])
            gal_to_ai_remover = gr.Button("Send Selected to AI Remover", size="sm", elem_classes=["send-btn"])
            gal_to_video = gr.Button("Send to Video", size="sm", elem_classes=["send-btn"])
            delete_btn = gr.Button("Delete Selected", size="sm", variant="stop")
            remove_all_img_btn = gr.Button("Remove All", size="sm", variant="stop")
        gallery = gr.Gallery(
            value=get_gallery_images(),
            label="Image Gallery",
            show_label=False,
            elem_id="gallery",
            columns=[4],
            rows=[3],
            object_fit="contain",
            height=600,
            allow_preview=True,
        )
        selected_gallery_item = gr.State(None)

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
_seal_runtime_module(globals())
