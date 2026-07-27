"""Gradio-facing gallery selection and send actions."""

from __future__ import annotations

from typing import Any

from image_studio.runtime_access import runtime_namespace as _runtime
from image_studio.ui.errors import ui_endpoint


def extract_video_path(evt: _runtime().gr.SelectData):
    """Resolve the real video path from a gallery thumbnail click.

    The gallery items are (thumb_path, video_filename).  Gradio copies
    thumbs into its own cache, so ``evt.value["image"]["path"]`` points
    at a cache copy - useless for finding the .mp4.  The *caption*,
    however, is the original video filename which we can join with
    OUTPUT_DIR.
    """
    caption = evt.value.get("caption", "") if isinstance(evt.value, dict) else ""
    if caption:
        candidate = _runtime().output_store.contained_path(_runtime().os.path.join(_runtime().OUTPUT_DIR, caption))
        if candidate and _runtime().os.path.isfile(candidate):
            return candidate

    # Fallback: try the image path and strip .thumb.jpg
    path = evt.value["image"]["path"] if isinstance(evt.value, dict) else evt.value
    if path and path.endswith(".thumb.jpg"):
        path = path[:-10]
    safe_path = _runtime().output_store.contained_path(path) if path else None
    if safe_path and _runtime().os.path.isfile(safe_path):
        return safe_path
    return None

def extract_gallery_path(evt: _runtime().gr.SelectData, cur_gal):
    if not cur_gal or evt.index is None or evt.index >= len(cur_gal):
        return None
    item = cur_gal[evt.index]
    path = item[0] if isinstance(item, (list, tuple)) else item
    if isinstance(path, dict) and "image" in path:
        path = path["image"]["path"] if isinstance(path["image"], dict) else path["image"]
    elif isinstance(path, dict) and "name" in path:
        path = path["name"]
    return _runtime().raw_image_path_for_preview(path) if isinstance(path, str) else path

def _gallery_download_update(path: Any):
    if not path:
        return _runtime().gr.update(value=None, interactive=False)
    raw_path = _runtime().resolve_raw_image_payload(path)
    safe_path = _runtime()._resolve_output_file_path(raw_path) if isinstance(raw_path, str) else None
    if not safe_path or not _runtime().os.path.isfile(safe_path):
        return _runtime().gr.update(value=None, interactive=False)
    return _runtime().gr.update(value=safe_path, interactive=True)

def select_gallery_path(evt: _runtime().gr.SelectData, cur_gal):
    path = extract_gallery_path(evt, cur_gal)
    return path, _gallery_download_update(path)

def refresh_gallery_selection():
    return _runtime().get_gallery_images(), None, _gallery_download_update(None)

def clear_gallery_selection():
    return None

def clear_gallery_download():
    return _gallery_download_update(None)

def send_to_edit_slots(new_img, e1, e2, e3):
    if new_img is None:
        raise _runtime().UserInputError("No image available to send.")
    new_img = _runtime().resolve_raw_image_payload(new_img)
    if e1 is None:
        return new_img, e2, e3, _runtime().gr.update(selected=_runtime().TAB_EDIT)
    if e2 is None:
        return e1, new_img, e3, _runtime().gr.update(selected=_runtime().TAB_EDIT)
    if e3 is None:
        return e1, e2, new_img, _runtime().gr.update(selected=_runtime().TAB_EDIT)
    return new_img, e2, e3, _runtime().gr.update(selected=_runtime().TAB_EDIT)

def _send_image_to_tab(img: Any, tab_id: int):
    if img is None:
        raise _runtime().UserInputError("No image available to send.")
    return _runtime().resolve_raw_image_payload(img), _runtime().gr.update(selected=tab_id)

@ui_endpoint
def _require_selected_image_to_tab(path: Any, tab_id: int):
    if not path:
        raise _runtime().UserInputError("Please select an image in the gallery first.")
    return _runtime().require_gallery_image_path(path), _runtime().gr.update(selected=tab_id)

def send_image_to_upscale(img):
    return _send_image_to_tab(img, _runtime().TAB_UPSCALE)

def send_image_to_video(img):
    return _send_image_to_tab(img, _runtime().TAB_VIDEO)

@ui_endpoint
def require_selected_to_video(path):
    return _require_selected_image_to_tab(path, _runtime().TAB_VIDEO)

def send_video_to_upscale(video):
    path = _runtime()._require_seedvr2_video_path(video)
    return path, _runtime().gr.update(selected=_runtime().TAB_UPSCALE)

@ui_endpoint
def send_gallery_to_edit_slots(path, e1, e2, e3):
    if not path:
        raise _runtime().UserInputError("Please select an image in the gallery first.")
    return send_to_edit_slots(_runtime().require_gallery_image_path(path), e1, e2, e3)

@ui_endpoint
def require_selected_to_upscale(path):
    return _require_selected_image_to_tab(path, _runtime().TAB_UPSCALE)

def send_image_to_ai_remover(img):
    return _send_image_to_tab(img, _runtime().TAB_AI_REMOVER)

@ui_endpoint
def require_selected_to_ai_remover(path):
    return _require_selected_image_to_tab(path, _runtime().TAB_AI_REMOVER)

__all__ = (
    'extract_video_path',
    'extract_gallery_path',
    '_gallery_download_update',
    'select_gallery_path',
    'refresh_gallery_selection',
    'clear_gallery_selection',
    'clear_gallery_download',
    'send_to_edit_slots',
    '_send_image_to_tab',
    '_require_selected_image_to_tab',
    'send_image_to_upscale',
    'send_image_to_video',
    'require_selected_to_video',
    'send_video_to_upscale',
    'send_gallery_to_edit_slots',
    'require_selected_to_upscale',
    'send_image_to_ai_remover',
    'require_selected_to_ai_remover',
)
