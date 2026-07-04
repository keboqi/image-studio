"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import UserInputError

# --- extracted runtime implementation ---
import sys as _runtime_sys
import logging
import os
import subprocess
from PIL import Image
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

log = logging.getLogger(__name__)


class VideoThumbnailer:
    """Create stable gallery thumbnails with a short-video fallback."""

    def __init__(self, timeout: float = 5):
        self.timeout = timeout

    def ensure(self, video_path: str) -> str | None:
        thumbnail = f"{video_path}.thumb.jpg"
        if os.path.isfile(thumbnail) and os.path.getsize(thumbnail) > 0:
            return thumbnail
        for seek in ("1", None):
            command = ["ffmpeg", "-y"]
            if seek is not None:
                command.extend(["-ss", seek])
            command.extend(["-i", video_path, "-vframes", "1", "-q:v", "2", thumbnail])
            try:
                subprocess.run(command, capture_output=True, timeout=self.timeout, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                log.error("FFmpeg failed to extract thumbnail for %s: %s", video_path, exc)
                break
            if os.path.isfile(thumbnail) and os.path.getsize(thumbnail) > 0:
                return thumbnail
        try:
            Image.new("RGB", (200, 200), (0, 0, 0)).save(thumbnail)
            return thumbnail
        except OSError as exc:
            log.error("Failed to write fallback thumbnail for %s: %s", video_path, exc)
            return None


video_thumbnailer = VideoThumbnailer()

def delete_image(path):
    if not path or not isinstance(path, str):
        return get_gallery_images()
    try:
        for artifact_path in related_image_artifact_paths(path):
            safe_path = _resolve_output_file_path(artifact_path)
            if safe_path is None:
                log.warning("Refusing to delete non-output image artifact: %s", artifact_path)
                continue
            if os.path.isfile(safe_path):
                os.remove(safe_path)
                log.info("Deleted image artifact: %s", safe_path)
    except Exception as e:
        log.error("Error deleting image %s: %s", path, e)
    return get_gallery_images()

def delete_all_images():
    if not os.path.exists(OUTPUT_DIR): return []
    for f in os.listdir(OUTPUT_DIR):
        if f.lower().endswith(GALLERY_IMAGE_EXTENSIONS) and not f.endswith(".thumb.jpg"):
            try: os.remove(os.path.join(OUTPUT_DIR, f))
            except: pass
    return get_gallery_images()

def delete_all_videos():
    if not os.path.exists(OUTPUT_DIR): return []
    for f in os.listdir(OUTPUT_DIR):
        if f.lower().endswith(GALLERY_VIDEO_EXTENSIONS) or f.endswith(".thumb.jpg"):
            try: os.remove(os.path.join(OUTPUT_DIR, f))
            except: pass
    return []

def get_video_gallery_images():
    if not os.path.exists(OUTPUT_DIR):
        return []
    with os.scandir(OUTPUT_DIR) as it:
        videos = [
            (e.path, e.stat().st_mtime)
            for e in it
            if e.name.lower().endswith(GALLERY_VIDEO_EXTENSIONS)
        ]
    videos.sort(key=lambda x: x[1], reverse=True)
    
    gallery_items = []
    for p, _ in videos[:MAX_GALLERY_IMAGES]:
        thumb_path = video_thumbnailer.ensure(p)
        if thumb_path:
            gallery_items.append((thumb_path, os.path.basename(p)))
    return gallery_items


def get_gallery_images():
    if not os.path.exists(OUTPUT_DIR):
        return []
    with os.scandir(OUTPUT_DIR) as it:
        raw_images = []
        for e in it:
            lower = e.name.lower()
            if not lower.endswith(GALLERY_IMAGE_EXTENSIONS):
                continue
            if e.name.endswith(".thumb.jpg") or lower.endswith(OUTPUT_PREVIEW_SUFFIX):
                continue
            raw_images.append((e.path, e.stat().st_mtime))

    images = []
    for raw_path, mtime in raw_images:
        preview_path = ensure_webp_preview(raw_path)
        images.append((preview_path, raw_path, mtime))
    images.sort(key=lambda x: x[2], reverse=True)
    
    # Enforce limit
    if len(images) > MAX_OUTPUT_FILES:
        for _, raw_path, _ in images[MAX_OUTPUT_FILES:]:
            for artifact_path in related_image_artifact_paths(raw_path):
                try:
                    if os.path.exists(artifact_path):
                        os.remove(artifact_path)
                except OSError:
                    pass
        images = images[:MAX_OUTPUT_FILES]
        
    return [(preview_path, os.path.basename(raw_path)) for preview_path, raw_path, _ in images[:MAX_GALLERY_IMAGES]]


















__all__ = (
    'VideoThumbnailer',
    'delete_image',
    'delete_all_images',
    'delete_all_videos',
    'get_video_gallery_images',
    'get_gallery_images',
)
_seal_runtime_module(globals())
