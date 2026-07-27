"""Generated-media gallery persistence."""

from __future__ import annotations

import logging
import os
import subprocess

from PIL import Image

from .output_store import (
    OUTPUT_PREVIEW_SUFFIX,
    _resolve_output_file_path,
    ensure_webp_preview,
    related_image_artifact_paths,
)

log = logging.getLogger(__name__)

MAX_GALLERY_IMAGES = 50
MAX_OUTPUT_FILES = 500
GALLERY_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
GALLERY_VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov")

_output_dir: str | None = None


def configure_gallery(output_dir: str) -> None:
    global _output_dir
    _output_dir = output_dir


def _root() -> str:
    if _output_dir is None:
        raise RuntimeError("Gallery storage is not configured.")
    return _output_dir


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
                subprocess.run(
                    command,
                    capture_output=True,
                    timeout=self.timeout,
                    check=False,
                )
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


def delete_image(path: object):
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
    except (OSError, ValueError) as exc:
        log.error("Error deleting image %s: %s", path, exc)
    return get_gallery_images()


def delete_all_images():
    root = _root()
    if not os.path.exists(root):
        return []
    for name in os.listdir(root):
        if name.lower().endswith(GALLERY_IMAGE_EXTENSIONS) and not name.endswith(
            ".thumb.jpg"
        ):
            try:
                os.remove(os.path.join(root, name))
            except OSError as exc:
                log.warning("Could not delete image %s: %s", name, exc)
    return get_gallery_images()


def delete_all_videos():
    root = _root()
    if not os.path.exists(root):
        return []
    for name in os.listdir(root):
        if name.lower().endswith(GALLERY_VIDEO_EXTENSIONS) or name.endswith(".thumb.jpg"):
            try:
                os.remove(os.path.join(root, name))
            except OSError as exc:
                log.warning("Could not delete video artifact %s: %s", name, exc)
    return []


def get_video_gallery_images():
    root = _root()
    if not os.path.exists(root):
        return []
    with os.scandir(root) as entries:
        videos = [
            (entry.path, entry.stat().st_mtime)
            for entry in entries
            if entry.name.lower().endswith(GALLERY_VIDEO_EXTENSIONS)
        ]
    videos.sort(key=lambda item: item[1], reverse=True)
    items = []
    for path, _mtime in videos[:MAX_GALLERY_IMAGES]:
        thumbnail = video_thumbnailer.ensure(path)
        if thumbnail:
            items.append((thumbnail, os.path.basename(path)))
    return items


def get_gallery_images():
    root = _root()
    if not os.path.exists(root):
        return []
    with os.scandir(root) as entries:
        raw_images = []
        for entry in entries:
            lower = entry.name.lower()
            if not lower.endswith(GALLERY_IMAGE_EXTENSIONS):
                continue
            if entry.name.endswith(".thumb.jpg") or lower.endswith(OUTPUT_PREVIEW_SUFFIX):
                continue
            raw_images.append((entry.path, entry.stat().st_mtime))

    images = [
        (ensure_webp_preview(raw_path), raw_path, modified)
        for raw_path, modified in raw_images
    ]
    images.sort(key=lambda item: item[2], reverse=True)

    if len(images) > MAX_OUTPUT_FILES:
        for _preview, raw_path, _modified in images[MAX_OUTPUT_FILES:]:
            for artifact_path in related_image_artifact_paths(raw_path):
                try:
                    if os.path.exists(artifact_path):
                        os.remove(artifact_path)
                except OSError as exc:
                    log.warning("Could not prune gallery artifact %s: %s", artifact_path, exc)
        images = images[:MAX_OUTPUT_FILES]

    return [
        (preview, os.path.basename(raw))
        for preview, raw, _modified in images[:MAX_GALLERY_IMAGES]
    ]


__all__ = (
    "VideoThumbnailer",
    "configure_gallery",
    "delete_all_images",
    "delete_all_videos",
    "delete_image",
    "get_gallery_images",
    "get_video_gallery_images",
)
