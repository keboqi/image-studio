"""Output paths, containment, previews, and atomic image saves."""

from __future__ import annotations
from image_studio.errors import UserInputError

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

OUTPUT_PREVIEW_SUFFIX = "_preview.webp"
OUTPUT_PREVIEW_QUALITY = 90


@dataclass(frozen=True)
class OutputStore:
    root: str
    preview_suffix: str = OUTPUT_PREVIEW_SUFFIX
    preview_quality: int = OUTPUT_PREVIEW_QUALITY

    def path_from_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if "image" in value:
                return self.path_from_value(value["image"])
            for key in ("path", "name"):
                if value.get(key):
                    return value[key]
        for key in ("path", "name"):
            path = getattr(value, key, None)
            if path:
                return path
        return value

    def path_for_basename(self, path: str) -> str | None:
        candidate = os.path.join(self.root, os.path.basename(path))
        return candidate if os.path.exists(candidate) else None

    def preview_path(self, raw_path: str) -> str:
        stem, _ = os.path.splitext(raw_path)
        return f"{stem}{self.preview_suffix}"

    def raw_path(self, path: str) -> str:
        if not path.lower().endswith(self.preview_suffix):
            return path if os.path.exists(path) else (self.path_for_basename(path) or path)
        base = path[: -len(self.preview_suffix)]
        for extension in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = f"{base}{extension}"
            existing = candidate if os.path.exists(candidate) else self.path_for_basename(candidate)
            if existing:
                return existing
        return path

    def resolve_payload(self, value: Any) -> Any:
        path = self.path_from_value(value)
        return self.raw_path(path) if isinstance(path, str) else value

    def contained_path(self, path: str) -> str | None:
        if not path:
            return None
        try:
            root = os.path.normcase(os.path.realpath(os.path.abspath(self.root)))
            candidate = os.path.realpath(os.path.abspath(path))
            if os.path.commonpath([root, os.path.normcase(candidate)]) != root:
                return None
            return candidate
        except (OSError, ValueError):
            return None

    def save_preview(self, preview_path: str, image: Image.Image) -> None:
        temporary = f"{preview_path}.tmp"
        image.convert("RGB").save(
            temporary, format="WEBP", quality=self.preview_quality, method=6
        )
        os.replace(temporary, preview_path)

    def ensure_preview(self, raw_path: str) -> str:
        if not raw_path or raw_path.endswith(".thumb.jpg"):
            return raw_path
        raw_path = self.raw_path(raw_path)
        if raw_path.lower().endswith(self.preview_suffix):
            return raw_path
        preview = self.preview_path(raw_path)
        try:
            if os.path.exists(preview) and os.path.getmtime(preview) >= os.path.getmtime(raw_path):
                return preview
            with Image.open(raw_path) as image:
                self.save_preview(preview, image)
            return preview
        except Exception as exc:
            log.warning("Could not create WebP preview for %s: %s", raw_path, exc)
            return raw_path

    def related_paths(self, path: str) -> list[str]:
        raw = self.raw_path(path)
        paths = [raw, self.preview_path(raw)]
        if path not in paths:
            paths.append(path)
        return list(dict.fromkeys(paths))

    def save_image_pair(self, prefix: str, image: Image.Image) -> tuple[str, str]:
        os.makedirs(self.root, exist_ok=True)
        stem = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        raw = os.path.join(self.root, f"{stem}.png")
        preview = self.preview_path(raw)
        temporary = f"{raw}.tmp"
        image.save(temporary, format="PNG")
        os.replace(temporary, raw)
        self.save_preview(preview, image)
        return preview, raw


# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _path_from_gradio_value(value: Any) -> Any:
    return output_store.path_from_value(value)

def _output_dir_path_for_basename(path: str) -> str | None:
    return output_store.path_for_basename(path)

def preview_path_for_raw_image(raw_path: str) -> str:
    return output_store.preview_path(raw_path)

def raw_image_path_for_preview(path: str) -> str:
    return output_store.raw_path(path)

def resolve_raw_image_payload(value: Any) -> Any:
    return output_store.resolve_payload(value)

def _resolve_output_file_path(path: str) -> str | None:
    return output_store.contained_path(path)

def _ideogram4_prompt_looks_like_json(prompt: Any) -> bool:
    if not isinstance(prompt, str):
        return False
    text = prompt.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text.startswith("{")

def _ideogram4_editor_prompt_from_candidates(*prompts: Any) -> str:
    for prompt in prompts:
        if _ideogram4_prompt_looks_like_json(prompt):
            return str(prompt).strip()
    return ""

def _write_ideogram4_prompt_metadata(raw_path: str, metadata: dict[str, Any]) -> None:
    try:
        raw_path = raw_image_path_for_preview(raw_path)
        _ideogram4_prompt_metadata_store.write(raw_path, metadata)
    except OSError as exc:
        log.warning("Could not write Ideogram prompt metadata for %s: %s", raw_path, exc)

def _read_ideogram4_prompt_metadata(raw_path: Any) -> dict[str, Any] | None:
    raw_path = resolve_raw_image_payload(raw_path)
    if not isinstance(raw_path, str):
        return None
    return _ideogram4_prompt_metadata_store.read(raw_image_path_for_preview(raw_path))

def require_gallery_image_path(path: Any) -> str:
    raw_path = resolve_raw_image_payload(path)
    if not isinstance(raw_path, str):
        raise UserInputError("Please select an image in the gallery first.")
    safe_path = _resolve_output_file_path(raw_path)
    if not safe_path or not os.path.isfile(safe_path):
        raise UserInputError("Selected gallery image is unavailable or outside the output folder.")
    return safe_path

def _save_webp_preview(preview_path: str, image: Image.Image) -> None:
    output_store.save_preview(preview_path, image)

def ensure_webp_preview(raw_path: str) -> str:
    return output_store.ensure_preview(raw_path)

def related_image_artifact_paths(path: str) -> list[str]:
    return output_store.related_paths(path)

def coerce_rgb_image(image: Any) -> Image.Image:
    image = resolve_raw_image_payload(image)
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image.convert("RGB")

def collect_rgb_images(*images: Any) -> list[Image.Image]:
    return [coerce_rgb_image(image) for image in images if image is not None]

def save_output_image_pair(prefix: str, image: Image.Image) -> tuple[str, str]:
    return output_store.save_image_pair(prefix, image)

__all__ = (
    '_path_from_gradio_value',
    '_output_dir_path_for_basename',
    'preview_path_for_raw_image',
    'raw_image_path_for_preview',
    'resolve_raw_image_payload',
    '_resolve_output_file_path',
    '_ideogram4_prompt_looks_like_json',
    '_ideogram4_editor_prompt_from_candidates',
    '_write_ideogram4_prompt_metadata',
    '_read_ideogram4_prompt_metadata',
    'require_gallery_image_path',
    '_save_webp_preview',
    'ensure_webp_preview',
    'related_image_artifact_paths',
    'coerce_rgb_image',
    'collect_rgb_images',
    'save_output_image_pair',
)
_seal_runtime_module(globals())
