"""Ideogram caption cleanup, normalization, and parsing."""

from __future__ import annotations
from image_studio.errors import UserInputError

import json
import re
from collections.abc import Callable
from typing import Any

from ...parsing import extract_json_object, strip_markdown_fences


def clean_malformed_json_caption(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = text.replace(',""}', "}").replace(',""]', "]")
        text = re.sub(r'([{\[,]\s*)"{2,}(?=[A-Za-z_][A-Za-z0-9_]*"\s*:)', r'\1"', text)
        text = re.sub(r',\s*"+\s*(?=[}\]])', "", text)
        text = re.sub(r'""([a-zA-Z0-9_]+)""?\s*:', r'"\1":', text)
    return text


def normalize_caption_object(
    caption: Any,
    aspect_ratio: str,
    *,
    reorder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(caption, dict):
        raise TypeError(f"caption must be a JSON object, got {type(caption).__name__}")
    normalized: dict[str, Any] = {"aspect_ratio": caption.get("aspect_ratio") or aspect_ratio}
    if isinstance(caption.get("high_level_description"), str):
        normalized["high_level_description"] = caption["high_level_description"]
    style = caption.get("style_description")
    if isinstance(style, dict):
        allowed = {"aesthetics", "lighting", "photo", "art_style", "medium", "color_palette"}
        normalized["style_description"] = {key: style[key] for key in style if key in allowed}
    composition = caption.get("compositional_deconstruction")
    if isinstance(composition, dict):
        normalized_composition: dict[str, Any] = {}
        if isinstance(composition.get("background"), str):
            normalized_composition["background"] = composition["background"]
        if isinstance(composition.get("elements"), list):
            elements = []
            for element in composition["elements"]:
                if not isinstance(element, dict):
                    continue
                element_type = element.get("type")
                if element_type not in {"obj", "text"}:
                    element_type = "text" if "text" in element else "obj"
                item: dict[str, Any] = {"type": element_type}
                if "bbox" in element:
                    item["bbox"] = element["bbox"]
                if element_type == "text":
                    item["text"] = element.get("text", "")
                item["desc"] = element.get("desc", "")
                if "color_palette" in element:
                    item["color_palette"] = element["color_palette"]
                elements.append(item)
            normalized_composition["elements"] = elements
        normalized["compositional_deconstruction"] = normalized_composition
    return reorder(normalized) if reorder else normalized


def parse_caption(
    text: str,
    aspect_ratio: str,
    *,
    reorder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> str:
    stripped = strip_markdown_fences(text)
    source = extract_json_object(stripped) or stripped
    candidates = [source]
    cleaned = clean_malformed_json_caption(source)
    if cleaned != source:
        candidates.append(cleaned)
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
    else:
        raise last_error or ValueError("No JSON object found.")
    normalized = normalize_caption_object(parsed, aspect_ratio, reorder=reorder)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


pure_extract_json_object = extract_json_object

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def extract_json_object(text: str) -> str | None:
    return pure_extract_json_object(text)

def _ideogram4_aspect_ratio(width: int, height: int) -> str:
    width = int(width)
    height = int(height)
    divisor = math.gcd(width, height) or 1
    return f"{width // divisor}:{height // divisor}"

def _ideogram4_strip_markdown_fences(text: str) -> str:
    return _strip_markdown_fences(text)

def _ideogram4_extract_json_object(text: str) -> str:
    stripped = _ideogram4_strip_markdown_fences(text)
    return extract_json_object(stripped) or stripped

def _ideogram4_clean_malformed_json_caption(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = text.replace(',""}', "}").replace(',""]', "]")
        text = re.sub(r'([{\[,])\s*"{2,}(?=[A-Za-z_][A-Za-z0-9_]*"\s*:)', r'\1"', text)
        text = re.sub(r',\s*"+\s*(?=[}\]])', "", text)
        text = re.sub(r'""([a-zA-Z0-9_]+)""?\s*:', r'"\1":', text)
    return text

def _ideogram4_normalize_caption_object(caption: Any, aspect_ratio: str) -> dict:
    if not isinstance(caption, dict):
        raise TypeError(f"caption must be a JSON object, got {type(caption).__name__}")

    normalized: dict[str, Any] = {"aspect_ratio": caption.get("aspect_ratio") or aspect_ratio}
    if isinstance(caption.get("high_level_description"), str):
        normalized["high_level_description"] = caption["high_level_description"]

    style_description = caption.get("style_description")
    if isinstance(style_description, dict):
        allowed_style_keys = {
            "aesthetics",
            "lighting",
            "photo",
            "art_style",
            "medium",
            "color_palette",
        }
        normalized["style_description"] = {
            key: style_description[key]
            for key in style_description
            if key in allowed_style_keys
        }

    cd = caption.get("compositional_deconstruction")
    if isinstance(cd, dict):
        normalized_cd: dict[str, Any] = {}
        if isinstance(cd.get("background"), str):
            normalized_cd["background"] = cd["background"]
        elements = cd.get("elements")
        if isinstance(elements, list):
            normalized_elements = []
            for element in elements:
                if not isinstance(element, dict):
                    continue
                element_type = element.get("type")
                if element_type not in {"obj", "text"}:
                    element_type = "text" if "text" in element else "obj"
                normalized_element: dict[str, Any] = {"type": element_type}
                if "bbox" in element:
                    normalized_element["bbox"] = element["bbox"]
                if element_type == "text":
                    normalized_element["text"] = element.get("text", "")
                normalized_element["desc"] = element.get("desc", "")
                if "color_palette" in element:
                    normalized_element["color_palette"] = element["color_palette"]
                normalized_elements.append(normalized_element)
            normalized_cd["elements"] = normalized_elements
        normalized["compositional_deconstruction"] = normalized_cd

    return _get_ideogram4()["reorder_caption_keys"](normalized)

def _ideogram4_parse_caption(text: str, aspect_ratio: str) -> str:
    return parse_caption(
        text,
        aspect_ratio,
        reorder=_get_ideogram4()["reorder_caption_keys"],
    )

def _ideogram4_repair_caption(raw_text: str, aspect_ratio: str) -> str:
    try:
        from json_repair import repair_json
    except ImportError as exc:
        raise UserInputError("Install json-repair on the server for malformed local Gemma JSON repair.") from exc

    text = _ideogram4_clean_malformed_json_caption(_ideogram4_extract_json_object(raw_text))
    repaired_text = repair_json(text, ensure_ascii=False, skip_json_loads=True)
    parsed = json.loads(repaired_text)
    parsed = _ideogram4_normalize_caption_object(parsed, aspect_ratio)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

def _ideogram4_load_upsample_cache() -> dict:
    return _ideogram4_upsample_cache.load()

def _ideogram4_upsample_cache_model(upsampler: str, chat_model: str | None = None) -> str:
    if upsampler == IDEOGRAM4_UPSAMPLE_GEMMA:
        return _normalize_chat_gemma_choice(chat_model or _chat_selector.choice)
    return ""

def _ideogram4_upsample_cache_key(
    prompt: str,
    upsampler: str,
    cache_model: str = "",
) -> str:
    payload = {
        "prompt": prompt or "",
        "upsampler": upsampler,
    }
    if cache_model:
        payload["chat_model"] = cache_model
    material = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

def _ideogram4_get_cached_upsample(
    prompt: str,
    upsampler: str,
    cache_model: str = "",
) -> str | None:
    key = _ideogram4_upsample_cache_key(prompt, upsampler, cache_model)
    entry = _ideogram4_load_upsample_cache().get(key)
    if isinstance(entry, dict) and isinstance(entry.get("result"), str):
        log.info("Reusing cached Ideogram prompt upsample (%s).", upsampler)
        return entry["result"]
    return None

def _ideogram4_store_cached_upsample(
    prompt: str,
    upsampler: str,
    result: str,
    cache_model: str = "",
):
    key = _ideogram4_upsample_cache_key(prompt, upsampler, cache_model)
    _ideogram4_upsample_cache.set(
        key,
        {
            "result": result,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

def _ideogram4_local_upsample_gemma(
    prompt: str,
    width: int,
    height: int,
    max_new_tokens: int,
    enable_thinking: bool,
    chat_model: str | None = None,
) -> str:
    aspect_ratio = _ideogram4_aspect_ratio(width, height)
    messages = _get_ideogram4()["build_messages"]("v1.txt", prompt, aspect_ratio)
    raw = _gemma_generate(
        messages,
        max_new_tokens=max(512, int(max_new_tokens)),
        enable_thinking=bool(enable_thinking),
        do_sample=False,
        chat_model=chat_model,
    )
    try:
        return _ideogram4_parse_caption(raw, aspect_ratio)
    except Exception as parse_error:
        log.warning("Local Gemma Ideogram caption parse failed; trying json-repair: %s", parse_error)
        return _ideogram4_repair_caption(raw, aspect_ratio)

def _ideogram4_remote_api_key(api_key: str = "") -> str:
    key = (api_key or "").strip() or APP_CONFIG.ideogram.api_key
    if key:
        return key
    local_key_path = os.path.join(BASE_DIR, "app_standalone_api_key.txt")
    try:
        with open(local_key_path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""

def _ideogram4_default_upsampler() -> str:
    return IDEOGRAM4_UPSAMPLE_REMOTE if _ideogram4_remote_api_key() else IDEOGRAM4_UPSAMPLE_GEMMA

def _ideogram4_remote_upsample(prompt: str, width: int, height: int, api_key: str = "") -> str:
    key = _ideogram4_remote_api_key(api_key)
    if not key:
        raise UserInputError("Set IDEOGRAM_API_KEY or enter an Ideogram API key for remote prompt upsampling.")
    aspect_ratio = _ideogram4_aspect_ratio(width, height)
    magic_prompt_cls = _get_ideogram4()["MAGIC_PROMPTS"]["ideogram-4-v1"]
    try:
        magic = magic_prompt_cls(api_key=key, strip_bboxes=False)
    except TypeError:
        magic = magic_prompt_cls(api_key=key)
    return magic.expand(prompt, aspect_ratio=aspect_ratio)

def _ideogram4_upsample_prompt(
    prompt: str,
    upsampler: str,
    width: int,
    height: int,
    gemma_max_new_tokens: int,
    gemma_enable_thinking: bool,
    reuse_cache: bool,
    api_key: str = "",
    chat_model: str | None = None,
) -> str:
    if upsampler == IDEOGRAM4_UPSAMPLE_NONE:
        return prompt
    if upsampler not in IDEOGRAM4_UPSAMPLERS:
        raise UserInputError(f"Unknown Ideogram prompt upsampler: {upsampler}")
    cache_model = _ideogram4_upsample_cache_model(upsampler, chat_model)
    upsample_t0 = time.time()
    log.info(
        "Ideogram prompt upsampling started | upsampler=%s | size=%sx%s | cache=%s | gemma_tokens=%s | gemma_thinking=%s",
        upsampler,
        int(width),
        int(height),
        bool(reuse_cache),
        int(gemma_max_new_tokens),
        bool(gemma_enable_thinking),
    )
    if reuse_cache:
        cached = _ideogram4_get_cached_upsample(prompt, upsampler, cache_model)
        if cached:
            log.info(
                "Ideogram prompt upsampling finished | upsampler=%s | source=cache | elapsed=%.2fs | prompt_chars=%d | result_chars=%d",
                upsampler,
                time.time() - upsample_t0,
                len(prompt or ""),
                len(cached or ""),
            )
            return cached

    if upsampler == IDEOGRAM4_UPSAMPLE_REMOTE:
        source = "ideogram_api"
    else:
        source = f"chat_model:{cache_model}"
    try:
        if upsampler == IDEOGRAM4_UPSAMPLE_REMOTE:
            result = _ideogram4_remote_upsample(prompt, width, height, api_key)
        else:
            result = _ideogram4_local_upsample_gemma(
                prompt, width, height, gemma_max_new_tokens, gemma_enable_thinking,
                chat_model=cache_model or chat_model,
            )
    except Exception:
        log.exception(
            "Ideogram prompt upsampling failed | upsampler=%s | source=%s | elapsed=%.2fs",
            upsampler,
            source,
            time.time() - upsample_t0,
        )
        raise

    log.info(
        "Ideogram prompt upsampling finished | upsampler=%s | source=%s | elapsed=%.2fs | prompt_chars=%d | result_chars=%d",
        upsampler,
        source,
        time.time() - upsample_t0,
        len(prompt or ""),
        len(result or ""),
    )
    _ideogram4_store_cached_upsample(prompt, upsampler, result, cache_model)
    return result

def _ideogram4_looks_like_json_caption(caption: str) -> bool:
    return _ideogram4_strip_markdown_fences(caption).lstrip().startswith("{")

def _ideogram4_normalize_caption_for_model(
    caption: str,
    width: int,
    height: int,
    strip_prompt: bool,
) -> str:
    if not _ideogram4_looks_like_json_caption(caption):
        return caption

    aspect_ratio = _ideogram4_aspect_ratio(width, height)
    try:
        caption = _ideogram4_parse_caption(caption, aspect_ratio)
    except Exception:
        caption = _ideogram4_repair_caption(caption, aspect_ratio)

    mods = _get_ideogram4()
    if strip_prompt:
        return mods["strip_aspect_ratio_and_bboxes"](caption)
    return mods["strip_aspect_ratio"](caption)

def _ideogram4_plain_prompt_designer_caption(prompt: str) -> str:
    caption = {
        "high_level_description": (prompt or "").strip(),
        "style_description": {
            "aesthetics": "",
            "lighting": "",
            "photo": "",
            "medium": "photograph",
            "color_palette": [],
        },
        "compositional_deconstruction": {
            "background": "",
            "elements": [],
        },
    }
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))

def _ideogram4_cached_editor_prompt(prompt: str, upsampler: str) -> tuple[str, str]:
    if not prompt or upsampler == IDEOGRAM4_UPSAMPLE_NONE:
        return "", ""
    if upsampler not in IDEOGRAM4_UPSAMPLERS:
        return "", ""
    cache_model = _ideogram4_upsample_cache_model(upsampler)
    cached = _ideogram4_get_cached_upsample(prompt, upsampler, cache_model)
    if not cached:
        return "", ""
    return cached, "upsample_cache"

def prepare_ideogram_json_designer_payload(
    prompt: str,
    width: int,
    height: int,
    upsampler: str,
    raw_path: Any,
) -> str:
    """Build the browser payload used to open the external Ideogram JSON editor."""
    width, height = int(width or 1024), int(height or 1024)
    aspect_ratio = _ideogram4_aspect_ratio(width, height)
    prompt = (prompt or "").strip()

    caption = ""
    source = ""
    metadata = _read_ideogram4_prompt_metadata(raw_path)
    if metadata:
        caption = _ideogram4_editor_prompt_from_candidates(
            metadata.get("editor_prompt"),
            metadata.get("upsampled_prompt"),
            metadata.get("model_prompt"),
            metadata.get("source_prompt"),
        )
        if caption:
            source = "last_generation"
            width = int(metadata.get("width") or width)
            height = int(metadata.get("height") or height)
            aspect_ratio = _ideogram4_aspect_ratio(width, height)

    if not caption:
        caption, source = _ideogram4_cached_editor_prompt(prompt, upsampler or IDEOGRAM4_UPSAMPLE_NONE)

    if not caption and _ideogram4_prompt_looks_like_json(prompt):
        caption = prompt
        source = "current_prompt_json"

    if not caption:
        caption = _ideogram4_plain_prompt_designer_caption(prompt)
        source = "plain_prompt"

    payload = {
        "url": IDEOGRAM4_JSON_DESIGNER_PATH,
        "caption": caption,
        "plain_prompt": prompt,
        "source": source,
        "aspect": aspect_ratio,
        "width": width,
        "height": height,
        "apply": {
            "upsampler": IDEOGRAM4_UPSAMPLE_NONE,
            "strip_prompt": False,
        },
    }
    return json.dumps(payload, ensure_ascii=False)

__all__ = (
    '_strip_markdown_fences',
    'extract_json_object',
    '_ideogram4_aspect_ratio',
    '_ideogram4_strip_markdown_fences',
    '_ideogram4_extract_json_object',
    '_ideogram4_clean_malformed_json_caption',
    '_ideogram4_normalize_caption_object',
    '_ideogram4_parse_caption',
    '_ideogram4_repair_caption',
    '_ideogram4_load_upsample_cache',
    '_ideogram4_upsample_cache_model',
    '_ideogram4_upsample_cache_key',
    '_ideogram4_get_cached_upsample',
    '_ideogram4_store_cached_upsample',
    '_ideogram4_local_upsample_gemma',
    '_ideogram4_remote_api_key',
    '_ideogram4_default_upsampler',
    '_ideogram4_remote_upsample',
    '_ideogram4_upsample_prompt',
    '_ideogram4_looks_like_json_caption',
    '_ideogram4_normalize_caption_for_model',
    '_ideogram4_plain_prompt_designer_caption',
    '_ideogram4_cached_editor_prompt',
    'prepare_ideogram_json_designer_payload',
)
_seal_runtime_module(globals())
