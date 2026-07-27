"""Extracted runtime implementation."""

from __future__ import annotations

from typing import Any

# --- extracted runtime implementation ---
from image_studio.infra.model_storage import NONE_CHOICE, format_storage_size
from image_studio.runtime_access import runtime_namespace as _runtime


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")

def _storage_location_label(paths: list[str]) -> str:
    if not paths:
        return ""
    first = paths[0]
    if len(paths) == 1:
        return first
    return f"{first} (+{len(paths) - 1} more)"

def _build_models_md() -> str:
    """Render the Models tab content as Markdown."""
    gpu = _runtime().model_mgr.gpu_summary()
    models = _runtime().model_mgr.status()
    downloaded = _runtime().MODEL_STORAGE.status()

    video_healthy, is_pipeline_loaded = _runtime().check_ltx_video_health()
    ltx_loaded = video_healthy and is_pipeline_loaded
    vllm_healthy = _runtime()._diffusiongemma_vllm_service.is_healthy()
    vllm_sleeping = _runtime()._diffusiongemma_vllm_service.is_sleeping()
    vllm_ready = vllm_healthy and not vllm_sleeping
    krea2_ready = _runtime()._krea2_comfy_service.is_healthy()
    tracked_names = {m["name"] for m in models}

    used_pct = (gpu["used_mb"] / gpu["total_mb"] * 100) if gpu["total_mb"] else 0
    bar_len = 30
    filled = int(bar_len * used_pct / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    lines = [
        "### GPU Memory",
        "```",
        f"{bar}  {used_pct:.0f}%",
        f"Used: {gpu['used_mb']:,} MiB  /  Total: {gpu['total_mb']:,} MiB  /  Free: {gpu['free_mb']:,} MiB",
        f"Tracked by manager: {gpu['tracked_mb']:,} MiB across {gpu['model_count']} model(s)",
        "```",
        "",
    ]

    if models or ltx_loaded or vllm_ready or krea2_ready:
        lines.append("### Active Models")
        lines.append("| Model | VRAM (MiB) | Loaded | Last Used |")
        lines.append("|-------|-----------|--------|-----------|")
        for m in models:
            display = _runtime()._model_display_name(m["name"])
            lines.append(
                f"| {display} | ~{m['vram_mb']:,} | {m['loaded_at']} | {m['last_used']} |"
            )
        if ltx_loaded:
            lines.append("| LTX-Web Video Pipeline | ~18,000 | (subprocess) | (subprocess) |")
        if vllm_ready and _runtime().MODEL_DIFFUSIONGEMMA_VLLM not in tracked_names:
            lines.append("| DiffusionGemma vLLM | ~26,000 | (Docker) | (OpenAI API) |")
        if krea2_ready and _runtime().MODEL_KREA2_TURBO_NVFP4 not in tracked_names:
            lines.append("| Krea2 Turbo (ComfyUI) | ~38,000 | (venv subprocess) | (ComfyUI API) |")
    else:
        lines.append("*No models currently loaded.*")

    if vllm_sleeping:
        lines.extend([
            "",
            "### Sleeping Services",
            "- DiffusionGemma vLLM is sleeping; the server process is retained and VRAM is mostly released.",
        ])

    lines.extend(["", "### Downloaded Model Files"])
    if downloaded:
        total_size = sum(int(item["size_bytes"]) for item in downloaded)
        lines.append(f"Total detected: **{format_storage_size(total_size)}**")
        lines.append("| Model | Disk Usage | Location |")
        lines.append("|-------|------------|----------|")
        for item in downloaded:
            lines.append(
                "| "
                f"{_md_cell(item['display_name'])} | "
                f"{format_storage_size(int(item['size_bytes']))} | "
                f"{_md_cell(_storage_location_label(item['paths']))} |"
            )
    else:
        lines.append("*No downloaded model files found in known cache locations.*")

    return "\n".join(lines)

def _unload_model_action(model_key: str) -> str:
    """Unload a specific model and return refreshed status."""
    if not model_key or model_key == "(none)":
        return _build_models_md()
    
    if model_key == "ltx_video_pipeline":
        _runtime().unload_ltx_video_pipeline()
    elif model_key == _runtime().MODEL_DIFFUSIONGEMMA_VLLM:
        if _runtime().model_mgr.is_loaded(_runtime().MODEL_DIFFUSIONGEMMA_VLLM):
            _runtime().model_mgr.unload(_runtime().MODEL_DIFFUSIONGEMMA_VLLM)
        else:
            _runtime()._diffusiongemma_vllm_service.stop()
    elif model_key == _runtime().MODEL_KREA2_TURBO_NVFP4:
        if _runtime().model_mgr.is_loaded(_runtime().MODEL_KREA2_TURBO_NVFP4):
            _runtime().model_mgr.unload(_runtime().MODEL_KREA2_TURBO_NVFP4)
        else:
            _runtime()._krea2_comfy_service.stop()
    else:
        _runtime().model_mgr.unload(model_key)
        
    return _build_models_md()

def _unload_all_action() -> str:
    _runtime().model_mgr.unload_all()
    _runtime().unload_ltx_video_pipeline()
    if _runtime()._diffusiongemma_vllm_service.is_healthy() or _runtime()._diffusiongemma_vllm_service.is_control_reachable():
        _runtime()._diffusiongemma_vllm_service.stop()
    if _runtime()._krea2_comfy_service.is_healthy():
        _runtime()._krea2_comfy_service.stop()
    return _build_models_md()

def _get_loaded_model_choices() -> list[str]:
    """Return the keys of currently loaded models for the dropdown."""
    keys = _runtime().model_mgr.keys()
    
    video_healthy, is_pipeline_loaded = _runtime().check_ltx_video_health()
    if video_healthy and is_pipeline_loaded:
        keys.append("ltx_video_pipeline")
    if _runtime()._diffusiongemma_vllm_service.is_ready() and _runtime().MODEL_DIFFUSIONGEMMA_VLLM not in keys:
        keys.append(_runtime().MODEL_DIFFUSIONGEMMA_VLLM)
    if _runtime()._krea2_comfy_service.is_healthy() and _runtime().MODEL_KREA2_TURBO_NVFP4 not in keys:
        keys.append(_runtime().MODEL_KREA2_TURBO_NVFP4)
        
    return keys if keys else ["(none)"]

def _get_downloaded_model_choices():
    entries = _runtime().MODEL_STORAGE.status()
    if not entries:
        return [NONE_CHOICE]
    return [
        (
            f"{entry['display_name']} ({format_storage_size(int(entry['size_bytes']))})",
            entry["key"],
        )
        for entry in entries
    ]

def _loaded_model_picker_update():
    return _runtime().gr.update(choices=_get_loaded_model_choices(), value=None)

def _downloaded_model_picker_update():
    return _runtime().gr.update(choices=_get_downloaded_model_choices(), value=None)

def _models_tab_refresh_result():
    return _build_models_md(), _loaded_model_picker_update(), _downloaded_model_picker_update()

def _stop_active_model_for_storage_cleanup(model_key: str) -> None:
    if model_key == _runtime().MODEL_DIFFUSIONGEMMA_VLLM:
        if _runtime().model_mgr.is_loaded(model_key):
            _runtime().model_mgr.unload(model_key)
        _runtime()._diffusiongemma_vllm_service.stop_process()
        return
    if model_key == _runtime().MODEL_KREA2_TURBO_NVFP4:
        if _runtime().model_mgr.is_loaded(model_key):
            _runtime().model_mgr.unload(model_key)
        else:
            _runtime()._krea2_comfy_service.stop()
        return
    if model_key == _runtime().MODEL_LTX_VIDEO:
        if _runtime().model_mgr.is_loaded(model_key):
            _runtime().model_mgr.unload(model_key)
        _runtime()._ltx_video_service.stop()
        return
    _runtime().model_mgr.unload(model_key)

def _prepare_storage_cleanup(target_key: str) -> None:
    if not target_key or target_key == NONE_CHOICE:
        return
    for model_key in _runtime().MODEL_STORAGE.active_model_keys(target_key):
        _stop_active_model_for_storage_cleanup(model_key)
    if target_key == "ideogram4_realism_lora":
        _runtime()._ideogram4_lora_cache.clear()

def _remove_downloaded_model_files_action(target_key: str) -> str:
    _prepare_storage_cleanup(target_key)
    _runtime().MODEL_STORAGE.remove(target_key)
    return _build_models_md()

def _remove_all_downloaded_model_files_action() -> str:
    for item in _runtime().MODEL_STORAGE.status():
        _prepare_storage_cleanup(item["key"])
    try:
        _runtime()._ideogram4_lora_cache.clear()
    except Exception:
        pass
    _runtime().MODEL_STORAGE.remove_all()
    return _build_models_md()

def _build_vram_widget_md() -> str:
    """Render a compact VRAM status for the header widget."""
    gpu = _runtime().model_mgr.gpu_summary()
    if not gpu["total_mb"]:
        return "**GPU VRAM:** N/A"
    used_pct = (gpu["used_mb"] / gpu["total_mb"] * 100) if gpu["total_mb"] else 0
    return (
        f"**VRAM Usage:**  \n"
        f"**{gpu['used_mb']:,}** / **{gpu['total_mb']:,}** MiB ({used_pct:.0f}%)  \n"
        f"Tracked: {gpu['tracked_mb']:,} MiB ({gpu['model_count']} models)"
    )

def refresh_models_tab():
    return _models_tab_refresh_result()

def unload_model_and_refresh(key):
    _unload_model_action(key)
    return _models_tab_refresh_result()

def unload_all_and_refresh():
    _unload_all_action()
    return _models_tab_refresh_result()

def remove_downloaded_model_files_and_refresh(key):
    _remove_downloaded_model_files_action(key)
    return _models_tab_refresh_result()

def remove_all_downloaded_model_files_and_refresh():
    _remove_all_downloaded_model_files_action()
    return _models_tab_refresh_result()

__all__ = (
    '_build_models_md',
    '_unload_model_action',
    '_unload_all_action',
    '_get_loaded_model_choices',
    '_get_downloaded_model_choices',
    '_build_vram_widget_md',
    'refresh_models_tab',
    'unload_model_and_refresh',
    'unload_all_and_refresh',
    'remove_downloaded_model_files_and_refresh',
    'remove_all_downloaded_model_files_and_refresh',
)
