"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _build_models_md() -> str:
    """Render the Models tab content as Markdown."""
    gpu = model_mgr.gpu_summary()
    models = model_mgr.status()

    video_healthy, is_pipeline_loaded = check_ltx_video_health()
    ltx_loaded = video_healthy and is_pipeline_loaded
    vllm_healthy = _diffusiongemma_vllm_service.is_healthy()
    vllm_sleeping = _diffusiongemma_vllm_service.is_sleeping()
    vllm_ready = vllm_healthy and not vllm_sleeping
    krea2_ready = _krea2_comfy_service.is_healthy()
    tracked_names = {m["name"] for m in models}

    used_pct = (gpu["used_mb"] / gpu["total_mb"] * 100) if gpu["total_mb"] else 0
    bar_len = 30
    filled = int(bar_len * used_pct / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    lines = [
        "### GPU Memory",
        f"```",
        f"{bar}  {used_pct:.0f}%",
        f"Used: {gpu['used_mb']:,} MiB  /  Total: {gpu['total_mb']:,} MiB  /  Free: {gpu['free_mb']:,} MiB",
        f"Tracked by manager: {gpu['tracked_mb']:,} MiB across {gpu['model_count']} model(s)",
        f"```",
        "",
    ]

    if models or ltx_loaded or vllm_ready or krea2_ready:
        lines.append("### Active Models")
        lines.append("| Model | VRAM (MiB) | Loaded | Last Used |")
        lines.append("|-------|-----------|--------|-----------|")
        for m in models:
            display = _model_display_name(m["name"])
            lines.append(
                f"| {display} | ~{m['vram_mb']:,} | {m['loaded_at']} | {m['last_used']} |"
            )
        if ltx_loaded:
            lines.append("| LTX-Web Video Pipeline | ~18,000 | (subprocess) | (subprocess) |")
        if vllm_ready and MODEL_DIFFUSIONGEMMA_VLLM not in tracked_names:
            lines.append("| DiffusionGemma vLLM | ~26,000 | (Docker) | (OpenAI API) |")
        if krea2_ready and MODEL_KREA2_TURBO_NVFP4 not in tracked_names:
            lines.append("| Krea2 Turbo (ComfyUI) | ~38,000 | (venv subprocess) | (ComfyUI API) |")
    else:
        lines.append("*No models currently loaded.*")

    if vllm_sleeping:
        lines.extend([
            "",
            "### Sleeping Services",
            "- DiffusionGemma vLLM is sleeping; the server process is retained and VRAM is mostly released.",
        ])

    return "\n".join(lines)

def _unload_model_action(model_key: str) -> str:
    """Unload a specific model and return refreshed status."""
    if not model_key or model_key == "(none)":
        return _build_models_md()
    
    if model_key == "ltx_video_pipeline":
        unload_ltx_video_pipeline()
    elif model_key == MODEL_DIFFUSIONGEMMA_VLLM:
        if model_mgr.is_loaded(MODEL_DIFFUSIONGEMMA_VLLM):
            model_mgr.unload(MODEL_DIFFUSIONGEMMA_VLLM)
        else:
            _diffusiongemma_vllm_service.stop()
    elif model_key == MODEL_KREA2_TURBO_NVFP4:
        if model_mgr.is_loaded(MODEL_KREA2_TURBO_NVFP4):
            model_mgr.unload(MODEL_KREA2_TURBO_NVFP4)
        else:
            _krea2_comfy_service.stop()
    else:
        model_mgr.unload(model_key)
        
    return _build_models_md()

def _unload_all_action() -> str:
    model_mgr.unload_all()
    unload_ltx_video_pipeline()
    if _diffusiongemma_vllm_service.is_healthy() or _diffusiongemma_vllm_service.is_control_reachable():
        _diffusiongemma_vllm_service.stop()
    if _krea2_comfy_service.is_healthy():
        _krea2_comfy_service.stop()
    return _build_models_md()

def _get_loaded_model_choices() -> list[str]:
    """Return the keys of currently loaded models for the dropdown."""
    keys = model_mgr.keys()
    
    video_healthy, is_pipeline_loaded = check_ltx_video_health()
    if video_healthy and is_pipeline_loaded:
        keys.append("ltx_video_pipeline")
    if _diffusiongemma_vllm_service.is_ready() and MODEL_DIFFUSIONGEMMA_VLLM not in keys:
        keys.append(MODEL_DIFFUSIONGEMMA_VLLM)
    if _krea2_comfy_service.is_healthy() and MODEL_KREA2_TURBO_NVFP4 not in keys:
        keys.append(MODEL_KREA2_TURBO_NVFP4)
        
    return keys if keys else ["(none)"]

def _build_vram_widget_md() -> str:
    """Render a compact VRAM status for the header widget."""
    gpu = model_mgr.gpu_summary()
    if not gpu["total_mb"]:
        return "**GPU VRAM:** N/A"
    used_pct = (gpu["used_mb"] / gpu["total_mb"] * 100) if gpu["total_mb"] else 0
    return (
        f"**VRAM Usage:**  \n"
        f"**{gpu['used_mb']:,}** / **{gpu['total_mb']:,}** MiB ({used_pct:.0f}%)  \n"
        f"Tracked: {gpu['tracked_mb']:,} MiB ({gpu['model_count']} models)"
    )

def refresh_models_tab():
    return _build_models_md(), gr.update(choices=_get_loaded_model_choices(), value=None)

def unload_model_and_refresh(key):
    return _unload_model_action(key), gr.update(choices=_get_loaded_model_choices(), value=None)

def unload_all_and_refresh():
    return _unload_all_action(), gr.update(choices=_get_loaded_model_choices(), value=None)

__all__ = (
    '_build_models_md',
    '_unload_model_action',
    '_unload_all_action',
    '_get_loaded_model_choices',
    '_build_vram_widget_md',
    'refresh_models_tab',
    'unload_model_and_refresh',
    'unload_all_and_refresh',
)
_seal_runtime_module(globals())
