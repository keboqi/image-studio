"""Shared route ordering and stable public API metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

PUBLIC_API_ENDPOINTS = (
    ("generate", 35, 4),
    ("edit", 18, 4),
    ("upscale", 9, 4),
    ("ai_remover", 3, 4),
    ("generate_video", 17, 3),
    ("upscale_video", 12, 3),
)


def promote_routes_before_fallback(app: Any, names: Iterable[str]) -> None:
    routes = getattr(getattr(app, "router", None), "routes", None)
    if not isinstance(routes, list):
        return
    names = set(names)
    promoted = [route for route in routes if getattr(route, "name", "") in names]
    if not promoted:
        return
    remaining = [route for route in routes if getattr(route, "name", "") not in names]
    insertion = len(remaining)
    for index, route in enumerate(remaining):
        path = str(getattr(route, "path", ""))
        if path in {"/{path:path}", "/{full_path:path}"} or path.endswith("{path:path}"):
            insertion = index
            break
    routes[:] = remaining[:insertion] + promoted + remaining[insertion:]


# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def attach_app_routes(app: Any, vllm_proxy: bool = False, api_key: str = "") -> Any:
    """Attach optional FastAPI routes and return Gradio's underlying app."""
    attach_ideogram_json_designer_route(app)
    if vllm_proxy:
        attach_vllm_proxy_routes(app, api_key=api_key)
    return getattr(app, "app", None)

__all__ = (
    'attach_app_routes',
)
_seal_runtime_module(globals())
