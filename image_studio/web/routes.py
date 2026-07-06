"""Shared route ordering and stable public API metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

PUBLIC_API_ENDPOINTS = (
    ("generate", 35, 4),
    ("edit", 18, 4),
    ("upscale", 9, 4),
    ("ai_remover", 3, 4),
    ("generate_video", 17, 3),
    ("upscale_video", 12, 3),
)

MODEL_CATALOG_PATH = "/api/models"
MODEL_CATALOG_ROUTE_NAME = "image_studio_model_catalog"


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


def attach_model_catalog_route(app: Any, provider: Callable[[], dict[str, Any]]) -> bool:
    """Attach an idempotent read-only model discovery endpoint."""
    fastapi_app = getattr(app, "app", None)
    router = getattr(fastapi_app, "router", None)
    routes = getattr(router, "routes", ())
    if fastapi_app is None or not callable(getattr(fastapi_app, "add_api_route", None)):
        return False
    if any(getattr(route, "name", "") == MODEL_CATALOG_ROUTE_NAME for route in routes):
        return True

    async def model_catalog() -> dict[str, Any]:
        return provider()

    fastapi_app.add_api_route(
        MODEL_CATALOG_PATH,
        model_catalog,
        methods=["GET"],
        name=MODEL_CATALOG_ROUTE_NAME,
        include_in_schema=True,
    )
    promote_routes_before_fallback(fastapi_app, {MODEL_CATALOG_ROUTE_NAME})
    return True


# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def attach_app_routes(
    app: Any,
    vllm_proxy: bool = False,
    api_key: str = "",
    model_catalog_provider: Callable[[], dict[str, Any]] | None = None,
) -> Any:
    """Attach optional FastAPI routes and return Gradio's underlying app."""
    attach_ideogram_json_designer_route(app)
    if vllm_proxy:
        attach_vllm_proxy_routes(app, api_key=api_key)
    if model_catalog_provider is not None:
        attach_model_catalog_route(app, model_catalog_provider)
    return getattr(app, "app", None)

__all__ = (
    'attach_app_routes',
    'attach_model_catalog_route',
)
_seal_runtime_module(globals())
