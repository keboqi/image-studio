"""Application route composition and stable public API metadata."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .designer import attach_ideogram_json_designer_route
from .proxy import VllmProxyBackend, attach_vllm_proxy_routes
from .routing import promote_routes_before_fallback

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


@dataclass(frozen=True)
class WebRouteDependencies:
    vllm_backend: VllmProxyBackend
    vllm_request_timeout: int


_dependencies: WebRouteDependencies | None = None


def configure_app_routes(dependencies: WebRouteDependencies) -> None:
    global _dependencies
    _dependencies = dependencies


def attach_model_catalog_route(
    app: Any,
    provider: Callable[[], dict[str, Any]],
) -> bool:
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


def attach_app_routes(
    app: Any,
    vllm_proxy: bool = False,
    api_key: str = "",
    model_catalog_provider: Callable[[], dict[str, Any]] | None = None,
) -> Any:
    """Attach optional FastAPI routes and return Gradio's underlying app."""
    attach_ideogram_json_designer_route(app)
    if vllm_proxy:
        if _dependencies is None:
            raise RuntimeError("Web route dependencies are not configured.")
        attach_vllm_proxy_routes(
            app,
            _dependencies.vllm_backend,
            request_timeout=_dependencies.vllm_request_timeout,
            api_key=api_key,
        )
    if model_catalog_provider is not None:
        attach_model_catalog_route(app, model_catalog_provider)
    return getattr(app, "app", None)


__all__ = (
    "MODEL_CATALOG_ROUTE_NAME",
    "PUBLIC_API_ENDPOINTS",
    "WebRouteDependencies",
    "attach_app_routes",
    "attach_model_catalog_route",
    "configure_app_routes",
    "promote_routes_before_fallback",
)
