"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _promote_ideogram_json_designer_route(fastapi_app: Any) -> None:
    promote_routes_before_fallback(fastapi_app, {IDEOGRAM4_JSON_DESIGNER_ROUTE_NAME})

def attach_ideogram_json_designer_route(blocks: Any) -> bool:
    """Serve the standalone Ideogram JSON designer from the same Gradio origin."""
    try:
        from fastapi.responses import Response
    except ImportError as exc:
        log.warning("Could not enable Ideogram JSON designer route because FastAPI is unavailable: %s", exc)
        return False

    fastapi_app = getattr(blocks, "app", None)
    if fastapi_app is None:
        log.warning("Could not enable Ideogram JSON designer route because this Gradio Blocks has no FastAPI app.")
        return False

    route_names = {getattr(route, "name", "") for route in getattr(fastapi_app, "routes", [])}
    if IDEOGRAM4_JSON_DESIGNER_ROUTE_NAME in route_names:
        _promote_ideogram_json_designer_route(fastapi_app)
        return True

    async def ideogram_json_designer():
        if not os.path.isfile(IDEOGRAM4_JSON_DESIGNER_FILE):
            return Response("jsondesigner.html not found.", status_code=404, media_type="text/plain")
        with open(IDEOGRAM4_JSON_DESIGNER_FILE, "rb") as fh:
            return Response(fh.read(), media_type="text/html; charset=utf-8")

    fastapi_app.add_api_route(
        IDEOGRAM4_JSON_DESIGNER_PATH,
        ideogram_json_designer,
        methods=["GET"],
        name=IDEOGRAM4_JSON_DESIGNER_ROUTE_NAME,
        include_in_schema=False,
    )
    _promote_ideogram_json_designer_route(fastapi_app)
    log.info("Ideogram JSON designer enabled at %s.", IDEOGRAM4_JSON_DESIGNER_PATH)
    return True

__all__ = (
    '_promote_ideogram_json_designer_route',
    'attach_ideogram_json_designer_route',
)
_seal_runtime_module(globals())
