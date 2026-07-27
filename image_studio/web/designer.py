"""Serve the standalone Ideogram JSON designer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .routing import promote_routes_before_fallback

log = logging.getLogger(__name__)

IDEOGRAM_JSON_DESIGNER_FILE = Path(__file__).resolve().parents[2] / "jsondesigner.html"
IDEOGRAM_JSON_DESIGNER_PATH = "/ideogram4/json-designer"
IDEOGRAM_JSON_DESIGNER_ROUTE_NAME = "ideogram4_json_designer"


def attach_ideogram_json_designer_route(
    blocks: Any,
    *,
    designer_file: Path = IDEOGRAM_JSON_DESIGNER_FILE,
) -> bool:
    """Serve the designer from the same Gradio origin."""
    try:
        from fastapi.responses import Response
    except ImportError as exc:
        log.warning("Could not enable Ideogram JSON designer route: %s", exc)
        return False

    fastapi_app = getattr(blocks, "app", None)
    if fastapi_app is None:
        log.warning("Could not enable Ideogram JSON designer route: no FastAPI app.")
        return False

    route_names = {
        getattr(route, "name", "")
        for route in getattr(fastapi_app, "routes", [])
    }
    if IDEOGRAM_JSON_DESIGNER_ROUTE_NAME in route_names:
        promote_routes_before_fallback(
            fastapi_app,
            {IDEOGRAM_JSON_DESIGNER_ROUTE_NAME},
        )
        return True

    async def ideogram_json_designer():
        if not designer_file.is_file():
            return Response(
                "jsondesigner.html not found.",
                status_code=404,
                media_type="text/plain",
            )
        return Response(
            designer_file.read_bytes(),
            media_type="text/html; charset=utf-8",
        )

    fastapi_app.add_api_route(
        IDEOGRAM_JSON_DESIGNER_PATH,
        ideogram_json_designer,
        methods=["GET"],
        name=IDEOGRAM_JSON_DESIGNER_ROUTE_NAME,
        include_in_schema=False,
    )
    promote_routes_before_fallback(
        fastapi_app,
        {IDEOGRAM_JSON_DESIGNER_ROUTE_NAME},
    )
    log.info("Ideogram JSON designer enabled at %s.", IDEOGRAM_JSON_DESIGNER_PATH)
    return True


__all__ = ("attach_ideogram_json_designer_route",)
