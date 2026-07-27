"""FastAPI reverse proxy for the managed DiffusionGemma backend."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from .routing import promote_routes_before_fallback

log = logging.getLogger(__name__)

VLLM_PROXY_ROUTE_NAME = "image_studio_vllm_proxy"
VLLM_PROXY_HEALTH_ROUTE_NAME = "image_studio_vllm_proxy_health"
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class VllmProxyBackend(Protocol):
    api_base: str
    model: str

    def ensure_running(self) -> Any: ...

    def is_healthy(self) -> bool: ...

    def is_sleeping(self) -> bool: ...


def _vllm_proxy_error(message: Any, error_type: str = "proxy_error") -> dict[str, Any]:
    return {
        "error": {
            "message": str(message),
            "type": error_type,
            "param": None,
            "code": None,
        }
    }


def _vllm_proxy_authorized(headers: Any, api_key: str) -> bool:
    if not api_key:
        return True
    auth = str(headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer ") and auth[7:].strip() == api_key:
        return True
    return str(headers.get("x-api-key") or "").strip() == api_key


def _vllm_proxy_target_url(
    api_base: str,
    path: str,
    query: str = "",
) -> str:
    quoted_path = urllib.parse.quote(path.strip("/"), safe="/:@._~-")
    url = f"{api_base}/{quoted_path}"
    return f"{url}?{query}" if query else url


def _vllm_proxy_request_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
        and key.lower() != "accept-encoding"
    }


def _vllm_proxy_response_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _vllm_proxy_request_wants_stream(body: bytes, content_type: str) -> bool:
    if content_type.lower().startswith("text/event-stream"):
        return True
    if not body:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(isinstance(payload, dict) and payload.get("stream") is True)


def _vllm_proxy_open(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: int,
):
    request = urllib.request.Request(
        url,
        data=body or None,
        headers=headers,
        method=method,
    )
    response = urllib.request.urlopen(request, timeout=timeout)
    return response, int(getattr(response, "status", 200)), response.headers


def _vllm_proxy_iter_response(response: Any):
    try:
        while chunk := response.read(65536):
            yield chunk
    finally:
        response.close()


def attach_vllm_proxy_routes(
    blocks: Any,
    backend: VllmProxyBackend,
    *,
    request_timeout: int,
    api_key: str = "",
) -> bool:
    """Expose a managed DiffusionGemma backend as ``/v1/*``."""
    try:
        import asyncio

        from fastapi import Request
        from fastapi.responses import JSONResponse, Response, StreamingResponse
    except ImportError as exc:
        log.warning("Could not enable vLLM proxy routes: %s", exc)
        return False

    fastapi_app = getattr(blocks, "app", None)
    if fastapi_app is None:
        log.warning("Could not enable vLLM proxy routes: no FastAPI app.")
        return False

    route_names = {
        getattr(route, "name", "")
        for route in getattr(fastapi_app, "routes", [])
    }
    proxy_names = {VLLM_PROXY_ROUTE_NAME, VLLM_PROXY_HEALTH_ROUTE_NAME}
    if VLLM_PROXY_ROUTE_NAME in route_names:
        promote_routes_before_fallback(fastapi_app, proxy_names)
        return True

    async def vllm_proxy(path: str, request: Request):
        if not _vllm_proxy_authorized(request.headers, api_key):
            return JSONResponse(
                _vllm_proxy_error(
                    "Invalid or missing vLLM proxy API key.",
                    "authentication_error",
                ),
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        body = await request.body()
        target_url = _vllm_proxy_target_url(
            backend.api_base,
            path,
            request.url.query,
        )
        headers = _vllm_proxy_request_headers(request.headers)
        try:
            await asyncio.to_thread(backend.ensure_running)
            response, status_code, response_headers = await asyncio.to_thread(
                _vllm_proxy_open,
                request.method,
                target_url,
                headers,
                body,
                request_timeout,
            )
        except urllib.error.HTTPError as exc:
            detail = await asyncio.to_thread(exc.read)
            return Response(
                content=detail,
                status_code=exc.code,
                media_type=exc.headers.get("content-type", "application/json"),
                headers=_vllm_proxy_response_headers(exc.headers),
            )
        except Exception as exc:
            log.exception("vLLM proxy request failed for %s %s", request.method, target_url)
            return JSONResponse(
                _vllm_proxy_error(exc, "backend_error"),
                status_code=503,
            )

        content_type = response_headers.get("content-type", "application/json")
        response_headers = _vllm_proxy_response_headers(response_headers)
        if _vllm_proxy_request_wants_stream(body, content_type):
            return StreamingResponse(
                _vllm_proxy_iter_response(response),
                status_code=status_code,
                media_type=content_type,
                headers=response_headers,
            )
        data = await asyncio.to_thread(response.read)
        response.close()
        return Response(
            content=data,
            status_code=status_code,
            media_type=content_type,
            headers=response_headers,
        )

    async def vllm_proxy_health(request: Request):
        if not _vllm_proxy_authorized(request.headers, api_key):
            return JSONResponse(
                _vllm_proxy_error(
                    "Invalid or missing vLLM proxy API key.",
                    "authentication_error",
                ),
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        healthy = await asyncio.to_thread(backend.is_healthy)
        sleeping = await asyncio.to_thread(backend.is_sleeping)
        return JSONResponse(
            {
                "ok": healthy and not sleeping,
                "healthy": healthy,
                "sleeping": sleeping,
                "backend": backend.api_base,
                "model": backend.model,
            }
        )

    fastapi_app.add_api_route(
        "/v1/{path:path}",
        vllm_proxy,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        name=VLLM_PROXY_ROUTE_NAME,
        include_in_schema=False,
    )
    fastapi_app.add_api_route(
        "/vllm/health",
        vllm_proxy_health,
        methods=["GET"],
        name=VLLM_PROXY_HEALTH_ROUTE_NAME,
        include_in_schema=False,
    )
    promote_routes_before_fallback(fastapi_app, proxy_names)
    log.info(
        "vLLM proxy enabled at /v1/* -> %s (model=%s).",
        backend.api_base,
        backend.model,
    )
    return True


__all__ = ("attach_vllm_proxy_routes",)
